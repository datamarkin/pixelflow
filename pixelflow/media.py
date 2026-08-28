"""
Media handling and display utilities for PixelFlow.

Provides purpose-built classes for reading video files, streaming from cameras,
and writing video output. Includes display utilities and image loading.

All image data uses **RGB** channel ordering throughout the library.
Conversion to/from OpenCV's BGR format happens at the I/O boundary.

## The contract

Everything here exists to serve one loop::

    video = pf.VideoReader("in.mp4")
    writer = pf.VideoWriter("out.mp4", like=video)

    for frame in video:
        detections = pf.from_ultralytics(model(frame))
        frame = pf.annotate.box(frame, detections)
        writer.write(frame)

Two rules follow from it, and they explain most of the decisions in this module.

**A source is anything that yields RGB uint8 HxWx3 arrays.** ``VideoReader`` and
``CameraStream`` are conveniences, not gates -- a list, a generator, or your own
reader for the one camera that needs special handling all work identically
everywhere else in PixelFlow.

**Size is discovered, rate is declared.** The loop is allowed to change a frame's
size -- resizing is the caller's business, and so is resizing their detections to
match -- so ``VideoWriter`` takes its size from the first frame it is given rather
than from the source. The loop cannot change the frame *rate*, so that has to be
stated, and ``like=`` exists to carry it across without arithmetic.

Reading and resizing are two jobs. Use ``pf.transform.resize`` for the second.

## What the errors mean

Reading something is a different kind of failure from being handed something, and
the distinction is load-bearing for a service that has to turn one into a status
code. One rule, everywhere in this module:

* ``FileNotFoundError`` -- the path is not there.
* ``IsADirectoryError`` -- the path is a directory.
* ``ValueError`` -- what you passed is not usable: the data is not an image or a
  video, an array is not RGB uint8 HxWx3, a frame rate is not a positive number.
  Bad input, not a broken program: this is the 400.
* ``TypeError`` -- the argument is not a kind of thing that could be an image.
* ``OSError`` -- the filesystem refused, for reasons that are not about the value.

A file that exists and is not what it claims to be raises ``ValueError`` whether it
was meant to be an image or a video, so one ``except`` covers both.

One honest imprecision: a live source that will not open is also ``ValueError``,
because ``cv2.VideoCapture`` reports failure as a bare ``False`` and cannot say
whether the URL was wrong (a bad value) or the device was busy or forbidden (not).
Given that it cannot be distinguished, it is reported as the more common case.
"""

import io
import math
import warnings
from pathlib import Path
from typing import Union, Optional, Iterator
import cv2
import numpy as np
from PIL import Image

from .transforms.image import resize


class DisplayExit(Exception):
    """Raised when the quit key is pressed during display."""
    pass


__all__ = [
    "DisplayExit",
    "VideoReader",
    "CameraStream",
    "VideoWriter",
    "read_image",
    "read_video",
    "encode_image",
    "display_video",
    "display_image",
    "save_image",
    "close_display",
    "to_pil",
    "from_pil",
]


class _Unset:
    """Sentinel for 'argument not supplied'.

    ``None`` cannot serve here: ``VideoWriter(path, fps=None)`` is a mistake worth a
    specific message, and it is a different mistake from omitting ``fps`` entirely.
    """

    def __repr__(self):
        return "<unset>"


_UNSET = _Unset()

# `width=` used to resize on the way through every reader, writer and image load. It
# is gone: it made the reader's own `.width` describe something other than the file,
# put the original frames out of reach, and existed only because `pf.transform` had
# no `resize`. It now does. The parameter survives as a sentinel purely so the error
# names the replacement -- a bare TypeError says what broke but not what to write.
_WIDTH_REMOVED = (
    "{cls} no longer resizes. Reading and resizing are two jobs, and combining them "
    "made .width describe something other than the source. Compose instead:\n\n"
    "    frame = pf.transform.resize(frame, width=640)\n"
)


def _reject_width(width, cls: str) -> None:
    """Raise a migration error if the removed ``width=`` parameter was passed."""
    if width is not _UNSET:
        raise TypeError(_WIDTH_REMOVED.format(cls=cls))


def _resolve_path(source: str) -> Path:
    """Return a local Path for source. Raises if it is missing or a directory.

    A directory is rejected explicitly -- ``exists()`` is true for one, so it would
    otherwise reach ``cv2.imread``, return None, and surface as a decode error. The
    message resolves the path because that is what reveals a bad *relative* path.
    """
    path = Path(source)
    if path.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"No such file: {source!r} (resolved to {path.resolve()})"
        )
    return path


# Only these can carry an alpha channel; JPEG cannot, so the header probe in
# read_image is skipped for it -- which is most of any real dataset.
_ALPHA_CAPABLE = frozenset({".png", ".gif", ".webp", ".tif", ".tiff", ".jp2"})


def _require_rgb(image, what: str = "frame") -> None:
    """Raise unless ``image`` is the RGB uint8 HxWx3 array this library passes around.

    ``cv2.cvtColor`` accepts a 2D or 4-channel array for ``RGB2BGR`` and quietly
    returns HxWx3, so a grayscale or RGBA array reaches an encoder as colour nonsense
    with nothing reported. Every boundary that hands an array to OpenCV goes through
    here, so the contract is stated once rather than per call site.

    Shape, dtype and channel count are checkable. Channel *order* is not -- RGB and
    BGR are the same bytes in a different sequence, and no inspection recovers which
    one you have. So every array entering this library is **taken** to be RGB, at
    every boundary, not just this one.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"expected an RGB uint8 HxWx3 {what}, got {type(image).__name__}"
        )
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"expected an RGB uint8 HxWx3 {what}, got shape {image.shape} "
            f"dtype {image.dtype}"
        )


def _warn_if_alpha_dropped(handle, label: str = "") -> None:
    """Warn if the source carried an alpha channel that decoding discarded.

    ``IMREAD_COLOR`` drops alpha without compositing, so transparent regions keep
    whatever colour was stored beneath them -- a real loss, and worth saying once
    rather than leaving to be discovered in the pixels.

    Reading the header costs no pixels, only a parse. Callers that know the format
    cannot carry alpha skip the call entirely; see ``_ALPHA_CAPABLE``.
    """
    try:
        with Image.open(handle) as probe:
            had_alpha = probe.mode in ("RGBA", "LA", "PA") or (
                probe.mode == "P" and "transparency" in probe.info
            )
    except Exception:
        return
    if had_alpha:
        warnings.warn(
            f"{label}alpha channel dropped. PixelFlow images are RGB uint8 HxWx3; "
            f"transparent areas keep the colour stored beneath them.",
            stacklevel=3,
        )


def _valid_fps(value) -> Optional[float]:
    """Return a usable frame rate, or None.

    OpenCV reports 0.0 for a rate it does not know, which is routine for cameras and
    common for network streams, and NaN for some malformed containers. Both flow
    happily into a VideoWriter and produce a file that will not play, so neither is
    allowed to masquerade as a number. None forces a caller to have a policy.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return value


class _FrameSource:
    """Shared machinery for the classes that own a ``cv2.VideoCapture``.

    Everything a consumer branches on lives here with the same name and the same
    meaning, so that swapping a file for a camera changes one line and nothing else.
    ``is_live`` is the only fact worth branching on.
    """

    # Declared here, not only in each subclass's __init__, so that the surface a
    # source must provide is visible in one place and a partial subclass answers
    # None rather than raising from inside a property.
    _cap: Optional[cv2.VideoCapture] = None
    _stride: int = 1
    _width: Optional[int] = None
    _height: Optional[int] = None
    _source_fps: Optional[float] = None
    _source_frames: Optional[int] = None
    _pending: Optional[np.ndarray] = None
    is_live: bool = False

    # -- facts ---------------------------------------------------------------

    @property
    def width(self) -> Optional[int]:
        """Width of the frames this source yields, in pixels."""
        return self._width

    @property
    def height(self) -> Optional[int]:
        """Height of the frames this source yields, in pixels."""
        return self._height

    @property
    def fps(self) -> Optional[float]:
        """Rate of the frames this source *yields*, or None if unknown.

        Already divided by ``stride``. Reading every 5th frame of a 25 fps file is a
        5 fps sequence, and a writer handed 25 would produce a file that plays five
        times too fast. Dividing here makes that mistake unconstructible rather than
        merely catchable.
        """
        if self._source_fps is None:
            return None
        return self._source_fps / self._stride

    @property
    def frames(self) -> Optional[int]:
        """Estimated number of frames this source will yield, or None if unknown.

        An estimate, deliberately named as one: several container formats derive the
        count from duration x rate rather than storing it. Good for a progress bar.
        Never use it to decide when to stop, or to preallocate.
        """
        if self._source_frames is None:
            return None
        return math.ceil(self._source_frames / self._stride)

    @property
    def duration(self) -> Optional[float]:
        """Length of the source in seconds, or None if unknown.

        Unaffected by ``stride`` -- striding changes how many frames you look at, not
        how much time the source covers, so this is always the source's own length.

        ``frames / fps`` approximates it but does not equal it: ``frames`` rounds up
        to count the frames you actually receive, so a source whose length is not a
        multiple of the stride reports up to one stride more time than it has. Use
        this property when you want the length; use ``frames`` for a progress total.
        """
        if self._source_fps is None or self._source_frames is None:
            return None
        return self._source_frames / self._source_fps

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # -- reading -------------------------------------------------------------

    def _retrieve_rgb(self) -> Optional[np.ndarray]:
        """Convert the most recently grabbed frame to RGB. None if it cannot be."""
        ok, frame = self._cap.retrieve()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _hand_out(self, frame: np.ndarray) -> np.ndarray:
        """Record a frame's size as fact, and return it.

        Container metadata and decoded reality normally agree -- OpenCV applies a
        rotation flag to both since 4.5, so a portrait phone video reports portrait
        dimensions. When they disagree the frame is right and the metadata is not,
        so the facts follow the frame rather than describing an array nobody
        received. Taking the size from every frame rather than only the first costs
        a tuple unpack against a full-frame colour conversion on the same line, and
        means a source that changes resolution mid-stream stays described correctly.
        """
        self._height, self._width = frame.shape[:2]
        return frame

    def read(self) -> Optional[np.ndarray]:
        """Grab a single RGB frame, ignoring ``stride``. None when the source ends."""
        if self._pending is not None:
            frame, self._pending = self._pending, None
            return frame
        if self._cap is None or not self._cap.grab():
            return None
        frame = self._retrieve_rgb()
        return None if frame is None else self._hand_out(frame)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield RGB frames from the current position until the source ends.

        Iteration does **not** rewind. Rewinding on every call made ``seek()``
        useless -- seek then iterate returned you to frame 0 -- and is impossible for
        a live source. Replay is ``source.seek(0)``, stated rather than implied.

        With a stride, skipped frames are grabbed but never retrieved or converted.
        Inter-frame coding means frame N is not decodable without its predecessors,
        so they must be walked; they need not be turned into arrays.
        """
        index = 0

        # A source that decoded a frame in order to learn its own size hands that
        # frame out here rather than discarding it. Dropping it would silently skip
        # a frame, and for a single-shot capture it is the only one there is.
        if self._pending is not None:
            frame, self._pending = self._pending, None
            yield self._hand_out(frame)
            index = 1

        while self._cap is not None:
            if not self._cap.grab():
                break
            if index % self._stride == 0:
                frame = self._retrieve_rgb()
                if frame is None:
                    break
                yield self._hand_out(frame)
            index += 1

    # -- removed surface, kept only to explain itself -------------------------

    def __len__(self):
        """Always raises. ``len()`` promised an exactness that cannot be kept.

        ``frames`` is an estimate for several container formats, so ``len(list(v))``
        could differ from ``len(v)``; and ``len()`` implies an indexable, re-iterable
        sequence, which a decode-forward stream is not.
        """
        raise TypeError(
            f"{type(self).__name__} has no len(): the frame count is an estimate, "
            f"not a length. Use .frames for a progress total, and iterate to count."
        )

    def __getattr__(self, name):
        if name == "frame_count":
            raise AttributeError(
                f"{type(self).__name__}.frame_count is now .frames, named as the "
                f"estimate it always was. It is divided by the stride."
            )
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    # -- cleanup -------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class VideoReader(_FrameSource):
    """Read a video file frame by frame.

    Args:
        source: Path to a video file.
        stride: Yield every Nth frame. ``fps`` and ``frames`` are divided to match,
            so a writer built with ``like=`` stays correct.
        start: Frame to begin at. Equivalent to ``seek(start)`` after opening.

    Example:
        >>> video = pf.VideoReader("input.mp4", stride=2)
        >>> writer = pf.VideoWriter("output.mp4", like=video)
        >>> for frame in video:
        ...     writer.write(frame)

    Note:
        Frames are yielded at the source's own resolution. To work at a smaller
        size, compose: ``pf.transform.resize(frame, width=640)``.
    """

    def __init__(self, source: str, *, stride: int = 1, start: int = 0,
                 width=_UNSET):
        _reject_width(width, "VideoReader")
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")

        path = _resolve_path(source)
        self._source = str(path)
        self._stride = stride

        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise ValueError(
                f"Could not open video file: {source} exists but is not a video "
                f"OpenCV can decode."
            )

        # Container metadata, read once. Nothing here is ever consulted per frame.
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._source_fps = _valid_fps(self._cap.get(cv2.CAP_PROP_FPS))
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._source_frames = count if count > 0 else None
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        self._codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        if start:
            self.seek(start)

    @property
    def codec(self) -> str:
        """FourCC of the source stream.

        Reported, not carried: a codec that decodes here is not necessarily one that
        *encodes* here, so passing it to a writer would make whether your output
        opens depend on what your input was.
        """
        return self._codec

    def seek(self, frame_number: int) -> None:
        """Jump to a frame. Iteration continues from there; ``seek(0)`` replays."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def __repr__(self):
        rate = "unknown fps" if self.fps is None else f"{self.fps:.2f}fps"
        count = "unknown length" if self.frames is None else f"{self.frames} frames"
        stride = "" if self._stride == 1 else f", stride={self._stride}"
        return (f"VideoReader({self._source!r}, {self.width}x{self.height}, "
                f"{rate}, {count}{stride})")


class CameraStream(_FrameSource):
    """Webcams and network streams.

    Presents the same facts and the same iteration contract as :class:`VideoReader`,
    so a consumer written against one works against the other and ``is_live`` is the
    only thing worth branching on.

    Args:
        source: Webcam index (int) or stream URL (str).
        stride: Yield every Nth frame.

    Example:
        >>> with pf.CameraStream(0) as cam:
        ...     for frame in cam:
        ...         pf.display_video(frame)

    Note:
        A live source has no reliable metadata, so ``width`` and ``height`` come from
        a frame decoded at open. ``fps`` is commonly ``None`` (devices report 0) and
        ``frames``/``duration`` are always ``None`` -- a stream has no length.
    """

    is_live = True

    def __init__(self, source: Union[int, str], *, stride: int = 1, width=_UNSET):
        _reject_width(width, "CameraStream")
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")

        self._source = source
        self._stride = stride

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise ValueError(
                f"Could not open camera/stream: {source!r}. The device or URL is "
                f"wrong, or it is already in use."
            )

        # A device's reported FRAME_WIDTH/HEIGHT is frequently the driver's default
        # rather than what it will hand over, so the only trustworthy source of the
        # size is a frame. One is decoded here and dropped: a camera is warming up at
        # this point anyway, and it buys facts that are complete and correct before
        # the first frame reaches the caller.
        ok, probe = self._cap.read()
        if ok:
            # Held, not discarded: the base loop hands a pending frame out first, so
            # learning the size costs no frame. _hand_out records it as fact there.
            self._pending = cv2.cvtColor(probe, cv2.COLOR_BGR2RGB)
            self._height, self._width = self._pending.shape[:2]
        else:
            self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None

        self._source_fps = _valid_fps(self._cap.get(cv2.CAP_PROP_FPS))
        self._source_frames = None  # a stream has no length, so duration is None too

        # Deferred, and marked rather than forgotten: a live source that blocks when
        # the consumer falls behind accumulates latency until the device's own buffer
        # overflows, so the frames being processed are minutes old. The policy for
        # that -- and reconnecting after a drop -- belongs on this class, not in the
        # consumer. Neither is implemented; there is no live consumer to design
        # against yet, and iteration is the base class's until there is.

    def __repr__(self):
        rate = "unknown fps" if self.fps is None else f"{self.fps:.2f}fps"
        return f"CameraStream({self._source!r}, {self.width}x{self.height}, {rate})"


class VideoWriter:
    """Write RGB frames to a video file.

    Exactly one of ``fps`` or ``like`` is required. ``like`` takes the rate from a
    source, which is the one value that has to cross from reader to writer and the
    one that is easy to get wrong -- a strided reader yields fewer frames per second
    than its file contains, and ``like`` carries the corrected rate.

    Size is not carried. It comes from the first frame written, because the loop
    between reader and writer is allowed to change it.

    Args:
        output_path: Path to the output video file.
        fps: Frames per second. Must be positive and finite.
        like: A source to take the frame rate from (anything with an ``fps``).
        codec: FourCC codec string. Default ``'mp4v'``.

    Example:
        >>> video = pf.VideoReader("input.mp4", stride=2)
        >>> with pf.VideoWriter("output.mp4", like=video) as writer:
        ...     for frame in video:
        ...         writer.write(frame)
    """

    def __init__(self, output_path: str, fps=_UNSET, codec: str = "mp4v", *,
                 like=None, width=_UNSET):
        _reject_width(width, "VideoWriter")

        if like is not None:
            if fps is not _UNSET:
                raise ValueError(
                    "VideoWriter() takes exactly one of fps= or like=; "
                    "like= already supplies the frame rate"
                )
            source_fps = getattr(like, "fps", _UNSET)
            if source_fps is _UNSET:
                raise TypeError(
                    f"like= expects a source with an .fps attribute, "
                    f"got {type(like).__name__}"
                )
            if source_fps is None:
                raise ValueError(
                    f"{type(like).__name__} reports no frame rate, so there is "
                    f"nothing to copy -- this is normal for cameras and streams. "
                    f"Pass fps= explicitly to state the rate you want."
                )
            fps = source_fps
        elif fps is _UNSET:
            raise ValueError(
                "VideoWriter() requires a frame rate: pass fps=, or like=<source> "
                "to take it from a reader"
            )

        # A rate that is zero, negative, NaN or infinite produces a file that will
        # not play. Rejecting it here rather than at the first write means the run
        # that would have wasted an hour fails on the line that was wrong.
        rate = _valid_fps(fps)
        if rate is None:
            raise ValueError(
                f"fps must be a positive finite number, got {fps!r}"
            )

        if not isinstance(codec, str) or len(codec) != 4:
            raise ValueError(
                f"codec must be a four-character FourCC string, got {codec!r}. "
                f"Try 'mp4v' or 'avc1'."
            )

        self._output_path = output_path
        self._fps = rate
        self._codec = codec
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[tuple] = None
        self._frames_written = 0

    # -- properties ----------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def is_opened(self) -> bool:
        return self._writer is not None and self._writer.isOpened()

    # -- write ---------------------------------------------------------------

    def write(self, frame: np.ndarray) -> None:
        """Write one RGB frame. The file's resolution is set by the first one.

        Raises:
            ValueError: If the frame is not RGB uint8 HxWx3, or if its size differs
                from the frame that opened the file.
        """
        _require_rgb(frame)
        height, width = frame.shape[:2]

        if self._writer is None:
            # Opened lazily because the only thing that knows the frame size is a
            # frame, and the loop above may have changed it.
            self._size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*self._codec)
            self._writer = cv2.VideoWriter(
                self._output_path, fourcc, self._fps, self._size
            )
            if not self._writer.isOpened():
                self._writer = None
                # Almost always the codec: a FourCC this OpenCV build cannot
                # encode. Named as a bad value so it joins the same except clause
                # as every other "what you passed will not work" in this module.
                raise ValueError(
                    f"Failed to open video writer for {self._output_path} "
                    f"(codec={self._codec!r}, fps={self._fps}, "
                    f"size={width}x{height}). The codec is most likely not "
                    f"available in this OpenCV build."
                )
        elif (width, height) != self._size:
            # cv2.VideoWriter drops a mismatched frame and reports nothing, so the
            # run ends with a short file and no error anywhere.
            raise ValueError(
                f"frame is {width}x{height} but {self._output_path!r} was opened at "
                f"{self._size[0]}x{self._size[1]}; every frame in a video file must "
                f"be the same size. Resize before writing, or open a second writer."
            )

        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self._frames_written += 1

    # -- cleanup -------------------------------------------------------------

    def close(self) -> None:
        """Finalize and release the video file."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return (f"VideoWriter({self._output_path!r}, "
                f"fps={self._fps}, frames={self._frames_written})")


# ---------------------------------------------------------------------------
# Reading and writing single images
# ---------------------------------------------------------------------------

def read_video(source: str, *, stride: int = 1, start: int = 0,
               width=_UNSET) -> "VideoReader":
    """Open a video file for frame-by-frame reading.

    Args:
        source: Path to a video file.
        stride: Yield every Nth frame.
        start: Frame to begin at.

    Returns:
        A VideoReader instance.
    """
    _reject_width(width, "read_video")
    return VideoReader(source, stride=stride, start=start)


def read_image(source, width=_UNSET) -> np.ndarray:
    """Return an RGB image from a path, an encoded buffer, or an array.

    Channel order is established at decode and invisible afterwards, so there has
    to be exactly one decoder. This is it: anything that can become a PixelFlow
    image comes through here, and everything that leaves is RGB ``uint8`` HxWx3.

    Args:
        source: One of --

            * ``str`` or ``Path`` -- a file on disk, decoded.
            * ``bytes``, ``bytearray`` or ``memoryview`` -- an encoded image, as an
              HTTP upload or a database blob gives it to you. Decoded in memory.
            * ``np.ndarray`` -- already pixels. Checked and returned unchanged.

    Returns:
        RGB uint8 array of shape (H, W, 3), always -- grayscale is expanded to
        three channels and any alpha channel is dropped, so that the result can go
        straight into the rest of PixelFlow.

    Raises:
        FileNotFoundError: If a path does not exist.
        IsADirectoryError: If a path is a directory.
        ValueError: If the data cannot be decoded, or an array is not RGB uint8
            HxWx3. Undecodable input is a bad *value*, not a runtime failure, which
            is what lets an HTTP layer turn it into a 400 without widening its
            catch.
        TypeError: If ``source`` is some other type.

    Note:
        **EXIF orientation is applied**, for a path and for a buffer alike -- they
        go through the same flag, so the same image does not come back rotated
        differently depending on which one you passed. A photo tagged as rotated is
        returned the way a viewer would show it, which matches what ``VideoReader``
        does with a rotated video and is what a model needs to produce upright
        coordinates. If annotations for an image were made against un-rotated pixels
        by a tool that ignores EXIF, their coordinates will not line up with this
        array -- the one case where the difference is visible, and worth checking
        before blaming the model.

        **An array is trusted, not verified.** This function decodes a path or a
        buffer, so it *knows* those results are RGB. An array's shape, dtype and
        channel count are checked, but channel *order* cannot be recovered from
        pixels -- so an array is taken to be RGB already and returned as given. If
        it came from ``cv2.imread`` it is BGR: pass the path instead and let this
        function decode it, which is what it is for.

        A URL is not a source. Reading a file must not make a network request --
        see the 0.4.0 note on why ``pf.assets`` was removed. Fetch it yourself and
        pass the bytes.
    """
    _reject_width(width, "read_image")

    # Already pixels. Nothing to decode, so the contract can only be checked, and
    # channel order is the one part of it that a check cannot reach.
    if isinstance(source, np.ndarray):
        _require_rgb(source, "image")
        return source

    # An encoded buffer: an upload, an S3 object, a blob. IMREAD_COLOR, not
    # IMREAD_UNCHANGED -- the latter silently ignores EXIF orientation, which would
    # hand back a sideways array for any phone photo and make this branch disagree
    # with the path branch about the very same file.
    if isinstance(source, (bytes, bytearray, memoryview)):
        buffer = np.frombuffer(source, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR) if buffer.size else None
        if image is None:
            raise ValueError(
                f"Could not decode image from {len(source)} bytes: the data is not "
                f"an image in a format OpenCV reads."
            )
        # No extension to gate on, so the probe always runs. It is a header parse,
        # measured at ~21us flat regardless of payload size.
        _warn_if_alpha_dropped(io.BytesIO(source))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if not isinstance(source, (str, Path)):
        raise TypeError(
            f"read_image() takes a path, an encoded buffer, or an RGB array; "
            f"got {type(source).__name__}"
        )

    path = _resolve_path(source)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"Could not decode image: {source} exists but is not an image in a "
            f"format OpenCV reads."
        )
    # A path carries its format in its name, so JPEG -- most of any real dataset --
    # skips the probe entirely.
    if path.suffix.lower() in _ALPHA_CAPABLE:
        _warn_if_alpha_dropped(path, f"{source}: ")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_image(image: np.ndarray, extension: str = ".png") -> bytes:
    """Encode an RGB image to bytes, for an HTTP response or a blob.

    The mirror of ``read_image``'s buffer branch, and it exists for the same reason:
    channel order is established at the codec boundary, so there has to be one
    encoder. Hand-rolling ``cv2.imencode`` means hand-rolling the RGB-to-BGR step
    that goes with it, and that is the step that gets forgotten.

    Args:
        image: RGB uint8 array of shape (H, W, 3).
        extension: Container/format extension, with or without the dot.

    Returns:
        The encoded bytes.

    Raises:
        ValueError: If the image is not RGB uint8 HxWx3, or OpenCV cannot encode
            the requested format.

    Example:
        >>> payload = pf.encode_image(frame, ".jpg")
        >>> return Response(payload, media_type="image/jpeg")
    """
    _require_rgb(image, "image")
    suffix = extension if extension.startswith(".") else f".{extension}"

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        success, buffer = cv2.imencode(suffix, bgr)
    except cv2.error as error:
        raise ValueError(
            f"Cannot encode {suffix!r}: OpenCV does not support that format"
        ) from error
    if not success:
        raise ValueError(f"Failed to encode image as {suffix!r}")
    return buffer.tobytes()


def save_image(path: str, image: np.ndarray) -> None:
    """Save an RGB numpy array to an image file.

    Args:
        path: Output file path (extension determines format).
        image: RGB numpy array.

    Raises:
        NotADirectoryError: If the parent directory does not exist.
        ValueError: If the image is not RGB uint8 HxWx3, or the extension is
            missing or not one OpenCV can write.
        OSError: If the file cannot be written for any other reason.
    """
    _require_rgb(image, "image")
    target = Path(path)

    # cv2.imwrite returns False for a missing directory and raises cv2.error for an
    # unknown extension. Both used to surface as "Failed to write image", which
    # names the symptom and never the cause.
    parent = target.parent
    if not parent.exists():
        raise NotADirectoryError(
            f"Cannot write {path!r}: directory {parent.resolve()} does not exist"
        )
    if not target.suffix:
        raise ValueError(
            f"Cannot write {path!r}: no file extension, so there is no way to know "
            f"what format to encode. Try '{target.name}.jpg' or '{target.name}.png'."
        )

    # Encoding goes through encode_image rather than cv2.imwrite, so that the
    # RGB-to-BGR step and the frame guard exist once. A second encoder here would
    # be exactly the duplication encode_image was added to remove.
    try:
        payload = encode_image(image, target.suffix)
    except ValueError as error:
        # The image was checked above, so anything left is the format.
        raise ValueError(
            f"Cannot write {path!r}: OpenCV cannot encode '{target.suffix}' files"
        ) from error
    target.write_bytes(payload)


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def _show(image: np.ndarray, window_name: str, width: Optional[int]) -> None:
    """Shrink for display if asked, convert to BGR, and show. Never mutates input."""
    if width is not None:
        image = resize(image, width=width)
    cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def display_video(frame: np.ndarray, window_name: str = "PixelFlow",
                  wait_key: int = 1, width: Optional[int] = None,
                  quit_key: Optional[int] = ord('q')) -> Optional[int]:
    """Display a frame in a video loop (non-blocking).

    Args:
        frame: RGB numpy array to display.
        window_name: Name of the display window. Default "PixelFlow".
        wait_key: Milliseconds to wait for key press. Default 1.
        width: Optional display width. Shrinks what is shown, nothing else.
        quit_key: Key code that triggers DisplayExit. Default ``ord('q')``.
            Set to ``None`` to disable auto-quit.

    Returns:
        The key code (int) if a key was pressed, otherwise None.

    Raises:
        DisplayExit: When the quit key is pressed.

    Note:
        ``width`` survives here, where it was removed from the readers and the
        writer, because a display is the end of the line. Shrinking a 4K frame to
        fit a laptop screen cannot affect anything downstream -- there is no
        downstream. The array you passed in is not modified.
    """
    _show(frame, window_name, width)
    key = cv2.waitKey(wait_key) & 0xFF
    if key == 255:
        return None
    if quit_key is not None and key == quit_key:
        cv2.destroyWindow(window_name)
        raise DisplayExit()
    return key


def display_image(image: np.ndarray, window_name: str = "PixelFlow",
                  width: Optional[int] = None) -> int:
    """Display an image and wait for any key press to close.

    Args:
        image: RGB numpy array to display.
        window_name: Name of the display window. Default "PixelFlow".
        width: Optional display width. Shrinks what is shown, nothing else.

    Returns:
        The key code pressed to dismiss the window.
    """
    _show(image, window_name, width)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(window_name)
    return key


def close_display() -> None:
    """Close all OpenCV display windows."""
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Format conversion helpers
# ---------------------------------------------------------------------------

def to_pil(image: np.ndarray) -> Image.Image:
    """Convert an RGB numpy array to a PIL Image."""
    return Image.fromarray(image)


def from_pil(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an RGB numpy array."""
    return np.array(image)
