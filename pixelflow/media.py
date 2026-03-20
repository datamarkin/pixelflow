"""
Media handling and display utilities for PixelFlow.

Provides purpose-built classes for reading video files, streaming from cameras,
and writing video output. Includes display utilities and image loading.

All image data uses **RGB** channel ordering throughout the library.
Conversion to/from OpenCV's BGR format happens at the I/O boundary.
"""

from pathlib import Path
from typing import Union, Optional, Iterator
import cv2
import numpy as np
from PIL import Image

from pixelflow import assets

__all__ = [
    "VideoReader",
    "CameraStream",
    "VideoWriter",
    "read_image",
    "read_video",
    "display_video",
    "display_image",
    "save_image",
    "close_display",
    "to_pil",
    "from_pil",
]


def _resize_frame(frame: np.ndarray, width: Optional[int]) -> np.ndarray:
    """Resize maintaining aspect ratio. Returns frame unchanged if width is None."""
    if width is None:
        return frame
    h, w = frame.shape[:2]
    height = int(h * width / w)
    return cv2.resize(frame, (width, height))


def _resolve_path(source: str) -> Path:
    """Return a local Path for source, downloading via assets if needed."""
    path = Path(source)
    if path.exists():
        return path
    try:
        return assets.download(source)
    except Exception:
        raise FileNotFoundError(
            f"File not found locally and download failed: {source}"
        )


def read_video(source: str, width: Optional[int] = None) -> "VideoReader":
    """Open a video file for frame-by-frame reading.

    Args:
        source: Path to a video file.
        width: Optional width for aspect-ratio frame resizing.

    Returns:
        A VideoReader instance.
    """
    return VideoReader(source, width=width)


def read_image(source: str, width: Optional[int] = None) -> np.ndarray:
    """Load a single image from disk with optional resizing.

    Args:
        source: Path to the image file.
        width: Optional width for aspect-ratio resize.

    Returns:
        RGB numpy array.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If OpenCV cannot decode the file.
    """
    path = _resolve_path(source)
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to decode image: {source}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return _resize_frame(image, width)


class VideoReader:
    """Read video files frame by frame.

    Args:
        source: Path to a video file.
        width: Optional width for aspect-ratio frame resizing.

    Example:
        >>> video = pf.VideoReader("input.mp4", width=640)
        >>> for frame in video:
        ...     process(frame)
    """

    def __init__(self, source: str, width: Optional[int] = None):
        self._cap = None
        path = _resolve_path(source)
        self._source = str(path)
        self._resize_width = width
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video file: {source}")

        # Cache raw metadata once
        self._raw_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._raw_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        self._codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    # -- properties ----------------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        """Post-resize width (what the user gets)."""
        if self._resize_width is not None:
            return self._resize_width
        return self._raw_width

    @property
    def height(self) -> int:
        """Post-resize height (what the user gets)."""
        if self._resize_width is not None:
            return int(self._raw_height * self._resize_width / self._raw_width)
        return self._raw_height

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        if self._fps > 0:
            return self._frame_count / self._fps
        return 0.0

    @property
    def codec(self) -> str:
        return self._codec

    # -- iteration / seek ----------------------------------------------------

    def __len__(self) -> int:
        return self._frame_count

    def _read_rgb_frame(self) -> Optional[np.ndarray]:
        """Read one frame, convert BGR→RGB, and resize."""
        ret, frame = self._cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return _resize_frame(frame, self._resize_width)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Iterate over frames. Resets to frame 0 on each call (replayable)."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            frame = self._read_rgb_frame()
            if frame is None:
                break
            yield frame

    def seek(self, frame_number: int) -> None:
        """Jump to a specific frame number."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # -- cleanup -------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying VideoCapture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def __repr__(self):
        return (f"VideoReader({self._source!r}, "
                f"{self.width}x{self.height}, "
                f"{self.fps:.2f}fps, {self.frame_count} frames)")


class CameraStream:
    """Webcams and network streams.

    Args:
        source: Webcam index (int) or stream URL (str).
        width: Optional width for aspect-ratio frame resizing.

    Example:
        >>> cam = pf.CameraStream(0, width=640)
        >>> for frame in cam:
        ...     if pf.display_video(frame) == ord('q'):
        ...         break
        >>> cam.close()
    """

    def __init__(self, source: Union[int, str], width: Optional[int] = None):
        self._source = source
        self._resize_width = width
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera/stream: {source}")

    # -- properties ----------------------------------------------------------

    @property
    def fps(self) -> float:
        """Best-effort FPS reported by the device/stream."""
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def width(self) -> int:
        if self._resize_width is not None:
            return self._resize_width
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        raw_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self._resize_width is not None and raw_w > 0:
            return int(raw_h * self._resize_width / raw_w)
        return raw_h

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # -- read / iterate ------------------------------------------------------

    def _read_rgb_frame(self) -> Optional[np.ndarray]:
        """Read one frame, convert BGR→RGB, and resize."""
        ret, frame = self._cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return _resize_frame(frame, self._resize_width)

    def read(self) -> Optional[np.ndarray]:
        """Grab a single frame. Returns None on failure."""
        return self._read_rgb_frame()

    def __iter__(self) -> Iterator[np.ndarray]:
        """Infinite iteration. Skips dropped frames, stops when stream closes."""
        while self._cap.isOpened():
            frame = self._read_rgb_frame()
            if frame is None:
                break
            yield frame

    # -- cleanup -------------------------------------------------------------

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def __repr__(self):
        return f"CameraStream({self._source!r}, {self.width}x{self.height})"


class VideoWriter:
    """Write frames to a video file.

    Args:
        output_path: Path to the output video file.
        fps: Frames per second (required).
        codec: FourCC codec string. Default ``'mp4v'``.
        width: Optional width for resize-on-write.

    Example:
        >>> writer = pf.VideoWriter("output.mp4", fps=30.0)
        >>> writer.write(frame)
        >>> writer.close()
    """

    def __init__(self, output_path: str, fps: float, codec: str = "mp4v",
                 width: Optional[int] = None):
        self._output_path = output_path
        self._fps = fps
        self._codec = codec
        self._resize_width = width
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_written = 0

    # -- properties ----------------------------------------------------------

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def is_opened(self) -> bool:
        return self._writer is not None and self._writer.isOpened()

    # -- write ---------------------------------------------------------------

    def write(self, frame: np.ndarray) -> None:
        """Write an RGB frame. Resolution auto-detected from first frame."""
        frame = _resize_frame(frame, self._resize_width)
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self._codec)
            self._writer = cv2.VideoWriter(
                self._output_path, fourcc, self._fps, (w, h)
            )
            if not self._writer.isOpened():
                raise RuntimeError(
                    f"Failed to open video writer for {self._output_path}"
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

    def __del__(self):
        self.close()

    def __repr__(self):
        return (f"VideoWriter({self._output_path!r}, "
                f"fps={self._fps}, frames={self._frames_written})")


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def display_video(frame: np.ndarray, window_name: str = "PixelFlow",
                  wait_key: int = 1, width: Optional[int] = None) -> Optional[int]:
    """Display a frame in a video loop (non-blocking).

    Args:
        frame: RGB numpy array to display.
        window_name: Name of the display window. Default "PixelFlow".
        wait_key: Milliseconds to wait for key press. Default 1.
        width: Optional display resize width.

    Returns:
        The key code (int) if a key was pressed, otherwise None.
    """
    display_frame = _resize_frame(frame, width)
    cv2.imshow(window_name, cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(wait_key) & 0xFF
    if key == 255:
        return None
    return key


def display_image(image: np.ndarray, window_name: str = "PixelFlow",
                  width: Optional[int] = None) -> int:
    """Display an image and wait for any key press to close.

    Args:
        image: RGB numpy array to display.
        window_name: Name of the display window. Default "PixelFlow".
        width: Optional display resize width.

    Returns:
        The key code pressed to dismiss the window.
    """
    display = _resize_frame(image, width)
    cv2.imshow(window_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(window_name)
    return key


def close_display() -> None:
    """Close all OpenCV display windows."""
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Format conversion helpers
# ---------------------------------------------------------------------------

def save_image(path: str, image: np.ndarray) -> None:
    """Save an RGB numpy array to an image file.

    Args:
        path: Output file path (extension determines format).
        image: RGB numpy array.

    Raises:
        RuntimeError: If the write fails.
    """
    success = cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError(f"Failed to write image: {path}")


def to_pil(image: np.ndarray) -> Image.Image:
    """Convert an RGB numpy array to a PIL Image."""
    return Image.fromarray(image)


def from_pil(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an RGB numpy array."""
    return np.array(image)
