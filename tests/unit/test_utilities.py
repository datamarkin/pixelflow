"""
Unit tests for pixelflow utility modules.

Tests Media, Buffer, SlicedInference, Smoother, Timer, and other
utility functionality.
"""

import pytest
import numpy as np
import cv2
import time
from unittest.mock import patch

from pathlib import Path

import pixelflow as pf


# ============================================================================
# VideoReader Tests
# ============================================================================

def decoded_frames(path):
    """Every frame of a file, as OpenCV actually decodes it.

    Frame identity has to be checked against this rather than against whatever was
    written: mp4v is lossy, so a frame authored as solid 50 comes back as 46. What a
    seek or a stride test needs to know is *which* frame arrived, and comparing to
    the decoded truth answers that exactly on every platform.
    """
    capture = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


class TestVideoReader:
    """Tests for VideoReader class."""

    def test_facts(self, temp_video_path):
        """Metadata is read once at open and describes what iteration yields."""
        with pf.VideoReader(temp_video_path) as video:
            assert video.frames == 10
            assert video.fps == 30.0
            assert video.width == 640
            assert video.height == 480
            assert video.duration == pytest.approx(10 / 30.0)
            assert video.is_live is False
            assert isinstance(video.codec, str) and len(video.codec) == 4

    def test_facts_match_the_frames_actually_yielded(self, temp_video_path,
                                                    portrait_video_path):
        """width/height describe the array you receive, on both orientations.

        Portrait is the case that catches a transposed size; a landscape-only
        suite cannot distinguish width from height.
        """
        for path in (temp_video_path, portrait_video_path):
            with pf.VideoReader(path) as video:
                frame = next(iter(video))
                assert (video.width, video.height) == (frame.shape[1], frame.shape[0])

    def test_iteration(self, temp_video_path):
        """Iterating yields RGB frames at the source's own resolution."""
        with pf.VideoReader(temp_video_path) as video:
            frames = list(video)
        assert len(frames) == 10
        assert all(f.shape == (480, 640, 3) and f.dtype == np.uint8 for f in frames)

    def test_iteration_does_not_rewind(self, counted_video_path):
        """Iteration continues from the current position rather than resetting.

        Rewinding on every __iter__ made seek() unobservable and is impossible for
        a live source. Replay is an explicit seek(0).
        """
        with pf.VideoReader(counted_video_path) as video:
            assert sum(1 for _ in video) == 20
            assert sum(1 for _ in video) == 0      # exhausted, not restarted
            video.seek(0)
            assert sum(1 for _ in video) == 20     # replay is explicit

    def test_seek_then_iterate_starts_there(self, counted_video_path):
        """seek(n) then iterating begins at frame n, not at 0."""
        truth = decoded_frames(counted_video_path)
        with pf.VideoReader(counted_video_path) as video:
            video.seek(7)
            assert np.array_equal(next(iter(video)), truth[7])

    def test_start_parameter(self, counted_video_path):
        """start= is seek() applied at open."""
        truth = decoded_frames(counted_video_path)
        with pf.VideoReader(counted_video_path, start=5) as video:
            assert np.array_equal(next(iter(video)), truth[5])

    def test_stride_divides_the_rate_and_the_count(self, counted_video_path):
        """fps and frames describe the strided stream, so a sink cannot be misfed.

        Reading every 5th frame of a 25 fps file is a 5 fps sequence. A writer
        handed 25 would produce a file that plays five times too fast.
        """
        with pf.VideoReader(counted_video_path, stride=5) as video:
            assert video.fps == 5.0
            assert video.frames == 4                 # ceil(20 / 5)
            assert sum(1 for _ in video) == video.frames

    @pytest.mark.parametrize("stride", [2, 3, 5, 7])
    def test_stride_leaves_duration_alone(self, counted_video_path, stride):
        """Striding changes how many frames you look at, not the source's length."""
        with pf.VideoReader(counted_video_path) as plain, \
                pf.VideoReader(counted_video_path, stride=stride) as strided:
            assert strided.duration == plain.duration

    @pytest.mark.parametrize("stride,frames", [(5, 4), (3, 7), (7, 3)])
    def test_frames_counts_what_arrives_even_when_it_does_not_divide(
            self, counted_video_path, stride, frames):
        """frames rounds up, because a partial stride still yields a frame.

        The consequence, which duration's docstring states: frames / fps only
        approximates duration. 20 frames at stride 3 yields 7, and 7 / (25/3) is
        0.84s against a real 0.8s. The source's length is duration's answer to give,
        not this one's.
        """
        with pf.VideoReader(counted_video_path, stride=stride) as video:
            assert video.frames == frames
            assert sum(1 for _ in video) == frames

    def test_context_manager(self, temp_video_path):
        with pf.VideoReader(temp_video_path) as video:
            assert video.is_opened
        assert not video.is_opened

    def test_invalid_path(self):
        """A path that is not on disk raises FileNotFoundError, without a network call."""
        with pytest.raises(FileNotFoundError):
            pf.VideoReader("nonexistent_file.mp4")

    def test_directory_path(self, tmp_path):
        """A directory raises IsADirectoryError rather than a decode failure."""
        with pytest.raises(IsADirectoryError):
            pf.VideoReader(str(tmp_path))

    def test_no_len(self, temp_video_path):
        """len() promised an exactness the frame count cannot keep."""
        with pf.VideoReader(temp_video_path) as video:
            with pytest.raises(TypeError, match="estimate"):
                len(video)

    def test_frame_count_renamed(self, temp_video_path):
        """The old name points at the new one."""
        with pf.VideoReader(temp_video_path) as video:
            with pytest.raises(AttributeError, match=r"\.frames"):
                video.frame_count


class TestUnknownFacts:
    """What the reader does when the container will not say.

    None is the whole point: OpenCV reports 0.0 for a rate it does not know, which
    is routine for cameras and common for streams, and 0.0 flows into a writer and
    produces a file that will not play. None cannot be mistaken for a number, so it
    forces a caller to have a policy.
    """

    @pytest.mark.parametrize("reported", [0.0, -1.0, float("nan"), float("inf"),
                                          None, "not a number"])
    def test_unusable_rates_become_none(self, reported):
        assert pf.media._valid_fps(reported) is None

    @pytest.mark.parametrize("reported", [25, 29.97, 1.0])
    def test_usable_rates_survive(self, reported):
        assert pf.media._valid_fps(reported) == pytest.approx(reported)

    def test_unknown_rate_makes_duration_unknown_too(self, temp_video_path,
                                                     monkeypatch):
        """duration is derived, so it cannot be more certain than what it is from."""
        with pf.VideoReader(temp_video_path) as video:
            monkeypatch.setattr(video, "_source_fps", None)
            assert video.fps is None
            assert video.duration is None
            assert sum(1 for _ in video) == 10      # still perfectly readable

    def test_unknown_length_makes_frames_none(self, temp_video_path, monkeypatch):
        with pf.VideoReader(temp_video_path) as video:
            monkeypatch.setattr(video, "_source_frames", None)
            assert video.frames is None
            assert video.duration is None

    def test_unknown_facts_are_readable_in_repr(self, temp_video_path, monkeypatch):
        with pf.VideoReader(temp_video_path) as video:
            monkeypatch.setattr(video, "_source_fps", None)
            monkeypatch.setattr(video, "_source_frames", None)
            assert "unknown fps" in repr(video)
            assert "unknown length" in repr(video)


class TestFactsFollowTheFrame:
    """When metadata and a decoded frame disagree, the frame wins.

    OpenCV applies a container's rotation flag to both the reported size and the
    decoded array, so on a supported version they agree even for portrait phone
    footage. This is the belt to that braces: if a source ever reports a size it
    does not deliver, the facts correct themselves on the first frame handed out
    rather than describing an array nobody received.
    """

    def test_size_corrects_itself_on_the_first_frame(self, temp_video_path,
                                                     monkeypatch):
        with pf.VideoReader(temp_video_path) as video:
            monkeypatch.setattr(video, "_width", 480)     # transposed, as a
            monkeypatch.setattr(video, "_height", 640)    # rotation flag would do

            frame = next(iter(video))

            assert (video.width, video.height) == (640, 480)
            assert (video.width, video.height) == (frame.shape[1], frame.shape[0])


class TestUndecodableSources:
    """A file that exists but is not what it claims to be."""

    def test_unreadable_video_is_not_a_missing_file(self, tmp_path):
        path = tmp_path / "not_really.mp4"
        path.write_text("this is not a video")
        with pytest.raises(RuntimeError, match="Could not open video"):
            pf.VideoReader(str(path))

    def test_unreadable_stream(self, tmp_path):
        path = tmp_path / "not_really.mp4"
        path.write_text("this is not a stream")
        with pytest.raises(RuntimeError, match="Could not open camera/stream"):
            pf.CameraStream(str(path))


# ============================================================================
# The shared source contract
# ============================================================================

@pytest.mark.parametrize("source_class", [pf.VideoReader, pf.CameraStream],
                         ids=["VideoReader", "CameraStream"])
class TestSharedSourceContract:
    """What every source promises, asserted once against each of them.

    The design claim is that swapping a file for a camera changes one line and
    nothing else. Copies of these tests, one per class, would let the two drift
    apart silently -- which is exactly the failure the shared base exists to
    prevent -- so the contract is stated once and parametrized.

    CameraStream is driven by a file path here: cv2.VideoCapture accepts one
    wherever it accepts a device, and the class cannot tell the difference.
    """

    def test_reports_the_same_facts(self, temp_video_path, source_class):
        with source_class(temp_video_path) as source:
            for name in ("width", "height", "fps", "frames", "duration", "is_live"):
                assert hasattr(source, name), f"{source_class.__name__} lacks {name}"
            assert isinstance(source.is_live, bool)

    def test_yields_rgb_uint8_hwc3(self, temp_video_path, source_class):
        with source_class(temp_video_path) as source:
            frame = next(iter(source))
        assert frame.dtype == np.uint8
        assert frame.ndim == 3 and frame.shape[2] == 3

    def test_facts_describe_the_frames_yielded(self, portrait_video_path,
                                               source_class):
        with source_class(portrait_video_path) as source:
            frame = next(iter(source))
            assert (source.width, source.height) == (frame.shape[1], frame.shape[0])

    def test_iterates_every_frame(self, temp_video_path, source_class):
        with source_class(temp_video_path) as source:
            assert sum(1 for _ in source) == 10

    def test_stride_yields_every_nth_frame(self, counted_video_path, source_class):
        """Skipped frames are grabbed but never retrieved or converted."""
        truth = decoded_frames(counted_video_path)
        with source_class(counted_video_path, stride=5) as source:
            got = list(source)
        assert len(got) == 4
        assert all(np.array_equal(a, truth[i]) for a, i in zip(got, [0, 5, 10, 15]))

    def test_stride_divides_the_rate(self, counted_video_path, source_class):
        with source_class(counted_video_path, stride=5) as source:
            assert source.fps == 5.0

    def test_stride_must_be_positive(self, temp_video_path, source_class):
        with pytest.raises(ValueError, match="stride"):
            source_class(temp_video_path, stride=0)

    def test_read_returns_frames_then_none(self, temp_video_path, source_class):
        """read() is on the base, so the webcam idiom works for a file too."""
        with source_class(temp_video_path) as source:
            assert source.read().shape == (480, 640, 3)
            for _ in range(9):
                source.read()
            assert source.read() is None

    def test_width_parameter_removed(self, temp_video_path, source_class):
        """The removed resize parameter names its replacement."""
        with pytest.raises(TypeError, match="transform.resize"):
            source_class(temp_video_path, width=320)

    def test_close_is_idempotent(self, temp_video_path, source_class):
        source = source_class(temp_video_path)
        source.close()
        source.close()
        assert not source.is_opened


# ============================================================================
# CameraStream Tests
# ============================================================================

class TestCameraStream:
    """Tests for CameraStream.

    Driven by a file: cv2.VideoCapture accepts a path wherever it accepts a device,
    and the class cannot tell the difference, so the whole contract is testable
    without hardware.
    """

    def test_size_comes_from_a_decoded_frame(self, portrait_video_path):
        """A live source has no trustworthy metadata, so a frame is decoded at open."""
        with pf.CameraStream(portrait_video_path) as cam:
            assert (cam.width, cam.height) == (480, 640)

    def test_has_no_length(self, temp_video_path):
        """A stream has no end, so neither count nor duration can be reported."""
        with pf.CameraStream(temp_video_path) as cam:
            assert cam.frames is None
            assert cam.duration is None

    def test_probe_frame_is_not_lost(self, temp_video_path):
        """The frame decoded at open is the first one handed out, not a discard."""
        with pf.CameraStream(temp_video_path) as cam:
            assert sum(1 for _ in cam) == 10


class TestReadVideo:
    """read_video is VideoReader by another name, kept for symmetry with read_image."""

    def test_returns_a_reader(self, temp_video_path):
        with pf.read_video(temp_video_path) as video:
            assert isinstance(video, pf.VideoReader)
            assert video.frames == 10

    def test_forwards_stride_and_start(self, counted_video_path):
        truth = decoded_frames(counted_video_path)
        with pf.read_video(counted_video_path, stride=5, start=5) as video:
            got = list(video)
        assert len(got) == 3                                  # frames 5, 10, 15
        assert all(np.array_equal(a, truth[i]) for a, i in zip(got, [5, 10, 15]))

    def test_width_parameter_removed(self, temp_video_path):
        with pytest.raises(TypeError, match="transform.resize"):
            pf.read_video(temp_video_path, width=320)


# ============================================================================
# VideoWriter Tests
# ============================================================================

class TestVideoWriter:
    """Tests for VideoWriter class."""

    def test_basic(self, tmp_path, sample_image):
        output = str(tmp_path / "output.mp4")
        with pf.VideoWriter(output, fps=30.0) as writer:
            for _ in range(5):
                writer.write(sample_image)
            assert writer.frames_written == 5
            assert writer.is_opened
        assert not writer.is_opened
        with pf.VideoReader(output) as video:
            assert video.frames == 5

    def test_size_comes_from_the_first_frame(self, tmp_path):
        """The loop may resize, so the writer takes its size from what it is given."""
        output = str(tmp_path / "sized.mp4")
        with pf.VideoWriter(output, fps=25.0) as writer:
            writer.write(np.zeros((360, 640, 3), dtype=np.uint8))
        with pf.VideoReader(output) as video:
            assert (video.width, video.height) == (640, 360)

    def test_mismatched_size_raises(self, tmp_path):
        """cv2.VideoWriter drops a mismatched frame and reports nothing.

        The result is a short file with no error anywhere -- discovered, if at all,
        hours later. This is the single most expensive silent failure in the module.
        """
        writer = pf.VideoWriter(str(tmp_path / "mismatch.mp4"), fps=25.0)
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="same size"):
            writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
        writer.close()

    @pytest.mark.parametrize("frame,description", [
        (np.zeros((480, 640, 4), dtype=np.uint8), "4 channels"),
        (np.zeros((480, 640), dtype=np.uint8), "grayscale"),
        (np.zeros((480, 640, 3), dtype=np.float32), "float32"),
        ([[0, 0, 0]], "not an array"),
    ])
    def test_rejects_non_rgb_frames(self, tmp_path, frame, description):
        """cv2.cvtColor accepts these and returns HxWx3 colour nonsense silently."""
        writer = pf.VideoWriter(str(tmp_path / f"{description}.mp4"), fps=25.0)
        with pytest.raises(ValueError, match="RGB uint8"):
            writer.write(frame)
        writer.close()

    @pytest.mark.parametrize("fps", [0.0, -5.0, float("nan"), float("inf"), None])
    def test_bad_rate_raises_at_construction(self, tmp_path, fps):
        """A rate that cannot produce a playable file fails on the line that set it."""
        with pytest.raises(ValueError, match="positive finite"):
            pf.VideoWriter(str(tmp_path / "bad.mp4"), fps=fps)

    def test_requires_a_rate(self, tmp_path):
        with pytest.raises(ValueError, match="requires a frame rate"):
            pf.VideoWriter(str(tmp_path / "none.mp4"))

    def test_like_carries_the_rate(self, tmp_path, counted_video_path):
        """like= takes the rate from a source: the one value that must cross."""
        with pf.VideoReader(counted_video_path) as video:
            writer = pf.VideoWriter(str(tmp_path / "like.mp4"), like=video)
            assert writer.fps == video.fps == 25.0
            writer.close()

    def test_like_carries_the_strided_rate(self, tmp_path, counted_video_path):
        """The corrected rate, not the file's -- otherwise the output plays fast."""
        output = str(tmp_path / "strided.mp4")
        with pf.VideoReader(counted_video_path, stride=5) as video:
            with pf.VideoWriter(output, like=video) as writer:
                for frame in video:
                    writer.write(frame)
        with pf.VideoReader(output) as result:
            assert result.fps == 5.0
            assert result.duration == pytest.approx(0.8, abs=0.05)  # 4 frames @ 5fps

    def test_like_and_fps_are_exclusive(self, tmp_path, temp_video_path):
        with pf.VideoReader(temp_video_path) as video:
            with pytest.raises(ValueError, match="exactly one"):
                pf.VideoWriter(str(tmp_path / "both.mp4"), fps=30.0, like=video)

    def test_like_refuses_a_source_without_a_rate(self, tmp_path):
        """None means unknown, and unknown forces the caller to state a policy."""
        class RatelessSource:
            fps = None

        with pytest.raises(ValueError, match="no frame rate"):
            pf.VideoWriter(str(tmp_path / "live.mp4"), like=RatelessSource())

    def test_like_needs_something_with_fps(self, tmp_path):
        with pytest.raises(TypeError, match="fps"):
            pf.VideoWriter(str(tmp_path / "x.mp4"), like=object())

    def test_width_parameter_removed(self, tmp_path):
        with pytest.raises(TypeError, match="transform.resize"):
            pf.VideoWriter(str(tmp_path / "w.mp4"), fps=30.0, width=320)


# ============================================================================
# Round-trip
# ============================================================================

class TestRoundTrip:
    """Reader and writer are two halves of one loop; test them as one."""

    def test_read_process_write_preserves_the_facts(self, tmp_path,
                                                    counted_video_path):
        output = str(tmp_path / "round.mp4")
        with pf.VideoReader(counted_video_path) as source:
            with pf.VideoWriter(output, like=source) as writer:
                for frame in source:
                    writer.write(frame)
            expected = (source.width, source.height, source.fps, source.frames)

        with pf.VideoReader(output) as result:
            assert (result.width, result.height, result.fps) == expected[:3]
            assert result.frames == expected[3]

    def test_resizing_in_the_loop_changes_size_and_nothing_else(
            self, tmp_path, counted_video_path):
        """Size is discovered, rate is declared -- the two do not interfere."""
        output = str(tmp_path / "resized.mp4")
        with pf.VideoReader(counted_video_path, stride=2) as source:
            with pf.VideoWriter(output, like=source) as writer:
                for frame in source:
                    writer.write(pf.transform.resize(frame, width=80))
            source_duration = source.duration

        with pf.VideoReader(output) as result:
            assert (result.width, result.height) == (80, 60)
            assert result.fps == 12.5
            assert result.duration == pytest.approx(source_duration, abs=0.05)


# ============================================================================
# read_image / save_image Tests
# ============================================================================

class TestReadImage:
    """Tests for read_image function."""

    def test_read_image(self, temp_image_path):
        image = pf.read_image(temp_image_path)
        assert isinstance(image, np.ndarray)
        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8

    def test_applies_exif_orientation(self, exif_rotated_image_path):
        """A rotated photo comes back the way a viewer shows it.

        This matches what VideoReader does with a rotated video, and it is what a
        model needs in order to produce upright coordinates. The stored array is
        200x100; the EXIF tag says display it rotated, so it arrives 100x200.
        """
        assert pf.read_image(exif_rotated_image_path).shape == (200, 100, 3)

    def test_grayscale_becomes_three_channels(self, grayscale_image_path):
        """The library's contract is RGB HxWx3; expanding is lossless."""
        assert pf.read_image(grayscale_image_path).shape == (40, 60, 3)

    def test_alpha_is_dropped_with_a_warning(self, rgba_image_path):
        """Dropping alpha is real information loss, so it is not done in silence."""
        with pytest.warns(UserWarning, match="alpha"):
            image = pf.read_image(rgba_image_path)
        assert image.shape == (40, 60, 3)

    def test_no_alpha_probe_for_formats_that_cannot_carry_it(self, temp_image_path):
        """JPEG has no alpha channel, so it never pays for the header probe."""
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")      # any warning here fails the test
            assert pf.read_image(temp_image_path).shape == (480, 640, 3)

    def test_undecodable_file_raises_value_error(self, tmp_path):
        """A file that exists but is not an image is a bad value, not a crash."""
        path = tmp_path / "not_really.jpg"
        path.write_text("this is not an image")
        with pytest.raises(ValueError, match="Could not decode"):
            pf.read_image(str(path))

    def test_invalid_path(self):
        """A missing file raises, and the message names the resolved absolute path."""
        with pytest.raises(FileNotFoundError, match="resolved to"):
            pf.read_image("nonexistent.jpg")

    def test_directory_path(self, tmp_path):
        """A directory raises IsADirectoryError rather than a decode failure."""
        with pytest.raises(IsADirectoryError):
            pf.read_image(str(tmp_path))

    def test_width_parameter_removed(self, temp_image_path):
        with pytest.raises(TypeError, match="transform.resize"):
            pf.read_image(temp_image_path, width=320)


class TestReadImageFromBuffer:
    """read_image accepting encoded bytes.

    HTTP uploads, S3 objects and database blobs arrive as bytes, never as a path.
    Without this branch every caller writes cv2.imdecode themselves and has to
    remember the BGR->RGB step -- which is the exact mistake read_image exists to
    make unmakeable.
    """

    def test_bytes_round_trip(self, sample_image):
        """The canonical round trip, and the channel-order check.

        sample_image carries pure red, green and blue rectangles, so exact equality
        here fails on any channel swap -- which is the failure this whole path
        exists to prevent.
        """
        payload = pf.encode_image(sample_image, ".png")
        assert np.array_equal(pf.read_image(payload), sample_image)

    @pytest.mark.parametrize("wrap", [bytearray, memoryview],
                             ids=["bytearray", "memoryview"])
    def test_any_bytes_like(self, sample_image, wrap):
        """An HTTP framework may hand you any of these; plain bytes is covered above."""
        payload = pf.encode_image(sample_image, ".png")
        assert pf.read_image(wrap(payload)).shape == sample_image.shape

    def test_path_and_bytes_of_the_same_file_agree(self, exif_rotated_image_path):
        """The property that would otherwise rot silently.

        cv2.imdecode honours EXIF orientation with IMREAD_COLOR and ignores it with
        IMREAD_UNCHANGED. If the two branches ever drift onto different flags, the
        same image comes back rotated differently depending on whether you passed
        the path or its bytes -- inside one function, with nothing to notice it.
        """
        from_path = pf.read_image(exif_rotated_image_path)
        from_bytes = pf.read_image(Path(exif_rotated_image_path).read_bytes())

        assert from_path.shape == from_bytes.shape == (200, 100, 3)
        assert np.array_equal(from_path, from_bytes)

    def test_undecodable_bytes_raise_value_error(self):
        """A bad value, not a runtime failure -- an HTTP layer turns this into 400."""
        with pytest.raises(ValueError, match="not an image"):
            pf.read_image(b"this is not an image")

    def test_empty_bytes_raise_value_error(self):
        with pytest.raises(ValueError):
            pf.read_image(b"")

    def test_alpha_warning_from_a_buffer_too(self, rgba_image_path):
        with pytest.warns(UserWarning, match="alpha"):
            pf.read_image(Path(rgba_image_path).read_bytes())


class TestReadImageFromArray:
    """read_image accepting an array it did not decode.

    This is the one input whose channel order cannot be verified: the function
    decodes a path or a buffer, so it knows those are RGB, but an array is taken
    on trust. Everything about it that *is* checkable is checked.
    """

    def test_array_returns_unchanged(self, sample_image):
        assert pf.read_image(sample_image) is sample_image

    @pytest.mark.parametrize("image", [
        np.zeros((40, 60, 4), dtype=np.uint8),
        np.zeros((40, 60), dtype=np.uint8),
        np.zeros((40, 60, 3), dtype=np.float32),
    ], ids=["4 channels", "grayscale", "float32"])
    def test_array_is_validated(self, image):
        """Stricter than a bare pass-through, and deliberately so.

        Shape, dtype and channel count are recoverable from the array; only RGB
        versus BGR is not. Checking what can be checked keeps grayscale and float
        arrays out of the annotators and the writer.
        """
        with pytest.raises(ValueError, match="RGB uint8"):
            pf.read_image(image)

    def test_other_types_raise_type_error(self):
        for value in (42, None, ["not", "an", "image"]):
            with pytest.raises(TypeError, match="path, an encoded buffer"):
                pf.read_image(value)


class TestEncodeImage:
    """encode_image -- the mirror of read_image's buffer branch.

    The asymmetry is what generates the bug: accept bytes in with nothing to give
    bytes out, and every HTTP handler hand-rolls cv2.imencode plus the RGB->BGR
    step that goes with it.
    """

    @pytest.mark.parametrize("extension", [".jpg", ".webp"])
    def test_round_trips_through_read_image(self, sample_image, extension):
        """PNG is covered exactly by TestReadImageFromBuffer; these are lossy."""
        decoded = pf.read_image(pf.encode_image(sample_image, extension))
        assert decoded.shape == sample_image.shape
        assert decoded.dtype == np.uint8

    def test_extension_dot_is_optional(self, sample_image):
        assert pf.encode_image(sample_image, "png") == \
               pf.encode_image(sample_image, ".png")

    def test_returns_bytes(self, sample_image):
        assert isinstance(pf.encode_image(sample_image, ".png"), bytes)

    def test_rejects_non_rgb(self):
        with pytest.raises(ValueError, match="RGB uint8"):
            pf.encode_image(np.zeros((8, 8, 4), dtype=np.uint8))

    def test_unknown_format_does_not_leak_cv2_error(self, sample_image):
        with pytest.raises(ValueError, match="does not support"):
            pf.encode_image(sample_image, ".xyz")


class TestSaveImage:
    """Tests for save_image function."""

    def test_round_trip(self, tmp_path, sample_image):
        path = str(tmp_path / "out.png")
        pf.save_image(path, sample_image)
        assert np.array_equal(pf.read_image(path), sample_image)

    def test_missing_directory_says_so(self, tmp_path, sample_image):
        """Previously surfaced as "Failed to write image", naming only the symptom."""
        with pytest.raises(NotADirectoryError, match="does not exist"):
            pf.save_image(str(tmp_path / "nope" / "out.png"), sample_image)

    def test_missing_extension_says_so(self, tmp_path, sample_image):
        with pytest.raises(ValueError, match="no file extension"):
            pf.save_image(str(tmp_path / "out"), sample_image)

    def test_unknown_extension_does_not_leak_cv2_error(self, tmp_path, sample_image):
        with pytest.raises(ValueError, match="cannot encode"):
            pf.save_image(str(tmp_path / "out.xyz"), sample_image)

    @pytest.mark.parametrize("image,description", [
        (np.zeros((40, 60, 4), dtype=np.uint8), "4 channels"),
        (np.zeros((40, 60), dtype=np.uint8), "grayscale"),
        (np.zeros((40, 60, 3), dtype=np.float32), "float32"),
    ])
    def test_rejects_non_rgb_images(self, tmp_path, image, description):
        """The same guard VideoWriter.write has, for the same reason.

        cv2.cvtColor turns any of these into HxWx3 without complaint, so the file
        written is a plausible-looking image full of colour nonsense.
        """
        with pytest.raises(ValueError, match="RGB uint8"):
            pf.save_image(str(tmp_path / f"{description}.png"), image)


# ============================================================================
# display_video Tests
# ============================================================================

class TestDisplayVideo:
    """Tests for display_video function."""

    def test_display_video_returns_none_or_int(self, sample_image):
        """display_video forwards to cv2 and returns the waitKey code.

        cv2.imshow/waitKey are stubbed: a real window needs a display server,
        which CI runners do not have, and on macOS the call blocks.
        """
        # 255 is OpenCV's "no key pressed" sentinel.
        with patch("cv2.imshow") as imshow, patch("cv2.waitKey", return_value=255) as wait_key:
            result = pf.display_video(sample_image, "test", wait_key=1)

        imshow.assert_called_once()
        wait_key.assert_called_once_with(1)
        assert result is None

    def test_display_video_raises_on_quit_key(self, sample_image):
        """Pressing the quit key raises DisplayExit so loops can break out."""
        with patch("cv2.imshow"), patch("cv2.waitKey", return_value=ord("q")), \
                patch("cv2.destroyWindow"):
            with pytest.raises(pf.media.DisplayExit):
                pf.display_video(sample_image, "test", wait_key=1)

    def test_display_width_does_not_touch_the_input(self, sample_image):
        """width survives on display because a display has no downstream."""
        original = sample_image.copy()
        with patch("cv2.imshow") as imshow, patch("cv2.waitKey", return_value=255):
            pf.display_video(sample_image, "test", width=320)
        assert np.array_equal(sample_image, original)
        assert imshow.call_args[0][1].shape[:2] == (240, 320)


# ============================================================================
# Buffer Tests
# ============================================================================

class TestBuffer:
    """Tests for Buffer — a rolling window that returns the delayed middle frame."""

    def test_buffer_creation(self):
        """A fresh buffer holds nothing and has a delay of frames // 2."""
        buffer = pf.Buffer(frames=5)

        assert buffer.current_size == 0
        assert buffer.frames_processed == 0
        assert buffer.delay == 2

    def test_buffer_fills_to_capacity(self, sample_image):
        """update() accumulates until capacity, then holds steady."""
        buffer = pf.Buffer(frames=3)

        buffer.update("r0", sample_image)
        assert buffer.current_size == 1

        buffer.update("r1", sample_image)
        buffer.update("r2", sample_image)
        assert buffer.current_size == 3

        # Beyond capacity the oldest entries are evicted.
        buffer.update("r3", sample_image)
        assert buffer.current_size == 3
        assert buffer.frames_processed == 4

    def test_buffer_returns_delayed_middle_frame(self, sample_image):
        """Once full, update() returns the middle (delayed) entry."""
        buffer = pf.Buffer(frames=3)

        buffer.update("r0", sample_image)
        buffer.update("r1", sample_image)
        results, frame = buffer.update("r2", sample_image)

        # With 3 frames buffered, the middle one is r1.
        assert results == "r1"
        assert isinstance(frame, np.ndarray)

    def test_buffer_get_buffer_contents(self, sample_image):
        """get_buffer_contents returns copies of both buffers."""
        buffer = pf.Buffer(frames=5)

        for i in range(3):
            buffer.update(f"r{i}", sample_image)

        results, frames = buffer.get_buffer_contents()
        assert results == ["r0", "r1", "r2"]
        assert len(frames) == 3
        assert all(isinstance(f, np.ndarray) for f in frames)

    def test_buffer_temporal_context_requires_full_buffer(self, sample_image):
        """Temporal context is only available once the buffer is full."""
        buffer = pf.Buffer(frames=3)

        buffer.update("r0", sample_image)
        assert buffer.get_temporal_context() is None

        buffer.update("r1", sample_image)
        buffer.update("r2", sample_image)

        context = buffer.get_temporal_context()
        assert context is not None
        assert context["current_results"] == "r1"
        assert context["past_results"] == ["r0"]
        assert context["future_results"] == ["r2"]

    def test_buffer_reset(self, sample_image):
        """reset() empties the buffer and the processed counter."""
        buffer = pf.Buffer(frames=5)

        buffer.update("r0", sample_image)
        buffer.update("r1", sample_image)

        buffer.reset()
        assert buffer.current_size == 0
        assert buffer.frames_processed == 0


# ============================================================================
# SlicedInference Tests
# ============================================================================

class TestSlicedInference:
    """Tests for SlicedInference (large image processing)."""

    def test_slicer_creation(self):
        """Slice geometry is configured on the slicer, not bound to an image."""
        slicer = pf.SlicedInference(
            slice_height=320,
            slice_width=320,
            overlap_ratio_h=0.2,
            overlap_ratio_w=0.2
        )

        assert slicer.slice_height == 320
        assert slicer.slice_width == 320
        assert slicer.overlap_ratio_h == 0.2

    def test_slicer_generates_slices(self, sample_image):
        """generate_slices() tiles the frame from its dimensions."""
        slicer = pf.SlicedInference(
            slice_height=320,
            slice_width=320,
            overlap_ratio_h=0.2,
            overlap_ratio_w=0.2
        )

        slices = slicer.generate_slices(
            image_height=sample_image.shape[0],
            image_width=sample_image.shape[1]
        )

        # Should have multiple slices for a 640x480 image
        assert len(slices) >= 4

    def test_slicer_slice_properties(self, sample_image):
        """Each slice is within bounds and no larger than the slice size."""
        slicer = pf.SlicedInference(slice_height=320, slice_width=320)

        h, w = sample_image.shape[:2]
        for x1, y1, x2, y2, slice_id in slicer.generate_slices(h, w):
            assert 0 <= x1 < x2 <= w
            assert 0 <= y1 < y2 <= h
            assert (x2 - x1) <= 320
            assert (y2 - y1) <= 320
            assert isinstance(slice_id, int)

    def test_auto_slice_size(self, sample_image):
        """auto_slice_size takes explicit height/width, not an image."""
        h, w = sample_image.shape[:2]
        slice_size = pf.auto_slice_size(image_height=h, image_width=w)

        # Should return reasonable slice size
        assert isinstance(slice_size, tuple)
        assert len(slice_size) == 2
        assert slice_size[0] > 0
        assert slice_size[1] > 0


# ============================================================================
# Smoother Tests
# ============================================================================

class TestSmoother:
    """Tests for smooth() — temporal detection smoothing over a Buffer."""

    @staticmethod
    def _dets(bbox, tracker_id=1):
        dets = pf.Detections()
        dets.add_detection(pf.Detection(
            bbox=bbox, class_id=0, confidence=0.9, tracker_id=tracker_id
        ))
        return dets

    def test_smooth_returns_raw_until_buffer_fills(self, sample_image):
        """Without full temporal context, smooth() passes results through."""
        buffer = pf.Buffer(frames=3)
        buffer.update(self._dets([100, 100, 200, 200]), sample_image)

        result = pf.smooth(buffer)

        assert isinstance(result, pf.Detections)

    def test_smooth_averages_jitter_across_frames(self, sample_image):
        """A jittery box is pulled toward its temporal neighbours."""
        buffer = pf.Buffer(frames=3)
        buffer.update(self._dets([100, 100, 200, 200]), sample_image)
        buffer.update(self._dets([140, 100, 240, 200]), sample_image)  # jitter
        buffer.update(self._dets([100, 100, 200, 200]), sample_image)

        smoothed = pf.smooth(buffer)

        assert len(smoothed) == 1
        # The middle frame's x1 of 140 is pulled back toward the 100s.
        assert smoothed[0].bbox[0] < 140
        assert smoothed[0].tracker_id == 1

    def test_smooth_rejects_out_of_range_decay(self, sample_image):
        """temporal_weight_decay is validated to [0.1, 1.0]."""
        buffer = pf.Buffer(frames=3)
        buffer.update(self._dets([100, 100, 200, 200]), sample_image)

        with pytest.raises(ValueError):
            pf.smooth(buffer, temporal_weight_decay=1.5)


# ============================================================================
# Timer Tests
# ============================================================================

class TestTimer:
    """Tests for TimeTracker — per-tracker_id duration accounting."""

    @staticmethod
    def _dets(tracker_ids):
        dets = pf.Detections()
        for tid in tracker_ids:
            dets.add_detection(pf.Detection(
                bbox=[100, 100, 200, 200], class_id=0, confidence=0.9, tracker_id=tid
            ))
        return dets

    def test_timer_creation(self):
        """Test creating TimeTracker."""
        timer = pf.TimeTracker()

        assert timer is not None

    def test_update_stamps_timing_fields(self):
        """update() populates total_time and first_seen_time in-place."""
        timer = pf.TimeTracker()

        dets = timer.update(self._dets([1, 2]))

        for det in dets:
            assert det.first_seen_time is not None
            assert det.total_time >= 0.0

    def test_total_time_accumulates_across_frames(self):
        """A tracker seen again later reports a larger total_time."""
        timer = pf.TimeTracker()

        timer.update(self._dets([1]))
        time.sleep(0.02)
        dets = timer.update(self._dets([1]))

        assert dets[0].total_time > 0.0

    def test_get_tracker_stats(self):
        """Per-tracker stats are queryable after an update."""
        timer = pf.TimeTracker()
        timer.update(self._dets([7]))

        stats = timer.get_tracker_stats(7)

        assert isinstance(stats, dict)

    def test_reset_clears_tracking_state(self):
        """reset() drops accumulated per-tracker state."""
        timer = pf.TimeTracker()
        timer.update(self._dets([1]))

        timer.reset()

        # After a reset the tracker is unknown again, so it restarts at ~0.
        dets = timer.update(self._dets([1]))
        assert dets[0].total_time == pytest.approx(0.0, abs=0.5)


# ============================================================================
# Colors Tests
# ============================================================================

class TestColors:
    """Tests for color palette functionality."""

    def test_default_palette_exists(self):
        """Test that default color palette exists."""
        assert pf.colors.DEFAULT_PALETTE is not None
        assert len(pf.colors.DEFAULT_PALETTE) > 0

    def test_palette_colors_are_tuples(self):
        """Test that palette colors are RGB tuples."""
        for color in pf.colors.DEFAULT_PALETTE:
            assert isinstance(color, tuple)
            assert len(color) == 3
            # Should be valid RGB values
            assert all(0 <= c <= 255 for c in color)

    def test_vibrant_palette_exists(self):
        """Test vibrant color palette."""
        assert pf.colors.VIBRANT_PALETTE is not None
        assert len(pf.colors.VIBRANT_PALETTE) > 0

    def test_pastel_palette_exists(self):
        """Test pastel color palette."""
        assert pf.colors.PASTEL_PALETTE is not None
        assert len(pf.colors.PASTEL_PALETTE) > 0


# ============================================================================
# Validators Tests
# ============================================================================

class TestValidators:
    """Tests for validation utilities."""

    def test_validate_bbox(self):
        """Test bbox validation."""
        from pixelflow.validators import validate_bbox

        # Valid bbox
        bbox = validate_bbox([100, 100, 200, 200])
        assert bbox == [100, 100, 200, 200]

    def test_validate_bbox_numpy(self):
        """Test bbox validation with numpy array."""
        from pixelflow.validators import validate_bbox

        bbox_np = np.array([100, 100, 200, 200])
        bbox = validate_bbox(bbox_np)

        assert isinstance(bbox, list)
        assert bbox == [100, 100, 200, 200]

    def test_validate_bbox_preserves_subpixel_precision(self):
        """Fractional coordinates survive validation instead of being truncated."""
        from pixelflow.validators import validate_bbox

        assert validate_bbox([10.7, 20.3, 110.9, 220.4]) == [10.7, 20.3, 110.9, 220.4]

    def test_validate_bbox_returns_floats(self):
        """Coordinates are floats, matching Detection's List[float] signature."""
        from pixelflow.validators import validate_bbox

        assert all(isinstance(v, float) for v in validate_bbox([1, 2, 3, 4]))

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # A truncating implementation returns [1, 2, 3, 4] for all three.
            ([1.9, 2.9, 3.9, 4.9], [1.9, 2.9, 3.9, 4.9]),
            ([1.999, 2.999, 3.999, 4.999], [2.0, 3.0, 4.0, 5.0]),
            ([-1.9, -2.9, 3.9, 4.9], [-1.9, -2.9, 3.9, 4.9]),
        ],
    )
    def test_validate_bbox_rounds_rather_than_truncates(self, raw, expected):
        """Validation must not shift boxes toward zero.

        Truncating moved every coordinate down by ~0.5 px on average, which
        reads as a systematic translation rather than as noise and costs
        roughly 1 mAP in the high-IoU bins.
        """
        from pixelflow.validators import validate_bbox

        assert validate_bbox(raw) == expected

    def test_validate_bbox_rounds_away_float32_artifacts(self):
        """float32 -> tolist() expansions are trimmed, keeping payloads small."""
        from pixelflow.validators import validate_bbox

        raw = np.array([10.7, 20.3, 110.9, 220.4], dtype=np.float32).tolist()
        assert raw[0] != 10.7  # float32 gives 10.699999809265137
        assert validate_bbox(raw) == [10.7, 20.3, 110.9, 220.4]

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_validate_bbox_rejects_non_finite(self, bad):
        """NaN/inf are rejected, not passed through.

        They would poison IoU maths silently, and strict JSON encoders -
        Starlette's JSONResponse among them - refuse to serialize them.
        """
        from pixelflow.validators import validate_bbox

        assert validate_bbox([0.0, 0.0, bad, 10.0]) is None

    @pytest.mark.parametrize(
        "bad", [None, "10,20,30,40", b"abcd", 42, [1, 2, 3], [1, 2, 3, 4, 5], ["a", 2, 3, 4]]
    )
    def test_validate_bbox_rejects_malformed(self, bad):
        """Malformed input still degrades to None rather than raising."""
        from pixelflow.validators import validate_bbox

        assert validate_bbox(bad) is None


    @pytest.mark.parametrize(
        "bad",
        [
            None,
            [],
            np.zeros((0, 2)),                 # vertexless mask from ultralytics
            [[1, 2, 3]],                      # points are not pairs
            [1, 2, 3],                        # flat, not points
            "abc",
            [[1, 2], [3]],                    # ragged
            [[1, 2], [float("nan"), 3]],
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],  # several polygons, not one
        ],
    )
    def test_validate_segments_rejects_non_polygons(self, bad):
        """Anything that is not one polygon of (x, y) points degrades to None."""
        from pixelflow.validators import validate_segments

        assert validate_segments(bad) is None

    def test_round_to_decimal(self):
        """Test decimal rounding utility."""
        from pixelflow.validators import round_to_decimal

        value = 3.14159265
        rounded = round_to_decimal(value, decimals=2)

        assert rounded == 3.14

    def test_simplify_polygon(self):
        """Test polygon simplification."""
        from pixelflow.validators import simplify_polygon

        # Complex polygon
        polygon = [
            (0, 0), (10, 0), (11, 0), (20, 0),
            (20, 10), (20, 20), (10, 20), (0, 20)
        ]

        simplified = simplify_polygon(polygon, tolerance=5.0)

        # Should have fewer points
        assert len(simplified) <= len(polygon)


# ============================================================================
# Edge Cases
# ============================================================================

class TestUtilityEdgeCases:
    """Test edge cases for utilities."""

    def test_buffer_with_zero_size(self):
        """Test buffer creation with zero size."""
        try:
            buffer = pf.Buffer(frames=0)
            # May raise error or handle gracefully
        except (ValueError, AssertionError):
            # Expected to fail
            pass

    def test_timer_stop_without_start(self):
        """Test stopping timer that wasn't started."""
        timer = pf.TimeTracker()

        try:
            timer.stop("nonexistent")
            # May raise error or return None
        except (KeyError, Exception):
            # Expected behavior
            pass

    def test_slicer_with_slice_larger_than_image(self, small_image):
        """Test slicer with slice size larger than image."""
        slicer = pf.SlicedInference(slice_height=500, slice_width=500)

        h, w = small_image.shape[:2]
        slices = slicer.generate_slices(image_height=h, image_width=w)

        # Should still return at least one slice (the whole image)
        assert len(slices) >= 1
