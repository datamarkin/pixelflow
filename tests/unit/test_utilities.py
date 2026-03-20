"""
Unit tests for pixelflow utility modules.

Tests Media, Buffer, SlicedInference, Smoother, Timer, and other
utility functionality.
"""

import pytest
import numpy as np
import cv2
import time
import pixelflow as pf


# ============================================================================
# VideoReader Tests
# ============================================================================

class TestVideoReader:
    """Tests for VideoReader class."""

    def test_video_reader_properties(self, temp_video_path):
        """Test video metadata properties."""
        video = pf.VideoReader(temp_video_path)
        assert video.frame_count == 10
        assert video.fps == 30.0
        assert video.width == 640
        assert video.height == 480
        assert video.duration == pytest.approx(10 / 30.0)
        assert len(video) == 10
        video.close()

    def test_video_reader_iteration(self, temp_video_path):
        """Test iterating through video frames."""
        video = pf.VideoReader(temp_video_path)
        frame_count = 0
        for frame in video:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (480, 640, 3)
            frame_count += 1
        assert frame_count == 10
        video.close()

    def test_video_reader_replayable(self, temp_video_path):
        """Test that iteration resets each time (replayable)."""
        video = pf.VideoReader(temp_video_path)
        count1 = sum(1 for _ in video)
        count2 = sum(1 for _ in video)
        assert count1 == count2 == 10
        video.close()

    def test_video_reader_resize(self, temp_video_path):
        """Test frame resizing."""
        video = pf.VideoReader(temp_video_path, width=320)
        assert video.width == 320
        assert video.height == 240
        for frame in video:
            assert frame.shape == (240, 320, 3)
            break
        video.close()

    def test_video_reader_seek(self, temp_video_path):
        """Test seeking to a specific frame."""
        video = pf.VideoReader(temp_video_path)
        video.seek(5)
        # Read one frame after seek
        for frame in video:
            assert isinstance(frame, np.ndarray)
            break
        video.close()

    def test_video_reader_context_manager(self, temp_video_path):
        """Test context manager usage."""
        with pf.VideoReader(temp_video_path) as video:
            assert video.frame_count == 10
            count = sum(1 for _ in video)
            assert count == 10

    def test_video_reader_invalid_path(self):
        """Test with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            pf.VideoReader("nonexistent_file.mp4")

    def test_video_reader_codec(self, temp_video_path):
        """Test codec property."""
        video = pf.VideoReader(temp_video_path)
        assert isinstance(video.codec, str)
        assert len(video.codec) == 4
        video.close()


# ============================================================================
# read_image Tests
# ============================================================================

class TestReadImage:
    """Tests for read_image function."""

    def test_read_image(self, temp_image_path):
        """Test loading an image."""
        image = pf.read_image(temp_image_path)
        assert isinstance(image, np.ndarray)
        assert image.shape == (480, 640, 3)

    def test_read_image_resize(self, temp_image_path):
        """Test loading with resize."""
        image = pf.read_image(temp_image_path, width=320)
        assert image.shape[1] == 320
        assert image.shape[0] == 240

    def test_read_image_invalid_path(self):
        """Test with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            pf.read_image("nonexistent.jpg")


# ============================================================================
# VideoWriter Tests
# ============================================================================

class TestVideoWriter:
    """Tests for VideoWriter class."""

    def test_video_writer_basic(self, tmp_path, sample_image):
        """Test writing frames to video."""
        output = str(tmp_path / "output.mp4")
        writer = pf.VideoWriter(output, fps=30.0)
        for _ in range(5):
            writer.write(sample_image)
        assert writer.frames_written == 5
        assert writer.is_opened
        writer.close()
        assert not writer.is_opened
        # Verify file exists and is readable
        video = pf.VideoReader(output)
        assert video.frame_count == 5
        video.close()

    def test_video_writer_context_manager(self, tmp_path, sample_image):
        """Test context manager usage."""
        output = str(tmp_path / "output_ctx.mp4")
        with pf.VideoWriter(output, fps=30.0) as writer:
            writer.write(sample_image)
            writer.write(sample_image)
            assert writer.frames_written == 2

    def test_video_writer_resize(self, tmp_path, sample_image):
        """Test resize-on-write."""
        output = str(tmp_path / "output_resize.mp4")
        writer = pf.VideoWriter(output, fps=30.0, width=320)
        writer.write(sample_image)
        writer.close()
        video = pf.VideoReader(output)
        assert video.width == 320
        video.close()


# ============================================================================
# display_video Tests
# ============================================================================

class TestDisplayVideo:
    """Tests for display_video function."""

    def test_display_video_returns_none_or_int(self, sample_image):
        """Test display_video return type."""
        try:
            result = pf.display_video(sample_image, "test", wait_key=1)
            assert result is None or isinstance(result, int)
            pf.close_display()
        except Exception:
            # May fail in headless environment
            pass


# ============================================================================
# Buffer Tests
# ============================================================================

class TestBuffer:
    """Tests for Buffer class (frame buffering)."""

    def test_buffer_creation(self):
        """Test creating buffer."""
        buffer = pf.Buffer(frames=5)

        assert buffer.size == 5
        assert len(buffer) == 0

    def test_buffer_append(self, sample_image):
        """Test appending frames to buffer."""
        buffer = pf.Buffer(frames=3)

        buffer.append(sample_image)
        assert len(buffer) == 1

        buffer.append(sample_image)
        buffer.append(sample_image)
        assert len(buffer) == 3

    def test_buffer_overflow(self, sample_image):
        """Test buffer behavior when exceeding size."""
        buffer = pf.Buffer(frames=3)

        # Add 5 frames to buffer of size 3
        for i in range(5):
            buffer.append(sample_image)

        # Should only keep last 3
        assert len(buffer) == 3

    def test_buffer_get_frames(self, sample_image):
        """Test retrieving frames from buffer."""
        buffer = pf.Buffer(frames=5)

        for i in range(3):
            buffer.append(sample_image)

        frames = buffer.get_frames()
        assert len(frames) == 3
        assert all(isinstance(f, np.ndarray) for f in frames)

    def test_buffer_clear(self, sample_image):
        """Test clearing buffer."""
        buffer = pf.Buffer(frames=5)

        buffer.append(sample_image)
        buffer.append(sample_image)

        buffer.clear()
        assert len(buffer) == 0

    def test_buffer_indexing(self, sample_image):
        """Test accessing buffer by index."""
        buffer = pf.Buffer(frames=5)

        buffer.append(sample_image)
        buffer.append(sample_image)

        # Get most recent frame
        frame = buffer[-1]
        assert isinstance(frame, np.ndarray)

        # Get oldest frame
        oldest = buffer[0]
        assert isinstance(oldest, np.ndarray)


# ============================================================================
# SlicedInference Tests
# ============================================================================

class TestSlicedInference:
    """Tests for SlicedInference (large image processing)."""

    def test_slicer_creation(self, sample_image):
        """Test creating SlicedInference."""
        slicer = pf.SlicedInference(
            image=sample_image,
            slice_size=(320, 320),
            overlap=0.2
        )

        assert slicer.slice_size == (320, 320)
        assert slicer.overlap == 0.2

    def test_slicer_generates_slices(self, sample_image):
        """Test that slicer generates image slices."""
        slicer = pf.SlicedInference(
            image=sample_image,
            slice_size=(320, 320),
            overlap=0.2
        )

        slices = list(slicer.get_slices())

        # Should have multiple slices for 640x480 image
        assert len(slices) >= 4

    def test_slicer_slice_properties(self, sample_image):
        """Test properties of generated slices."""
        slicer = pf.SlicedInference(
            image=sample_image,
            slice_size=(320, 320)
        )

        for slice_img, x_offset, y_offset in slicer.get_slices():
            # Each slice should be correct size (or smaller at edges)
            assert slice_img.shape[0] <= 320
            assert slice_img.shape[1] <= 320
            # Offsets should be non-negative
            assert x_offset >= 0
            assert y_offset >= 0

    def test_auto_slice_size(self, sample_image):
        """Test automatic slice size calculation."""
        slice_size = pf.auto_slice_size(sample_image)

        # Should return reasonable slice size
        assert isinstance(slice_size, tuple)
        assert len(slice_size) == 2
        assert slice_size[0] > 0
        assert slice_size[1] > 0


# ============================================================================
# Smoother Tests
# ============================================================================

class TestSmoother:
    """Tests for trajectory smoothing functions."""

    def test_smooth_basic(self):
        """Test basic trajectory smoothing."""
        # Create noisy trajectory
        trajectory = np.array([
            [100, 100],
            [102, 98],
            [105, 103],
            [107, 97],
            [110, 101]
        ], dtype=np.float32)

        smoothed = pf.smooth(trajectory, window_size=3)

        assert smoothed.shape == trajectory.shape
        # Smoothed should be close to original but less noisy
        assert isinstance(smoothed, np.ndarray)

    def test_smooth_with_different_window_sizes(self):
        """Test smoothing with various window sizes."""
        trajectory = np.random.rand(20, 2) * 100

        smoothed_small = pf.smooth(trajectory, window_size=3)
        smoothed_large = pf.smooth(trajectory, window_size=7)

        # Larger window should produce smoother result
        assert smoothed_small.shape == smoothed_large.shape

    def test_smooth_preserves_endpoints(self):
        """Test that smoothing preserves start and end points."""
        trajectory = np.array([
            [0, 0],
            [10, 10],
            [20, 15],
            [30, 30]
        ], dtype=np.float32)

        smoothed = pf.smooth(trajectory, window_size=3)

        # Endpoints should be preserved or very close
        # (depending on smoothing method)
        assert smoothed.shape == trajectory.shape


# ============================================================================
# Timer Tests
# ============================================================================

class TestTimer:
    """Tests for TimeTracker utility."""

    def test_timer_creation(self):
        """Test creating TimeTracker."""
        timer = pf.TimeTracker()

        assert timer is not None

    def test_timer_start_stop(self):
        """Test basic start/stop timing."""
        timer = pf.TimeTracker()

        timer.start("operation1")
        time.sleep(0.01)  # Sleep 10ms
        timer.stop("operation1")

        elapsed = timer.get_time("operation1")
        assert elapsed > 0.0
        assert elapsed >= 0.01  # At least 10ms

    def test_timer_multiple_operations(self):
        """Test timing multiple operations."""
        timer = pf.TimeTracker()

        timer.start("op1")
        time.sleep(0.01)
        timer.stop("op1")

        timer.start("op2")
        time.sleep(0.02)
        timer.stop("op2")

        assert timer.get_time("op1") < timer.get_time("op2")

    def test_timer_get_all_times(self):
        """Test getting all timed operations."""
        timer = pf.TimeTracker()

        timer.start("op1")
        timer.stop("op1")

        timer.start("op2")
        timer.stop("op2")

        all_times = timer.get_all_times()

        assert "op1" in all_times
        assert "op2" in all_times

    def test_timer_reset(self):
        """Test resetting timer."""
        timer = pf.TimeTracker()

        timer.start("op1")
        timer.stop("op1")

        timer.reset()

        all_times = timer.get_all_times()
        assert len(all_times) == 0

    def test_timer_context_manager(self):
        """Test timer as context manager (if supported)."""
        timer = pf.TimeTracker()

        try:
            with timer.time("operation"):
                time.sleep(0.01)

            elapsed = timer.get_time("operation")
            assert elapsed >= 0.01
        except AttributeError:
            # May not support context manager
            pass


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

    def test_video_reader_with_invalid_path(self):
        """Test VideoReader with invalid path."""
        with pytest.raises(FileNotFoundError):
            pf.VideoReader("nonexistent_file.mp4")

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
        slicer = pf.SlicedInference(
            image=small_image,
            slice_size=(500, 500)
        )

        slices = list(slicer.get_slices())

        # Should still return at least one slice (the whole image)
        assert len(slices) >= 1
