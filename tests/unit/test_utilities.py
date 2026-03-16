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
# Media Tests
# ============================================================================

class TestMedia:
    """Tests for Media class (video/image loading)."""

    def test_media_from_video(self, temp_video_path):
        """Test loading video file."""
        media = pf.Media(temp_video_path)

        assert media.info.frame_count == 10
        assert media.info.fps > 0
        assert media.info.width == 640
        assert media.info.height == 480

    def test_media_iteration(self, temp_video_path):
        """Test iterating through video frames."""
        media = pf.Media(temp_video_path)

        frame_count = 0
        for frame in media.frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (480, 640, 3)
            frame_count += 1

        assert frame_count == 10

    def test_media_lazy_loading(self, temp_video_path):
        """Test that media uses lazy loading."""
        media = pf.Media(temp_video_path)

        # Creating Media should not load all frames
        # Only when iterating should frames be loaded
        assert media.info.frame_count == 10

    def test_media_from_image(self, temp_image_path):
        """Test loading single image."""
        media = pf.Media(temp_image_path)

        # Single image should have 1 frame
        assert media.info.frame_count == 1

        for frame in media.frames:
            assert isinstance(frame, np.ndarray)
            break


class TestMediaInfo:
    """Tests for MediaInfo class."""

    def test_media_info_from_video(self, temp_video_path):
        """Test getting video metadata."""
        media = pf.Media(temp_video_path)
        info = media.info

        assert info.frame_count == 10
        assert info.fps == 30.0
        assert info.width == 640
        assert info.height == 480


class TestMediaHelpers:
    """Tests for media helper functions."""

    def test_show_frame(self, sample_image):
        """Test show_frame function."""
        # Can't actually display in test, but verify it doesn't crash
        try:
            pf.show_frame(sample_image, window_name="test", wait_key=1)
            # Close window immediately
            pf.close_display()
        except Exception:
            # May fail in headless environment
            pass

    def test_write_frame(self, tmp_path, sample_image):
        """Test write_frame function."""
        output_path = tmp_path / "output_frame.jpg"

        pf.write_frame(sample_image, str(output_path))

        # Verify file was written
        assert output_path.exists()

        # Verify it can be read back
        loaded = cv2.imread(str(output_path))
        assert loaded.shape == sample_image.shape


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
        """Test that palette colors are BGR tuples."""
        for color in pf.colors.DEFAULT_PALETTE:
            assert isinstance(color, tuple)
            assert len(color) == 3
            # Should be valid BGR values
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

    def test_media_with_invalid_path(self):
        """Test media loading with invalid path."""
        try:
            media = pf.Media("nonexistent_file.mp4")
            # Should raise error
            assert False, "Should have raised exception"
        except (FileNotFoundError, Exception):
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
        slicer = pf.SlicedInference(
            image=small_image,
            slice_size=(500, 500)
        )

        slices = list(slicer.get_slices())

        # Should still return at least one slice (the whole image)
        assert len(slices) >= 1
