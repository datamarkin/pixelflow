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
        """A path that is neither on disk nor downloadable raises FileNotFoundError.

        media._resolve_path falls back to assets.download() for any missing
        path, so the download is stubbed out here — otherwise this test would
        make a real network request to dtmfiles.com.
        """
        with patch("pixelflow.media.assets.download", side_effect=OSError("offline")):
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
        with patch("pixelflow.media.assets.download", side_effect=OSError("offline")):
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
        dets = pf.detections.Detections()
        dets.add_detection(pf.detections.Detection(
            bbox=bbox, class_id=0, confidence=0.9, tracker_id=tracker_id
        ))
        return dets

    def test_smooth_returns_raw_until_buffer_fills(self, sample_image):
        """Without full temporal context, smooth() passes results through."""
        buffer = pf.Buffer(frames=3)
        buffer.update(self._dets([100, 100, 200, 200]), sample_image)

        result = pf.smooth(buffer)

        assert isinstance(result, pf.detections.Detections)

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
        dets = pf.detections.Detections()
        for tid in tracker_ids:
            dets.add_detection(pf.detections.Detection(
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
        with patch("pixelflow.media.assets.download", side_effect=OSError("offline")):
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
        slicer = pf.SlicedInference(slice_height=500, slice_width=500)

        h, w = small_image.shape[:2]
        slices = slicer.generate_slices(image_height=h, image_width=w)

        # Should still return at least one slice (the whole image)
        assert len(slices) >= 1
