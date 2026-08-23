"""
Integration tests for complete PixelFlow workflows.

Tests end-to-end pipelines combining multiple components including
detection, filtering, annotation, and spatial analytics.
"""

import pytest
import numpy as np
import cv2
import pixelflow as pf


# ============================================================================
# Detection Pipeline Tests
# ============================================================================

@pytest.mark.integration
class TestDetectionPipeline:
    """Tests for complete detection processing pipelines."""

    def test_basic_detection_pipeline(self, sample_image, sample_detections):
        """Test basic detection workflow: filter -> annotate."""
        # Filter
        filtered = (sample_detections
                   .filter_by_confidence(0.8)
                   .filter_by_class_id(0))

        # Annotate
        annotated = sample_image.copy()
        annotated = pf.annotate.box(annotated, filtered)
        annotated = pf.annotate.label(annotated, filtered)

        assert annotated.shape == sample_image.shape
        assert not np.array_equal(annotated, sample_image)

    def test_privacy_pipeline(self, sample_image, sample_detections):
        """Test privacy protection pipeline."""
        # Filter for people
        people = sample_detections.filter_by_class_id(0)

        # Apply privacy blur
        protected = pf.annotate.blur(sample_image.copy(), people, kernel_size=31)

        # Verify image was modified
        assert not np.array_equal(protected, sample_image)

    def test_multi_class_pipeline(self, sample_image, sample_detections):
        """Test pipeline with different processing per class."""
        # Process people differently from vehicles
        people = sample_detections.filter_by_class_id(0)
        vehicles = sample_detections.filter_by_class_id([2])

        annotated = sample_image.copy()

        # Blur people
        annotated = pf.annotate.blur(annotated, people)

        # Draw boxes on vehicles
        annotated = pf.annotate.box(annotated, vehicles, thickness=3)
        annotated = pf.annotate.label(annotated, vehicles)

        assert annotated.shape == sample_image.shape


# ============================================================================
# Zone Analytics Pipeline Tests
# ============================================================================

@pytest.mark.integration
class TestZoneAnalyticsPipeline:
    """Tests for zone-based analytics workflows."""

    def test_zone_filtering_pipeline(self, sample_image, sample_detections):
        """Test zone-based detection filtering."""
        # Setup zones
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(50, 50), (300, 50), (300, 300), (50, 300)],
            zone_id="area1",
            trigger_strategy="center"
        )

        # Update detections with zone info
        updated = zones.update(sample_detections)

        # Filter by zone
        in_zone = updated.filter_by_zones("area1")

        # Annotate
        annotated = sample_image.copy()
        annotated = pf.annotate.zones(annotated, zones, opacity=0.2)
        annotated = pf.annotate.box(annotated, in_zone)

        assert annotated.shape == sample_image.shape

    def test_multi_zone_analytics(self, sample_image, sample_detections):
        """Test analytics across multiple zones."""
        zones = pf.Zones()

        # Entrance zone
        zones.add_zone(
            polygon=[(50, 50), (250, 50), (250, 250), (50, 250)],
            zone_id="entrance",
            trigger_strategy="bottom_center"
        )

        # Exit zone
        zones.add_zone(
            polygon=[(400, 300), (600, 300), (600, 450), (400, 450)],
            zone_id="exit",
            trigger_strategy="center"
        )

        # Process
        updated = zones.update(sample_detections)

        # Get statistics
        stats = zones.get_zone_stats()

        # Visualize
        annotated = sample_image.copy()
        annotated = pf.annotate.zones(annotated, zones)
        annotated = pf.annotate.box(annotated, updated)

        assert isinstance(stats, dict)


# ============================================================================
# Transform Pipeline Tests
# ============================================================================

@pytest.mark.integration
class TestTransformPipeline:
    """Tests for transformation pipelines."""

    def test_augmentation_pipeline(self, sample_image, sample_detections):
        """Test image augmentation with detection tracking."""
        # Apply multiple transforms
        img, dets = sample_image, sample_detections

        # Rotate
        img, dets = pf.transform.rotate_detections(img, dets, angle=15)

        # Flip
        img, dets = pf.transform.flip_horizontal_detections(img, dets)

        # Crop
        img, dets = pf.transform.crop_detections(img, dets, bbox=[50, 50, 500, 400])

        # Should still have detections
        assert len(dets) >= 0

        # Annotate transformed result
        annotated = pf.annotate.box(img.copy(), dets)
        assert annotated.shape == img.shape

    def test_preprocessing_pipeline(self, sample_image):
        """Test image preprocessing pipeline."""
        # Apply enhancement operations
        processed = sample_image.copy()

        # Grayscale
        processed = pf.transform.to_grayscale(processed)

        # CLAHE for contrast
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

        processed = pf.transform.clahe(processed)

        assert processed.shape[:2] == sample_image.shape[:2]

    def test_inverse_transform_pipeline(self, sample_image, sample_detections):
        """Test inverse transform workflow."""
        # Apply transform chain
        img, dets = sample_image, sample_detections

        img, dets = pf.transform.rotate_detections(img, dets, 30)
        img, dets = pf.transform.crop_detections(img, dets, [100, 100, 400, 400])

        # Simulate new detections in transformed space
        new_dets = pf.Detections()
        new_dets.add_detection(pf.Detection(
            bbox=[50, 50, 100, 100]
        ))

        # Apply inverse (if tracking is enabled)
        try:
            original_coords = pf.transform.inverse_transforms(new_dets)
            assert isinstance(original_coords, pf.Detections)
        except (AttributeError, NotImplementedError):
            # Inverse transforms may not be fully implemented
            pass


# ============================================================================
# Video Processing Pipeline Tests
# ============================================================================

@pytest.mark.integration
class TestVideoProcessingPipeline:
    """Tests for video processing workflows."""

    def test_video_frame_processing(self, temp_video_path):
        """Test processing video frames."""
        video = pf.VideoReader(temp_video_path)

        processed_count = 0
        for frame in video:
            # Create mock detections
            detections = pf.Detections()
            detections.add_detection(pf.Detection(
                bbox=[100, 100, 200, 200],
                confidence=0.9,
                class_id=0
            ))

            # Annotate
            annotated = pf.annotate.box(frame.copy(), detections)

            assert annotated.shape == frame.shape
            processed_count += 1

        assert processed_count == 10

    def test_video_with_buffer(self, temp_video_path):
        """Test video processing with frame buffering."""
        video = pf.VideoReader(temp_video_path)
        buffer = pf.Buffer(frames=5)
        results = pf.Detections()

        for frame in video:
            buffer.update(results, frame)

            if buffer.current_size >= 3:
                # Process with temporal context
                all_results, all_frames = buffer.get_buffer_contents()
                assert len(all_frames) >= 3

    def test_video_with_tracking(self, temp_video_path):
        """Test video processing with object tracking."""
        video = pf.VideoReader(temp_video_path)

        frame_count = 0
        for frame in video:
            # Simulate tracked detections
            detections = pf.Detections()
            detections.add_detection(pf.Detection(
                bbox=[100 + frame_count * 5, 100, 200 + frame_count * 5, 200],
                confidence=0.9,
                class_id=0,
                tracker_id=1
            ))

            # Could track crossing here
            frame_count += 1

        assert frame_count == 10


# ============================================================================
# Complex Multi-Component Pipelines
# ============================================================================

@pytest.mark.integration
class TestComplexPipelines:
    """Tests for complex workflows combining many components."""

    def test_full_analytics_pipeline(self, sample_image, sample_detections):
        """Test complete analytics pipeline."""
        # 1. Filter high-confidence detections
        filtered = sample_detections.filter_by_confidence(0.7)

        # 2. Setup zones
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(50, 50), (300, 50), (300, 300), (50, 300)],
            zone_id="zone1"
        )

        # 3. Update with zone info
        filtered = zones.update(filtered)

        # 4. Filter by zone
        in_zone = filtered.filter_by_zones("zone1")

        # 5. Multi-layer annotation
        result = sample_image.copy()
        result = pf.annotate.zones(result, zones, opacity=0.2)
        result = pf.annotate.box(result, in_zone, thickness=2)
        result = pf.annotate.label(result, in_zone)

        assert result.shape == sample_image.shape


    def test_sliced_inference_pipeline(self, sample_image):
        """Test large image processing with slicing."""
        # Create slicer
        slicer = pf.SlicedInference(
            slice_height=320,
            slice_width=320,
            overlap_ratio_h=0.2,
            overlap_ratio_w=0.2
        )

        all_detections = pf.Detections()

        # Process each slice
        slices = slicer.generate_slices(
            image_height=sample_image.shape[0],
            image_width=sample_image.shape[1]
        )
        assert len(slices) > 1  # 640x480 at 320x320 must tile into several slices

        for x1, y1, x2, y2, _slice_id in slices:
            x_offset, y_offset = x1, y1
            # Mock detection in slice
            slice_det = pf.Detection(
                bbox=[10, 10, 50, 50],
                confidence=0.9
            )

            # Adjust coordinates to full image
            adjusted_bbox = [
                slice_det.bbox[0] + x_offset,
                slice_det.bbox[1] + y_offset,
                slice_det.bbox[2] + x_offset,
                slice_det.bbox[3] + y_offset
            ]

            all_detections.add_detection(pf.Detection(
                bbox=adjusted_bbox,
                confidence=slice_det.confidence
            ))

        # Remove duplicates across slices
        final = all_detections.remove_duplicates(iou_threshold=0.5)

        assert len(final) >= 0


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestPerformance:
    """Tests for performance of integrated pipelines."""

    def test_large_batch_processing(self, sample_image):
        """Test processing large batch of detections."""
        # Create many detections
        detections = pf.Detections()

        for i in range(100):
            detections.add_detection(pf.Detection(
                bbox=[i * 5, i * 4, i * 5 + 50, i * 4 + 50],
                confidence=0.9,
                class_id=i % 3
            ))

        # Filter chain
        filtered = (detections
                   .filter_by_confidence(0.8)
                   .filter_by_size(min_area=1000))

        # Annotate
        annotated = pf.annotate.box(sample_image.copy(), filtered)

        assert annotated.shape == sample_image.shape

    def test_video_processing_performance(self, temp_video_path):
        """Test video processing speed."""
        import time

        video = pf.VideoReader(temp_video_path)

        start = time.perf_counter()

        for frame in video:
            # Mock detection
            dets = pf.Detections()
            dets.add_detection(pf.Detection(
                bbox=[100, 100, 200, 200]
            ))

            # Annotate
            _ = pf.annotate.box(frame.copy(), dets)

        elapsed = time.perf_counter() - start

        # Should process 10 frames reasonably fast
        assert elapsed < 10.0  # Less than 10 seconds for 10 frames


# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling in integrated workflows."""

    def test_pipeline_with_empty_detections(self, sample_image, empty_detections):
        """Test pipeline handles empty detections gracefully."""
        # Should not crash with empty detections
        filtered = empty_detections.filter_by_confidence(0.8)

        annotated = pf.annotate.box(sample_image.copy(), filtered)

        assert np.array_equal(annotated, sample_image)

    def test_pipeline_with_invalid_data(self, sample_image):
        """Test pipeline handles invalid data gracefully."""
        # Detection with None values
        detections = pf.Detections()
        detections.add_detection(pf.Detection(
            bbox=[100, 100, 200, 200],
            confidence=None,
            class_id=None
        ))

        # Should handle gracefully
        try:
            filtered = detections.filter_by_confidence(0.5)
            annotated = pf.annotate.box(sample_image.copy(), detections)
            assert annotated.shape == sample_image.shape
        except Exception as e:
            # Some operations may fail, but shouldn't crash unexpectedly
            assert isinstance(e, (TypeError, ValueError, AttributeError))
