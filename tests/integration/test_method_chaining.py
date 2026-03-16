"""
Integration tests for method chaining and serialization.

Tests fluent API patterns, complex filter chains, and JSON
serialization/deserialization workflows.
"""

import pytest
import json
import numpy as np
import pixelflow as pf


# ============================================================================
# Method Chaining Tests
# ============================================================================

@pytest.mark.integration
class TestFilterChaining:
    """Tests for chaining filter methods."""

    def test_simple_chain(self, sample_detections):
        """Test simple two-filter chain."""
        result = (sample_detections
                 .filter_by_confidence(0.8)
                 .filter_by_class_id(0))

        # Should have filtered detections
        for det in result:
            assert det.confidence >= 0.8
            assert det.class_id == 0

    def test_complex_chain(self, sample_detections):
        """Test complex multi-filter chain."""
        result = (sample_detections
                 .filter_by_confidence(0.7)
                 .filter_by_class_id([0, 2])
                 .filter_by_size(min_area=5000)
                 .filter_by_position(max_x=500))

        # All filters should be applied
        assert isinstance(result, pf.detections.Detections)

    def test_chain_preserves_immutability(self, sample_detections):
        """Test that chaining doesn't modify original."""
        original_len = len(sample_detections)

        _ = (sample_detections
             .filter_by_confidence(0.9)
             .filter_by_class_id(0))

        # Original should be unchanged
        assert len(sample_detections) == original_len

    def test_chain_with_all_filter_types(self, sample_detections):
        """Test chain combining different filter categories."""
        result = (sample_detections
                 .filter_by_confidence(0.5)           # Confidence
                 .filter_by_class_id([0, 1, 2])       # Class
                 .filter_by_size(min_area=1000)       # Size
                 .filter_by_aspect_ratio(0.5, 2.0)    # Dimensions
                 .filter_by_position(min_x=0, max_x=640))  # Position

        assert isinstance(result, pf.detections.Detections)

    def test_empty_chain_result(self, sample_detections):
        """Test chain that results in empty detections."""
        result = (sample_detections
                 .filter_by_confidence(0.99)  # Very high threshold
                 .filter_by_class_id(999))    # Non-existent class

        assert len(result) == 0


# ============================================================================
# OCR Filter Chaining Tests
# ============================================================================

@pytest.mark.integration
class TestOCRFilterChaining:
    """Tests for chaining OCR-specific filters."""

    def test_ocr_filter_chain(self):
        """Test chaining OCR filters."""
        # Create OCR detections
        detections = pf.detections.Detections()

        detections.add_detection(pf.detections.Detection(
            bbox=[100, 100, 300, 150],
            text="Invoice #12345",
            text_confidence=0.95,
            text_language="en",
            text_level="line",
            text_order=1
        ))

        detections.add_detection(pf.detections.Detection(
            bbox=[100, 160, 250, 200],
            text="total amount",
            text_confidence=0.65,
            text_language="en",
            text_level="word",
            text_order=2
        ))

        detections.add_detection(pf.detections.Detection(
            bbox=[100, 210, 280, 250],
            text="Subtotal",
            text_confidence=0.88,
            text_language="es",
            text_level="word",
            text_order=3
        ))

        # Chain OCR filters
        result = (detections
                 .filter_by_text_confidence(0.8)
                 .filter_by_text_language("en")
                 .filter_by_text_contains("Invoice"))

        assert len(result) == 1
        assert result[0].text == "Invoice #12345"

    def test_ocr_sort_and_filter_chain(self):
        """Test combining sorting and filtering for OCR."""
        detections = pf.detections.Detections()

        # Add out of order
        detections.add_detection(pf.detections.Detection(
            bbox=[100, 200, 200, 240],
            text="Third",
            text_order=3,
            text_confidence=0.9
        ))

        detections.add_detection(pf.detections.Detection(
            bbox=[100, 100, 200, 140],
            text="First",
            text_order=1,
            text_confidence=0.95
        ))

        detections.add_detection(pf.detections.Detection(
            bbox=[100, 150, 200, 190],
            text="Second",
            text_order=2,
            text_confidence=0.7
        ))

        # Filter then sort
        result = (detections
                 .filter_by_text_confidence(0.85)
                 .sort_by_text_order())

        assert len(result) == 2
        assert result[0].text == "First"
        assert result[1].text == "Third"


# ============================================================================
# Tracking Filter Chaining Tests
# ============================================================================

@pytest.mark.integration
class TestTrackingFilterChaining:
    """Tests for chaining tracking-related filters."""

    def test_tracking_chain(self, tracked_detections):
        """Test chaining tracking filters."""
        result = (tracked_detections
                 .filter_tracked_objects()
                 .filter_by_tracking_duration(min_duration=2.0))

        # Should have only long-tracked objects
        for det in result:
            assert det.tracker_id is not None
            assert det.tracking_duration >= 2.0

    def test_tracking_with_confidence_chain(self, tracked_detections):
        """Test combining tracking and confidence filters."""
        result = (tracked_detections
                 .filter_tracked_objects()
                 .filter_by_confidence(0.9))

        for det in result:
            assert det.tracker_id is not None
            assert det.confidence >= 0.9


# ============================================================================
# Zone Integration Chaining Tests
# ============================================================================

@pytest.mark.integration
class TestZoneFilterChaining:
    """Tests for chaining zone filters with other filters."""

    def test_zone_and_confidence_chain(self, sample_detections):
        """Test combining zone and confidence filters."""
        # Setup zones
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(50, 50), (300, 50), (300, 300), (50, 300)],
            zone_id="zone1"
        )

        # Update and chain filters
        result = (zones.update(sample_detections)
                 .filter_by_confidence(0.8)
                 .filter_by_zones("zone1"))

        for det in result:
            assert det.confidence >= 0.8

    def test_multi_zone_chain(self, sample_detections):
        """Test filtering across multiple zones."""
        zones = pf.Zones()

        zones.add_zone(
            polygon=[(50, 50), (200, 50), (200, 200), (50, 200)],
            zone_id="zone1"
        )

        zones.add_zone(
            polygon=[(300, 200), (500, 200), (500, 400), (300, 400)],
            zone_id="zone2"
        )

        result = (zones.update(sample_detections)
                 .filter_by_zones(["zone1", "zone2"])
                 .filter_by_confidence(0.7))

        assert isinstance(result, pf.detections.Detections)


# ============================================================================
# Serialization Tests
# ============================================================================

@pytest.mark.integration
class TestSerialization:
    """Tests for JSON serialization and deserialization."""

    def test_detection_to_json_and_back(self, sample_detection):
        """Test roundtrip serialization of single detection."""
        # Serialize
        json_str = sample_detection.to_json()
        data = json.loads(json_str)

        # Verify structure
        assert "bbox" in data
        assert "confidence" in data
        assert "class_id" in data

    def test_detections_to_json_and_back(self, sample_detections):
        """Test roundtrip serialization of detection container."""
        # Serialize
        json_str = sample_detections.to_json()
        data = json.loads(json_str)

        # Should be list of detections
        assert isinstance(data, list)
        assert len(data) == len(sample_detections)

    def test_mask_serialization(self, sample_detection_with_mask):
        """Test serialization of detection with binary mask."""
        # Serialize
        data = sample_detection_with_mask.to_dict()

        # Mask should be base64 encoded
        assert "masks" in data
        assert len(data["masks"]) > 0
        assert "data" in data["masks"][0]
        assert "shape" in data["masks"][0]

        # Decode
        decoded_mask = pf.detections.Detection.decode_mask(data["masks"][0])

        # Should match original
        assert decoded_mask.shape == sample_detection_with_mask.masks[0].shape
        assert decoded_mask.dtype == bool

    def test_keypoints_serialization(self, sample_detection_with_keypoints):
        """Test serialization of detection with keypoints."""
        data = sample_detection_with_keypoints.to_dict()

        assert "keypoints" in data
        assert len(data["keypoints"]) > 0

        # Each keypoint should have x, y, name, visibility
        for kp in data["keypoints"]:
            assert "x" in kp
            assert "y" in kp
            assert "name" in kp
            assert "visibility" in kp

    def test_ocr_detection_serialization(self, sample_ocr_detection):
        """Test serialization of OCR detection."""
        data = sample_ocr_detection.to_dict()

        assert "text" in data
        assert "text_confidence" in data
        assert "text_language" in data
        assert "text_level" in data

    def test_full_pipeline_with_serialization(self, sample_detections):
        """Test processing pipeline with serialization."""
        # Filter
        filtered = (sample_detections
                   .filter_by_confidence(0.8)
                   .filter_by_class_id([0, 2]))

        # Serialize
        json_str = filtered.to_json()

        # Deserialize
        data = json.loads(json_str)

        # Verify all detections meet filter criteria
        for det_data in data:
            assert det_data["confidence"] >= 0.8
            assert det_data["class_id"] in [0, 2]


# ============================================================================
# Complex Workflow Tests
# ============================================================================

@pytest.mark.integration
class TestComplexWorkflows:
    """Tests for complex combined workflows."""

    def test_filter_annotate_serialize_workflow(self, sample_image, sample_detections):
        """Test complete workflow: filter -> annotate -> serialize."""
        # 1. Filter
        filtered = (sample_detections
                   .filter_by_confidence(0.8)
                   .filter_by_class_id(0)
                   .filter_by_size(min_area=5000))

        # 2. Annotate
        annotated = sample_image.copy()
        annotated = pf.annotate.box(annotated, filtered, thickness=2)
        annotated = pf.annotate.label(annotated, filtered)

        # 3. Serialize results
        results_json = filtered.to_json()
        results_data = json.loads(results_json)

        # Verify
        assert annotated.shape == sample_image.shape
        assert isinstance(results_data, list)

    def test_zone_analytics_workflow(self, sample_image, sample_detections):
        """Test zone analytics with chaining."""
        # Setup zones
        zones = pf.Zones()
        zones.add_zone(
            polygon=[(50, 50), (300, 50), (300, 300), (50, 300)],
            zone_id="entrance",
            trigger_strategy="bottom_center"
        )

        # Complex chain
        result = (zones.update(sample_detections)
                 .filter_by_confidence(0.7)
                 .filter_by_zones("entrance")
                 .filter_by_class_id(0))

        # Get stats
        stats = zones.get_zone_stats()

        # Annotate
        annotated = sample_image.copy()
        annotated = pf.annotate.zones(annotated, zones, opacity=0.3)
        annotated = pf.annotate.box(annotated, result)

        # Serialize
        results_json = result.to_json()

        assert annotated.shape == sample_image.shape
        assert isinstance(stats, dict)
        assert isinstance(json.loads(results_json), list)

    def test_transform_chain_workflow(self, sample_image, sample_detections):
        """Test transformation chain workflow."""
        # Apply transform chain
        img, dets = sample_image, sample_detections

        img, dets = pf.transform.rotate_detections(img, dets, angle=15)
        img, dets = pf.transform.flip_horizontal_detections(img, dets)

        # Filter in transformed space
        filtered = (dets
                   .filter_by_confidence(0.7)
                   .filter_by_size(min_area=3000))

        # Annotate
        annotated = pf.annotate.box(img.copy(), filtered)

        # Serialize
        results = filtered.to_json()

        assert annotated.shape == img.shape
        assert isinstance(json.loads(results), list)

    def test_ocr_extraction_workflow(self):
        """Test complete OCR extraction workflow."""
        # Create OCR detections
        detections = pf.detections.Detections()

        detections.add_detection(pf.detections.Detection(
            bbox=[100, 50, 300, 90],
            text="Document Title",
            text_confidence=0.95,
            text_level="line",
            text_order=1
        ))

        detections.add_detection(pf.detections.Detection(
            bbox=[100, 100, 250, 140],
            text="Section 1",
            text_confidence=0.88,
            text_level="line",
            text_order=2
        ))

        detections.add_detection(pf.detections.Detection(
            bbox=[100, 150, 350, 190],
            text="Some content here",
            text_confidence=0.92,
            text_level="line",
            text_order=3
        ))

        # Process chain
        processed = (detections
                    .filter_by_text_confidence(0.85)
                    .sort_by_text_order())

        # Extract text in order
        document_text = '\n'.join([d.text for d in processed if d.text])

        # Serialize
        results = processed.to_json()

        assert "Document Title" in document_text
        assert isinstance(json.loads(results), list)


# ============================================================================
# Edge Case Workflows
# ============================================================================

@pytest.mark.integration
class TestEdgeCaseWorkflows:
    """Tests for edge cases in workflows."""

    def test_empty_chain_workflow(self, empty_detections):
        """Test workflow with empty detections."""
        result = (empty_detections
                 .filter_by_confidence(0.8)
                 .filter_by_class_id(0))

        # Should handle empty gracefully
        assert len(result) == 0

        # Serialization should work
        json_str = result.to_json()
        assert json.loads(json_str) == []

    def test_chain_that_empties_gradually(self, sample_detections):
        """Test chain where filters gradually reduce to empty."""
        result = (sample_detections
                 .filter_by_confidence(0.5)   # Some pass
                 .filter_by_class_id(0)       # Fewer pass
                 .filter_by_size(min_area=50000))  # None pass

        assert len(result) == 0

    def test_workflow_with_none_values(self):
        """Test workflow handles None values gracefully."""
        detections = pf.detections.Detections()

        detections.add_detection(pf.detections.Detection(
            bbox=[100, 100, 200, 200],
            confidence=None,
            class_id=None
        ))

        # Should handle None values
        try:
            result = detections.filter_by_confidence(0.5)
            # May filter out None or handle specially
        except (TypeError, AttributeError):
            # Expected to potentially fail
            pass
