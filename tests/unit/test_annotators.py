"""
Unit tests for pixelflow.annotators module.

Tests all annotation functions including box, label, mask, blur,
pixelate, polygon, oval, zones, and crossings.
"""

import pytest
import numpy as np
import cv2
import pixelflow as pf


# ============================================================================
# Box Annotator Tests
# ============================================================================

class TestBoxAnnotator:
    """Tests for box annotator."""

    def test_box_basic(self, blank_image, sample_detections):
        """Test basic box annotation."""
        annotated = pf.annotate.box(blank_image.copy(), sample_detections)

        # Image should be modified
        assert annotated.shape == blank_image.shape
        # Should have drawn something (image changed)
        assert not np.array_equal(annotated, blank_image)

    def test_box_with_thickness(self, blank_image, sample_detection):
        """Test box with custom thickness."""
        detections = pf.Detections()
        detections.add_detection(sample_detection)

        annotated = pf.annotate.box(
            blank_image.copy(),
            detections,
            thickness=5
        )

        assert annotated.shape == blank_image.shape

    def test_box_with_color(self, blank_image, sample_detection):
        """Test box with custom color."""
        detections = pf.Detections()
        detections.add_detection(sample_detection)

        annotated = pf.annotate.box(
            blank_image.copy(),
            detections,
            colors=[(255, 0, 0)]  # Red in RGB
        )

        assert annotated.shape == blank_image.shape

    def test_box_empty_detections(self, blank_image, empty_detections):
        """Test box annotation with no detections."""
        annotated = pf.annotate.box(blank_image.copy(), empty_detections)

        # Should return unchanged image
        assert np.array_equal(annotated, blank_image)

    def test_box_returns_ndarray(self, blank_image, sample_detections):
        """Test that box returns numpy array."""
        annotated = pf.annotate.box(blank_image.copy(), sample_detections)
        assert isinstance(annotated, np.ndarray)
        assert annotated.dtype == np.uint8


# ============================================================================
# Label Annotator Tests
# ============================================================================

class TestLabelAnnotator:
    """Tests for label annotator."""

    def test_label_basic(self, blank_image, sample_detections):
        """Test basic label annotation."""
        annotated = pf.annotate.label(blank_image.copy(), sample_detections)

        assert annotated.shape == blank_image.shape
        assert not np.array_equal(annotated, blank_image)

    def test_label_with_custom_labels(self, blank_image, sample_detection):
        """Test labels with custom text."""
        sample_detection.label = "Custom Label"
        detections = pf.Detections()
        detections.add_detection(sample_detection)

        annotated = pf.annotate.label(blank_image.copy(), detections)
        assert annotated.shape == blank_image.shape

    def test_label_position(self, blank_image, sample_detection):
        """Test label with different positions."""
        detections = pf.Detections()
        detections.add_detection(sample_detection)

        # Top position
        annotated_top = pf.annotate.label(
            blank_image.copy(),
            detections,
            position="top"
        )

        # Bottom position
        annotated_bottom = pf.annotate.label(
            blank_image.copy(),
            detections,
            position="bottom"
        )

        # Results should be different
        assert not np.array_equal(annotated_top, annotated_bottom)

    def test_label_empty_detections(self, blank_image, empty_detections):
        """Test label with no detections."""
        annotated = pf.annotate.label(blank_image.copy(), empty_detections)
        assert np.array_equal(annotated, blank_image)


# ============================================================================
# Mask Annotator Tests
# ============================================================================

class TestMaskAnnotator:
    """Tests for mask annotator."""

    def test_mask_basic(self, blank_image, sample_detection_with_mask):
        """Test basic mask annotation."""
        detections = pf.Detections()
        detections.add_detection(sample_detection_with_mask)

        annotated = pf.annotate.mask(blank_image.copy(), detections)

        assert annotated.shape == blank_image.shape
        assert not np.array_equal(annotated, blank_image)

    def test_mask_with_opacity(self, blank_image, sample_detection_with_mask):
        """Test mask with different opacity values."""
        detections = pf.Detections()
        detections.add_detection(sample_detection_with_mask)

        annotated_low = pf.annotate.mask(
            blank_image.copy(),
            detections,
            opacity=0.2
        )

        annotated_high = pf.annotate.mask(
            blank_image.copy(),
            detections,
            opacity=0.8
        )

        # Different opacities should produce different results
        assert not np.array_equal(annotated_low, annotated_high)

    def test_mask_no_masks(self, blank_image, sample_detection):
        """Test mask annotator with detections without masks."""
        detections = pf.Detections()
        detections.add_detection(sample_detection)  # No mask

        annotated = pf.annotate.mask(blank_image.copy(), detections)

        # Should return image unchanged if no masks
        # Or handle gracefully
        assert annotated.shape == blank_image.shape


# ============================================================================
# Blur Annotator Tests
# ============================================================================

class TestBlurAnnotator:
    """Tests for blur (privacy) annotator."""

    def test_blur_basic(self, sample_image, sample_detections):
        """Test basic blur annotation."""
        annotated = pf.annotate.blur(sample_image.copy(), sample_detections)

        assert annotated.shape == sample_image.shape
        assert not np.array_equal(annotated, sample_image)

    def test_blur_strength(self, sample_image, sample_detection):
        """Test blur with different strengths."""
        detections = pf.Detections()
        detections.add_detection(sample_detection)

        annotated_weak = pf.annotate.blur(
            sample_image.copy(),
            detections,
            kernel_size=15
        )

        annotated_strong = pf.annotate.blur(
            sample_image.copy(),
            detections,
            kernel_size=51
        )

        # Different strengths should produce different results
        assert not np.array_equal(annotated_weak, annotated_strong)

    def test_blur_empty_detections(self, sample_image, empty_detections):
        """Test blur with no detections."""
        annotated = pf.annotate.blur(sample_image.copy(), empty_detections)
        assert np.array_equal(annotated, sample_image)


# ============================================================================
# Pixelate Annotator Tests
# ============================================================================

class TestPixelateAnnotator:
    """Tests for pixelate (privacy) annotator."""

    def test_pixelate_basic(self, sample_image, sample_detections):
        """Test basic pixelate annotation."""
        annotated = pf.annotate.pixelate(sample_image.copy(), sample_detections)

        assert annotated.shape == sample_image.shape
        assert not np.array_equal(annotated, sample_image)

    def test_pixelate_pixel_size(self, sample_image, sample_detection):
        """Test pixelate with different pixel sizes."""
        detections = pf.Detections()
        detections.add_detection(sample_detection)

        annotated_small = pf.annotate.pixelate(
            sample_image.copy(),
            detections,
            pixel_size=10
        )

        annotated_large = pf.annotate.pixelate(
            sample_image.copy(),
            detections,
            pixel_size=30
        )

        # Different pixel sizes should produce different results
        assert not np.array_equal(annotated_small, annotated_large)

    def test_pixelate_empty_detections(self, sample_image, empty_detections):
        """Test pixelate with no detections."""
        annotated = pf.annotate.pixelate(sample_image.copy(), empty_detections)
        assert np.array_equal(annotated, sample_image)


# ============================================================================
# Polygon Annotator Tests
# ============================================================================

class TestPolygonAnnotator:
    """Tests for polygon annotator."""

    def test_polygon_basic(self, blank_image):
        """Test basic polygon annotation."""
        # Create detection with polygon segments
        polygon = [(100, 100), (200, 100), (200, 200), (100, 200)]
        det = pf.Detection(
            bbox=[100, 100, 200, 200],
            segments=polygon
        )
        detections = pf.Detections()
        detections.add_detection(det)

        annotated = pf.annotate.polygon(blank_image.copy(), detections)

        assert annotated.shape == blank_image.shape
        assert not np.array_equal(annotated, blank_image)

    def test_polygon_thickness(self, blank_image):
        """Test polygon with custom thickness."""
        polygon = [(100, 100), (200, 100), (200, 200), (100, 200)]
        det = pf.Detection(bbox=[100, 100, 200, 200], segments=polygon)
        detections = pf.Detections()
        detections.add_detection(det)

        annotated = pf.annotate.polygon(
            blank_image.copy(),
            detections,
            thickness=3
        )

        assert annotated.shape == blank_image.shape

    def test_polygon_no_segments(self, blank_image, sample_detection):
        """Test polygon with detections without segments."""
        detections = pf.Detections()
        detections.add_detection(sample_detection)  # No segments

        annotated = pf.annotate.polygon(blank_image.copy(), detections)
        # Should handle gracefully
        assert annotated.shape == blank_image.shape


# ============================================================================
# Oval Annotator Tests
# ============================================================================

class TestOvalAnnotator:
    """Tests for oval/ellipse annotator."""

    def test_oval_basic(self, blank_image, sample_detections):
        """Test basic oval annotation."""
        annotated = pf.annotate.oval(blank_image.copy(), sample_detections)

        assert annotated.shape == blank_image.shape
        assert not np.array_equal(annotated, blank_image)

    def test_oval_thickness(self, blank_image, sample_detection):
        """Test oval with custom thickness."""
        detections = pf.Detections()
        detections.add_detection(sample_detection)

        annotated = pf.annotate.oval(
            blank_image.copy(),
            detections,
            thickness=3
        )

        assert annotated.shape == blank_image.shape

    def test_oval_filled(self, blank_image, sample_detection):
        """Test filled oval."""
        detections = pf.Detections()
        detections.add_detection(sample_detection)

        annotated = pf.annotate.oval(
            blank_image.copy(),
            detections,
            thickness=-1  # Filled
        )

        assert annotated.shape == blank_image.shape


# ============================================================================
# Anchors Annotator Tests
# ============================================================================

class TestAnchorsAnnotator:
    """Tests for anchors annotator."""

    def test_anchors_basic(self, blank_image, sample_detections):
        """Test basic anchor points visualization."""
        annotated = pf.annotate.anchors(blank_image.copy(), sample_detections)

        assert annotated.shape == blank_image.shape
        assert not np.array_equal(annotated, blank_image)

    def test_anchors_radius(self, blank_image, sample_detection):
        """Test anchors with custom radius."""
        detections = pf.Detections()
        detections.add_detection(sample_detection)

        annotated = pf.annotate.anchors(
            blank_image.copy(),
            detections,
            radius=10
        )

        assert annotated.shape == blank_image.shape


# ============================================================================
# Zones Annotator Tests
# ============================================================================

class TestZonesAnnotator:
    """Tests for zones annotator."""

    def test_zones_basic(self, blank_image, sample_zones):
        """Test basic zone visualization."""
        annotated = pf.annotate.zones(blank_image.copy(), sample_zones)

        assert annotated.shape == blank_image.shape
        assert not np.array_equal(annotated, blank_image)

    def test_zones_opacity(self, blank_image, sample_zones):
        """Test zones with different opacity."""
        annotated_low = pf.annotate.zones(
            blank_image.copy(),
            sample_zones,
            opacity=0.2
        )

        annotated_high = pf.annotate.zones(
            blank_image.copy(),
            sample_zones,
            opacity=0.8
        )

        # Different opacities should produce different results
        assert not np.array_equal(annotated_low, annotated_high)


# ============================================================================
# Crossings Annotator Tests
# ============================================================================

class TestCrossingsAnnotator:
    """Tests for crossings (line crossing) annotator."""

    def test_crossings_basic(self, blank_image):
        """Test basic crossings visualization."""
        # Create crossings
        crossings = pf.Crossings()
        crossings.add_crossing(
            start=(100, 240),
            end=(540, 240),
            line_id="line1"
        )

        annotated = pf.annotate.crossings(blank_image.copy(), crossings)

        assert annotated.shape == blank_image.shape
        assert not np.array_equal(annotated, blank_image)

    def test_crossings_multiple(self, blank_image):
        """Test crossings with multiple lines."""
        crossings = pf.Crossings()
        crossings.add_crossing(
            start=(100, 240),
            end=(540, 240),
            line_id="line1"
        )
        crossings.add_crossing(
            start=(320, 100),
            end=(320, 400),
            line_id="line2"
        )

        annotated = pf.annotate.crossings(
            blank_image.copy(),
            crossings
        )

        assert annotated.shape == blank_image.shape


# ============================================================================
# Multi-Layer Annotation Tests
# ============================================================================

class TestMultiLayerAnnotation:
    """Tests for combining multiple annotators."""

    def test_layer_mask_box_label(self, blank_image, sample_detection_with_mask):
        """Test layering mask, box, and label."""
        detections = pf.Detections()
        detections.add_detection(sample_detection_with_mask)

        # Layer annotations
        annotated = blank_image.copy()
        annotated = pf.annotate.mask(annotated, detections, opacity=0.3)
        annotated = pf.annotate.box(annotated, detections, thickness=2)
        annotated = pf.annotate.label(annotated, detections)

        assert annotated.shape == blank_image.shape
        assert not np.array_equal(annotated, blank_image)

    def test_layer_blur_box(self, sample_image, sample_detections):
        """Test layering blur and box."""
        annotated = sample_image.copy()
        annotated = pf.annotate.blur(annotated, sample_detections)
        annotated = pf.annotate.box(annotated, sample_detections)

        assert annotated.shape == sample_image.shape

    def test_privacy_pipeline(self, sample_image, sample_detections):
        """Test privacy annotation pipeline."""
        # Half blur, half pixelate
        blur_dets = sample_detections[:2]
        pixelate_dets = sample_detections[2:]

        annotated = sample_image.copy()
        annotated = pf.annotate.blur(annotated, blur_dets)
        annotated = pf.annotate.pixelate(annotated, pixelate_dets)

        assert annotated.shape == sample_image.shape


# ============================================================================
# Performance and Edge Cases
# ============================================================================

class TestAnnotatorEdgeCases:
    """Test edge cases for annotators."""

    def test_annotator_with_small_image(self, small_image, sample_detection):
        """Test annotators on very small images."""
        # Adjust detection to fit small image
        det = pf.Detection(bbox=[10, 10, 90, 90])
        detections = pf.Detections()
        detections.add_detection(det)

        # Should handle small images gracefully
        annotated = pf.annotate.box(small_image.copy(), detections)
        assert annotated.shape == small_image.shape

    def test_annotator_with_out_of_bounds_bbox(self, blank_image):
        """Test annotator with bbox outside image boundaries."""
        # Detection extending beyond image
        det = pf.Detection(bbox=[600, 450, 700, 550])
        detections = pf.Detections()
        detections.add_detection(det)

        # Should handle gracefully without crashing
        annotated = pf.annotate.box(blank_image.copy(), detections)
        assert annotated.shape == blank_image.shape

    def test_annotator_modifies_copy_not_original(self, blank_image, sample_detection):
        """Test that annotators don't modify original if copy is passed."""
        detections = pf.Detections()
        detections.add_detection(sample_detection)

        original = blank_image.copy()
        annotated = pf.annotate.box(original.copy(), detections)

        # Original should be unchanged if we passed a copy
        # (Though annotators may modify in-place, so this tests the pattern)
        assert annotated.shape == original.shape
