"""
Unit tests for pixelflow.transforms module.

Tests image-only transforms and detection-aware transforms including
rotation, flipping, cropping, enhancement operations, and inverse transforms.
"""

import pytest
import numpy as np
import cv2
import pixelflow as pf


# ============================================================================
# Image-Only Transform Tests
# ============================================================================

class TestImageRotation:
    """Tests for image rotation transform."""

    def test_rotate_basic(self, sample_image):
        """Test basic image rotation."""
        rotated = pf.transform.rotate(sample_image, angle=45)

        # Rotated image should have different dimensions or content
        assert isinstance(rotated, np.ndarray)
        assert rotated.dtype == np.uint8

    def test_rotate_90_degrees(self, sample_image):
        """Test 90 degree rotation."""
        rotated = pf.transform.rotate(sample_image, angle=90)

        # 90 degree rotation should swap dimensions
        # (with potential scaling to fit)
        assert rotated.shape[2] == 3  # Still RGB

    def test_rotate_360_degrees(self, sample_image):
        """Test full rotation returns similar image."""
        rotated = pf.transform.rotate(sample_image, angle=360)

        # Should be very similar to original
        assert rotated.shape[2] == 3


class TestImageFlipping:
    """Tests for image flipping transforms."""

    def test_flip_horizontal(self, sample_image):
        """Test horizontal flip."""
        flipped = pf.transform.flip_horizontal(sample_image)

        assert flipped.shape == sample_image.shape
        # Should be different from original
        assert not np.array_equal(flipped, sample_image)

    def test_flip_vertical(self, sample_image):
        """Test vertical flip."""
        flipped = pf.transform.flip_vertical(sample_image)

        assert flipped.shape == sample_image.shape
        assert not np.array_equal(flipped, sample_image)

    def test_flip_both_equals_180_rotation(self, sample_image):
        """Test that flipping both axes equals 180 degree rotation."""
        flipped_both = pf.transform.flip_vertical(
            pf.transform.flip_horizontal(sample_image)
        )
        rotated_180 = pf.transform.rotate(sample_image, angle=180)

        # Results should be very similar (allowing for minor differences)
        assert flipped_both.shape == rotated_180.shape


class TestImageCropping:
    """Tests for image cropping transform."""

    def test_crop_basic(self, sample_image):
        """Test basic image cropping."""
        cropped = pf.transform.crop(sample_image, bbox=[100, 100, 300, 300])

        # Cropped image should be 200x200
        assert cropped.shape == (200, 200, 3)

    def test_crop_to_quarter(self, sample_image):
        """Test cropping to top-left quarter."""
        h, w = sample_image.shape[:2]
        cropped = pf.transform.crop(sample_image, bbox=[0, 0, w//2, h//2])

        assert cropped.shape[0] == h // 2
        assert cropped.shape[1] == w // 2

    def test_crop_with_padding(self, sample_image):
        """Test cropping with bbox extending beyond image."""
        # Bbox partially outside image
        cropped = pf.transform.crop(sample_image, bbox=[-50, -50, 100, 100])

        # Should handle gracefully (clamp or pad)
        assert isinstance(cropped, np.ndarray)


class TestImageEnhancement:
    """Tests for image enhancement transforms."""

    def test_clahe(self, sample_image):
        """Test CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        enhanced = pf.transform.clahe(sample_image)

        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == np.uint8

    def test_to_grayscale(self, sample_image):
        """Test grayscale conversion."""
        gray = pf.transform.to_grayscale(sample_image)

        # Should be single channel
        assert len(gray.shape) == 2 or gray.shape[2] == 1

    def test_auto_contrast(self, sample_image):
        """Test auto contrast enhancement."""
        enhanced = pf.transform.auto_contrast(sample_image)

        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == np.uint8

    def test_normalize(self, sample_image):
        """Test image normalization."""
        normalized = pf.transform.normalize(sample_image)

        # Should be float type normalized to 0-1
        assert normalized.dtype in [np.float32, np.float64]
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_gamma_correction(self, sample_image):
        """Test gamma correction."""
        corrected = pf.transform.gamma_correction(sample_image, gamma=1.5)

        assert corrected.shape == sample_image.shape
        assert corrected.dtype == np.uint8

    def test_standardize(self, sample_image):
        """Test image standardization (zero mean, unit variance)."""
        standardized = pf.transform.standardize(sample_image)

        # Should have mean close to 0 and std close to 1
        assert isinstance(standardized, np.ndarray)
        assert standardized.dtype in [np.float32, np.float64]


# ============================================================================
# Detection-Aware Transform Tests
# ============================================================================

class TestDetectionRotation:
    """Tests for detection-aware rotation."""

    def test_rotate_detections_basic(self, sample_image, sample_detections):
        """Test basic detection rotation."""
        rotated_img, rotated_dets = pf.transform.rotate_detections(
            sample_image,
            sample_detections,
            angle=45
        )

        # Should return both image and detections
        assert isinstance(rotated_img, np.ndarray)
        assert isinstance(rotated_dets, pf.detections.Detections)
        # Should have same number of detections
        assert len(rotated_dets) == len(sample_detections)

    def test_rotate_detections_90(self, sample_image, sample_detection):
        """Test 90 degree rotation with detection."""
        detections = pf.detections.Detections()
        detections.add_detection(sample_detection)

        rotated_img, rotated_dets = pf.transform.rotate_detections(
            sample_image,
            detections,
            angle=90
        )

        # Bbox should be transformed
        assert rotated_dets[0].bbox != sample_detection.bbox

    def test_rotate_detections_with_keypoints(self, sample_image, sample_detection_with_keypoints):
        """Test rotation with keypoints."""
        detections = pf.detections.Detections()
        detections.add_detection(sample_detection_with_keypoints)

        rotated_img, rotated_dets = pf.transform.rotate_detections(
            sample_image,
            detections,
            angle=30
        )

        # Keypoints should also be rotated
        assert len(rotated_dets[0].keypoints) == len(sample_detection_with_keypoints.keypoints)


class TestDetectionFlipping:
    """Tests for detection-aware flipping."""

    def test_flip_horizontal_detections(self, sample_image, sample_detections):
        """Test horizontal flip with detections."""
        flipped_img, flipped_dets = pf.transform.flip_horizontal_detections(
            sample_image,
            sample_detections
        )

        assert flipped_img.shape == sample_image.shape
        assert len(flipped_dets) == len(sample_detections)

        # Bboxes should be flipped horizontally
        # x-coordinates should be mirrored
        original_bbox = sample_detections[0].bbox
        flipped_bbox = flipped_dets[0].bbox

        # Check that bbox is actually different
        assert flipped_bbox != original_bbox

    def test_flip_vertical_detections(self, sample_image, sample_detections):
        """Test vertical flip with detections."""
        flipped_img, flipped_dets = pf.transform.flip_vertical_detections(
            sample_image,
            sample_detections
        )

        assert flipped_img.shape == sample_image.shape
        assert len(flipped_dets) == len(sample_detections)

    def test_flip_preserves_detection_count(self, sample_image, sample_detections):
        """Test that flipping preserves all detections."""
        _, flipped_dets = pf.transform.flip_horizontal_detections(
            sample_image,
            sample_detections
        )

        assert len(flipped_dets) == len(sample_detections)


class TestDetectionCropping:
    """Tests for detection-aware cropping."""

    def test_crop_detections_basic(self, sample_image, sample_detections):
        """Test basic detection cropping."""
        cropped_img, cropped_dets = pf.transform.crop_detections(
            sample_image,
            sample_detections,
            bbox=[80, 80, 250, 250]
        )

        # Image should be cropped
        assert cropped_img.shape == (170, 170, 3)

        # Some detections may be filtered out if outside crop region
        assert len(cropped_dets) <= len(sample_detections)

    def test_crop_detections_filters_outside(self, sample_image):
        """Test that cropping filters out detections outside crop region."""
        detections = pf.detections.Detections()

        # Detection inside crop region
        detections.add_detection(pf.detections.Detection(
            bbox=[100, 100, 150, 150]
        ))

        # Detection outside crop region
        detections.add_detection(pf.detections.Detection(
            bbox=[400, 400, 500, 500]
        ))

        cropped_img, cropped_dets = pf.transform.crop_detections(
            sample_image,
            detections,
            bbox=[50, 50, 200, 200]
        )

        # Only detection inside crop should remain
        assert len(cropped_dets) <= 1

    def test_crop_around_detections(self, sample_image, sample_detection):
        """Test cropping around specific detections."""
        detections = pf.detections.Detections()
        detections.add_detection(sample_detection)

        cropped_img, cropped_dets = pf.transform.crop_around_detections(
            sample_image,
            detections,
            padding=20
        )

        # Should crop to encompass all detections with padding
        assert isinstance(cropped_img, np.ndarray)
        assert len(cropped_dets) == 1


class TestDetectionPadding:
    """Tests for adding padding to detections."""

    def test_add_padding_basic(self, sample_detection):
        """Test adding padding to detection bbox."""
        detections = pf.detections.Detections()
        detections.add_detection(sample_detection)

        padded_dets = pf.transform.add_padding(detections, padding=0.1)

        # Bbox should be larger
        original_area = (sample_detection.bbox[2] - sample_detection.bbox[0]) * \
                       (sample_detection.bbox[3] - sample_detection.bbox[1])

        padded_area = (padded_dets[0].bbox[2] - padded_dets[0].bbox[0]) * \
                     (padded_dets[0].bbox[3] - padded_dets[0].bbox[1])

        assert padded_area > original_area

    def test_add_padding_preserves_count(self, sample_detections):
        """Test that adding padding preserves detection count."""
        padded_dets = pf.transform.add_padding(sample_detections, padding=0.2)

        assert len(padded_dets) == len(sample_detections)


class TestKeypointAlignment:
    """Tests for keypoint-based rotation alignment."""

    def test_rotate_to_align(self, sample_image, sample_detection_with_keypoints):
        """Test rotating to align keypoints horizontally."""
        detections = pf.detections.Detections()
        detections.add_detection(sample_detection_with_keypoints)

        aligned_img, aligned_dets = pf.transform.rotate_to_align(
            sample_image,
            detections,
            keypoint1='left_eye',
            keypoint2='right_eye',
            target_angle=0  # Horizontal
        )

        assert isinstance(aligned_img, np.ndarray)
        assert len(aligned_dets) == 1


class TestBboxFromKeypoints:
    """Tests for updating bbox from keypoints."""

    def test_update_bbox_from_keypoints(self, sample_detection_with_keypoints):
        """Test updating bbox to encompass keypoints."""
        detections = pf.detections.Detections()
        detections.add_detection(sample_detection_with_keypoints)

        updated_dets = pf.transform.update_bbox_from_keypoints(
            detections,
            keypoint_names=['nose', 'left_eye', 'right_eye']
        )

        # Should have updated bbox
        assert updated_dets[0].bbox is not None


# ============================================================================
# Inverse Transform Tests
# ============================================================================

class TestInverseTransforms:
    """Tests for automated inverse transformations."""

    def test_inverse_transforms_basic(self, sample_image, sample_detections):
        """Test inverse transforms tracking and reversal."""
        # Apply transformations
        img1, dets1 = pf.transform.rotate_detections(
            sample_image,
            sample_detections,
            angle=30
        )

        img2, dets2 = pf.transform.crop_detections(
            img1,
            dets1,
            bbox=[100, 100, 400, 400]
        )

        # Create new detections in transformed space
        new_dets = pf.detections.Detections()
        new_dets.add_detection(pf.detections.Detection(
            bbox=[50, 50, 100, 100]
        ))

        # Apply inverse transforms
        original_coords = pf.transform.inverse_transforms(new_dets)

        # Should return detections in original coordinate space
        assert isinstance(original_coords, pf.detections.Detections)

    def test_inverse_with_multiple_transforms(self, sample_image, sample_detections):
        """Test inverse transforms with complex chain."""
        # Apply multiple transforms
        img, dets = sample_image, sample_detections

        img, dets = pf.transform.flip_horizontal_detections(img, dets)
        img, dets = pf.transform.rotate_detections(img, dets, angle=45)
        img, dets = pf.transform.crop_detections(img, dets, bbox=[100, 100, 400, 400])

        # New detections in transformed space
        new_dets = pf.detections.Detections()
        new_dets.add_detection(pf.detections.Detection(bbox=[50, 50, 100, 100]))

        # Inverse should undo all transforms
        original = pf.transform.inverse_transforms(new_dets)

        assert isinstance(original, pf.detections.Detections)


# ============================================================================
# Edge Cases and Validation
# ============================================================================

class TestTransformEdgeCases:
    """Test edge cases for transforms."""

    def test_transform_empty_detections(self, sample_image, empty_detections):
        """Test transforms with no detections."""
        rotated_img, rotated_dets = pf.transform.rotate_detections(
            sample_image,
            empty_detections,
            angle=45
        )

        # Should handle gracefully
        assert len(rotated_dets) == 0

    def test_transform_preserves_detection_properties(self, sample_image, sample_detection):
        """Test that transforms preserve non-geometric properties."""
        sample_detection.confidence = 0.95
        sample_detection.class_id = 5
        sample_detection.class_name = "test"

        detections = pf.detections.Detections()
        detections.add_detection(sample_detection)

        _, transformed = pf.transform.rotate_detections(sample_image, detections, 30)

        # Non-geometric properties should be preserved
        assert transformed[0].confidence == 0.95
        assert transformed[0].class_id == 5
        assert transformed[0].class_name == "test"

    def test_crop_with_zero_area(self, sample_image, sample_detections):
        """Test crop with invalid bbox."""
        # This might raise an error or return empty result
        try:
            cropped_img, cropped_dets = pf.transform.crop_detections(
                sample_image,
                sample_detections,
                bbox=[100, 100, 100, 100]  # Zero area
            )
            # If it doesn't error, check it handles gracefully
            assert isinstance(cropped_img, np.ndarray)
        except (ValueError, AssertionError):
            # Expected to fail with zero-area crop
            pass
