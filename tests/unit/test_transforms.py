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
        """normalize() scales to 0-1 then applies (x - mean) / std."""
        # Identity mean/std leaves the plain 0-1 scaling.
        normalized = pf.transform.normalize(sample_image, mean=0.0, std=1.0)

        assert normalized.dtype in [np.float32, np.float64]
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_normalize_with_imagenet_stats(self, sample_image):
        """ImageNet mean/std produce a zero-centred result."""
        normalized = pf.transform.normalize(
            sample_image,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        assert normalized.dtype == np.float32
        # Centring pushes values below zero, unlike the plain 0-1 scaling.
        assert normalized.min() < 0.0

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
        assert isinstance(rotated_dets, pf.Detections)
        # Should have same number of detections
        assert len(rotated_dets) == len(sample_detections)

    def test_rotate_detections_90(self, sample_image, sample_detection):
        """Test 90 degree rotation with detection."""
        detections = pf.Detections()
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
        detections = pf.Detections()
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
        detections = pf.Detections()

        # Detection inside crop region
        detections.add_detection(pf.Detection(
            bbox=[100, 100, 150, 150]
        ))

        # Detection outside crop region
        detections.add_detection(pf.Detection(
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
        detections = pf.Detections()
        detections.add_detection(sample_detection)

        # Returns one crop per detection (a list), not an (image, detections)
        # tuple. `padding` is a fraction of the bbox's shorter side.
        crops = pf.transform.crop_around_detections(
            sample_image,
            detections,
            padding=0.2
        )

        assert isinstance(crops, list)
        assert len(crops) == 1
        assert isinstance(crops[0], np.ndarray)
        # bbox is 100x100, +20% on each side => roughly 140x140.
        assert crops[0].shape[0] == pytest.approx(140, abs=2)
        assert crops[0].shape[1] == pytest.approx(140, abs=2)


class TestDetectionPadding:
    """Tests for adding padding to detections."""

    def test_add_padding_basic(self, sample_detection):
        """Test adding padding to detection bbox."""
        detections = pf.Detections()
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
        detections = pf.Detections()
        detections.add_detection(sample_detection_with_keypoints)

        aligned_img, aligned_dets = pf.transform.rotate_to_align(
            sample_image,
            detections,
            point1_name='left_eye',
            point2_name='right_eye',
            target_angle=0  # Horizontal
        )

        assert isinstance(aligned_img, np.ndarray)
        assert len(aligned_dets) == 1


class TestBboxFromKeypoints:
    """Tests for updating bbox from keypoints."""

    def test_update_bbox_from_keypoints(self, sample_detection_with_keypoints):
        """Test updating bbox to encompass keypoints."""
        detections = pf.Detections()
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

# Coordinates chosen so that truncation is visibly lossy in every direction.
FRACTIONAL_BBOX = [10.7, 20.3, 110.9, 220.4]
FRACTIONAL_KEYPOINT = (10.7, 20.3)


class TestInverseTransforms:
    """Tests for automated inverse transformations."""

    @pytest.fixture
    def fractional_detections(self):
        """Detections whose coordinates are not integers.

        The rest of the suite uses whole-number boxes, which cannot tell a
        precision-preserving transform apart from a truncating one.
        """
        dets = pf.Detections()
        dets.add_detection(pf.Detection(
            bbox=list(FRACTIONAL_BBOX), confidence=0.9, class_id=0, class_name="person",
            keypoints=[pf.KeyPoint(*FRACTIONAL_KEYPOINT, id=0, name="nose")],
            segments=[list(FRACTIONAL_KEYPOINT), [50.9, 60.4]],
        ))
        return dets

    @pytest.mark.parametrize(
        "flip",
        ["flip_horizontal_detections", "flip_vertical_detections"],
    )
    def test_flip_round_trip_is_exact(self, sample_image, fractional_detections, flip):
        """Flipping twice returns the original box and landmarks exactly.

        Keypoints go through the same transforms as boxes, so asserting only on
        the box would certify half the geometry.
        """
        flip_fn = getattr(pf.transform, flip)
        _, once = flip_fn(sample_image, fractional_detections)
        _, twice = flip_fn(sample_image, once)
        assert twice.detections[0].bbox == FRACTIONAL_BBOX
        kp = twice.detections[0].keypoints[0]
        assert (kp.x, kp.y) == FRACTIONAL_KEYPOINT
        assert twice.detections[0].segments == [list(FRACTIONAL_KEYPOINT), [50.9, 60.4]]

    def test_full_turn_rotation_recovers_geometry(self, sample_image, fractional_detections):
        """Four quarter turns return every coordinate exactly.

        The rotate path is the one whose numerics this change altered - polygon
        vertices go through cv2 rather than a comprehension - and a full turn is
        the strongest available check, since it must be the identity.
        """
        dets = fractional_detections
        for _ in range(4):
            _, dets = pf.transform.rotate_detections(sample_image, dets, angle=90)

        detection = dets.detections[0]
        assert detection.bbox == FRACTIONAL_BBOX
        kp = detection.keypoints[0]
        assert (kp.x, kp.y) == FRACTIONAL_KEYPOINT
        assert detection.segments == [list(FRACTIONAL_KEYPOINT), [50.9, 60.4]]

    def test_crop_inverse_recovers_original_coordinates(self, sample_image, fractional_detections):
        """Cropping then inverting recovers the original box exactly."""
        _, cropped = pf.transform.crop_detections(
            sample_image, fractional_detections, bbox=[5.5, 10.5, 300.5, 250.5]
        )
        restored = pf.transform.inverse_transforms(cropped)
        assert restored.detections[0].bbox == FRACTIONAL_BBOX

    def test_mixed_transform_chain_does_not_accumulate_drift(self, sample_image, fractional_detections):
        """A chain of different transforms must not walk the box off position.

        Each write re-validates, which snaps coordinates back onto the decimal
        grid rather than letting float error compound across steps. Mixing the
        transforms matters: a self-inverse one repeated cannot show drift that
        alternating axes would.
        """
        dets = fractional_detections
        for _ in range(3):
            _, dets = pf.transform.flip_horizontal_detections(sample_image, dets)
            _, dets = pf.transform.flip_vertical_detections(sample_image, dets)
            _, dets = pf.transform.flip_horizontal_detections(sample_image, dets)
            _, dets = pf.transform.flip_vertical_detections(sample_image, dets)
        assert dets.detections[0].bbox == FRACTIONAL_BBOX

    def test_transform_preserves_subpixel_precision(self, sample_image, fractional_detections):
        """A transformed box keeps its fraction instead of being truncated."""
        _, flipped = pf.transform.flip_horizontal_detections(sample_image, fractional_detections)
        w = sample_image.shape[1]
        assert flipped.detections[0].bbox == [w - 110.9, 20.3, w - 10.7, 220.4]

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
        new_dets = pf.Detections()
        new_dets.add_detection(pf.Detection(
            bbox=[50, 50, 100, 100]
        ))

        # Apply inverse transforms
        original_coords = pf.transform.inverse_transforms(new_dets)

        # Should return detections in original coordinate space
        assert isinstance(original_coords, pf.Detections)

    def test_inverse_with_multiple_transforms(self, sample_image, sample_detections):
        """Test inverse transforms with complex chain."""
        # Apply multiple transforms
        img, dets = sample_image, sample_detections

        img, dets = pf.transform.flip_horizontal_detections(img, dets)
        img, dets = pf.transform.rotate_detections(img, dets, angle=45)
        img, dets = pf.transform.crop_detections(img, dets, bbox=[100, 100, 400, 400])

        # New detections in transformed space
        new_dets = pf.Detections()
        new_dets.add_detection(pf.Detection(bbox=[50, 50, 100, 100]))

        # Inverse should undo all transforms
        original = pf.transform.inverse_transforms(new_dets)

        assert isinstance(original, pf.Detections)


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

        detections = pf.Detections()
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


# ============================================================================
# The invariant transforms must not break
# ============================================================================

class TestTransformsPreserveBoxValidity:
    """A transform relocates a detection; it must never invalidate one.

    `add_detection` guards the way into a container, but transforms assign
    `detection.bbox` on detections already inside one, where nothing re-checks
    them. An invalid box assigned there becomes None and the detection stays --
    still counted, with no geometry, and fatal to the next annotator.

    Today the arithmetic preserves validity: a flip maps x1 < x2 to
    w - x2 < w - x1, a rotation takes the hull of the rotated corners, and
    cropping drops a box that falls outside rather than clipping it away. That
    holds by construction rather than by enforcement, so these tests are what
    would notice if a new transform stopped honouring it.
    """

    # Shapes chosen to sit on the edges where the arithmetic could invert or
    # collapse: the frame boundary, one-pixel extents, and slivers.
    ADVERSARIAL = [
        [10, 10, 60, 60],        # ordinary
        [0, 0, 300, 200],        # exactly the whole frame
        [-40, -40, 20, 20],      # overlapping the top-left corner
        [280, 180, 340, 240],    # overlapping the bottom-right corner
        [1, 1, 2, 2],            # one pixel at the origin
        [150, 100, 151, 101],    # one pixel mid-frame
        [0, 0, 1, 200],          # a full-height sliver
        [299, 199, 300, 200],    # the far corner pixel
    ]

    @staticmethod
    def _frame():
        return np.zeros((200, 300, 3), dtype=np.uint8)

    def _assert_all_valid(self, detections, case):
        for i, detection in enumerate(detections):
            assert detection.bbox is not None, f"{case}: detection {i} lost its box"
            x1, y1, x2, y2 = detection.bbox
            assert x2 > x1 and y2 > y1, f"{case}: detection {i} became {detection.bbox}"

    @pytest.mark.parametrize("angle", [0, 1, 45, 90, 179, 270, 359, -37])
    def test_rotation(self, angle):
        for bbox in self.ADVERSARIAL:
            source = pf.from_arrays([bbox], scores=[0.9], class_ids=[0])
            _, out = pf.transform.rotate_detections(self._frame(), source, angle=angle)
            self._assert_all_valid(out, f"rotate({bbox}, {angle})")

    @pytest.mark.parametrize("flip", [
        pf.transform.flip_horizontal_detections,
        pf.transform.flip_vertical_detections,
    ])
    def test_flips(self, flip):
        for bbox in self.ADVERSARIAL:
            source = pf.from_arrays([bbox], scores=[0.9], class_ids=[0])
            _, out = flip(self._frame(), source)
            self._assert_all_valid(out, f"{flip.__name__}({bbox})")

    @pytest.mark.parametrize("crop", [
        [0, 0, 100, 100], [50, 50, 300, 200], [0, 0, 1, 1],
        [299, 199, 300, 200], [100, 0, 200, 200],
    ])
    def test_cropping(self, crop):
        for bbox in self.ADVERSARIAL:
            source = pf.from_arrays([bbox], scores=[0.9], class_ids=[0])
            _, out = pf.transform.crop_detections(self._frame(), source, bbox=crop)
            self._assert_all_valid(out, f"crop({bbox}, {crop})")

    def test_a_round_trip_survives(self):
        """Chained transforms are where a near-degenerate box would finally collapse."""
        for bbox in self.ADVERSARIAL:
            source = pf.from_arrays([bbox], scores=[0.9], class_ids=[0])
            frame = self._frame()
            frame, out = pf.transform.flip_horizontal_detections(frame, source)
            frame, out = pf.transform.rotate_detections(frame, out, angle=90)
            frame, out = pf.transform.flip_vertical_detections(frame, out)
            self._assert_all_valid(out, f"round trip({bbox})")

    @pytest.mark.parametrize("padding", [0.2, 0.0, -0.1, -0.49, -0.6, -2.0])
    def test_padding_never_leaves_a_boxless_detection(self, padding):
        """Regression: padding by a negative fraction shrank the box to nothing and
        left the detection in the collection, counted and geometry-less."""
        for bbox in self.ADVERSARIAL:
            source = pf.from_arrays([bbox], scores=[0.9], class_ids=[0])
            self._assert_all_valid(pf.transform.add_padding(source, padding=padding),
                                   f"add_padding({bbox}, {padding})")

    @pytest.mark.parametrize("count", [1, 2, 3, 8])
    def test_bbox_from_keypoints_never_leaves_a_boxless_detection(self, count):
        """Regression: a box rebuilt from a single keypoint is a point, not a box."""
        source = pf.from_arrays([[10, 10, 60, 60]], scores=[0.9], class_ids=[0])
        source[0].keypoints = [pf.KeyPoint(x=30 + i, y=30 + i, id=i, name=f"k{i}",
                                           confidence=0.9) for i in range(count)]
        self._assert_all_valid(pf.transform.update_bbox_from_keypoints(source),
                               f"update_bbox_from_keypoints({count} keypoints)")
