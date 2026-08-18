"""
Unit tests for pixelflow.detections module.

Tests the core Detection, Detections, and KeyPoint classes including
creation, serialization, and basic operations.
"""

import pytest
import numpy as np
import json
import pixelflow as pf


# ============================================================================
# KeyPoint Tests
# ============================================================================

class TestKeyPoint:
    """Tests for the KeyPoint class."""

    def test_keypoint_creation(self):
        """Test basic keypoint creation."""
        kp = pf.detections.KeyPoint(x=100, y=200, id=0, name="nose", confidence=0.9)
        assert kp.x == 100
        assert kp.y == 200
        assert kp.name == "nose"
        assert kp.id == 0
        assert kp.confidence == 0.9

    def test_keypoint_to_dict(self, sample_keypoint):
        """Test keypoint serialization to dict."""
        data = sample_keypoint.to_dict()
        assert data["x"] == 100
        assert data["y"] == 200
        assert data["name"] == "nose"
        assert data["id"] == 0
        assert data["confidence"] == 0.9

    def test_keypoint_numpy_conversion(self):
        """Test keypoint with numpy integer types."""
        kp = pf.detections.KeyPoint(
            x=np.int64(150),
            y=np.int32(250),
            name="point",
            id=0,
            confidence=0.9
        )
        data = kp.to_dict()
        # Should convert numpy types to Python native types
        assert isinstance(data["x"], int)
        assert isinstance(data["y"], int)


# ============================================================================
# Detection Tests
# ============================================================================

class TestDetection:
    """Tests for the Detection class."""

    def test_detection_creation_minimal(self):
        """Test detection with minimal required fields."""
        det = pf.detections.Detection(bbox=[100, 100, 200, 200])
        assert det.bbox == [100, 100, 200, 200]
        assert det.confidence is None
        assert det.class_id is None
        assert det.class_name is None

    def test_detection_creation_full(self):
        """Test detection with all common fields."""
        det = pf.detections.Detection(
            bbox=[100, 100, 200, 200],
            confidence=0.95,
            class_id=0,
            class_name="person",
            labels=["person 0.95"]
        )
        assert det.bbox == [100, 100, 200, 200]
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == "person"
        assert det.labels == ["person 0.95"]

    def test_detection_with_mask(self):
        """Test detection with binary mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        det = pf.detections.Detection(
            bbox=[100, 100, 200, 200],
            masks=[mask]
        )
        assert len(det.masks) == 1
        assert det.masks[0].shape == (100, 100)
        assert det.masks[0].dtype == bool

    def test_detection_with_keypoints(self, sample_keypoints):
        """Test detection with keypoints."""
        det = pf.detections.Detection(
            bbox=[50, 50, 150, 200],
            keypoints=sample_keypoints
        )
        assert len(det.keypoints) == 5
        assert det.keypoints[0].name == "nose"


    def test_detection_with_tracking(self):
        """Test detection with tracking information."""
        det = pf.detections.Detection(
            bbox=[100, 100, 200, 200],
            tracker_id=42,
            first_seen_time=1.5,
            total_time=3.2
        )
        assert det.tracker_id == 42
        assert det.first_seen_time == 1.5
        assert det.total_time == 3.2

    def test_detection_to_dict_basic(self, sample_detection):
        """Test detection serialization to dict."""
        data = sample_detection.to_dict()
        assert data["bbox"] == [100, 100, 200, 200]
        assert data["confidence"] == 0.95
        assert data["class_id"] == 0
        assert data["class_name"] == "person"

    def test_detection_to_dict_with_mask(self, sample_detection_with_mask):
        """Test detection with mask serialization."""
        data = sample_detection_with_mask.to_dict()
        assert "masks" in data
        assert len(data["masks"]) == 1
        # Mask should be base64 encoded
        assert isinstance(data["masks"][0], dict)
        assert "data" in data["masks"][0]
        assert "shape" in data["masks"][0]

    def test_detection_decode_mask(self):
        """Test mask decoding from base64."""
        # Create and encode mask
        original_mask = np.zeros((50, 50), dtype=bool)
        original_mask[10:40, 10:40] = True

        det = pf.detections.Detection(bbox=[0, 0, 50, 50], masks=[original_mask])
        data = det.to_dict()

        # Decode mask
        decoded_mask = pf.detections.Detection.decode_mask(data["masks"][0])
        assert decoded_mask.shape == original_mask.shape
        assert decoded_mask.dtype == bool
        assert np.array_equal(decoded_mask, original_mask)

    def test_detection_to_dict_is_json_serializable(self, sample_detection):
        """Detection exposes to_dict(); to_json() lives on the Detections container."""
        assert not hasattr(sample_detection, "to_json")

        data = json.loads(json.dumps(sample_detection.to_dict()))
        assert data["bbox"] == [100, 100, 200, 200]
        assert data["confidence"] == 0.95


# ============================================================================
# Detections Container Tests
# ============================================================================

class TestDetections:
    """Tests for the Detections container class."""

    def test_detections_creation_empty(self):
        """Test creating empty Detections container."""
        dets = pf.detections.Detections()
        assert len(dets) == 0

    def test_detections_add_detection(self, empty_detections, sample_detection):
        """Test adding detection to container."""
        empty_detections.add_detection(sample_detection)
        assert len(empty_detections) == 1

    def test_detections_iteration(self, sample_detections):
        """Test iterating over detections."""
        count = 0
        for det in sample_detections:
            assert isinstance(det, pf.detections.Detection)
            count += 1
        assert count == 4

    def test_detections_indexing(self, sample_detections):
        """Test accessing detections by index."""
        first = sample_detections[0]
        assert isinstance(first, pf.detections.Detection)
        assert first.bbox == [100, 100, 200, 200]

    def test_detections_len(self, sample_detections):
        """Test length of detections container."""
        assert len(sample_detections) == 4

    def test_detections_bool(self, sample_detections, empty_detections):
        """Test boolean conversion of detections."""
        assert bool(sample_detections) is True
        assert bool(empty_detections) is False

    def test_detections_to_list(self, sample_detections):
        """Test converting detections to list of dicts."""
        det_list = [d.to_dict() for d in sample_detections]
        assert len(det_list) == 4
        assert all(isinstance(d, dict) for d in det_list)

    def test_detections_to_json(self, sample_detections):
        """Test JSON serialization of detections container."""
        json_str = sample_detections.to_json()
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == 4

    def test_detections_class_names_property(self, sample_detections):
        """Test extracting class names from detections."""
        class_names = [d.class_name for d in sample_detections if d.class_name]
        assert "person" in class_names
        assert "car" in class_names
        assert "bicycle" in class_names

    def test_detections_confidence_scores_property(self, sample_detections):
        """Test extracting confidence scores from detections."""
        confidences = [d.confidence for d in sample_detections if d.confidence is not None]
        assert len(confidences) == 4
        assert max(confidences) == 0.95
        assert min(confidences) == 0.45

    def test_detections_bboxes_property(self, sample_detections):
        """Test extracting bboxes from detections."""
        bboxes = [d.bbox for d in sample_detections if d.bbox is not None]
        assert len(bboxes) == 4
        assert bboxes[0] == [100, 100, 200, 200]


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestDetectionEdgeCases:
    """Test edge cases and error handling."""

    def test_detection_with_none_values(self):
        """Test detection with explicitly None values."""
        det = pf.detections.Detection(
            bbox=[0, 0, 10, 10],
            confidence=None,
            class_id=None,
            class_name=None
        )
        assert det.confidence is None
        assert det.class_id is None
        assert det.class_name is None

    def test_detection_empty_masks_list(self):
        """Test detection with empty masks list."""
        det = pf.detections.Detection(bbox=[0, 0, 10, 10], masks=[])
        assert det.masks == []

    def test_detection_empty_keypoints_list(self):
        """Test detection with empty keypoints list."""
        det = pf.detections.Detection(bbox=[0, 0, 10, 10], keypoints=[])
        assert det.keypoints == []

    def test_detections_slicing(self, sample_detections):
        """Test slicing detections container."""
        subset = sample_detections[1:3]
        # Slicing should work like a list
        assert len(subset) == 2
