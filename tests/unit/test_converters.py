"""
Unit tests for pixelflow.detections.converters module.

Tests framework-specific converters including Ultralytics, Detectron2,
Transformers, SAM, Datamarkin, and Falcon Perception converters.
"""

import pytest
import numpy as np
import pixelflow as pf


# ============================================================================
# Mock Framework Outputs
# ============================================================================

@pytest.fixture
def mock_ultralytics_result():
    """Create mock Ultralytics YOLO result."""
    class MockTensor:
        def __init__(self, data):
            self._data = np.array(data)
        def cpu(self):
            return self
        def numpy(self):
            return self._data

    class MockBox:
        def __init__(self):
            # data format: [x1, y1, x2, y2, conf, class_id]
            self.data = MockTensor([
                [100, 100, 200, 200, 0.95, 0],
                [300, 150, 400, 280, 0.87, 2],
            ])
            self.id = None
        def __len__(self):
            return 2

    class MockResult:
        def __init__(self):
            self.boxes = MockBox()
            self.names = {0: "person", 2: "car"}
            self.probs = None
            self.masks = None
            self.keypoints = None

    return [MockResult()]


@pytest.fixture
def mock_detectron2_output():
    """Create mock Detectron2 output."""
    class MockTensor:
        def __init__(self, data):
            self._data = np.array(data)
        def numpy(self):
            return self._data

    class MockBoxes:
        def __init__(self, data):
            self.tensor = MockTensor(data)

    class MockInstances:
        def __init__(self):
            self.pred_boxes = MockBoxes([[100, 100, 200, 200], [300, 150, 400, 280]])
            self.scores = MockTensor([0.95, 0.87])
            self.pred_classes = MockTensor([0, 2])
            self._fields = {"pred_boxes", "scores", "pred_classes"}

        def to(self, device):
            return self

        def has(self, field):
            return field in self._fields

        def __len__(self):
            return 2

    return {"instances": MockInstances()}


@pytest.fixture
def mock_mayaku_output():
    """Create mock Mayaku output. Mayaku returns Instances directly (no dict wrap)."""
    class MockTensor:
        def __init__(self, data):
            self._data = np.array(data)
        def numpy(self):
            return self._data

    class MockBoxes:
        def __init__(self, data):
            self.tensor = MockTensor(data)

    class MockInstances:
        def __init__(self):
            self.pred_boxes = MockBoxes([[100, 100, 200, 200], [300, 150, 400, 280]])
            self.scores = MockTensor([0.95, 0.87])
            self.pred_classes = MockTensor([0, 2])
            self._fields = {"pred_boxes", "scores", "pred_classes"}

        def to(self, device):
            return self

        def has(self, field):
            return field in self._fields

        def __len__(self):
            return 2

    return MockInstances()


# ============================================================================
# Ultralytics Converter Tests
# ============================================================================

class TestUltralyticsConverter:
    """Tests for from_ultralytics converter."""

    def test_from_ultralytics_basic(self, mock_ultralytics_result):
        """Test basic Ultralytics conversion."""
        detections = pf.detections.from_ultralytics(mock_ultralytics_result)

        assert len(detections) == 2
        assert detections[0].bbox == [100, 100, 200, 200]
        assert detections[0].confidence == 0.95
        assert detections[0].class_id == 0
        assert detections[0].class_name == "person"

    def test_from_ultralytics_multiple_results(self, mock_ultralytics_result):
        """Test conversion with multiple result objects."""
        # Simulate batch processing
        detections = pf.detections.from_ultralytics(mock_ultralytics_result)
        assert isinstance(detections, pf.detections.Detections)

    def test_from_ultralytics_empty(self):
        """Test conversion with empty results."""
        class MockBox:
            def __init__(self):
                pass
            def __len__(self):
                return 0

        class MockResult:
            def __init__(self):
                self.boxes = MockBox()
                self.names = {}
                self.probs = None

        detections = pf.detections.from_ultralytics([MockResult()])
        assert len(detections) == 0

    def test_from_ultralytics_with_labels(self, mock_ultralytics_result):
        """Test that labels parameter overrides result.names."""
        labels = {0: "human", 2: "vehicle"}
        detections = pf.detections.from_ultralytics(mock_ultralytics_result, labels=labels)
        assert detections[0].class_name == "human"
        assert detections[1].class_name == "vehicle"

    def test_from_ultralytics_with_rich_labels(self, mock_ultralytics_result):
        """Test ultralytics conversion with rich Datamarkin format labels."""
        labels = [
            {"id": 0, "name": "person"},
            {"id": 2, "name": "car"},
        ]
        detections = pf.detections.from_ultralytics(mock_ultralytics_result, labels=labels)
        assert detections[0].class_name == "person"
        assert detections[1].class_name == "car"


# ============================================================================
# Detectron2 Converter Tests
# ============================================================================

class TestDetectron2Converter:
    """Tests for from_detectron2 converter."""

    def test_from_detectron2_basic(self, mock_detectron2_output):
        """Test basic Detectron2 conversion."""
        detections = pf.detections.from_detectron2(mock_detectron2_output)

        assert len(detections) == 2
        assert detections[0].confidence == 0.95
        assert detections[0].class_id == 0

    def test_from_detectron2_with_labels_dict(self, mock_detectron2_output):
        """Test Detectron2 conversion with Dict[int, str] labels."""
        labels = {0: "person", 2: "car"}
        detections = pf.detections.from_detectron2(
            mock_detectron2_output, labels=labels
        )
        assert detections[0].class_name == "person"
        assert detections[1].class_name == "car"

    def test_from_detectron2_with_labels_list(self, mock_detectron2_output):
        """Test Detectron2 conversion with List[str] labels."""
        labels = ["person", "bicycle", "car"]
        detections = pf.detections.from_detectron2(
            mock_detectron2_output, labels=labels
        )
        assert detections[0].class_name == "person"
        assert detections[1].class_name == "car"

    def test_from_detectron2_with_rich_labels(self, mock_detectron2_output):
        """Test Detectron2 conversion with rich Datamarkin format labels."""
        labels = [
            {"id": 0, "name": "person", "keypoints": [{"id": 0, "name": "nose"}]},
            {"id": 2, "name": "car"},
        ]
        detections = pf.detections.from_detectron2(
            mock_detectron2_output, labels=labels
        )
        assert detections[0].class_name == "person"
        assert detections[1].class_name == "car"

    def test_from_detectron2_with_keypoints(self):
        """Test Detectron2 keypoint conversion to KeyPoint objects."""
        class MockTensor:
            def __init__(self, data):
                self._data = np.array(data)
            def numpy(self):
                return self._data

        class MockBoxes:
            def __init__(self, data):
                self.tensor = MockTensor(data)

        class MockInstances:
            def __init__(self):
                self.pred_boxes = MockBoxes([[100, 100, 200, 200]])
                self.scores = MockTensor([0.95])
                self.pred_classes = MockTensor([0])
                # 3 keypoints: (x, y, visibility)
                self.pred_keypoints = MockTensor([[[150, 120, 2.0], [140, 115, 1.0], [160, 115, 0.0]]])
                self._fields = {"pred_boxes", "scores", "pred_classes", "pred_keypoints"}

            def to(self, device):
                return self
            def has(self, field):
                return field in self._fields
            def __len__(self):
                return 1

        labels = [{"id": 0, "name": "person", "keypoints": [
            {"id": 0, "name": "nose"},
            {"id": 1, "name": "left_eye"},
            {"id": 2, "name": "right_eye"},
        ]}]

        detections = pf.detections.from_detectron2(
            {"instances": MockInstances()}, labels=labels
        )
        assert len(detections) == 1
        assert detections[0].keypoints is not None
        assert len(detections[0].keypoints) == 3
        assert detections[0].keypoints[0].name == "nose"
        assert detections[0].keypoints[0].x == 150
        assert detections[0].keypoints[0].visibility is True  # 2.0 > 0
        assert detections[0].keypoints[1].name == "left_eye"
        assert detections[0].keypoints[1].visibility is True  # 1.0 > 0
        assert detections[0].keypoints[2].name == "right_eye"
        assert detections[0].keypoints[2].visibility is False  # 0.0 > 0 = False


# ============================================================================
# Mayaku Converter Tests
# ============================================================================

class TestMayakuConverter:
    """Tests for from_mayaku converter.

    Mayaku is a clean Detectron2 reimplementation; the runtime output schema
    is identical except that mayaku's Predictor returns the Instances object
    directly (not wrapped in a {"instances": ...} dict).
    """

    def test_from_mayaku_basic(self, mock_mayaku_output):
        """Test basic Mayaku conversion."""
        detections = pf.detections.from_mayaku(mock_mayaku_output)

        assert len(detections) == 2
        assert detections[0].bbox == [100, 100, 200, 200]
        assert detections[0].confidence == 0.95
        assert detections[0].class_id == 0
        assert detections[1].class_id == 2

    def test_from_mayaku_with_labels_dict(self, mock_mayaku_output):
        """Test Mayaku conversion with Dict[int, str] labels."""
        labels = {0: "person", 2: "car"}
        detections = pf.detections.from_mayaku(mock_mayaku_output, labels=labels)
        assert detections[0].class_name == "person"
        assert detections[1].class_name == "car"

    def test_from_mayaku_with_labels_list(self, mock_mayaku_output):
        """Test Mayaku conversion with List[str] labels."""
        labels = ["person", "bicycle", "car"]
        detections = pf.detections.from_mayaku(mock_mayaku_output, labels=labels)
        assert detections[0].class_name == "person"
        assert detections[1].class_name == "car"

    def test_from_mayaku_with_rich_labels(self, mock_mayaku_output):
        """Test Mayaku conversion with rich Datamarkin format labels."""
        labels = [
            {"id": 0, "name": "person", "keypoints": [{"id": 0, "name": "nose"}]},
            {"id": 2, "name": "car"},
        ]
        detections = pf.detections.from_mayaku(mock_mayaku_output, labels=labels)
        assert detections[0].class_name == "person"
        assert detections[1].class_name == "car"

    def test_from_mayaku_empty(self):
        """Test conversion with empty Instances returns empty Detections."""
        class MockInstances:
            def to(self, device):
                return self
            def has(self, field):
                return False
            def __len__(self):
                return 0

        detections = pf.detections.from_mayaku(MockInstances())
        assert len(detections) == 0

    def test_from_mayaku_with_keypoints(self):
        """Test Mayaku keypoint conversion to KeyPoint objects.

        Mayaku stores keypoints as (N, K, 3) with (x, y, score) — same as D2.
        """
        class MockTensor:
            def __init__(self, data):
                self._data = np.array(data)
            def numpy(self):
                return self._data

        class MockBoxes:
            def __init__(self, data):
                self.tensor = MockTensor(data)

        class MockInstances:
            def __init__(self):
                self.pred_boxes = MockBoxes([[100, 100, 200, 200]])
                self.scores = MockTensor([0.95])
                self.pred_classes = MockTensor([0])
                self.pred_keypoints = MockTensor(
                    [[[150, 120, 0.9], [140, 115, 0.5], [160, 115, 0.0]]]
                )
                self._fields = {"pred_boxes", "scores", "pred_classes", "pred_keypoints"}

            def to(self, device):
                return self
            def has(self, field):
                return field in self._fields
            def __len__(self):
                return 1

        labels = [{"id": 0, "name": "person", "keypoints": [
            {"id": 0, "name": "nose"},
            {"id": 1, "name": "left_eye"},
            {"id": 2, "name": "right_eye"},
        ]}]

        detections = pf.detections.from_mayaku(MockInstances(), labels=labels)
        assert len(detections) == 1
        assert detections[0].keypoints is not None
        assert len(detections[0].keypoints) == 3
        assert detections[0].keypoints[0].name == "nose"
        assert detections[0].keypoints[0].x == 150
        assert detections[0].keypoints[0].visibility is True   # 0.9 > 0
        assert detections[0].keypoints[1].visibility is True   # 0.5 > 0
        assert detections[0].keypoints[2].visibility is False  # 0.0 > 0 = False

    def test_from_mayaku_with_masks(self):
        """Test Mayaku mask conversion. Masks are (N, H, W) bool after postprocess."""
        class MockTensor:
            def __init__(self, data):
                self._data = np.array(data)
            def numpy(self):
                return self._data

        class MockBoxes:
            def __init__(self, data):
                self.tensor = MockTensor(data)

        class MockInstances:
            def __init__(self):
                self.pred_boxes = MockBoxes([[10, 10, 20, 20]])
                self.scores = MockTensor([0.9])
                self.pred_classes = MockTensor([0])
                # (N=1, H=4, W=4) bool mask
                self.pred_masks = MockTensor([[[True, True, False, False],
                                                [True, True, False, False],
                                                [False, False, False, False],
                                                [False, False, False, False]]])
                self._fields = {"pred_boxes", "scores", "pred_classes", "pred_masks"}

            def to(self, device):
                return self
            def has(self, field):
                return field in self._fields
            def __len__(self):
                return 1

        detections = pf.detections.from_mayaku(MockInstances())
        assert len(detections) == 1
        assert detections[0].masks is not None
        assert len(detections[0].masks) == 1
        mask = detections[0].masks[0]
        assert mask.dtype == bool
        assert mask.shape == (4, 4)
        assert mask[0, 0] == True
        assert mask[3, 3] == False


# ============================================================================
# _get_label_info Helper Tests
# ============================================================================

class TestGetLabelInfo:
    """Tests for _get_label_info helper function."""

    def test_list_format(self):
        from pixelflow.detections.converters import _get_label_info
        name, kp_names = _get_label_info(["person", "car", "dog"], 1)
        assert name == "car"
        assert kp_names is None

    def test_dict_format(self):
        from pixelflow.detections.converters import _get_label_info
        name, kp_names = _get_label_info({0: "person", 5: "car"}, 5)
        assert name == "car"
        assert kp_names is None

    def test_dict_nonsequential_keys(self):
        from pixelflow.detections.converters import _get_label_info
        name, kp_names = _get_label_info({0: "person", 91: "banana"}, 91)
        assert name == "banana"

    def test_rich_format(self):
        from pixelflow.detections.converters import _get_label_info
        labels = [{"id": 0, "name": "person", "keypoints": [{"id": 0, "name": "nose"}]}]
        name, kp_names = _get_label_info(labels, 0)
        assert name == "person"
        assert kp_names == ["nose"]

    def test_rich_format_no_keypoints(self):
        from pixelflow.detections.converters import _get_label_info
        labels = [{"id": 1, "name": "bicycle"}]
        name, kp_names = _get_label_info(labels, 1)
        assert name == "bicycle"
        assert kp_names is None

    def test_missing_id_returns_none(self):
        from pixelflow.detections.converters import _get_label_info
        name, kp_names = _get_label_info(["person", "car"], 99)
        assert name is None
        assert kp_names is None

    def test_none_labels(self):
        from pixelflow.detections.converters import _get_label_info
        name, kp_names = _get_label_info(None, 0)
        assert name is None
        assert kp_names is None

    def test_none_class_id(self):
        from pixelflow.detections.converters import _get_label_info
        name, kp_names = _get_label_info(["person"], None)
        assert name is None

    def test_empty_list(self):
        from pixelflow.detections.converters import _get_label_info
        name, kp_names = _get_label_info([], 0)
        assert name is None


# ============================================================================
# SAM Converter Tests
# ============================================================================

class TestSAMConverter:
    """Tests for from_sam / from_efficienttam converters."""

    def test_from_sam_basic(self):
        """Test basic SAM conversion with masks and scores."""
        masks = np.array([
            np.ones((100, 100), dtype=bool),
            np.zeros((100, 100), dtype=bool)
        ])
        scores = np.array([0.95, 0.87])

        detections = pf.detections.from_sam(masks, scores)

        assert len(detections) == 1  # second mask is empty, skipped
        assert detections[0].masks is not None
        assert detections[0].confidence == 0.95
        assert detections[0].bbox == [0, 0, 99, 99]

    def test_from_efficienttam_basic(self):
        """Test EfficientTAM conversion with multiple masks."""
        masks = np.zeros((3, 50, 80), dtype=bool)
        masks[0, 10:30, 20:60] = True
        masks[1, 5:15, 0:10] = True
        scores = np.array([0.92, 0.85, 0.73])

        detections = pf.detections.from_efficienttam(masks, scores)

        assert len(detections) == 2  # third mask is empty
        assert detections[0].confidence == 0.92
        assert detections[0].bbox == [20, 10, 59, 29]
        assert detections[1].confidence == 0.85
        assert detections[1].bbox == [0, 5, 9, 14]

    def test_from_efficienttam_empty(self):
        """Test EfficientTAM with no masks."""
        masks = np.zeros((0, 100, 100), dtype=bool)
        scores = np.zeros(0)

        detections = pf.detections.from_efficienttam(masks, scores)
        assert len(detections) == 0

    def test_from_efficienttam_mask_is_bool(self):
        """Test that output masks are boolean."""
        masks = np.ones((1, 50, 50), dtype=np.uint8)
        scores = np.array([0.9])

        detections = pf.detections.from_efficienttam(masks, scores)
        assert detections[0].masks[0].dtype == bool


# ============================================================================
# Datamarkin API Converter Tests
# ============================================================================

class TestDatamarkinAPIConverter:
    """Tests for from_datamarkin converter."""

    def test_from_datamarkin_basic(self):
        """Test conversion from Datamarkin API format."""
        api_response = [
            {
                'bbox': [100, 100, 200, 200],
                'confidence': 0.95,
                'class_id': 0,
                'class_name': 'person'
            },
            {
                'bbox': [300, 150, 400, 280],
                'confidence': 0.87,
                'class_id': 2,
                'class_name': 'car'
            }
        ]

        detections = pf.detections.from_datamarkin(api_response)

        assert len(detections) == 2
        assert detections[0].bbox == [100, 100, 200, 200]
        assert detections[0].confidence == 0.95
        assert detections[0].class_name == "person"


# ============================================================================
# CSV Converter Tests
# ============================================================================

class TestDatamarkinCSVConverter:
    """Tests for from_datamarkin_csv converter."""

    def test_from_datamarkin_csv_basic(self, tmp_path):
        """Test conversion from CSV file."""
        # Create temporary CSV file
        csv_content = """bbox_x1,bbox_y1,bbox_x2,bbox_y2,confidence,class_id,class_name
100,100,200,200,0.95,0,person
300,150,400,280,0.87,2,car"""

        csv_file = tmp_path / "detections.csv"
        csv_file.write_text(csv_content)

        detections = pf.detections.from_datamarkin_csv(str(csv_file))

        assert len(detections) == 2
        assert detections[0].bbox == [100, 100, 200, 200]
        assert detections[0].confidence == 0.95


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestConverterEdgeCases:
    """Test edge cases and error handling for converters."""

    def test_converter_with_none_values(self):
        """Test converters handle None/null values gracefully."""
        api_response = [
            {
                'bbox': [100, 100, 200, 200],
                'confidence': None,
                'class_id': None,
                'class_name': None
            }
        ]

        detections = pf.detections.from_datamarkin(api_response)
        assert len(detections) == 1
        assert detections[0].confidence is None

    def test_converter_empty_input(self):
        """Test converters with empty input."""
        detections = pf.detections.from_datamarkin([])
        assert len(detections) == 0


# ============================================================================
# Falcon Perception Converter Tests
# ============================================================================

class MockAuxOutput:
    def __init__(self, bboxes_raw, masks_rle=None, text=None):
        self.bboxes_raw = bboxes_raw
        self.masks_rle = masks_rle
        self.text = text


class TestFalconPerceptionConverter:
    """Test from_falcon_perception converter."""

    def test_even_length_returns_correct_detections(self):
        """Even-length bboxes_raw produces correct detections."""
        output = MockAuxOutput(bboxes_raw=[
            {"x": 0.5, "y": 0.5},
            {"h": 0.2, "w": 0.4},
        ])
        detections = pf.detections.from_falcon_perception(
            output, image_size=(100, 100), label="cat"
        )
        assert len(detections) == 1
        bbox = detections[0].bbox
        # normalized cxcywh (0.5,0.5,0.4,0.2) on 100x100 → pixel xyxy
        assert bbox[0] == pytest.approx(30.0)
        assert bbox[1] == pytest.approx(40.0)
        assert bbox[2] == pytest.approx(70.0)
        assert bbox[3] == pytest.approx(60.0)

    def test_odd_length_warns_and_drops_last_entry(self):
        """Odd-length bboxes_raw warns and drops the trailing entry."""
        output = MockAuxOutput(bboxes_raw=[
            {"x": 0.5, "y": 0.5},
            {"h": 0.2, "w": 0.4},
            {"x": 0.3, "y": 0.3},  # incomplete — no size dict
        ])
        with pytest.warns(UserWarning, match="odd length"):
            detections = pf.detections.from_falcon_perception(
                output, image_size=(100, 100), label="cat"
            )
        assert len(detections) == 1

    def test_empty_bboxes_raw_returns_zero_detections(self):
        """Empty bboxes_raw returns 0 detections without warning."""
        output = MockAuxOutput(bboxes_raw=[])
        detections = pf.detections.from_falcon_perception(
            output, image_size=(100, 100), label="cat"
        )
        assert len(detections) == 0

    def test_single_entry_warns_and_returns_zero_detections(self):
        """Single entry (just center, no size) warns and returns 0 detections."""
        output = MockAuxOutput(bboxes_raw=[{"x": 0.5, "y": 0.5}])
        with pytest.warns(UserWarning, match="odd length"):
            detections = pf.detections.from_falcon_perception(
                output, image_size=(100, 100), label="cat"
            )
        assert len(detections) == 0
