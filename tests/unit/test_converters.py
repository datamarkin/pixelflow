"""
Unit tests for pixelflow.detections.converters module.

Tests framework-specific converters including Ultralytics, Detectron2,
Transformers, SAM, and OCR converters (Tesseract, EasyOCR, PaddleOCR).
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
def mock_tesseract_data():
    """Create mock Tesseract OCR output."""
    return {
        'text': ['', 'Hello', 'World', 'Test'],
        'conf': [-1, 92, 88, 75],
        'left': [0, 100, 200, 300],
        'top': [0, 100, 100, 100],
        'width': [0, 80, 80, 60],
        'height': [0, 40, 40, 40],
        'level': [1, 5, 5, 5],  # 5 = word level
        'page_num': [1, 1, 1, 1],
        'block_num': [0, 1, 1, 1],
        'par_num': [0, 1, 1, 1],
        'line_num': [0, 1, 1, 1],
        'word_num': [0, 1, 2, 3]
    }


@pytest.fixture
def mock_easyocr_result():
    """Create mock EasyOCR output."""
    return [
        (
            [[100, 100], [200, 100], [200, 150], [100, 150]],  # Polygon
            "Hello",  # Text
            0.92  # Confidence
        ),
        (
            [[220, 100], [320, 100], [320, 150], [220, 150]],
            "World",
            0.88
        )
    ]


@pytest.fixture
def mock_paddleocr_result():
    """Create mock PaddleOCR output."""
    return [[
        [
            [[100, 100], [200, 100], [200, 150], [100, 150]],  # Box
            ("Hello", 0.92)  # Text and confidence
        ],
        [
            [[220, 100], [320, 105], [318, 150], [218, 145]],  # Rotated box
            ("World", 0.88)
        ]
    ]]


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
# Tesseract OCR Converter Tests
# ============================================================================

class TestTesseractConverter:
    """Tests for from_tesseract OCR converter."""

    def test_from_tesseract_word_level(self, mock_tesseract_data):
        """Test Tesseract conversion at word level."""
        detections = pf.detections.from_tesseract(
            mock_tesseract_data,
            level='word',
            language='en'
        )

        # Should have 3 words (empty text filtered out)
        assert len(detections) == 3

        # Check first word
        assert detections[0].text == "Hello"
        assert detections[0].text_confidence == 0.92
        assert detections[0].text_language == "en"
        assert detections[0].text_level == "word"
        assert detections[0].bbox == [100, 100, 180, 140]

    def test_from_tesseract_confidence_filter(self, mock_tesseract_data):
        """Test Tesseract with minimum confidence filter."""
        detections = pf.detections.from_tesseract(
            mock_tesseract_data,
            level='word',
            min_confidence=0.80
        )

        # Only Hello (92) and World (88) should pass
        assert len(detections) == 2
        for det in detections:
            assert det.text_confidence >= 0.80

    def test_from_tesseract_line_level(self, mock_tesseract_data):
        """Test Tesseract conversion at line level."""
        # Modify data to have line-level entries
        line_data = mock_tesseract_data.copy()
        line_data['level'] = [1, 4, 4, 4]  # 4 = line level

        detections = pf.detections.from_tesseract(
            line_data,
            level='line',
            language='en'
        )

        # Should group words into lines
        assert len(detections) >= 1

    def test_from_tesseract_text_order(self, mock_tesseract_data):
        """Test that text_order is set for reading order."""
        detections = pf.detections.from_tesseract(
            mock_tesseract_data,
            level='word'
        )

        # Each detection should have text_order
        for i, det in enumerate(detections):
            assert det.text_order is not None


# ============================================================================
# EasyOCR Converter Tests
# ============================================================================

class TestEasyOCRConverter:
    """Tests for from_easyocr OCR converter."""

    def test_from_easyocr_basic(self, mock_easyocr_result):
        """Test basic EasyOCR conversion."""
        detections = pf.detections.from_easyocr(
            mock_easyocr_result,
            language='en'
        )

        assert len(detections) == 2
        assert detections[0].text == "Hello"
        assert detections[0].text_confidence == 0.92
        assert detections[0].text_language == "en"

    def test_from_easyocr_polygon_preserved(self, mock_easyocr_result):
        """Test that EasyOCR polygons are preserved."""
        detections = pf.detections.from_easyocr(mock_easyocr_result)

        # Check that segments (polygons) are stored
        assert detections[0].segments is not None
        assert len(detections[0].segments) == 4  # 4 points for quadrilateral

    def test_from_easyocr_bbox_calculated(self, mock_easyocr_result):
        """Test that bbox is calculated from polygon."""
        detections = pf.detections.from_easyocr(mock_easyocr_result)

        # Bbox should be axis-aligned bounding box of polygon
        assert detections[0].bbox == [100, 100, 200, 150]

    def test_from_easyocr_confidence_filter(self, mock_easyocr_result):
        """Test EasyOCR with minimum confidence."""
        detections = pf.detections.from_easyocr(
            mock_easyocr_result,
            min_confidence=0.90
        )

        # Only "Hello" (0.92) should pass
        assert len(detections) == 1
        assert detections[0].text == "Hello"


# ============================================================================
# PaddleOCR Converter Tests
# ============================================================================

class TestPaddleOCRConverter:
    """Tests for from_paddleocr OCR converter."""

    def test_from_paddleocr_basic(self, mock_paddleocr_result):
        """Test basic PaddleOCR conversion."""
        detections = pf.detections.from_paddleocr(
            mock_paddleocr_result,
            language='en'
        )

        assert len(detections) == 2
        assert detections[0].text == "Hello"
        assert detections[0].text_confidence == 0.92

    def test_from_paddleocr_angle_detection(self, mock_paddleocr_result):
        """Test that PaddleOCR detects text angle."""
        detections = pf.detections.from_paddleocr(mock_paddleocr_result)

        # Second detection has rotated box, should have text_angle set
        # First detection is horizontal (angle ≈ 0)
        assert detections[0].text_angle is not None

    def test_from_paddleocr_direction_detection(self, mock_paddleocr_result):
        """Test that text direction is detected."""
        detections = pf.detections.from_paddleocr(mock_paddleocr_result)

        # Should have text_direction set based on angle
        for det in detections:
            assert det.text_direction in ['ltr', 'rtl', 'vertical-ttb', 'vertical-btt', None]


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
# SAM3 Converter Tests
# ============================================================================

@pytest.fixture
def mock_sam3_result():
    """Create mock SAM3 (HuggingFace transformers) output."""
    return {
        'masks': [
            np.ones((100, 100), dtype=bool),
            np.zeros((100, 100), dtype=bool)
        ],
        'boxes': [
            [10.0, 20.0, 80.0, 90.0],
            [50.0, 60.0, 150.0, 160.0]
        ],
        'scores': [0.95, 0.87]
    }


@pytest.fixture
def mock_sam3_result_tensors():
    """Create mock SAM3 output with tensor-like objects."""
    class MockTensor:
        """Simple mock for PyTorch tensor behavior."""
        def __init__(self, data):
            self._data = np.array(data)

        def cpu(self):
            return self

        def numpy(self):
            return self._data

        def item(self):
            return float(self._data)

        def tolist(self):
            return self._data.tolist()

    return {
        'masks': [
            MockTensor(np.ones((100, 100), dtype=bool)),
            MockTensor(np.zeros((100, 100), dtype=bool))
        ],
        'boxes': [
            MockTensor([10.0, 20.0, 80.0, 90.0]),
            MockTensor([50.0, 60.0, 150.0, 160.0])
        ],
        'scores': [
            MockTensor(0.95),
            MockTensor(0.87)
        ]
    }


class TestSam3Converter:
    """Tests for from_sam3 (SAM3 via HuggingFace transformers) converter."""

    def test_from_sam3_basic(self, mock_sam3_result):
        """Test basic SAM3 conversion with masks, boxes, and scores."""
        detections = pf.detections.from_sam3(mock_sam3_result, prompt="person")

        assert len(detections) == 2
        assert detections[0].masks is not None
        assert detections[0].masks[0].shape == (100, 100)
        assert detections[0].confidence == 0.95
        assert detections[0].class_name == "person"
        assert detections[0].class_id == 0

    def test_from_sam3_bbox(self, mock_sam3_result):
        """Test that bounding boxes are correctly converted."""
        detections = pf.detections.from_sam3(mock_sam3_result)

        assert detections[0].bbox == [10.0, 20.0, 80.0, 90.0]
        assert detections[1].bbox == [50.0, 60.0, 150.0, 160.0]

    def test_from_sam3_mask_dtype(self, mock_sam3_result):
        """Test that masks are converted to boolean dtype."""
        detections = pf.detections.from_sam3(mock_sam3_result)

        assert detections[0].masks[0].dtype == bool
        assert detections[1].masks[0].dtype == bool

    def test_from_sam3_custom_prompt(self, mock_sam3_result):
        """Test custom prompt is used as class_name."""
        detections = pf.detections.from_sam3(
            mock_sam3_result,
            prompt="yellow school bus"
        )

        for det in detections:
            assert det.class_name == "yellow school bus"

    def test_from_sam3_custom_class_id(self, mock_sam3_result):
        """Test custom class_id is applied."""
        detections = pf.detections.from_sam3(
            mock_sam3_result,
            prompt="car",
            class_id=5
        )

        for det in detections:
            assert det.class_id == 5
            assert det.class_name == "car"

    def test_from_sam3_empty_results(self):
        """Test conversion with empty results."""
        empty_result = {'masks': [], 'boxes': [], 'scores': []}
        detections = pf.detections.from_sam3(empty_result)

        assert len(detections) == 0

    def test_from_sam3_none_results(self):
        """Test conversion with None results."""
        detections = pf.detections.from_sam3(None)

        assert len(detections) == 0

    def test_from_sam3_empty_dict(self):
        """Test conversion with empty dict."""
        detections = pf.detections.from_sam3({})

        assert len(detections) == 0

    def test_from_sam3_tensor_conversion(self, mock_sam3_result_tensors):
        """Test that tensor-like objects are correctly converted."""
        detections = pf.detections.from_sam3(mock_sam3_result_tensors)

        assert len(detections) == 2
        assert detections[0].masks[0].dtype == bool
        assert detections[0].confidence == 0.95
        assert detections[0].bbox == [10.0, 20.0, 80.0, 90.0]

    def test_from_sam3_single_detection(self):
        """Test conversion with single detection."""
        result = {
            'masks': [np.ones((50, 50), dtype=bool)],
            'boxes': [[0.0, 0.0, 50.0, 50.0]],
            'scores': [0.99]
        }
        detections = pf.detections.from_sam3(result, prompt="dog")

        assert len(detections) == 1
        assert detections[0].confidence == 0.99
        assert detections[0].class_name == "dog"

    def test_from_sam3_default_prompt(self, mock_sam3_result):
        """Test default prompt is 'object'."""
        detections = pf.detections.from_sam3(mock_sam3_result)

        for det in detections:
            assert det.class_name == "object"

    def test_from_sam3_method_chaining(self, mock_sam3_result):
        """Test that returned Detections supports method chaining."""
        detections = pf.detections.from_sam3(mock_sam3_result, prompt="car")

        # Test filter chaining
        filtered = detections.filter_by_confidence(0.90)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.95


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
