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
    class MockBox:
        def __init__(self):
            self.xyxy = np.array([[100, 100, 200, 200], [300, 150, 400, 280]])
            self.conf = np.array([0.95, 0.87])
            self.cls = np.array([0, 2])

    class MockResult:
        def __init__(self):
            self.boxes = MockBox()
            self.names = {0: "person", 2: "car"}

    return [MockResult()]


@pytest.fixture
def mock_detectron2_output():
    """Create mock Detectron2 output."""
    return {
        "instances": type('obj', (object,), {
            "pred_boxes": type('obj', (object,), {
                "tensor": np.array([[100, 100, 200, 200], [300, 150, 400, 280]])
            })(),
            "scores": np.array([0.95, 0.87]),
            "pred_classes": np.array([0, 2])
        })()
    }


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
                self.xyxy = np.array([])
                self.conf = np.array([])
                self.cls = np.array([])

        class MockResult:
            def __init__(self):
                self.boxes = MockBox()
                self.names = {}

        detections = pf.detections.from_ultralytics([MockResult()])
        assert len(detections) == 0


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

    def test_from_detectron2_with_class_names(self, mock_detectron2_output):
        """Test Detectron2 conversion with class name mapping."""
        class_names = {0: "person", 2: "car"}
        detections = pf.detections.from_detectron2(
            mock_detectron2_output,
            class_names=class_names
        )

        assert detections[0].class_name == "person"
        assert detections[1].class_name == "car"


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
    """Tests for from_sam (Segment Anything Model) converter."""

    def test_from_sam_basic(self):
        """Test basic SAM conversion with masks."""
        # Mock SAM output
        sam_output = {
            'masks': np.array([
                np.ones((100, 100), dtype=bool),
                np.zeros((100, 100), dtype=bool)
            ]),
            'scores': np.array([0.95, 0.87])
        }

        detections = pf.detections.from_sam(sam_output)

        assert len(detections) == 2
        assert detections[0].masks is not None
        assert detections[0].confidence == 0.95


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
