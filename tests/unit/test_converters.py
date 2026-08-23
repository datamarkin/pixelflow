"""
Unit tests for pixelflow.detections.converters module.

Tests framework-specific converters including Ultralytics, Detectron2,
Transformers, SAM, Datamarkin, and Falcon Perception converters.
"""

import pytest
import numpy as np
from types import SimpleNamespace
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
# Florence-2 Converter Tests
# ============================================================================

class TestFlorence2Converter:
    """Tests for from_florence2 converter."""

    def test_from_florence2_object_detection(self):
        """The <OD> task yields boxes with sequential class_ids."""
        parsed = {
            "<OD>": {
                "bboxes": [[10, 10, 50, 50], [60, 60, 90, 90]],
                "labels": ["cat", "dog"],
            }
        }

        detections = pf.detections.from_florence2(parsed, task_prompt="<OD>")

        assert len(detections) == 2
        assert detections[0].bbox == [10, 10, 50, 50]
        assert detections[0].class_name == "cat"
        assert detections[1].class_name == "dog"
        # Sequential class_ids drive consistent colour mapping.
        assert [d.class_id for d in detections] == [0, 1]
        assert detections[0].confidence == 1.0

    def test_from_florence2_segmentation_polygons(self):
        """Polygon tasks populate segments and a derived bbox, both sub-pixel.

        Fractional vertices matter here: the bbox is their hull, so truncating a
        vertex truncated the box with it, past the validator that never saw the
        fraction.
        """
        parsed = {
            "<REFERRING_EXPRESSION_SEGMENTATION>": {
                "polygons": [[[10.7, 20.3, 50.9, 20.3, 50.9, 60.4, 10.7, 60.4]]],
                "labels": ["cat"],
            }
        }

        detections = pf.detections.from_florence2(
            parsed, task_prompt="<REFERRING_EXPRESSION_SEGMENTATION>"
        )

        assert len(detections) == 1
        assert detections[0].segments[0] == [10.7, 20.3]
        # bbox is the axis-aligned hull of the polygon.
        assert detections[0].bbox == [10.7, 20.3, 50.9, 60.4]

    @pytest.mark.parametrize("polygons", [
        [[10, 10, 50, 10, 50, 50, 10, 50]],        # flat
        [[[10, 10, 50, 10, 50, 50, 10, 50]]],      # nested once
        [[[[10, 10, 50, 10, 50, 50, 10, 50]]]],    # nested twice
    ], ids=["flat", "nested", "deep"])
    def test_from_florence2_polygon_nesting_depths(self, polygons):
        """Florence-2 nests polygons inconsistently; all depths must parse."""
        parsed = {"<SEG>": {"polygons": polygons, "labels": ["cat"]}}

        detections = pf.detections.from_florence2(parsed, task_prompt="<SEG>")

        assert len(detections) == 1
        assert detections[0].bbox == [10, 10, 50, 50]

    def test_from_florence2_keeps_all_polygon_parts(self):
        """An instance split into several polygons keeps every part."""
        parsed = {
            "<SEG>": {
                # One instance, two disjoint parts (e.g. split by occlusion).
                "polygons": [[[0, 0, 10, 0, 10, 10], [90, 90, 100, 90, 100, 100]]],
                "labels": ["cat"],
            }
        }

        detections = pf.detections.from_florence2(parsed, task_prompt="<SEG>")

        assert len(detections) == 1
        assert len(detections[0].masks) == 2
        # bbox spans both parts rather than just the first.
        assert detections[0].bbox == [0, 0, 100, 100]

    @pytest.mark.parametrize("task", [
        "<DENSE_REGION_CAPTION>",       # describes: "a red car parked"
        "<CAPTION_TO_PHRASE_GROUNDING>",  # locates the caller's own words
        "<SOME_NEW_TASK>",              # unrecognised: assumed to emit prose
    ])
    def test_from_florence2_prose_goes_to_text(self, task):
        """Tasks that do not name a class fill `text` and leave class_* None.

        Their output shape is identical to <OD>'s, so only task_prompt tells them
        apart. A caption in class_name would also mint a fake class_id per unique
        string. Unrecognised tasks default here because Florence-2's region tasks
        emit prose by default, and a category name sitting in text is inert where
        prose in class_name leaks into the crossings class-name map.
        """
        parsed = {task: {
            "bboxes": [[10, 10, 50, 50], [60, 60, 90, 90]],
            "labels": ["a red car parked", "a man in a blue jacket"],
        }}

        detections = pf.detections.from_florence2(parsed, task_prompt=task)

        assert [d.text for d in detections] == [
            "a red car parked", "a man in a blue jacket"
        ]
        assert [d.class_name for d in detections] == [None, None]
        assert [d.class_id for d in detections] == [None, None]

    def test_from_florence2_region_proposal_names_a_class(self):
        """<REGION_PROPOSAL> is a vocabulary task and keeps class_name/class_id."""
        parsed = {
            "<REGION_PROPOSAL>": {
                "bboxes": [[10, 10, 50, 50], [60, 60, 90, 90]],
                "labels": ["region", "region"],
            }
        }

        detections = pf.detections.from_florence2(
            parsed, task_prompt="<REGION_PROPOSAL>"
        )

        assert [d.class_name for d in detections] == ["region", "region"]
        # Repeated labels share an id - that is what makes the id a class and not
        # a row number.
        assert [d.class_id for d in detections] == [0, 0]
        assert [d.text for d in detections] == [None, None]

    def test_from_florence2_ocr_with_region(self):
        """<OCR_WITH_REGION> returns quads, which are kept as read."""
        parsed = {
            "<OCR_WITH_REGION>": {
                # Flat [x1, y1, x2, y2, x3, y3, x4, y4] per region.
                "quad_boxes": [[12, 22, 115, 15, 118, 48, 15, 55]],
                "labels": ["Main St"],
            }
        }

        detections = pf.detections.from_florence2(
            parsed, task_prompt="<OCR_WITH_REGION>"
        )

        assert len(detections) == 1
        assert detections[0].text == "Main St"
        assert detections[0].segments == [[12, 22], [115, 15], [118, 48], [15, 55]]
        # bbox is the axis-aligned hull, so zones and filters keep working.
        assert detections[0].bbox == [12, 15, 118, 55]
        assert detections[0].class_name is None

    def test_from_florence2_referring_expression_goes_to_text(self):
        """The polygon branch routes labels the same way the bbox branch does."""
        parsed = {
            "<REFERRING_EXPRESSION_SEGMENTATION>": {
                "polygons": [[[10, 20, 50, 20, 50, 60, 10, 60]]],
                "labels": ["the red car on the left"],
            }
        }

        detections = pf.detections.from_florence2(
            parsed, task_prompt="<REFERRING_EXPRESSION_SEGMENTATION>"
        )

        assert detections[0].text == "the red car on the left"
        assert detections[0].class_name is None
        assert detections[0].class_id is None

    @pytest.mark.parametrize("task", [
        "<CAPTION>", "<OCR>", "<REGION_TO_CATEGORY>",
        "<REGION_TO_DESCRIPTION>", "<REGION_TO_OCR>",
    ])
    def test_from_florence2_pure_text_tasks_raise(self, task):
        """Tasks the processor treats as pure_text carry no geometry to locate."""
        with pytest.raises(ValueError, match="text only"):
            pf.detections.from_florence2({task: "some string"}, task_prompt=task)

    def test_from_florence2_missing_task_prompt_raises(self):
        """A task_prompt absent from the parsed result is an error."""
        with pytest.raises(ValueError):
            pf.detections.from_florence2({"<OD>": {}}, task_prompt="<CAPTION>")

    def test_from_florence2_unsupported_data_shape_raises(self):
        """Data without a recognised field combination raises ValueError."""
        parsed = {"<OD>": {"something_else": []}}

        with pytest.raises(ValueError):
            pf.detections.from_florence2(parsed, task_prompt="<OD>")


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

    def test_from_ultralytics_segments_keep_subpixel_precision(self, mock_ultralytics_result):
        """Polygon vertices keep their fraction, in a compact representation.

        The vertices arrive as float32, whose nearest value to 10.7 is
        10.699999809265137 - rounding without widening to float64 first leaves
        that expansion intact and .tolist() writes every digit of it.
        """
        polygon = np.array([[10.7, 20.3], [50.9, 20.3], [50.9, 60.4]], dtype=np.float32)
        result = mock_ultralytics_result[0]
        result.masks = SimpleNamespace(xy=[polygon, polygon])  # one per detection

        detections = pf.detections.from_ultralytics([result])

        assert detections[0].segments[0] == [10.7, 20.3]

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

    def test_build_keypoints_preserves_subpixel_precision(self):
        """The shared keypoint intake does not truncate.

        _build_keypoints feeds from_arrays, from_detectron2, from_mayaku and
        from_ultralytics, so it is the highest-fan-in coordinate boundary in
        this module - and every other keypoint test here uses whole numbers,
        which cannot tell a truncating intake from a faithful one.
        """
        from pixelflow.detections.converters import _build_keypoints

        keypoints = _build_keypoints(
            np.array([[10.7, 20.3, 0.9], [50.9, 60.4, 0.8]], dtype=np.float32),
            kp_names=["nose", "eye"],
        )

        assert [(kp.x, kp.y) for kp in keypoints] == [(10.7, 20.3), (50.9, 60.4)]

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
                # 3 keypoints: (x, y, confidence)
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
        assert detections[0].keypoints[0].id == 0
        assert detections[0].keypoints[0].confidence == 2.0
        assert detections[0].keypoints[1].name == "left_eye"
        assert detections[0].keypoints[1].confidence == 1.0
        assert detections[0].keypoints[2].name == "right_eye"
        assert detections[0].keypoints[2].confidence == 0.0


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

    def test_from_mayaku_uses_checkpoint_vocabulary(self, mock_mayaku_output):
        """predictor.class_names is a plain List[str] indexed by class_id.

        Mayaku's pretrained checkpoints are Objects365 (365 classes), so the
        vocabulary must come from the checkpoint. Verified against the real
        mayaku-n-det weights: class 0 is "Person" and class 5 is "Car",
        whereas COCO puts "bus" at 5.
        """
        # Truncated stand-in for predictor.class_names.
        class_names = ["Person", "Sneakers", "Chair", "Other Shoes", "Hat", "Car"]

        detections = pf.detections.from_mayaku(mock_mayaku_output, labels=class_names)

        assert detections[0].class_id == 0
        assert detections[0].class_name == "Person"
        assert detections[1].class_id == 2
        assert detections[1].class_name == "Chair"

    def test_from_mayaku_out_of_range_class_id_is_none(self):
        """A class_id past the end of the vocabulary yields None, not a crash."""
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
                self.pred_boxes = MockBoxes([[10, 10, 50, 50]])
                self.scores = MockTensor([0.9])
                self.pred_classes = MockTensor([364])  # last Objects365 id
                self._fields = {"pred_boxes", "scores", "pred_classes"}
            def to(self, device):
                return self
            def has(self, field):
                return field in self._fields
            def __len__(self):
                return 1

        # Passing a COCO-sized vocabulary to a 365-class model is the mistake
        # this guards against: it must degrade to None rather than raise.
        detections = pf.detections.from_mayaku(MockInstances(), labels=["person", "car"])

        assert len(detections) == 1
        assert detections[0].class_id == 364
        assert detections[0].class_name is None

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
        assert [kp.id for kp in detections[0].keypoints] == [0, 1, 2]
        assert detections[0].keypoints[0].confidence == pytest.approx(0.9)
        assert detections[0].keypoints[1].confidence == pytest.approx(0.5)
        assert detections[0].keypoints[2].confidence == pytest.approx(0.0)

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
# get_label_info Helper Tests
# ============================================================================

class TestGetLabelInfo:
    """Tests for get_label_info helper function."""

    def test_list_format(self):
        from pixelflow.labels import get_label_info
        name, kp_names = get_label_info(["person", "car", "dog"], 1)
        assert name == "car"
        assert kp_names is None

    def test_dict_format(self):
        from pixelflow.labels import get_label_info
        name, kp_names = get_label_info({0: "person", 5: "car"}, 5)
        assert name == "car"
        assert kp_names is None

    def test_dict_nonsequential_keys(self):
        from pixelflow.labels import get_label_info
        name, kp_names = get_label_info({0: "person", 91: "banana"}, 91)
        assert name == "banana"

    def test_rich_format(self):
        from pixelflow.labels import get_label_info
        labels = [{"id": 0, "name": "person", "keypoints": [{"id": 0, "name": "nose"}]}]
        name, kp_names = get_label_info(labels, 0)
        assert name == "person"
        assert kp_names == ["nose"]

    def test_rich_format_no_keypoints(self):
        from pixelflow.labels import get_label_info
        labels = [{"id": 1, "name": "bicycle"}]
        name, kp_names = get_label_info(labels, 1)
        assert name == "bicycle"
        assert kp_names is None

    def test_missing_id_returns_none(self):
        from pixelflow.labels import get_label_info
        name, kp_names = get_label_info(["person", "car"], 99)
        assert name is None
        assert kp_names is None

    def test_none_labels(self):
        from pixelflow.labels import get_label_info
        name, kp_names = get_label_info(None, 0)
        assert name is None
        assert kp_names is None

    def test_none_class_id(self):
        from pixelflow.labels import get_label_info
        name, kp_names = get_label_info(["person"], None)
        assert name is None

    def test_empty_list(self):
        from pixelflow.labels import get_label_info
        name, kp_names = get_label_info([], 0)
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
        api_response = {
            "predictions": {
                "objects": [
                    {
                        "bbox": [100, 100, 200, 200],
                        "bbox_score": 0.95,
                        "class": "person",
                    },
                    {
                        "bbox": [300, 150, 400, 280],
                        "bbox_score": 0.87,
                        "class": "car",
                    },
                ]
            }
        }

        detections = pf.detections.from_datamarkin(api_response)

        assert len(detections) == 2
        assert detections[0].bbox == [100, 100, 200, 200]
        assert detections[0].confidence == 0.95
        assert detections[0].class_name == "person"
        assert detections[1].class_name == "car"

    def test_from_datamarkin_keypoints(self):
        """Keypoint probability is carried through as confidence, not thresholded."""
        api_response = {
            "predictions": {
                "objects": [
                    {
                        "bbox": [100, 100, 200, 200],
                        "bbox_score": 0.9,
                        "class": "person",
                        "keypoints": [
                            {"name": "nose", "point": [150, 120], "probability": 0.8},
                            {"name": "left_eye", "point": [140, 115], "probability": 0.0},
                        ],
                    }
                ]
            }
        }

        detections = pf.detections.from_datamarkin(api_response)

        assert len(detections[0].keypoints) == 2
        assert detections[0].keypoints[0].name == "nose"
        assert detections[0].keypoints[0].x == 150
        assert detections[0].keypoints[0].confidence == pytest.approx(0.8)
        assert detections[0].keypoints[1].confidence == pytest.approx(0.0)
        assert [kp.id for kp in detections[0].keypoints] == [0, 1]


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestConverterEdgeCases:
    """Test edge cases and error handling for converters."""

    def test_converter_with_none_values(self):
        """Test converters handle None/null values gracefully."""
        api_response = {
            "predictions": {
                "objects": [
                    {"bbox": [100, 100, 200, 200], "bbox_score": None, "class": None}
                ]
            }
        }

        detections = pf.detections.from_datamarkin(api_response)
        assert len(detections) == 1
        assert detections[0].confidence is None

    def test_converter_empty_input(self):
        """Test converters with empty input."""
        assert len(pf.detections.from_datamarkin({})) == 0
        assert len(pf.detections.from_datamarkin({"predictions": {"objects": []}})) == 0


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


class TestArraysConverter:
    """Tests for from_arrays, the framework-free converter."""

    def test_boxes_scores_and_classes(self):
        detections = pf.detections.from_arrays(
            boxes=[[10, 20, 110, 220], [30, 40, 130, 240]],
            scores=[0.9, 0.8],
            class_ids=[0, 2],
        )
        assert len(detections) == 2
        assert detections[0].bbox == [10, 20, 110, 220]
        assert detections[0].confidence == 0.9
        assert detections[1].class_id == 2

    def test_labels_resolve_class_names(self):
        detections = pf.detections.from_arrays(
            boxes=[[0, 0, 1, 1]], scores=[0.5], class_ids=[2],
            labels=["person", "bike", "car"],
        )
        assert detections[0].class_name == "car"

    def test_numpy_input(self):
        detections = pf.detections.from_arrays(
            boxes=np.array([[0.0, 0.0, 5.0, 5.0]]),
            scores=np.array([0.7]),
            class_ids=np.array([1]),
        )
        assert len(detections) == 1
        assert detections[0].class_id == 1

    def test_masks_are_cast_to_bool(self):
        detections = pf.detections.from_arrays(
            boxes=[[0, 0, 4, 4]], scores=[0.6], class_ids=[0],
            masks=np.ones((1, 4, 4), dtype=np.float32),
        )
        assert detections[0].masks[0].dtype == bool
        assert detections[0].masks[0].shape == (4, 4)

    def test_keypoints_carry_id_and_score(self):
        keypoints = np.zeros((1, 17, 3), dtype=np.float32)
        keypoints[0, 0] = [5, 6, 0.9]
        detections = pf.detections.from_arrays(
            boxes=[[0, 0, 10, 10]], scores=[0.8], class_ids=[0], keypoints=keypoints,
        )
        first = detections[0].keypoints[0]
        assert (first.x, first.y) == (5, 6)
        assert first.id == 0
        assert first.confidence == pytest.approx(0.9)

    def test_keypoints_are_unnamed_without_labels(self):
        """A model whose vocabulary nobody stated gets ids, not COCO's pose names."""
        keypoints = np.zeros((1, 21, 3), dtype=np.float32)
        detections = pf.detections.from_arrays(
            boxes=[[0, 0, 10, 10]], scores=[0.8], class_ids=[0], keypoints=keypoints,
        )
        kps = detections[0].keypoints
        assert len(kps) == 21
        assert [kp.id for kp in kps] == list(range(21))
        assert all(kp.name is None for kp in kps)

    def test_empty_input(self):
        assert len(pf.detections.from_arrays(boxes=[], scores=[], class_ids=[])) == 0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="scores describes 2"):
            pf.detections.from_arrays(boxes=[[0, 0, 1, 1]], scores=[0.5, 0.6], class_ids=[0])

    def test_torch_tensors_are_detached(self):
        torch = pytest.importorskip("torch")
        detections = pf.detections.from_arrays(
            boxes=torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True),
            scores=torch.tensor([0.5]),
            class_ids=torch.tensor([1]),
            labels=["a", "b"],
        )
        assert detections[0].class_name == "b"

    def test_texts_and_segments_ride_along(self):
        """A vendored OCR model returns arrays, so from_arrays has to carry the read."""
        quads = [
            [[10, 20], [110, 22], [108, 60], [8, 58]],
            [[30, 80], [130, 80], [130, 120], [30, 120]],
        ]
        detections = pf.detections.from_arrays(
            boxes=[[8, 20, 110, 60], [30, 80, 130, 120]],
            scores=[0.91, 0.55],
            texts=["Hello", "world"],
            segments=quads,
        )
        assert [d.text for d in detections] == ["Hello", "world"]
        assert [len(d.segments) for d in detections] == [4, 4]
        assert detections[0].segments[1] == [110.0, 22.0]

    def test_class_ids_are_optional(self):
        """OCR reads content; it does not pick a class out of a vocabulary."""
        detections = pf.detections.from_arrays(
            boxes=[[0, 0, 1, 1]], scores=[0.9], texts=["read"],
        )
        assert detections[0].class_id is None
        assert detections[0].class_name is None
        assert detections[0].text == "read"

    def test_an_empty_read_is_kept_but_a_missing_one_is_not(self):
        """A located region that decoded to nothing was still located."""
        detections = pf.detections.from_arrays(
            boxes=[[0, 0, 1, 1], [2, 2, 3, 3]], scores=[0.9, 0.8], texts=["", None],
        )
        assert detections[0].text == ""
        assert detections[1].text is None

    def test_text_and_class_name_coexist(self):
        """They answer different questions, so supplying one must not clear the other."""
        detections = pf.detections.from_arrays(
            boxes=[[0, 0, 1, 1]], scores=[0.9], class_ids=[1],
            labels=["plate", "sign"], texts=["ABC-123"],
        )
        assert detections[0].class_name == "sign"
        assert detections[0].text == "ABC-123"

    def test_text_survives_to_dict(self):
        detections = pf.detections.from_arrays(
            boxes=[[0, 0, 1, 1]], scores=[0.9], texts=["Hello"],
            segments=[[[0, 0], [1, 0], [1, 1], [0, 1]]],
        )
        payload = detections[0].to_dict()
        assert payload["text"] == "Hello"
        assert len(payload["segments"]) == 4

    def test_mismatched_texts_and_segments_raise(self):
        with pytest.raises(ValueError, match="texts describes 1"):
            pf.detections.from_arrays(
                boxes=[[0, 0, 1, 1], [2, 2, 3, 3]], scores=[0.9, 0.8], texts=["one"],
            )
        with pytest.raises(ValueError, match="segments describes 1"):
            pf.detections.from_arrays(
                boxes=[[0, 0, 1, 1], [2, 2, 3, 3]], scores=[0.9, 0.8],
                segments=[[[0, 0], [1, 0], [1, 1]]],
            )

    def test_numpy_strings_become_str(self):
        """A caller who did hand over a numpy array should not get numpy.str_ back."""
        detections = pf.detections.from_arrays(
            boxes=[[0, 0, 1, 1]], scores=[0.9], texts=np.array(["Hello"]),
        )
        assert type(detections[0].text) is str


class TestNoBundledVocabulary:
    """pixelflow holds no dataset's class names. Callers state their model's vocabulary."""

    def test_no_label_constants_are_exported(self):
        """A container library knowing about COCO is how the mislabelling bug got written."""
        assert not [name for name in dir(pf) if "COCO" in name.upper()]

    def test_names_come_only_from_the_caller(self):
        ids = [1, 73]
        unnamed = pf.detections.from_arrays(
            boxes=[[0, 0, 1, 1], [2, 2, 3, 3]], scores=[0.9, 0.8], class_ids=ids,
        )
        assert [d.class_id for d in unnamed] == ids
        assert all(d.class_name is None for d in unnamed)

        named = pf.detections.from_arrays(
            boxes=[[0, 0, 1, 1], [2, 2, 3, 3]], scores=[0.9, 0.8], class_ids=ids,
            labels={1: "person", 73: "laptop"},
        )
        assert [d.class_name for d in named] == ["person", "laptop"]


# ============================================================================
# EasyOCR Converter Tests
# ============================================================================

# What easyocr.Reader.readtext actually returns: horizontal text comes back as the
# axis-aligned rectangle expressed as four corners, rotated text as a genuine
# quadrilateral. Module-level so the parametrize decorators below can see them.
HORIZONTAL = ([[10, 20], [110, 20], [110, 50], [10, 50]], "STOP", 0.9812)
ROTATED = ([[12, 22], [115, 15], [118, 48], [15, 55]], "Main St", 0.7431)


class TestEasyOCRConverter:
    """Tests for from_easyocr converter."""

    def test_bbox_is_the_hull_of_the_quad(self):
        """bbox is the axis-aligned hull, so zones and box filters keep working."""
        detections = pf.detections.from_easyocr([HORIZONTAL, ROTATED])

        assert len(detections) == 2
        # A horizontal quad's hull is the quad; a rotated one's is strictly larger.
        assert detections[0].bbox == [10, 20, 110, 50]
        assert detections[1].bbox == [12, 15, 118, 55]

    @pytest.mark.parametrize("item, expected_text, expected_confidence", [
        (HORIZONTAL, "STOP", 0.981),
        (list(HORIZONTAL), "STOP", 0.981),
        ([HORIZONTAL[0], "STOP AHEAD"], "STOP AHEAD", None),
        ({"boxes": HORIZONTAL[0], "text": "STOP", "confident": 0.9812}, "STOP", 0.981),
        ({"boxes": HORIZONTAL[0], "text": "STOP"}, "STOP", None),
    ], ids=["standard", "arabic-lists", "paragraph", "dict", "dict-paragraph"])
    def test_readtext_output_shapes(self, item, expected_text, expected_confidence):
        """Every shape readtext returns converts to the same thing.

        'standard' is the default (quad, text, confidence) tuple; the Arabic path
        rebuilds each result as a list; paragraph=True merges lines and returns no
        confidence at all, so it stays None rather than a fabricated 0 claiming the
        read was certainly wrong; output_format='dict' renames the fields.
        """
        detections = pf.detections.from_easyocr([item])

        assert len(detections) == 1
        assert detections[0].text == expected_text
        assert detections[0].confidence == expected_confidence

    def test_quad_is_preserved_in_segments(self):
        """The corners are kept as read, not collapsed into the xyxy hull."""
        detections = pf.detections.from_easyocr([ROTATED])

        assert detections[0].segments == [[12, 22], [115, 15], [118, 48], [15, 55]]

    def test_no_class_is_invented(self):
        """OCR reads content; it does not pick a class out of a vocabulary."""
        detections = pf.detections.from_easyocr([HORIZONTAL])

        assert detections[0].class_id is None
        assert detections[0].class_name is None

    def test_numpy_coordinates(self):
        """Rotated boxes arrive with numpy ints, not Python ints."""
        quad = [[np.int32(12), np.int32(22)], [np.int32(115), np.int32(15)],
                [np.int32(118), np.int32(48)], [np.int32(15), np.int32(55)]]
        detections = pf.detections.from_easyocr([(quad, "Main St", np.float32(0.74))])

        assert detections[0].segments == [[12, 22], [115, 15], [118, 48], [15, 55]]
        assert isinstance(detections[0].bbox[0], float)

    def test_empty_results(self):
        """No text found is not an error."""
        assert len(pf.detections.from_easyocr([])) == 0

    def test_empty_read_is_kept(self):
        """A located region read as '' is still a region the detector found."""
        detections = pf.detections.from_easyocr([(HORIZONTAL[0], "", 0.12)])

        assert len(detections) == 1
        assert detections[0].text == ""

    def test_detail_zero_raises(self):
        """detail=0 returns strings with no geometry to convert."""
        with pytest.raises(ValueError, match="detail=0"):
            pf.detections.from_easyocr(["STOP", "Main St"])

    def test_json_output_format_raises(self):
        """output_format='json' is also a list of strings."""
        with pytest.raises(ValueError, match="output_format"):
            pf.detections.from_easyocr(['{"boxes": [], "text": "STOP"}'])

    def test_malformed_geometry_warns_and_skips(self):
        """One unusable result does not abort the rest of the conversion."""
        with pytest.warns(UserWarning, match="four corner points"):
            detections = pf.detections.from_easyocr([
                ([[10, 20], [110, 50]], "STOP", 0.98),  # two points, not four
                ROTATED,
            ])

        assert len(detections) == 1
        assert detections[0].text == "Main St"

    def test_non_finite_geometry_warns_and_skips(self):
        """NaN corners are rejected rather than poisoning downstream IoU maths."""
        with pytest.warns(UserWarning):
            detections = pf.detections.from_easyocr([
                ([[10, 20], [float("nan"), 20], [110, 50], [10, 50]], "STOP", 0.98),
                ROTATED,
            ])

        assert len(detections) == 1
        assert detections[0].text == "Main St"

    def test_every_result_unusable_raises(self):
        """A wrong input type is a bad thing to learn from an empty container.

        One degenerate quad is skipped, but nothing surviving means the caller
        almost certainly passed something that is not EasyOCR output.
        """
        with pytest.raises(ValueError, match="not EasyOCR output"):
            with pytest.warns(UserWarning):
                pf.detections.from_easyocr([
                    ([[10, 20], [110, 50]], "STOP", 0.98),
                    ([[1, 2]], "Main St", 0.74),
                ])

    def test_quad_survives_rotation(self):
        """segments is transform-aware, so the orientation is not lost on rotate."""
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        detections = pf.detections.from_easyocr([ROTATED])

        _, rotated = pf.transforms.rotate_detections(image, detections, 30)

        assert len(rotated[0].segments) == 4
        assert rotated[0].segments != detections[0].segments
        assert rotated[0].text == "Main St"

    def test_text_survives_copy(self):
        """copy() must carry text, or transforms silently drop the read."""
        detections = pf.detections.from_easyocr([HORIZONTAL])

        assert detections.copy()[0].text == "STOP"

    def test_text_is_serialized(self):
        """to_dict exposes text as its own key, not buried in metadata."""
        detections = pf.detections.from_easyocr([HORIZONTAL])

        assert detections[0].to_dict()["text"] == "STOP"
