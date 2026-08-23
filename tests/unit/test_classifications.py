"""
Unit tests for pixelflow.classifications.

Covers the Classification row, the Classifications container, both converters, and
the annotator. The rounding-parity test is the important one: it pins Classification
to the same precision policy as Detection so the two can never drift apart.
"""

import json

import numpy as np
import pytest
from types import SimpleNamespace

import pixelflow as pf


@pytest.fixture
def sample_classifications():
    """Three candidates, deliberately not in ranked order."""
    return pf.Classifications([
        pf.Classification(class_id=0, class_name="cat", confidence=0.15),
        pf.Classification(class_id=1, class_name="golden retriever", confidence=0.82),
        pf.Classification(class_id=2, class_name="dingo", confidence=0.03),
    ])


@pytest.fixture
def mock_classification_result():
    """Mock Ultralytics YOLO classification result."""
    class MockProbs:
        def __init__(self, data):
            self.data = np.array(data)

    class MockResult:
        def __init__(self):
            self.probs = MockProbs([0.15, 0.82, 0.03])
            self.names = {0: "cat", 1: "golden retriever", 2: "dingo"}
            self.boxes = None
            self.masks = None
            self.keypoints = None

    return [MockResult()]


# ============================================================================
# Classification
# ============================================================================

class TestClassification:
    """Tests for the single-row Classification type."""

    def test_classification_creation(self):
        """Test creating a classification with all fields."""
        c = pf.Classification(class_id=207, class_name="golden retriever", confidence=0.94)
        assert c.class_id == 207
        assert c.class_name == "golden retriever"
        assert c.confidence == 0.94

    def test_classification_defaults_to_none(self):
        """Test that every field is optional."""
        c = pf.Classification()
        assert c.class_id is None
        assert c.class_name is None
        assert c.confidence is None
        assert c.metadata is None

    def test_classification_keeps_id_without_name(self):
        """An id with no known name keeps the id and leaves the name None."""
        c = pf.Classification(class_id=207, confidence=0.94)
        assert c.class_id == 207
        assert c.class_name is None

    def test_classification_coerces_numpy_scalars(self):
        """Numpy scalars are converted at construction, not at serialization."""
        c = pf.Classification(class_id=np.int64(3), confidence=np.float32(0.5))
        assert isinstance(c.class_id, int)
        assert isinstance(c.confidence, float)
        json.dumps(c.to_dict())  # must not raise

    def test_classification_to_dict(self):
        """Test dictionary conversion."""
        c = pf.Classification(207, "golden retriever", 0.94)
        assert c.to_dict() == {
            "class_id": 207,
            "class_name": "golden retriever",
            "confidence": 0.94,
            "metadata": None,
        }

    def test_classification_accepts_scores_outside_zero_one(self):
        """Cosine similarity is [-1, 1] and logits are unbounded; neither is clamped."""
        assert pf.Classification(confidence=-0.42).confidence == -0.42
        assert pf.Classification(confidence=7.3).confidence == 7.3

    def test_classification_copy_is_independent(self):
        """Test that copies do not share metadata."""
        c = pf.Classification(0, "cat", 0.5, metadata={"task": "zero-shot"})
        twin = c.copy()
        twin.metadata["task"] = "changed"
        assert c.metadata["task"] == "zero-shot"


class TestRoundingParity:
    """Confidence must round identically in Classification and Detection."""

    @pytest.mark.parametrize("raw", [
        0.123456789, 0.9999999, 0.0001234, 1 / 3, 0.5, 0.0, 1.0,
        np.float32(0.87654321), 0.6666666666666666,
    ])
    def test_confidence_matches_detection(self, raw):
        """The same score through either type must produce the same value."""
        assert pf.Classification(confidence=raw).confidence == pf.Detection(confidence=raw).confidence

    def test_confidence_uses_the_shared_policy(self):
        """The precision comes from validators, not from a local constant."""
        from pixelflow.validators import CONFIDENCE_DECIMALS

        raw = 0.123456789
        assert pf.Classification(confidence=raw).confidence == round(raw, CONFIDENCE_DECIMALS)


# ============================================================================
# Classifications
# ============================================================================

class TestClassifications:
    """Tests for the Classifications container."""

    def test_creation_empty(self):
        """Test creating an empty container."""
        result = pf.Classifications()
        assert len(result) == 0
        assert result.top1 is None

    def test_add_classification(self):
        """Test appending a row."""
        result = pf.Classifications()
        result.add_classification(pf.Classification(0, "cat", 0.7))
        assert len(result) == 1
        assert result[0].class_name == "cat"

    def test_len_iteration_and_indexing(self, sample_classifications):
        """Test the container protocols Detections also implements."""
        assert len(sample_classifications) == 3
        assert [c.class_name for c in sample_classifications] == ["cat", "golden retriever", "dingo"]
        assert sample_classifications[1].class_name == "golden retriever"
        assert sample_classifications[-1].class_name == "dingo"

    def test_slicing_returns_classifications(self, sample_classifications):
        """A slice of a result is still a result, unlike Detections."""
        subset = sample_classifications[:2]
        assert isinstance(subset, pf.Classifications)
        assert len(subset) == 2

    def test_top1_ranks_by_score_not_position(self, sample_classifications):
        """top1 is the best row, not the first one."""
        assert sample_classifications.top1.class_name == "golden retriever"

    def test_top1_is_none_when_nothing_is_scored(self):
        """Rows carrying no confidence cannot win."""
        result = pf.Classifications([pf.Classification(0, "cat"), pf.Classification(1, "dog")])
        assert result.top1 is None

    def test_top_k_ranks_best_first(self, sample_classifications):
        """Test that top_k sorts descending."""
        top2 = sample_classifications.top_k(2)
        assert isinstance(top2, pf.Classifications)
        assert [c.class_name for c in top2] == ["golden retriever", "cat"]

    def test_top_k_larger_than_collection(self, sample_classifications):
        """Asking for more rows than exist returns what there is."""
        assert len(sample_classifications.top_k(99)) == 3

    def test_top_k_sorts_unscored_rows_last(self):
        """An unscored row ranks below every scored one rather than raising."""
        result = pf.Classifications([
            pf.Classification(0, "unscored"),
            pf.Classification(1, "scored", 0.1),
        ])
        assert [c.class_name for c in result.top_k(2)] == ["scored", "unscored"]

    def test_top_k_does_not_mutate_source(self, sample_classifications):
        """Ranking returns a new collection and leaves the original order alone."""
        sample_classifications.top_k(3)
        assert [c.class_name for c in sample_classifications] == ["cat", "golden retriever", "dingo"]

    def test_filter_by_confidence(self, sample_classifications):
        """Test thresholding, which is what multi-label results need."""
        kept = sample_classifications.filter_by_confidence(0.1)
        assert isinstance(kept, pf.Classifications)
        assert [c.class_name for c in kept] == ["cat", "golden retriever"]

    def test_filter_by_confidence_preserves_input_order(self):
        """Thresholding ranks nothing."""
        result = pf.Classifications([
            pf.Classification(0, "low", 0.4),
            pf.Classification(1, "high", 0.9),
        ])
        assert [c.class_name for c in result.filter_by_confidence(0.3)] == ["low", "high"]

    def test_filter_by_confidence_excludes_unscored(self):
        """Matches how Detections treats a missing confidence."""
        result = pf.Classifications([pf.Classification(0, "cat")])
        assert len(result.filter_by_confidence(0.0)) == 0

    def test_copy_is_independent(self, sample_classifications):
        """Test that copies do not share rows."""
        twin = sample_classifications.copy()
        twin[0].class_name = "changed"
        assert sample_classifications[0].class_name == "cat"

    def test_derived_collections_do_not_share_rows(self, sample_classifications):
        """top_k and filter_by_confidence copy their rows, as the Detections filters do."""
        sample_classifications.top_k(1)[0].class_name = "changed"
        assert sample_classifications[1].class_name == "golden retriever"

    def test_to_dict(self, sample_classifications):
        """Test conversion to a list of dictionaries."""
        data = sample_classifications.to_dict()
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[1]["class_name"] == "golden retriever"

    def test_to_json_is_parseable(self, sample_classifications):
        """Test JSON serialization."""
        data = json.loads(sample_classifications.to_json())
        assert isinstance(data, list)
        assert data[1]["confidence"] == 0.82

    @pytest.mark.parametrize("derive", [
        lambda r: r.top_k(1),
        lambda r: r.filter_by_confidence(0.1),
        lambda r: r.copy(),
        lambda r: r[:1],
    ])
    def test_derived_collections_do_not_share_metadata(self, derive):
        """Every path that builds a new collection must copy the metadata dict."""
        result = pf.Classifications([pf.Classification(0, "cat", 0.9)], metadata={"k": 1})
        derive(result).metadata["k"] = 99
        assert result.metadata == {"k": 1}

    def test_identifiers_survive_derivation(self):
        """A ranked or filtered result is still the same inference."""
        result = pf.Classifications(
            [pf.Classification(0, "cat", 0.9)], inference_id="abc-123"
        )
        assert result.top_k(1).inference_id == "abc-123"
        assert result.filter_by_confidence(0.1).inference_id == "abc-123"


# ============================================================================
# from_scores
# ============================================================================

class TestFromScores:
    """Tests for the framework-free converter."""

    def test_class_ids_default_to_position(self):
        """scores[i] is the score for class i when the full vector is passed."""
        result = pf.from_scores([0.1, 0.2, 0.7])
        assert [c.class_id for c in result] == [0, 1, 2]

    def test_explicit_class_ids(self):
        """A partial vector carries its own ids."""
        result = pf.from_scores([0.1, 0.9], class_ids=[7, 9])
        assert [c.class_id for c in result] == [7, 9]

    def test_labels_as_list(self):
        """Test list label format."""
        result = pf.from_scores([0.1, 0.9], labels=["cat", "dog"])
        assert [c.class_name for c in result] == ["cat", "dog"]

    def test_labels_as_dict(self):
        """Test dict label format."""
        result = pf.from_scores([0.1, 0.9], class_ids=[7, 9], labels={7: "cat", 9: "dog"})
        assert [c.class_name for c in result] == ["cat", "dog"]

    def test_labels_as_dicts_list(self):
        """Test rich label format."""
        result = pf.from_scores([0.1, 0.9], labels=[{"id": 0, "name": "cat"},
                                                    {"id": 1, "name": "dog"}])
        assert [c.class_name for c in result] == ["cat", "dog"]

    def test_missing_label_leaves_name_none(self):
        """A name is never invented for an id the labels do not cover."""
        result = pf.from_scores([0.1, 0.9], labels=["cat"])
        assert result[1].class_id == 1
        assert result[1].class_name is None

    def test_no_labels_at_all(self):
        """Ids alone are a complete answer."""
        result = pf.from_scores([0.1, 0.9])
        assert all(c.class_name is None for c in result)

    @pytest.mark.parametrize("scores", [
        [0.1, 0.9, 0.5],       # order preserved, nothing ranked
        [0.7, 0.2, 0.1],       # a softmax the caller already applied
        [0.31, 0.28, 0.44],    # independent similarities, summing to anything
        [-0.12, 0.44],         # cosine similarity is legitimately negative
    ])
    def test_scores_pass_through_untouched(self, scores):
        """Nothing normalises, re-softmaxes, ranks or clamps."""
        assert [c.confidence for c in pf.from_scores(scores)] == scores

    def test_accepts_numpy(self):
        """Test numpy input."""
        result = pf.from_scores(np.array([0.1, 0.9], dtype=np.float32))
        assert len(result) == 2
        assert isinstance(result[0].confidence, float)

    def test_empty_scores(self):
        """Test empty input."""
        assert len(pf.from_scores([])) == 0

    def test_class_ids_length_mismatch_raises(self):
        """Test validation of mismatched arrays."""
        with pytest.raises(ValueError, match="class_ids describes"):
            pf.from_scores([0.1, 0.9], class_ids=[0])


# ============================================================================
# from_ultralytics_classification
# ============================================================================

class TestFromUltralyticsClassification:
    """Tests for the Ultralytics classification converter."""

    def test_basic_conversion(self, mock_classification_result):
        """Test that every scored class becomes a row."""
        result = pf.from_ultralytics_classification(mock_classification_result)
        assert isinstance(result, pf.Classifications)
        assert len(result) == 3
        assert result.top1.class_name == "golden retriever"
        assert result.top1.confidence == 0.82

    def test_returns_every_class_in_id_order_by_default(self, mock_classification_result):
        """Faithful by default: no truncation and no ranking."""
        result = pf.from_ultralytics_classification(mock_classification_result)
        assert [c.class_id for c in result] == [0, 1, 2]

    def test_top_k_ranks_and_truncates(self, mock_classification_result):
        """Test the per-frame escape hatch."""
        result = pf.from_ultralytics_classification(mock_classification_result, top_k=2)
        assert [c.class_name for c in result] == ["golden retriever", "cat"]

    def test_labels_override_model_names(self, mock_classification_result):
        """A fine-tuned checkpoint's own vocabulary wins when the caller supplies one."""
        result = pf.from_ultralytics_classification(
            mock_classification_result, labels=["a", "b", "c"]
        )
        assert [c.class_name for c in result] == ["a", "b", "c"]

    def test_accepts_a_bare_result(self, mock_classification_result):
        """Test that the list wrapper is optional."""
        result = pf.from_ultralytics_classification(mock_classification_result[0])
        assert len(result) == 3

    def test_empty_input(self):
        """Test empty input."""
        assert len(pf.from_ultralytics_classification([])) == 0

    def test_multiple_images_raise(self, mock_classification_result):
        """Silently reporting only the first image would drop the rest."""
        with pytest.raises(ValueError, match="only one can be converted"):
            pf.from_ultralytics_classification(mock_classification_result * 2)

    def test_detection_result_raises(self):
        """A detection Result names the converter that does handle it."""
        detection_result = SimpleNamespace(probs=None, boxes=[], names={})
        with pytest.raises(ValueError, match="pf.from_ultralytics"):
            pf.from_ultralytics_classification(detection_result)
