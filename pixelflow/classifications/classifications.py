"""
Core Classification Data Structures for Computer Vision Processing.

A classification row is not a detection row. Three detections mean three objects;
three classifications mean three competing answers about one image. Nothing here
carries geometry, because the model did not localise anything - and a whole-image
rectangle standing in for "no box" reads to every downstream consumer as a
localisation that never happened.
"""

import copy as copy_module
import json

from pixelflow.validators import round_to_decimal
from typing import Any, Dict, Iterator, List, Optional, Union

__all__ = ["Classification", "Classifications"]


class Classification:
    """
    One candidate answer about an image: a class, and how strongly the model backs it.

    Identified by its `class_id` - the position it occupies in the vocabulary the model
    was trained on. That number comes out of the weights and is always right. Its
    `class_name` is metadata about that number, and metadata has to come from somewhere:
    the checkpoint, or the caller. Where neither supplies it, `class_name` is None rather
    than a plausible guess.

    Args:
        class_id (Optional[int]): Index of this class in the model's vocabulary.
        class_name (Optional[str]): Human-readable name, when the model or caller supplies one.
        confidence (Optional[float]): The score the model emitted for this class, rounded to
            the shared precision in `pixelflow.validators`.
        metadata (Optional[Dict[str, Any]]): Additional custom metadata.

    Example:
        >>> import pixelflow as pf
        >>>
        >>> pf.Classification(class_id=207, class_name="golden retriever", confidence=0.94)
        >>>
        >>> # Unnamed: the id still identifies it exactly
        >>> unnamed = pf.Classification(class_id=207, confidence=0.94)
        >>> unnamed.class_name is None
        True

    Notes:
        - `confidence` is whatever the model emitted, reported as given. It is not
          constrained to [0, 1]: a softmax probability, a raw logit and a cosine
          similarity are all legitimate here, and only the caller knows which it is.
        - `class_id` is meaningful only within the model that produced it. Two models'
          index 5 are unrelated unless they share a vocabulary.
    """

    def __init__(self,
                 class_id: Optional[int] = None,
                 class_name: Optional[str] = None,
                 confidence: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        # Coercing here rather than at serialization keeps numpy scalars - which every
        # framework hands over - from reaching to_dict() and breaking json.dumps().
        self.class_id = int(class_id) if class_id is not None else None
        self.class_name = str(class_name) if class_name is not None else None
        self.confidence = round_to_decimal(confidence)
        self.metadata = metadata

    def copy(self) -> "Classification":
        """
        Create an independent copy of this classification.

        Returns:
            Classification: A new classification with the same values and its own metadata.
        """
        # The values were coerced and rounded on the way in, so re-running __init__ on
        # them is a guaranteed no-op. top_k and filter_by_confidence clone every retained
        # row, which made that no-op measurable on full 1000-class results. Same reasoning
        # as KeyPoint._clone.
        twin = object.__new__(Classification)
        twin.__dict__.update(self.__dict__)
        if self.metadata is not None:
            twin.metadata = copy_module.deepcopy(self.metadata)
        return twin

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Classification to dictionary format for JSON serialization and storage.

        Returns:
            Dict[str, Any]: The classification's class_id, class_name, confidence and
                metadata. `class_name` is None when nothing supplied one, so a consumer
                can tell "unnamed" from a real name.

        Example:
            >>> import pixelflow as pf
            >>> pf.Classification(207, "golden retriever", 0.94).to_dict()
            {'class_id': 207, 'class_name': 'golden retriever', 'confidence': 0.94, 'metadata': None}
        """
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (f"Classification(class_id={self.class_id!r}, "
                f"class_name={self.class_name!r}, confidence={self.confidence!r})")


class Classifications:
    """
    Container for the competing answers one model gave about one image.

    Implements the same container protocols as `Detections` - `len()`, indexing,
    iteration - but none of its spatial surface, because none of it means anything
    here. What remains is ranking (`top1`, `top_k`) and thresholding
    (`filter_by_confidence`), which are the two questions a classification result
    actually answers.

    Row order is never load-bearing: converters preserve whatever order the model
    used, and `top1` is defined as the highest-scoring row rather than the first one.

    Args:
        classifications (Optional[List[Classification]]): Rows to start with.
        inference_id (Optional[str]): Unique identifier for the inference session or batch.
        metadata (Optional[Dict[str, Any]]): Additional custom metadata for the whole result.

    Example:
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>>
        >>> model = YOLO("yolo11n-cls.pt")
        >>> result = pf.from_ultralytics_classification(model.predict("dog.jpg"))
        >>>
        >>> result.top1.class_name
        'golden retriever'
        >>> for c in result.top_k(5):
        ...     print(f"{c.class_name}: {c.confidence}")
        >>>
        >>> # Multi-label CLIP: threshold rather than rank
        >>> matches = result.filter_by_confidence(0.3)

    Notes:
        - Scores are reported exactly as the model emitted them. Nothing here
          normalises, re-softmaxes, or assumes they sum to 1, so single-label
          (softmax) and multi-label (independent similarity) results both survive
          intact.
        - Slicing returns a Classifications, so `result[:3]` and `result.top_k(3)`
          agree on type.
    """

    def __init__(self,
                 classifications: Optional[List[Classification]] = None,
                 inference_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.classifications: List[Classification] = list(classifications) if classifications else []
        self.inference_id = inference_id
        self.metadata = metadata

    def add_classification(self, classification: Classification) -> None:
        """
        Add a Classification object to the collection.

        Args:
            classification (Classification): Candidate answer to add.

        Example:
            >>> import pixelflow as pf
            >>> result = pf.Classifications()
            >>> result.add_classification(pf.Classification(0, "cat", 0.7))
            >>> len(result)
            1
        """
        self.classifications.append(classification)

    def __len__(self) -> int:
        return len(self.classifications)

    def __iter__(self) -> Iterator[Classification]:
        return iter(self.classifications)

    def __getitem__(self, index) -> Union[Classification, "Classifications"]:
        # A slice of a result is still a result, so it keeps the type and the
        # identifiers with it. Detections returns a bare list here; that is a wart
        # worth not reproducing.
        if isinstance(index, slice):
            return self._rebuild(self.classifications[index])
        return self.classifications[index]

    def _rebuild(self, rows: List[Classification]) -> "Classifications":
        """Wrap `rows` in a Classifications carrying this one's identifiers.

        `metadata` is copied rather than shared: a ranked or filtered result is a
        separate object, and handing it the parent's dict let a write on one reach
        the other. Every derived collection is built here so none of them can
        rediscover that on its own.
        """
        return Classifications(
            rows,
            inference_id=self.inference_id,
            metadata=copy_module.deepcopy(self.metadata) if self.metadata is not None else None,
        )

    @property
    def top1(self) -> Optional[Classification]:
        """
        The highest-scoring answer, or None when there is nothing to rank.

        Defined by score rather than by position, so it stays correct no matter what
        order a converter produced. Rows carrying no confidence cannot win, since
        there is nothing to compare them on.

        Returns:
            Optional[Classification]: The best-scoring row, or None if the collection
                is empty or no row carries a confidence.

        Example:
            >>> result.top1.class_name
            'golden retriever'
        """
        scored = [c for c in self.classifications if c.confidence is not None]
        if not scored:
            return None
        return max(scored, key=lambda c: c.confidence)

    def top_k(self, n: int) -> "Classifications":
        """
        The `n` highest-scoring answers, best first.

        Args:
            n (int): How many rows to keep. Fewer are returned if the collection is
                smaller.

        Returns:
            Classifications: A new collection, sorted by confidence descending.

        Example:
            >>> for c in result.top_k(5):
            ...     print(c.class_name, c.confidence)

        Notes:
            - Rows with no confidence sort last rather than raising.
            - Ranking is the right question for a single-label model. For multi-label
              scores, where the number of true labels varies per image,
              `filter_by_confidence` is the one that answers it.
        """
        # Unscored rows sort to the bottom rather than raising on the comparison.
        ranked = sorted(
            self.classifications,
            key=lambda c: c.confidence if c.confidence is not None else float("-inf"),
            reverse=True,
        )
        return self._rebuild([c.copy() for c in ranked[:n]])

    def filter_by_confidence(self, threshold: float) -> "Classifications":
        """
        Keep the answers scoring at or above `threshold`.

        Args:
            threshold (float): Minimum score to keep.

        Returns:
            Classifications: A new collection containing the rows that clear it.

        Example:
            >>> # CLIP zero-shot: however many labels genuinely match
            >>> matches = result.filter_by_confidence(0.3)

        Notes:
            - Rows with no confidence are excluded, matching `Detections`.
            - Input order is preserved; this ranks nothing.
        """
        kept = [c.copy() for c in self.classifications
                if c.confidence is not None and c.confidence >= threshold]
        return self._rebuild(kept)

    def copy(self) -> "Classifications":
        """
        Create a deep copy of the collection.

        Returns:
            Classifications: A new collection, independent of this one.
        """
        return self._rebuild([c.copy() for c in self.classifications])

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Convert all classifications to a list of dictionaries.

        Returns:
            List[Dict[str, Any]]: One dictionary per candidate answer, in the order
                they are held.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame(result.to_dict())
        """
        return [c.to_dict() for c in self.classifications]

    def to_json(self) -> str:
        """
        Convert all classifications to a JSON string.

        Returns:
            str: JSON array of the candidate answers, indented to match `Detections`.

        Example:
            >>> with open("classifications.json", "w") as f:
            ...     f.write(result.to_json())
        """
        return json.dumps(self.to_dict(), indent=4)

    def __repr__(self) -> str:
        return f"Classifications({len(self.classifications)} rows)"
