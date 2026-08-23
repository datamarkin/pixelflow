"""
Classification Converters for Machine Learning Framework Integration.

Turns what a classifier emitted into PixelFlow's unified Classifications format.

Scores pass through exactly as the model produced them. Nothing here normalises,
re-softmaxes or assumes the scores sum to 1, because both conventions are real and
the arrays look identical: an ImageNet head emits a softmax over a fixed class list,
while CLIP zero-shot emits an independent similarity per prompt. Only the caller
knows which one they have, so only the caller may transform it.
"""

from typing import Any, Optional

from pixelflow.arrays import to_numpy
from pixelflow.labels import build_name_index

__all__ = [
    "from_scores",
    "from_ultralytics_classification",
]


def from_scores(scores, class_ids=None, labels=None):
    """Convert a plain vector of class scores to a Classifications object.

    The framework-free entry point, and the one every classifier can reach: a CLIP
    zero-shot run is a matrix multiply and a list of prompts, a vendored ResNet is a
    forward pass and a tensor. Neither has a framework container to name a converter
    after.

    Torch tensors and numpy arrays are both accepted; tensors are detached and moved
    to CPU automatically.

    Args:
        scores: `(N,)` scores, one per candidate class, reported exactly as given.
        class_ids: Optional `(N,)` integer class IDs. Defaults to each score's own
            position, which is what a full score vector means -- `scores[i]` is the
            score for class `i`. Pass them explicitly when the vector is not the full
            one, such as a top-5 slice the caller already extracted.
        labels: Optional label definitions for class name resolution. Accepts the same
            three formats as the detection converters:
            - List[str]: ["cat", "dog"] — index = class_id
            - Dict[int, str]: {0: "cat", 1: "dog"} — key = class_id
            - List[dict]: [{"id": 0, "name": "cat"}]

    Returns:
        Classifications: One row per score, in the order they arrived.

    Raises:
        ValueError: If `class_ids` and `scores` describe different numbers of classes.

    Example:
        >>> import pixelflow as pf
        >>>
        >>> # CLIP zero-shot: independent similarities, summing to nothing in particular
        >>> prompts = ["a photo of a cat", "a photo of a dog"]
        >>> result = pf.from_scores([0.21, 0.34], labels=prompts)
        >>> result.top1.class_name
        'a photo of a dog'
        >>>
        >>> # ImageNet head: a softmax the caller already applied
        >>> result = pf.from_scores(probs, labels=class_names)

    Notes:
        - Scores are not transformed. A model emitting logits must be converted by the
          caller -- `torch.softmax(logits, dim=-1)` for a single-label head,
          `torch.sigmoid(logits)` for a multi-label one -- because the raw values alone
          cannot say which was intended.
        - Rows keep their input order. Use `top_k()` to rank them.
        - Pass the model's own class names as `labels`. Where a name cannot be resolved
          the id is carried and `class_name` stays None; it is never invented.
    """
    from .classifications import Classification, Classifications

    scores = to_numpy(scores)
    class_ids = to_numpy(class_ids)

    if class_ids is not None and len(class_ids) != len(scores):
        raise ValueError(
            f"class_ids describes {len(class_ids)} classes but scores describes {len(scores)}"
        )

    # Resolved once rather than per row: a full score vector is every class the model
    # knows, and scanning the labels for each of them is quadratic.
    names = build_name_index(labels)

    result = Classifications()
    for i, score in enumerate(scores):
        class_id = int(class_ids[i]) if class_ids is not None else i
        result.add_classification(Classification(
            class_id=class_id,
            class_name=names.get(class_id) if names else None,
            confidence=float(score),
        ))

    return result


def from_ultralytics_classification(ultralytics_results: Any,
                                    labels=None,
                                    top_k: Optional[int] = None):
    """Convert Ultralytics YOLO classification results to a Classifications object.

    The counterpart to `from_ultralytics`, which handles every Ultralytics task that
    localises something. A `-cls` checkpoint localises nothing, so its output is a
    different kind of answer and gets a different type.

    Args:
        ultralytics_results: A single Result, or the one-element list `model.predict()`
            returns, from a classification model.
        labels: Optional label definitions overriding `result.names`. Accepts List[str],
            Dict[int, str], or List[dict] with "id"/"name" keys.
        top_k: Optional number of rows to keep, ranked best first. None keeps every
            class the model scored, in class-id order.

    Returns:
        Classifications: One row per class the model scored.

    Raises:
        ValueError: If the result carries boxes rather than class probabilities, or if
            more than one image's results are passed.

    Example:
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>>
        >>> model = YOLO("yolo11n-cls.pt")
        >>> result = pf.from_ultralytics_classification(model.predict("dog.jpg"))
        >>> result.top1.class_name
        'golden retriever'
        >>>
        >>> # Per-frame on video, where 1000 rows an image is not worth building
        >>> result = pf.from_ultralytics_classification(model.predict(frame), top_k=5)

    Notes:
        - `model.predict()` returns a list even for one image, so the one-element list
          is the normal case and is unwrapped. More than one image raises rather than
          silently reporting only the first.
        - Every class is returned by default, which is ~1000 rows for an ImageNet head.
          Pass `top_k` on per-frame paths where that cost matters.
    """
    from .classifications import Classifications

    if not ultralytics_results:
        return Classifications()

    if isinstance(ultralytics_results, list):
        if len(ultralytics_results) > 1:
            raise ValueError(
                f"{len(ultralytics_results)} images were passed, and only one can be "
                f"converted -- the rest would be silently dropped. Convert each result "
                f"separately: [pf.from_ultralytics_classification(r) for r in results]"
            )
        result = ultralytics_results[0]
    else:
        result = ultralytics_results

    probs = getattr(result, "probs", None)
    if probs is None:
        raise ValueError(
            "this Result carries no class probabilities, so it did not come from a "
            "classification model. Use pf.from_ultralytics() for detection, "
            "segmentation, pose and OBB results."
        )

    # result.names is {int: str}, one of the three label formats already understood.
    names = labels if labels is not None else getattr(result, "names", None)

    classifications = from_scores(probs.data, labels=names)
    return classifications if top_k is None else classifications.top_k(top_k)
