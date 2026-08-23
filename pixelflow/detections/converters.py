"""
Detection Converters for Machine Learning Framework Integration.

Provides standardized conversion utilities to transform detection outputs from
various machine learning frameworks (Detectron2, Ultralytics YOLO, Datamarkin API,
Transformers) into PixelFlow's unified Detections format. This module enables seamless
integration with different ML backends while maintaining consistent data structures
for downstream processing, visualization, and analysis workflows.

Framework-free deployment code -- code with no framework container to convert from --
is served by `from_arrays`, which accepts plain numpy or torch arrays directly.
"""

import cv2
import warnings
import numpy as np
from typing import (List, Dict, Any, Union, Optional)
from pixelflow.arrays import to_numpy
from pixelflow.labels import get_label_info
from pixelflow.validators import round_coord, validate_segments

__all__ = [
    "from_arrays",
    "from_datamarkin",
    "from_florence2",
    "from_detectron2",
    "from_mayaku",
    "from_ultralytics",
    "from_transformers",
    "from_sam",
    "from_supervision",
    "from_rfdetr",
    "from_falcon_perception",
    "from_efficienttam",
    "from_easyocr"
]


def _build_keypoints(kpt_data, kp_names):
    """Build KeyPoint objects from a `(K, 3)` array of `(x, y, score)` rows.

    `id` is the row's index in the model's keypoint vocabulary. `name` is taken from
    `kp_names` when the caller supplied one for that index and is None otherwise -- a
    landmark's name is metadata that has to come from somewhere, and guessing it is how
    a 21-point hand model ends up reporting `left_shoulder`.
    """
    from .detections import KeyPoint

    return [
        KeyPoint(
            x=kpt[0], y=kpt[1], id=idx,
            name=kp_names[idx] if kp_names and idx < len(kp_names) else None,
            confidence=float(kpt[2]),
        )
        for idx, kpt in enumerate(kpt_data)
    ]


def _hull_bbox(points):
    """Axis-aligned hull of an ``[[x, y], ...]`` outline, as ``[x1, y1, x2, y2]``.

    Converters that read richer geometry than a box - polygons, OCR quads - still owe
    `bbox` a value, because zones, filters and every box annotator read it. One
    definition keeps the three call sites from drifting on rounding and type.
    """
    xs, ys = zip(*points)
    return [min(xs), min(ys), max(xs), max(ys)]


def _detection_from_quad(quad, text, confidence=None):
    """Build a Detection from one text quadrilateral, or None if it is unusable.

    The quad is kept in `segments` because real-world text is rotated and `bbox`
    alone throws the orientation away; since it lives there, transforms move all
    four corners and the polygon annotator draws it with no further work.

    Note that the quad is handed to the constructor raw. `Detection.segments` is a
    validating property, so normalizing it here as well would run the numpy
    round-trip twice on a per-frame path.
    """
    from .detections import Detection

    detection = Detection(segments=quad, text=text, confidence=confidence)
    if detection.segments is None or len(detection.segments) != 4:
        return None

    detection.bbox = _hull_bbox(detection.segments)
    return detection


def _flat_coord_lists(nested):
    """Yield every flat [x1, y1, x2, y2, ...] coordinate list inside `nested`.

    Florence-2 nests segmentation polygons inconsistently across versions and
    tasks — an instance may arrive as [x1, y1, ...], [[x1, y1, ...]], or
    [[[x1, y1, ...]]]. Descending until the first numeric element makes the
    converter indifferent to the depth.
    """
    if nested is None:
        return

    # A flat run of numbers is a polygon in itself.
    if not isinstance(nested, (list, tuple)):
        return
    if all(isinstance(v, (int, float)) for v in nested):
        if nested:
            yield list(nested)
        return

    for item in nested:
        yield from _flat_coord_lists(item)


def from_datamarkin(api_response: Dict[str, Any]):
    """Convert Datamarkin API response to a Detections object.

    Args:
        api_response: Dict with 'predictions' -> 'objects' structure from the Datamarkin API.

    Returns:
        Detections: Bounding boxes, polygon masks, keypoints, class names, and confidence scores.

    Example:
        >>> import pixelflow as pf
        >>> detections = pf.from_datamarkin(api_response)
        >>> for det in detections:
        ...     print(f"{det.class_name}: {det.confidence:.2f}")

    Notes:
        - Keypoint probability is carried through as `confidence`, not thresholded.
    """
    from .detections import Detections, Detection, KeyPoint

    detections_obj = Detections()

    for obj in api_response.get("predictions", {}).get("objects", []):
        bbox = obj.get("bbox", [])
        mask = obj.get("mask", [])
        keypoints_api = obj.get("keypoints", [])
        class_name = obj.get("class", "")
        confidence = obj.get("bbox_score", None)

        # API format: {"name": "p0", "point": [x, y], "probability": 0.375}
        # The API names its own keypoints, so nothing here has to be inferred. Position in the
        # list is the id, and an absent name stays absent rather than becoming "".
        keypoints = None
        if keypoints_api:
            keypoints = []
            for idx, kp in enumerate(keypoints_api):
                point = kp.get("point", (0, 0))
                keypoints.append(KeyPoint(
                    x=point[0],
                    y=point[1],
                    id=idx,
                    name=kp.get("name") or None,
                    confidence=kp.get("probability"),
                ))

        # Create the Detection object
        detection = Detection(
            bbox=bbox,
            masks=mask,
            keypoints=keypoints,
            class_id=0,  # API doesn't provide numeric class IDs, default to 0
            class_name=class_name,
            confidence=confidence,
        )

        # Add the prediction to the list
        detections_obj.add_detection(detection)

    return detections_obj


# Florence-2 tasks whose `labels` are category names drawn from a vocabulary, so
# they belong in `class_name`. Every other region task returns a free-form string
# the model wrote or the caller supplied as a phrase, which belongs in `text`:
# <DENSE_REGION_CAPTION> describes ("a red car parked"), <OCR_WITH_REGION>
# transcribes, and <CAPTION_TO_PHRASE_GROUNDING> and
# <REFERRING_EXPRESSION_SEGMENTATION> hand back the caller's own words located in
# the image. None of those is a class, and putting them in `class_name` made that
# field mean two things.
#
# The output shapes cannot tell these apart - <OD> and <DENSE_REGION_CAPTION> both
# return {"bboxes": [...], "labels": [...]} - so this branches on task_prompt,
# which the caller passes in. Unrecognised tasks fall to `text`: Florence-2 is a
# captioning model whose region tasks emit prose by default, and a category name
# sitting in `text` is inert, where prose in `class_name` mints fake class_ids and
# leaks into the crossings class-name map.
# <OPEN_VOCABULARY_DETECTION> cannot reach the vocabulary branch yet - it returns
# `bboxes_labels`/`polygons_labels` rather than `labels`, so it raises before the
# routing runs. It is listed anyway so that whoever adds that branch gets the
# routing for free rather than having to rediscover this decision.
_FLORENCE2_VOCABULARY_TASKS = frozenset((
    '<OD>',
    '<REGION_PROPOSAL>',
    '<OPEN_VOCABULARY_DETECTION>',
))

# Tasks the processor post-processes as "pure_text": they return a string with no
# geometry at all, so there is nothing to locate. Declared beside the set above so
# that everything known about a task prompt lives in one place.
_FLORENCE2_TEXT_ONLY_TASKS = frozenset((
    '<CAPTION>',
    '<DETAILED_CAPTION>',
    '<MORE_DETAILED_CAPTION>',
    '<OCR>',  # Pure OCR without regions
    '<REGION_TO_CATEGORY>',
    '<REGION_TO_DESCRIPTION>',
    '<REGION_TO_OCR>',
))


def _florence2_label_fields(label, label_to_id, names_a_class):
    """Where this task's label belongs on the Detection.

    Both the bboxes and the polygons branch face the same question, so the answer
    lives once. See _FLORENCE2_VOCABULARY_TASKS for how the task decides.
    """
    if names_a_class:
        return {"class_id": label_to_id[label], "class_name": label}
    return {"text": label}


def from_florence2(
    parsed_result: Dict[str, Any],
    task_prompt: str,
    image_size: Union[tuple, None] = None
):
    """Convert Florence-2 model output to a Detections object.

    Args:
        parsed_result: Dict from processor.post_process_generation(). Task prompt is the top-level key.
        task_prompt: Florence-2 task string, e.g. '<OD>', '<REFERRING_EXPRESSION_SEGMENTATION>'.
        image_size: Image dimensions as (width, height). Default is None.

    Returns:
        Detections: Bounding boxes, polygon segments, quadrilaterals for OCR, and either
            class names or `text` depending on the task.

    Raises:
        ValueError: If task_prompt is not in parsed_result, is a text-only task, or the data
                   fields don't match any supported format.

    Example:
        >>> import pixelflow as pf
        >>> task = "<OD>"
        >>> parsed = processor.post_process_generation(outputs, task=task, image_size=(w, h))
        >>> detections = pf.from_florence2(parsed, task_prompt=task)
        >>> for det in detections:
        ...     print(f"{det.class_name}: {det.bbox}")
        >>>
        >>> # Captioning and OCR tasks fill `text` instead, and leave class_name None
        >>> task = "<DENSE_REGION_CAPTION>"
        >>> parsed = processor.post_process_generation(outputs, task=task, image_size=(w, h))
        >>> for det in pf.from_florence2(parsed, task_prompt=task):
        ...     print(f"{det.text}: {det.bbox}")

    Notes:
        - Supported tasks: <OD>, <REGION_PROPOSAL>, <CAPTION_TO_PHRASE_GROUNDING>,
          <DENSE_REGION_CAPTION>, <OCR_WITH_REGION>, <REFERRING_EXPRESSION_SEGMENTATION>,
          <REGION_TO_SEGMENTATION>.
        - Where the label goes depends on the task. <OD>, <REGION_PROPOSAL> and
          <OPEN_VOCABULARY_DETECTION> name a class, so their labels become `class_name` with
          sequential class_ids for stable colouring. Every other task returns a free-form
          string, which becomes `text` with `class_id` and `class_name` left None -- the model
          picked nothing out of a vocabulary, so there is no class to report.
        - <OCR_WITH_REGION> returns quadrilaterals rather than boxes. The four corners are kept
          in `segments` and the axis-aligned hull in `bbox`, so rotated text keeps its
          orientation through transforms.
        - Text-only tasks (<CAPTION>, <OCR>, <REGION_TO_DESCRIPTION>, etc.) raise ValueError.
        - <OPEN_VOCABULARY_DETECTION> is not supported: it returns `bboxes_labels` and
          `polygons_labels` rather than `labels` and needs its own branch.
        - Confidence defaults to 1.0 when not provided.
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    # Validate that task_prompt exists in parsed_result
    if task_prompt not in parsed_result:
        raise ValueError(
            f"Task prompt '{task_prompt}' not found in parsed_result. "
            f"Available keys: {list(parsed_result.keys())}"
        )

    # Reject text-only tasks
    if task_prompt in _FLORENCE2_TEXT_ONLY_TASKS:
        raise ValueError(
            f"Task '{task_prompt}' returns text only and cannot be converted to Detections. "
            f"Text-only tasks: {sorted(_FLORENCE2_TEXT_ONLY_TASKS)}"
        )

    # Extract the task-specific data
    task_data = parsed_result[task_prompt]

    # Whether this task's labels name a class or are free-form strings. See
    # _FLORENCE2_VOCABULARY_TASKS for why this branches on the prompt and not the shape.
    names_a_class = task_prompt in _FLORENCE2_VOCABULARY_TASKS

    # Sequential class IDs give stable colours per class. Only meaningful when the
    # labels are a vocabulary and repeat; for captions every label is unique, so an
    # id would just be the row number dressed up as a class.
    label_to_id = {}
    if names_a_class:
        for label in task_data.get('labels') or []:
            label_to_id.setdefault(label, len(label_to_id))

    # Handle different task types based on available data fields
    if 'bboxes' in task_data and 'labels' in task_data:
        # Detection tasks: <OD>, <REGION_PROPOSAL>, <CAPTION_TO_PHRASE_GROUNDING>,
        # <DENSE_REGION_CAPTION> - all the same shape, told apart by task_prompt.
        bboxes = task_data['bboxes']
        labels = task_data['labels']
        scores = task_data.get('scores', None)  # Optional confidence scores

        for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
            confidence = float(scores[idx]) if scores is not None else 1.0

            # Florence-2 bboxes are already in XYXY format [x1, y1, x2, y2]
            detection = Detection(
                bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                confidence=confidence,
                **_florence2_label_fields(label, label_to_id, names_a_class)
            )
            detections_obj.add_detection(detection)

    elif 'quad_boxes' in task_data and 'labels' in task_data:
        # <OCR_WITH_REGION>: the processor builds labels from inst["text"], and each
        # quad_box is a flat [x1, y1, x2, y2, x3, y3, x4, y4] run of corners. Same
        # treatment as from_easyocr - the quad is kept because rotated text is the
        # normal case and bbox alone throws the orientation away.
        for quad, label in zip(task_data['quad_boxes'], task_data['labels']):
            # Pair the flat run into [x, y]; _detection_from_quad rounds and validates.
            points = [quad[i:i + 2] for i in range(0, len(quad) - 1, 2)]
            detection = _detection_from_quad(points, label, confidence=1.0)
            if detection is None:
                warnings.warn(
                    f"Skipping Florence-2 OCR region with unusable geometry: {quad!r}"
                )
                continue
            detections_obj.add_detection(detection)

    elif 'polygons' in task_data and 'labels' in task_data:
        # Segmentation tasks: <REFERRING_EXPRESSION_SEGMENTATION>
        # polygons[i] holds every polygon belonging to instance i, each a flat
        # [x1,y1,x2,y2,...] list. Nesting depth varies between Florence-2
        # versions, so _flat_coord_lists normalises it.
        polygons = task_data['polygons']
        labels = task_data['labels']

        for poly_nested, label in zip(polygons, labels):
            # Every polygon for this instance is kept — an instance split by
            # occlusion has more than one, and dropping the rest loses geometry.
            all_polygons = []
            for coords in _flat_coord_lists(poly_nested):
                points = [
                    (round_coord(coords[i]), round_coord(coords[i + 1]))
                    for i in range(0, len(coords) - 1, 2)
                ]
                if points:
                    all_polygons.append(points)

            if not all_polygons:
                continue

            detection = Detection(
                # The hull spans every polygon of the instance, not just the first.
                bbox=_hull_bbox([p for poly in all_polygons for p in poly]),
                segments=all_polygons[0],  # segments holds a single polygon
                masks=all_polygons,        # masks keeps every part
                confidence=1.0,
                **_florence2_label_fields(label, label_to_id, names_a_class)
            )
            detections_obj.add_detection(detection)

    else:
        # Unknown task format
        raise ValueError(
            f"Unsupported Florence-2 task format for '{task_prompt}'. "
            f"Available data fields: {list(task_data.keys())}. "
            f"Expected 'bboxes+labels', 'quad_boxes+labels' or 'polygons+labels'."
        )

    return detections_obj


def from_arrays(boxes, scores, class_ids=None, masks=None, keypoints=None, labels=None,
                texts=None, segments=None):
    """Convert plain detection arrays to a Detections object.

    The framework-free entry point. Every other converter in this module is named
    after a framework's output container — `sv.Detections`, D2's `Instances`, an
    Ultralytics `Results`. Deployment code that has been extracted from a research
    repository has no such container: it returns arrays, because stripping the
    framework wrapper is what extraction is for. This converter takes those arrays
    directly, so a vendored model does not need a converter of its own.

    Torch tensors and numpy arrays are both accepted; tensors are detached and moved
    to CPU automatically.

    Args:
        boxes: `(N, 4)` bounding boxes in absolute XYXY pixel coordinates.
        scores: `(N,)` confidence scores.
        class_ids: Optional `(N,)` integer class IDs. None for a model that locates
            things without naming them -- OCR reads content, SAM segments whatever it
            was pointed at, and neither picks a class out of a vocabulary.
        masks: Optional `(N, H, W)` masks, cast to boolean. Instance segmentation only.
        keypoints: Optional `(N, K, 3)` keypoints with `(x, y, score)` columns.
        labels: Optional label definitions for class and keypoint name resolution.
            Accepts the same three formats as `from_detectron2`:
            - List[str]: ["person", "car"] — index = class_id
            - Dict[int, str]: {0: "person", 1: "car"} — key = class_id
            - List[dict]: [{"id": 0, "name": "person", "keypoints": [...]}]
        texts: Optional `(N,)` free-form strings, one per detection -- what an OCR engine
            read inside the box, or a region caption. An empty string is kept as an empty
            string; None leaves the field unset.
        segments: Optional `(N,)` polygons, one per detection, each `(P, 2)` points in
            absolute pixel coordinates. Pass the shape the model actually produced -- a
            text quadrilateral, an oriented box -- where `bbox` alone would throw the
            orientation away.

    Returns:
        Detections: Bounding boxes, polygons, text, masks, keypoints, class IDs and
        confidence scores in PixelFlow's unified format.

    Raises:
        ValueError: If the arrays do not all describe the same number of detections.

    Example:
        >>> import pixelflow as pf
        >>> boxes = [[10, 20, 110, 220], [30, 40, 130, 240]]
        >>> detections = pf.from_arrays(
        ...     boxes, scores=[0.9, 0.8], class_ids=[0, 2], labels=["person", "bike", "car"]
        ... )
        >>> len(detections)
        2
        >>> detections[0].class_name
        'person'

    Notes:
        - Boxes must already be in absolute pixel coordinates. Normalised boxes, or
          the CXCYWH layout many DETR-style models emit internally, must be converted
          by the caller — this function cannot tell the layouts apart.
        - No score threshold is applied. Chain `filter_by_confidence()` to cut the tail.
        - Pass the model's own class names as `labels`. A checkpoint fine-tuned on your
          data has its own vocabulary, and another model's names would mislabel every
          detection rather than fail.
        - `texts` is not a substitute for `labels`. `class_name` is which class out of a
          fixed vocabulary the model was trained on; `text` is content it produced that
          belongs to no vocabulary. A model can supply either, both, or neither.
        - `boxes` is required even when `segments` is given: the axis-aligned hull is what
          zones, filters and most annotators read, and only the caller knows whether the
          polygon or the box is the authoritative geometry.
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    boxes = to_numpy(boxes)
    scores = to_numpy(scores)
    class_ids = to_numpy(class_ids)
    masks = to_numpy(masks)
    keypoints = to_numpy(keypoints)
    # texts and segments stay in whatever sequence they arrived in. numpy would turn
    # strings into `numpy.str_` and would collapse polygons of differing vertex counts
    # into an object array, so neither gains anything from the round-trip.

    count = len(boxes)
    for name, array in (("scores", scores), ("class_ids", class_ids),
                        ("masks", masks), ("keypoints", keypoints),
                        ("texts", texts), ("segments", segments)):
        if array is not None and len(array) != count:
            raise ValueError(
                f"{name} describes {len(array)} detections but boxes describes {count}"
            )

    for i in range(count):
        # A model that locates without naming has no class_id, and `get_label_info`
        # already answers (None, None) for one -- so an unnamed detection costs no branch
        # here beyond not casting None to int.
        class_id = int(class_ids[i]) if class_ids is not None else None
        class_name, kp_names = get_label_info(labels, class_id)

        kpts = _build_keypoints(keypoints[i], kp_names) if keypoints is not None else None

        # Handed over raw: `Detection.segments` is a validating property, and normalizing
        # here as well would run the numpy round-trip twice on a per-frame path.
        detections_obj.add_detection(Detection(
            bbox=boxes[i].tolist(),
            masks=[masks[i].astype(bool)] if masks is not None else None,
            segments=segments[i] if segments is not None else None,
            keypoints=kpts,
            class_id=class_id,
            class_name=class_name,
            confidence=float(scores[i]),
            # str() rather than the value itself, so a caller who did pass a numpy array
            # of strings gets `str` out. An empty read is content; only None is absence.
            text=None if texts is None or texts[i] is None else str(texts[i]),
        ))

    return detections_obj


def from_detectron2(detectron2_results: Dict[str, Any], labels=None):
    """Convert Detectron2 inference results to a Detections object.

    Args:
        detectron2_results: Output dict from DefaultPredictor with an 'instances' key.
        labels: Optional label definitions for class name and keypoint name resolution.
            Accepts three formats:
            - List[str]: ["person", "car"] — index = class_id
            - Dict[int, str]: {0: "person", 1: "car"} — key = class_id
            - List[dict]: [{"id": 0, "name": "person", "keypoints": [...]}] — rich format

    Returns:
        Detections: Bounding boxes, boolean masks, keypoints, class IDs, and confidence scores.

    Example:
        >>> import pixelflow as pf
        >>> outputs = predictor(image)
        >>> detections = pf.from_detectron2(outputs, labels=predictor.class_names)
        >>> for det in detections:
        ...     print(f"{det.class_name}: {det.confidence:.2f}")

    Notes:
        - All tensors are moved to CPU automatically before conversion.
        - Segmentation masks are converted to boolean arrays.
        - Keypoints are converted to KeyPoint objects with names from labels.
    """
    from .detections import Detections, Detection
    
    detections_obj = Detections()
    
    # Get instances and ensure they're on CPU for processing
    instances = detectron2_results["instances"].to("cpu")
    
    # Check if we have any instances
    if len(instances) == 0:
        return detections_obj

    # Extract prediction data
    # Bounding boxes - Detectron2 uses XYXY format
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None
    
    # Confidence scores
    scores = instances.scores.numpy() if instances.has("scores") else None
    
    # Class IDs  
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None
    
    # Segmentation masks
    masks = None
    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()
    
    # Keypoints if available
    keypoints = None
    if instances.has("pred_keypoints"):
        keypoints = instances.pred_keypoints.numpy()

    # Iterate over each detection
    for i in range(len(instances)):
        # Extract bounding box in XYXY format
        bbox = boxes[i].tolist() if boxes is not None else None
        
        # Extract confidence score
        confidence = float(scores[i]) if scores is not None else None
        
        # Extract class ID
        class_id = int(classes[i]) if classes is not None else None

        # Extract class name and keypoint names from labels
        class_name, kp_names = get_label_info(labels, class_id)

        # Handle segmentation masks
        mask = None
        if masks is not None:
            mask_data = masks[i].astype(bool)
            mask = mask_data

        # Handle keypoints if available
        # keypoints[i] has shape (K, 3): x, y, confidence
        kpts = _build_keypoints(keypoints[i], kp_names) if keypoints is not None else None
        
        # Create a Detection object
        detection = Detection(
            bbox=bbox,
            masks=[mask] if mask is not None else None,
            segments=None,
            keypoints=kpts,
            class_id=class_id,
            class_name=class_name,
            confidence=confidence
        )

        # Add the detection to the Detections object
        detections_obj.add_detection(detection)

    return detections_obj


def from_mayaku(mayaku_instances, labels=None):
    """Convert Mayaku inference results to a Detections object.

    Mayaku shares Detectron2's runtime layout: the same `Instances`
    container, the same field names (`pred_boxes`, `scores`,
    `pred_classes`, `pred_masks`, `pred_keypoints`), xyxy absolute boxes,
    `(N, H, W)` boolean masks after postprocess, and `(N, K, 3)`
    keypoints with `(x, y, score)` columns. The divergence that matters
    here: Mayaku's predictor returns the `Instances` directly, whereas
    Detectron2's `DefaultPredictor` wraps it in a `{"instances": ...}`
    dict.

    Args:
        mayaku_instances: The `Instances` returned by calling a Mayaku
            predictor (NOT a dict). If you have a wrapper that mimics
            D2's dict shape, pass `wrapper["instances"]`.
        labels: Optional label definitions for class name and keypoint
            name resolution. Pass `predictor.class_names` — see the note
            below. Accepts the same three formats as `from_detectron2`:
            - List[str]: ["person", "car"] — index = class_id
            - Dict[int, str]: {0: "person", 1: "car"} — key = class_id
            - List[dict]: [{"id": 0, "name": "person", "keypoints": [...]}]

    Returns:
        Detections: Bounding boxes, boolean masks, keypoints, class IDs,
        and confidence scores in PixelFlow's unified format.

    Example:
        >>> import pixelflow as pf
        >>> from mayaku import from_pretrained
        >>>
        >>> predictor = from_pretrained("mayaku-n-det")
        >>> instances = predictor("photo.jpg")
        >>>
        >>> # Take the vocabulary from the checkpoint, not a hardcoded list.
        >>> detections = pf.from_mayaku(
        ...     instances, labels=predictor.class_names
        ... )
        >>> for det in detections:
        ...     print(f"{det.class_name}: {det.confidence:.2f}")

    Notes:
        - Use `predictor.class_names`. Mayaku checkpoints carry their own
          vocabulary in the sidecar, and the pretrained models are trained on
          Objects365 (365 classes). Another dataset's names mislabel every
          detection — class 5 is "Car" in Objects365 but "bus" in COCO.
        - A predictor accepts a path directly and decodes it as RGB.
          When passing an array instead, it must be RGB: `cv2.imread`
          gives BGR, so convert with `mayaku.utils.bgr_to_rgb` first.
        - Predictors return a fixed-size candidate set (typically 100
          rows) with no score threshold applied. Chain
          `filter_by_confidence()` to cut the low-scoring tail.
        - All tensors are moved to CPU automatically before conversion.
        - Mayaku auto-runs `detector_postprocess`, so coordinates and
          masks are already in original image space.
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    # Mayaku's Predictor returns Instances directly — no dict unwrap.
    instances = mayaku_instances.to("cpu")

    if len(instances) == 0:
        return detections_obj

    # Bounding boxes — XYXY absolute pixel coords (same as D2).
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else None

    # Confidence scores — float ∈ [0, 1].
    scores = instances.scores.numpy() if instances.has("scores") else None

    # Class IDs — int64, 0-indexed.
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else None

    # Segmentation masks — (N, H, W) bool after postprocess.
    masks = None
    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()

    # Keypoints — (N, K, 3) with (x, y, score) columns.
    keypoints = None
    if instances.has("pred_keypoints"):
        keypoints = instances.pred_keypoints.numpy()

    for i in range(len(instances)):
        bbox = boxes[i].tolist() if boxes is not None else None
        confidence = float(scores[i]) if scores is not None else None
        class_id = int(classes[i]) if classes is not None else None

        class_name, kp_names = get_label_info(labels, class_id)

        mask = None
        if masks is not None:
            mask = masks[i].astype(bool)

        # keypoints[i] has shape (K, 3): x, y, confidence
        kpts = _build_keypoints(keypoints[i], kp_names) if keypoints is not None else None

        detection = Detection(
            bbox=bbox,
            masks=[mask] if mask is not None else None,
            segments=None,
            keypoints=kpts,
            class_id=class_id,
            class_name=class_name,
            confidence=confidence
        )

        detections_obj.add_detection(detection)

    return detections_obj


def from_ultralytics(ultralytics_results: Union[Any, List[Any]], labels=None):
    """Convert Ultralytics YOLO results to a Detections object.

    Args:
        ultralytics_results: Single Result or list[Result] from model.predict() or model.track().
        labels: Optional label definitions to override model class/keypoint names.
            Accepts three formats:
            - List[str]: ["person", "car"] — index = class_id
            - Dict[int, str]: {0: "person", 1: "car"} — key = class_id
            - List[dict]: [{"id": 0, "name": "person", "keypoints": [...]}] — rich format
            When provided, overrides result.names for class names and COCO keypoint names.

    Returns:
        Detections: Bounding boxes, binary masks, polygon segments, and tracker IDs when available.

    Example:
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> outputs = model.predict(image)
        >>> detections = pf.from_ultralytics(outputs)
        >>> for det in detections:
        ...     print(f"{det.class_name}: {det.confidence:.2f}, bbox={det.bbox}")

    Notes:
        - tracker_id is set when using model.track() with persist=True.
        - Classification results raise; they belong to pf.from_ultralytics_classification().
        - Binary masks are resized to original image dimensions with letterbox padding removed.
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    # Handle empty results
    if not ultralytics_results:
        return detections_obj

    # Handle both single result and list of results
    if isinstance(ultralytics_results, list):
        # Get the first result (YOLO returns a list with one result per image)
        result = ultralytics_results[0]
    else:
        # Already a single result object
        result = ultralytics_results

    # Handle classification models (no boxes, only probs)
    if hasattr(result, 'probs') and result.probs is not None:
        probs = result.probs

        # Get top-1 prediction
        top1_idx = int(probs.top1)
        top1_conf = float(probs.top1conf)

        # Get class name
        class_name = None
        if hasattr(result, 'names') and result.names:
            class_name = result.names.get(top1_idx, str(top1_idx))

        # Get top-5 predictions for metadata
        top5_indices = [int(idx) for idx in probs.top5]
        top5_confs = probs.top5conf.cpu().numpy().tolist()
        top5_names = []
        if hasattr(result, 'names') and result.names:
            top5_names = [result.names.get(idx, str(idx)) for idx in top5_indices]

        # Create single detection for classification result
        detection = Detection(
            bbox=None,  # Classification has no bbox
            class_id=top1_idx,
            class_name=class_name,
            confidence=top1_conf,
            metadata={
                'task': 'classification',
                'top5_indices': top5_indices,
                'top5_confidences': top5_confs,
                'top5_names': top5_names
            }
        )
        detections_obj.add_detection(detection)
        return detections_obj

    # Handle case where there are no detections
    if result.boxes is None or len(result.boxes) == 0:
        return detections_obj
    
    # Get all box data in one tensor transfer (more efficient)
    boxes_data = result.boxes.data.cpu().numpy()
    
    # Extract components from the tensor
    # Format: [x1, y1, x2, y2, conf, class_id, ...] or [x1, y1, x2, y2, conf, class_id, track_id]
    xyxy = boxes_data[:, :4]  # Bounding boxes
    confidences = boxes_data[:, 4]  # Confidence scores  
    class_ids = boxes_data[:, 5].astype(int)  # Class IDs
    
    # Check if tracker IDs are available (when using model.track())
    tracker_ids = None
    if hasattr(result.boxes, 'id') and result.boxes.id is not None:
        tracker_ids = result.boxes.id.cpu().numpy().astype(int)
    
    # Check if we have segmentation masks
    has_masks = hasattr(result, 'masks') and result.masks is not None
    
    # Process each detection
    num_detections = len(xyxy)
    
    # Get binary masks if available (shape: [num_masks, height, width])
    binary_masks = None
    orig_shape = None
    if has_masks and hasattr(result.masks, 'data'):
        # Convert masks to numpy arrays
        binary_masks = result.masks.data.cpu().numpy()
        # Get original image shape from masks or result
        if hasattr(result.masks, 'orig_shape'):
            orig_shape = result.masks.orig_shape  # (height, width)
        elif hasattr(result, 'orig_shape'):
            orig_shape = result.orig_shape  # (height, width)

    # Check if we have keypoints (pose estimation)
    has_keypoints = hasattr(result, 'keypoints') and result.keypoints is not None
    keypoints_data = None
    if has_keypoints:
        keypoints_data = result.keypoints.data.cpu().numpy()  # Shape: [N, 17, 3]

    for i in range(num_detections):
        # Basic detection info
        bbox = xyxy[i].tolist()
        confidence = float(confidences[i])
        class_id = int(class_ids[i])
        
        # Extract class name — labels overrides result.names when provided
        class_name = None
        if labels is not None:
            class_name, _ = get_label_info(labels, class_id)
        elif hasattr(result, 'names') and result.names:
            if class_id in result.names:
                class_name = result.names[class_id]
        
        # Get tracker ID if available
        tracker_id = None
        if tracker_ids is not None:
            tracker_id = int(tracker_ids[i])
        
        # Handle masks if available
        masks = None
        segments = None
        if has_masks:
            # Store polygon format (xy) for segments
            # Detection would normalize this anyway; doing it here too means the
            # mask fallback below also gets plain rounded lists.
            segments = validate_segments(result.masks.xy[i])
            
            # Store binary mask if available
            if binary_masks is not None:
                # Get the binary mask for this detection
                mask = binary_masks[i]
                
                # Handle letterbox padding and resize mask to original shape
                if orig_shape is not None and mask.shape[:2] != orig_shape:
                    # YOLO uses letterboxing: it pads to square then resizes
                    # We need to remove padding and resize to original dimensions
                    mask_h, mask_w = mask.shape[:2]  # Should be 640x640
                    orig_h, orig_w = orig_shape  # Original image dimensions
                    
                    # Calculate the scale and padding used by YOLO
                    scale = min(mask_h / orig_h, mask_w / orig_w)
                    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
                    
                    # Calculate padding
                    pad_h = (mask_h - new_h) // 2
                    pad_w = (mask_w - new_w) // 2
                    
                    # Remove padding
                    mask = mask[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
                    
                    # Resize to original dimensions
                    mask = cv2.resize(mask.astype(np.uint8), 
                                    (orig_w, orig_h),  # cv2 uses (width, height)
                                    interpolation=cv2.INTER_NEAREST)
                
                mask = mask.astype(bool)
                masks = [mask]  # Wrap in list for consistency with API
            elif segments is not None:
                # Fallback to polygon format if binary not available
                masks = [segments]

        # Handle keypoints if available (pose estimation)
        keypoints_list = None
        if has_keypoints and keypoints_data is not None:
            # keypoints_data[i] has shape (K, 3): x, y, confidence
            _, kpt_names = get_label_info(labels, class_id)
            keypoints_list = _build_keypoints(keypoints_data[i], kpt_names)

        # Create detection object
        detection = Detection(
            bbox=bbox,
            masks=masks,  # Can be either binary mask or polygon coordinates
            segments=segments,  # Always polygon coordinates
            keypoints=keypoints_list,
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            tracker_id=tracker_id
        )
        
        detections_obj.add_detection(detection)
    
    # Store the original YOLO masks data for later use if needed
    # This avoids processing masks until they're actually used
    if has_masks:
        detections_obj._ultralytics_masks = result.masks
    
    return detections_obj


def from_transformers(transformers_results: Any):
    """Convert Hugging Face Transformers object detection results to a Detections object.

    Not yet implemented.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError("from_transformers converter not yet implemented")


def from_efficienttam(masks, scores):
    """Convert EfficientTAM/SAM image predictor output to a Detections object.

    Args:
        masks: Binary masks, shape (C, H, W), dtype bool or uint8.
               Can be numpy ndarray or torch Tensor.
        scores: Quality scores, shape (C,), values in [0, 1].
               Can be numpy ndarray or torch Tensor.

    Returns:
        Detections: One Detection per candidate mask with confidence and bbox
                    derived from mask extent. class_id and class_name are None.
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    masks = to_numpy(masks)
    scores = to_numpy(scores)

    if masks.shape[0] == 0:
        return detections_obj

    for i in range(masks.shape[0]):
        mask = masks[i].astype(bool)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any():
            continue

        y1, y2 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        x1, x2 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

        detections_obj.add_detection(Detection(
            bbox=[x1, y1, x2, y2],
            masks=[mask],
            confidence=float(scores[i]),
        ))

    return detections_obj


def from_sam(masks, scores):
    """Convert SAM image predictor output to a Detections object.

    Alias for from_efficienttam(). SAM returns identical (masks, scores, logits) format.

    Args:
        masks: Binary masks, shape (C, H, W). Can be numpy ndarray or torch Tensor.
        scores: Quality scores, shape (C,). Can be numpy ndarray or torch Tensor.

    Returns:
        Detections: One Detection per candidate mask.
    """
    return from_efficienttam(masks, scores)


def from_supervision(
    supervision_detections: Any,
    labels=None,
) -> "Detections":
    """Convert supervision library sv.Detections to a PixelFlow Detections object.

    Args:
        supervision_detections: sv.Detections with xyxy, confidence, class_id, mask attributes.
        labels: Optional label definitions for class name resolution.
            Accepts three formats:
            - List[str]: ["person", "car"] — index = class_id
            - Dict[int, str]: {0: "person", 1: "car"} — key = class_id
            - List[dict]: [{"id": 0, "name": "person"}] — rich format

    Returns:
        Detections: Bounding boxes, optional masks, class IDs, and confidence scores.

    Raises:
        AttributeError: If supervision_detections lacks the required xyxy attribute.

    Example:
        >>> import pixelflow as pf
        >>> detections = pf.from_supervision(sv_detections, labels=model.class_names)
        >>> for det in detections:
        ...     print(f"{det.class_name}: {det.confidence:.2f}")

    Notes:
        - labels accepts List[str], Dict[int, str], or List[dict] (rich Datamarkin format).
        - confidence, class_id, and mask are optional; missing fields are stored as None.
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    # Check if input has required xyxy attribute
    if not hasattr(supervision_detections, 'xyxy'):
        raise AttributeError(
            "Input must be supervision library's sv.Detections with 'xyxy' attribute"
        )

    xyxy = supervision_detections.xyxy  # shape (n, 4) numpy array

    # Handle empty detections
    if len(xyxy) == 0:
        return detections_obj

    # Extract optional fields from supervision detections
    confidence = getattr(supervision_detections, 'confidence', None)  # Optional, shape (n,) or None
    class_id = getattr(supervision_detections, 'class_id', None)  # Optional, shape (n,) or None
    mask = getattr(supervision_detections, 'mask', None)  # Optional, list of masks or None

    for i in range(len(xyxy)):
        cid = int(class_id[i]) if class_id is not None else None

        # Resolve class_name from labels
        class_name, _ = get_label_info(labels, cid)
        if class_name is None and cid is not None:
            class_name = str(cid)

        detection = Detection(
            bbox=xyxy[i].tolist(),  # Convert [x1,y1,x2,y2] numpy array to list
            confidence=float(confidence[i]) if confidence is not None else None,
            class_id=cid,
            class_name=class_name,
        )

        # Handle masks if present - store as list of mask arrays
        if mask is not None and i < len(mask) and mask[i] is not None:
            detection.masks = [mask[i]]  # Store single mask as list element

        detections_obj.add_detection(detection)

    return detections_obj


def from_rfdetr(
    supervision_detections: Any,
    labels=None,
) -> "Detections":
    """
    Convert RF-DETR output to a unified PixelFlow Detections object.

    Alias for from_supervision(). RF-DETR models return supervision library's
    sv.Detections format, so this function provides a more discoverable name
    for users working with RF-DETR who may not know the underlying format.

    Args:
        supervision_detections: RF-DETR output (sv.Detections from supervision).
                               Expected to have attributes: xyxy (n,4),
                               confidence (n,), class_id (n,), mask (optional).
        labels: Optional label definitions for class name resolution.
            Accepts three formats:
            - List[str]: ["person", "car"] — index = class_id
            - Dict[int, str]: {0: "person", 1: "car"} — key = class_id
            - List[dict]: [{"id": 0, "name": "person"}] — rich format

    Returns:
        Detections: PixelFlow Detections container with converted Detection objects.

    Example:
        >>> import pixelflow as pf
        >>> from rfdetr import RFDETRMedium
        >>> from PIL import Image
        >>>
        >>> # Load and run RF-DETR model
        >>> model = RFDETRMedium()
        >>> image = Image.open("image.jpg")
        >>> rfdetr_output = model.predict(image, threshold=0.5)
        >>>
        >>> # Convert with COCO labels for meaningful names
        >>> pf_detections = pf.from_rfdetr(rfdetr_output, labels=model.class_names)
        >>> print(f"Detected {len(pf_detections)} objects")

    Notes:
        - This is an alias for from_supervision()
        - RF-DETR returns supervision library's sv.Detections format
        - Use this function name if you're working with RF-DETR specifically
    """
    # Delegate to from_supervision for actual conversion
    return from_supervision(supervision_detections, labels=labels)


def from_easyocr(easyocr_results: List[Any]):
    """Convert EasyOCR readtext() output to a Detections object.

    Args:
        easyocr_results: The list returned by ``reader.readtext(...)``, called with
            ``detail=1`` (the default) and ``output_format`` of 'standard', 'free_merge'
            or 'dict'.

    Returns:
        Detections: One detection per text region -- the quadrilateral as EasyOCR read it in
            `segments`, its axis-aligned hull in `bbox`, the recognized string in `text`.

    Raises:
        ValueError: If the results are plain strings, which carry no geometry to convert.

    Example:
        >>> import easyocr
        >>> import pixelflow as pf
        >>>
        >>> reader = easyocr.Reader(['en'])
        >>> detections = pf.from_easyocr(reader.readtext("sign.jpg"))
        >>> for det in detections:
        ...     print(det.text, det.bbox)

    Notes:
        - `class_id` and `class_name` stay None. OCR reads content, it does not pick a class
          out of a vocabulary, and `text` is where the content belongs.
        - The quad is kept because real-world text is rotated. For a horizontal line EasyOCR
          emits the axis-aligned rectangle as four corners, so the quad adds nothing over
          `bbox`; for a line off the horizontal it is the only record of the orientation.
          Transforms move all four corners, so it survives rotation and flipping.
        - `paragraph=True` merges lines and returns no confidence, so `confidence` is None
          there. Note that confidence filters drop None-confidence detections.
        - `detail=0` and ``output_format='json'`` both return strings rather than located
          text and raise.
        - A region EasyOCR located but read as an empty string is kept: the box was still
          detected, and whether an empty read is worth keeping is the caller's call.
        - Results with unusable geometry are skipped with a warning rather than aborting
          the whole conversion.
    """
    from .detections import Detections

    detections_obj = Detections()

    for item in easyocr_results:
        if isinstance(item, str):
            raise ValueError(
                "EasyOCR returned strings, not located text: readtext() was called with "
                "detail=0 or output_format='json'. pixelflow needs the boxes -- use "
                "detail=1 with output_format='standard' or 'dict'."
            )

        if isinstance(item, dict):
            quad, text, conf = item.get("boxes"), item.get("text"), item.get("confident")
        else:
            # 'standard' and 'free_merge' give (quad, text, confidence) tuples. paragraph=True
            # merges lines and drops the confidence, leaving a 2-element item, and the Arabic
            # path rebuilds every item as a list -- so index rather than unpack.
            quad, text = item[0], item[1]
            conf = item[2] if len(item) > 2 else None

        # The quad arrives in whichever shape EasyOCR used - numpy ints for rotated
        # boxes, Python ints for horizontal ones - and is normalized on the way in.
        detection = _detection_from_quad(quad, text, confidence=conf)
        if detection is None:
            # One degenerate quad should not cost the other 499 regions, so skip it.
            # But every quad failing means this is not EasyOCR output at all, and a
            # silent empty result is a bad way to find that out - see below.
            warnings.warn(
                f"Skipping EasyOCR result with unusable geometry: {quad!r}. "
                "Expected four corner points."
            )
            continue

        detections_obj.add_detection(detection)

    if easyocr_results and len(detections_obj) == 0:
        # Cannot tell "every region was degenerate" from "this is not EasyOCR
        # output" - both look identical from here - so the message names both.
        raise ValueError(
            f"None of the {len(easyocr_results)} results had usable geometry. Either "
            "every region was degenerate, or this is not EasyOCR output -- expected "
            "[(quad, text, confidence), ...] from reader.readtext(...)."
        )

    return detections_obj


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a boolean mask to (height, width) using nearest-neighbor interpolation."""
    if mask.shape[:2] == (height, width):
        return mask
    return cv2.resize(
        mask.astype(np.uint8), (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """Decode a COCO RLE dict to a boolean H×W numpy array.

    Args:
        rle (Dict[str, Any]): COCO RLE dictionary with 'counts' and 'size' keys.

    Returns:
        np.ndarray: Boolean mask array of shape (H, W).

    Raises:
        ImportError: If pycocotools is not installed.
    """
    try:
        from pycocotools import mask as coco_mask
    except ImportError:
        raise ImportError(
            "pycocotools is required to decode COCO RLE masks from falcon-perception. "
            "Install it with: pip install pycocotools"
        )
    decoded = coco_mask.decode(rle)  # uint8, shape (H, W), Fortran-order
    return decoded.astype(bool)


def from_falcon_perception(
    output: Any,
    image_size: Optional[tuple] = None,
    label: Optional[str] = None,
) -> "Detections":
    """Convert Falcon Perception model output to a Detections object.

    Args:
        output: AuxOutput object with 'bboxes_raw' attribute (local engine), or dict
                with 'masks' key (falcon-perception FastAPI server output).
        image_size: Image dimensions as (width, height). Required for local AuxOutput
                   (bboxes are normalized). For dict format, used to resize masks
                   if provided. Default is None.
        label: Class label for all detections when using local AuxOutput format.
               Pass the query string you gave the model, e.g. "cat". Default is None.

    Returns:
        Detections: Bounding boxes, optional boolean masks, class names, and metadata.

    Raises:
        ValueError: If output format is unrecognized.
        ValueError: If output is AuxOutput and image_size is not provided.
        ImportError: If COCO RLE masks are present but pycocotools is not installed.

    Warns:
        UserWarning: If bboxes_raw has odd length; last incomplete entry is dropped.

    Example:
        >>> import pixelflow as pf
        >>> from PIL import Image
        >>> from falcon_perception import load_and_prepare_model, build_prompt_for_task
        >>> from falcon_perception.paged_inference import PagedInferenceEngine
        >>>
        >>> model, tokenizer, args = load_and_prepare_model("perception", backend="torch")
        >>> engine = PagedInferenceEngine(model, tokenizer, args)
        >>> image = Image.open("photo.jpg")
        >>> prompt = build_prompt_for_task(query="cat", task="detection")
        >>> output = engine.generate(image=image, prompt=prompt, max_tokens=1024)
        >>> detections = pf.from_falcon_perception(
        ...     output, image_size=(image.width, image.height), label="cat"
        ... )
        >>> for det in detections:
        ...     print(f"{det.class_name}: {det.bbox}")

    Notes:
        - Format auto-detected: AuxOutput has .bboxes_raw; dict format has 'masks' key.
        - AuxOutput bboxes are normalized cxcywh, converted to pixel xyxy using image_size.
        - Dict format bboxes are already pixel xyxy; image_size is ignored.
        - Confidence scores are not provided by either format; stored as None.
        - Class IDs for dict format assigned sequentially by label first-appearance.
        - COCO RLE masks decoded to boolean H×W arrays (same format as Detectron2/YOLO).
          Requires pycocotools: pip install pycocotools. Skipped if rle is None.
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    # Format detection
    is_api_format = isinstance(output, dict) and "masks" in output
    is_raw_format = not is_api_format and hasattr(output, "bboxes_raw")

    if not is_api_format and not is_raw_format:
        raise ValueError(
            "Unrecognized falcon-perception output format. "
            "Expected either a dict with a 'masks' key (API server response) "
            "or an object with a 'bboxes_raw' attribute (raw AuxOutput)."
        )

    # API dict format
    if is_api_format:
        mask_entries = output.get("masks", [])

        if not mask_entries:
            return detections_obj

        # Build label-to-sequential-int mapping for consistent class_id assignment
        # (matches from_florence2 pattern for open-vocabulary models)
        label_to_id: Dict[str, int] = {}
        for entry in mask_entries:
            lbl = entry.get("label", "")
            if lbl not in label_to_id:
                label_to_id[lbl] = len(label_to_id)

        response_model = output.get("model", "falcon-perception")
        response_query = output.get("query", None)
        response_id = output.get("id", None)
        if image_size is not None:
            api_w, api_h = image_size

        for entry in mask_entries:
            lbl = entry.get("label", "")
            class_id = label_to_id.get(lbl, 0)

            raw_bbox = entry.get("bbox", None)
            bbox = list(raw_bbox) if raw_bbox is not None else None

            masks = None
            rle = entry.get("rle", None)
            if rle is not None:
                mask = _decode_rle(rle)
                if image_size is not None:
                    mask = _resize_mask(mask, api_w, api_h)
                masks = [mask]

            detection = Detection(
                bbox=bbox,
                masks=masks,
                class_id=class_id,
                class_name=lbl if lbl else None,
                confidence=None,
                metadata={
                    "source": "falcon-perception-api",
                    "model": response_model,
                    "query": response_query,
                    "inference_id": response_id,
                    "color": entry.get("color", None),
                },
            )
            detections_obj.add_detection(detection)

        return detections_obj

    # Raw AuxOutput format
    if image_size is None:
        raise ValueError(
            "image_size=(width, height) is required when output is a raw AuxOutput "
            "object, because bounding boxes are in normalized coordinates."
        )

    bboxes_raw = getattr(output, "bboxes_raw", None)
    masks_rle = getattr(output, "masks_rle", None)
    output_text = getattr(output, "text", None)

    if not bboxes_raw:
        return detections_obj

    if len(bboxes_raw) % 2 != 0:
        warnings.warn(
            f"bboxes_raw has odd length ({len(bboxes_raw)}); "
            "dropping last incomplete entry (likely truncated generation)."
        )
        bboxes_raw = bboxes_raw[:-1]

    img_w, img_h = image_size
    num_detections = len(bboxes_raw) // 2
    class_id = 0 if label is not None else None

    for i in range(num_detections):
        center = bboxes_raw[2 * i]      # {"x": cx, "y": cy}  normalized
        size   = bboxes_raw[2 * i + 1]  # {"h": h,  "w": w}   normalized

        cx = center.get("x", 0.0)
        cy = center.get("y", 0.0)
        bw = size.get("w", 0.0)
        bh = size.get("h", 0.0)

        # Convert normalized cxcywh → pixel xyxy
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h

        masks = None
        if masks_rle is not None and i < len(masks_rle):
            rle = masks_rle[i]
            if rle is not None:
                masks = [_resize_mask(_decode_rle(rle), img_w, img_h)]

        detection = Detection(
            bbox=[x1, y1, x2, y2],
            masks=masks,
            class_id=class_id,
            class_name=label,
            confidence=None,
            metadata={
                "source": "falcon-perception-raw",
                "query_text": output_text,
            },
        )
        detections_obj.add_detection(detection)

    return detections_obj
