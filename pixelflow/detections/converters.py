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
    "from_efficienttam"
]


def _to_numpy(array):
    """Return `array` as numpy, detaching torch tensors and passing None through."""
    if array is None:
        return None
    if hasattr(array, "detach"):
        array = array.detach().cpu()
    return np.asarray(array)


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


def _get_label_info(labels, class_id):
    """Extract class name and keypoint names from various label formats.

    Supports:
        List[str]      — ["person", "car"] — index = class_id
        Dict[int, str] — {0: "person", 1: "car"} — key = class_id
        List[dict]     — [{"id": 0, "name": "person", "keypoints": [...]}] — search by "id" field

    Returns:
        tuple: (class_name, kp_names) where kp_names is List[str] or None.
    """
    if labels is None or class_id is None:
        return None, None

    # Rich label format: List[dict] with "id"/"name" keys
    if isinstance(labels, list) and labels and isinstance(labels[0], dict):
        for label in labels:
            if label["id"] == class_id:
                name = label["name"]
                kp_names = [kp["name"] for kp in label.get("keypoints", [])] or None
                return name, kp_names
        return None, None

    # Dict format: {int: str}
    if isinstance(labels, dict):
        return labels.get(class_id), None

    # Simple list format: ["person", "car"]
    if isinstance(labels, list) and class_id < len(labels):
        return labels[class_id], None

    return None, None


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
        >>> detections = pf.detections.from_datamarkin(api_response)
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
        Detections: Bounding boxes, polygon segments, OCR data, and sequential class IDs.

    Raises:
        ValueError: If task_prompt is not in parsed_result, is a text-only task, or the data
                   fields don't match any supported format.

    Example:
        >>> import pixelflow as pf
        >>> task = "<OD>"
        >>> parsed = processor.post_process_generation(outputs, task=task, image_size=(w, h))
        >>> detections = pf.detections.from_florence2(parsed, task_prompt=task)
        >>> for det in detections:
        ...     print(f"{det.class_name}: {det.bbox}")

    Notes:
        - Supported tasks: <OD>, <CAPTION_TO_PHRASE_GROUNDING>, <DENSE_REGION_CAPTION>,
          <REFERRING_EXPRESSION_SEGMENTATION>.
        - Text-only tasks (<CAPTION>, <OCR>, etc.) raise ValueError.
        - Sequential class_ids (0, 1, 2, ...) assigned for consistent color mapping.
        - Confidence defaults to 1.0 when not provided.
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    # Text-only tasks that should not be converted to Detections
    TEXT_ONLY_TASKS = {
        '<CAPTION>',
        '<DETAILED_CAPTION>',
        '<MORE_DETAILED_CAPTION>',
        '<OCR>',  # Pure OCR without regions
    }

    # Validate that task_prompt exists in parsed_result
    if task_prompt not in parsed_result:
        raise ValueError(
            f"Task prompt '{task_prompt}' not found in parsed_result. "
            f"Available keys: {list(parsed_result.keys())}"
        )

    # Reject text-only tasks
    if task_prompt in TEXT_ONLY_TASKS:
        raise ValueError(
            f"Task '{task_prompt}' returns text only and cannot be converted to Detections. "
            f"Text-only tasks: {TEXT_ONLY_TASKS}"
        )

    # Extract the task-specific data
    task_data = parsed_result[task_prompt]

    # Handle different task types based on available data fields
    if 'bboxes' in task_data and 'labels' in task_data:
        # Detection tasks: <OD>, <CAPTION_TO_PHRASE_GROUNDING>, <DENSE_REGION_CAPTION>, etc.
        bboxes = task_data['bboxes']
        labels = task_data['labels']
        scores = task_data.get('scores', None)  # Optional confidence scores

        # Assign sequential class IDs for consistent coloring
        unique_labels = []
        label_to_id = {}
        for label in labels:
            if label not in label_to_id:
                label_to_id[label] = len(unique_labels)
                unique_labels.append(label)

        for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
            class_id = label_to_id[label]
            confidence = float(scores[idx]) if scores is not None else 1.0

            # Florence-2 bboxes are already in XYXY format [x1, y1, x2, y2]
            detection = Detection(
                bbox=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                class_id=class_id,
                class_name=label,
                confidence=confidence
            )
            detections_obj.add_detection(detection)

    elif 'polygons' in task_data and 'labels' in task_data:
        # Segmentation tasks: <REFERRING_EXPRESSION_SEGMENTATION>
        # polygons[i] holds every polygon belonging to instance i, each a flat
        # [x1,y1,x2,y2,...] list. Nesting depth varies between Florence-2
        # versions, so _flat_coord_lists normalises it.
        polygons = task_data['polygons']
        labels = task_data['labels']

        # Assign sequential class IDs
        unique_labels = []
        label_to_id = {}
        for label in labels:
            if label not in label_to_id:
                label_to_id[label] = len(unique_labels)
                unique_labels.append(label)

        for idx, (poly_nested, label) in enumerate(zip(polygons, labels)):
            class_id = label_to_id[label]

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

            # bbox is the axis-aligned hull across every polygon of the instance.
            x_coords = [p[0] for poly in all_polygons for p in poly]
            y_coords = [p[1] for poly in all_polygons for p in poly]
            bbox = [
                float(min(x_coords)),
                float(min(y_coords)),
                float(max(x_coords)),
                float(max(y_coords))
            ]

            detection = Detection(
                bbox=bbox,
                segments=all_polygons[0],  # segments holds a single polygon
                masks=all_polygons,        # masks keeps every part
                class_id=class_id,
                class_name=label,
                confidence=1.0
            )
            detections_obj.add_detection(detection)

    else:
        # Unknown task format
        raise ValueError(
            f"Unsupported Florence-2 task format for '{task_prompt}'. "
            f"Available data fields: {list(task_data.keys())}. "
            f"Expected 'bboxes+labels' or 'polygons+labels'."
        )

    return detections_obj


def from_arrays(boxes, scores, class_ids, masks=None, keypoints=None, labels=None):
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
        class_ids: `(N,)` integer class IDs.
        masks: Optional `(N, H, W)` masks, cast to boolean. Instance segmentation only.
        keypoints: Optional `(N, K, 3)` keypoints with `(x, y, score)` columns.
        labels: Optional label definitions for class and keypoint name resolution.
            Accepts the same three formats as `from_detectron2`:
            - List[str]: ["person", "car"] — index = class_id
            - Dict[int, str]: {0: "person", 1: "car"} — key = class_id
            - List[dict]: [{"id": 0, "name": "person", "keypoints": [...]}]

    Returns:
        Detections: Bounding boxes, masks, keypoints, class IDs and confidence
        scores in PixelFlow's unified format.

    Raises:
        ValueError: If the arrays do not all describe the same number of detections.

    Example:
        >>> import pixelflow as pf
        >>> boxes = [[10, 20, 110, 220], [30, 40, 130, 240]]
        >>> detections = pf.detections.from_arrays(
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
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    boxes = _to_numpy(boxes)
    scores = _to_numpy(scores)
    class_ids = _to_numpy(class_ids)
    masks = _to_numpy(masks)
    keypoints = _to_numpy(keypoints)

    count = len(boxes)
    for name, array in (("scores", scores), ("class_ids", class_ids),
                        ("masks", masks), ("keypoints", keypoints)):
        if array is not None and len(array) != count:
            raise ValueError(
                f"{name} describes {len(array)} detections but boxes describes {count}"
            )

    for i in range(count):
        class_id = int(class_ids[i])
        class_name, kp_names = _get_label_info(labels, class_id)

        kpts = _build_keypoints(keypoints[i], kp_names) if keypoints is not None else None

        detections_obj.add_detection(Detection(
            bbox=boxes[i].tolist(),
            masks=[masks[i].astype(bool)] if masks is not None else None,
            segments=None,
            keypoints=kpts,
            class_id=class_id,
            class_name=class_name,
            confidence=float(scores[i])
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
        >>> detections = pf.detections.from_detectron2(outputs, labels=predictor.class_names)
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
        class_name, kp_names = _get_label_info(labels, class_id)

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
        >>> detections = pf.detections.from_mayaku(
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

        class_name, kp_names = _get_label_info(labels, class_id)

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
        >>> detections = pf.detections.from_ultralytics(outputs)
        >>> for det in detections:
        ...     print(f"{det.class_name}: {det.confidence:.2f}, bbox={det.bbox}")

    Notes:
        - tracker_id is set when using model.track() with persist=True.
        - Classification models produce a single detection with top-5 predictions in metadata.
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
            class_name, _ = _get_label_info(labels, class_id)
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
            _, kpt_names = _get_label_info(labels, class_id)
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

    if hasattr(masks, 'cpu'):
        masks = masks.cpu().numpy()
    if hasattr(scores, 'cpu'):
        scores = scores.cpu().numpy()

    masks = np.asarray(masks)
    scores = np.asarray(scores)

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
        >>> detections = pf.detections.from_supervision(sv_detections, labels=model.class_names)
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
        class_name, _ = _get_label_info(labels, cid)
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
        >>> pf_detections = pf.detections.from_rfdetr(rfdetr_output, labels=model.class_names)
        >>> print(f"Detected {len(pf_detections)} objects")

    Notes:
        - This is an alias for from_supervision()
        - RF-DETR returns supervision library's sv.Detections format
        - Use this function name if you're working with RF-DETR specifically
    """
    # Delegate to from_supervision for actual conversion
    return from_supervision(supervision_detections, labels=labels)


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
        >>> detections = pf.detections.from_falcon_perception(
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
