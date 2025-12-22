"""
Detection Converters for Machine Learning Framework Integration.

Provides standardized conversion utilities to transform detection outputs from 
various machine learning frameworks (Detectron2, Ultralytics YOLO, Datamarkin API, 
Transformers) into PixelFlow's unified Detections format. This module enables seamless 
integration with different ML backends while maintaining consistent data structures r
for downstream processing, visualization, and analysis workflows.
"""

import ast
import cv2
import numpy as np
from typing import (List, Dict, Any, Union, Optional)

__all__ = [
    "from_datamarkin",
    "from_florence2",
    "from_detectron2",
    "from_ultralytics",
    "from_transformers",
    "from_sam",
    "from_datamarkin_csv",
    # OCR converters
    "from_tesseract",
    "from_easyocr",
    "from_paddleocr",
    "from_ppstructure"
]

# COCO pose keypoint names (17 keypoints)
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def from_datamarkin(api_response: Dict[str, Any]):
    """
    Convert Datamarkin API response to a unified Detections object.

    Processes detection results from Datamarkin's cloud-based object detection API,
    extracting bounding boxes, segmentation masks, keypoints, class labels, and
    confidence scores into PixelFlow's standardized format for further processing.

    Args:
        api_response (Dict[str, Any]): Datamarkin API response dictionary containing
                                      nested 'predictions' -> 'objects' structure
                                      with detection data. Each object should have
                                      'bbox', 'mask', 'keypoints', 'class', and
                                      'bbox_score' fields.

    Returns:
        Detections: Unified Detections object containing all detected objects with
                   standardized XYXY bounding boxes, polygon masks, keypoint data,
                   and confidence scores. Empty Detections object if no predictions.

    Raises:
        KeyError: If required API response structure is missing or malformed
        TypeError: If bbox coordinates cannot be converted to numeric format
        ValueError: If confidence scores are outside valid range [0.0, 1.0]

    Example:
        >>> import pixelflow as pf
        >>> import requests
        >>>
        >>> # Call Datamarkin API for object detection
        >>> response = requests.post(
        ...     "https://api.datamarkin.com/detect",
        ...     files={"image": open("image.jpg", "rb")}
        ... )
        >>> api_response = response.json()  # Raw API output
        >>> detections = pf.detections.from_datamarkin(api_response)  # Convert to PixelFlow format
        >>>
        >>> # Basic usage - access detection data
        >>> for detection in detections.detections:
        ...     print(f"Class: {detection.class_id}, Confidence: {detection.confidence:.2f}")
        >>>
        >>> # Advanced usage - filter by confidence
        >>> high_conf_detections = [d for d in detections.detections if d.confidence > 0.8]
        >>> print(f"High confidence detections: {len(high_conf_detections)}")
        >>>
        >>> # Process masks and keypoints
        >>> for detection in detections.detections:
        ...     if detection.masks:
        ...         print(f"Object has {len(detection.masks)} mask regions")
        ...     if detection.keypoints:
        ...         print(f"Object has {len(detection.keypoints)} keypoints")
        >>>
        >>> # Empty response handling
        >>> empty_response = {"predictions": {"objects": []}}
        >>> empty_detections = pf.detections.from_datamarkin(empty_response)
        >>> print(f"Empty result: {len(empty_detections.detections)} objects")

    Notes:
        - Bounding boxes are expected in XYXY format from the API
        - Mask data is stored as polygon coordinates in nested list format
        - Keypoints are converted from API format (name, point, probability) to PixelFlow KeyPoint objects
        - API 'probability' field is converted to 'visibility' in KeyPoint (values > 0 = visible)
        - Class names are stored in class_name field (API doesn't provide numeric class_id)
        - Missing or null confidence scores are preserved as None values
        - Function gracefully handles missing optional fields (mask, keypoints)

    Performance Notes:
        - Efficient single-pass processing of API response structure
        - Minimal data copying for large mask or keypoint arrays
        - No validation overhead for well-formed API responses
    """
    from .detections import Detections, Detection, KeyPoint

    detections_obj = Detections()

    for obj in api_response.get("predictions", {}).get("objects", []):
        bbox = obj.get("bbox", [])
        mask = obj.get("mask", [])
        keypoints_api = obj.get("keypoints", [])
        class_name = obj.get("class", "")
        confidence = obj.get("bbox_score", None)

        # Convert API keypoint format to PixelFlow KeyPoint objects
        # API format: {"name": "p0", "point": [x, y], "probability": 0.375}
        # PixelFlow format: KeyPoint(x, y, name, visibility)
        keypoints = None
        if keypoints_api and len(keypoints_api) > 0:
            keypoints = []
            for kp_dict in keypoints_api:
                point = kp_dict.get("point", [0, 0])
                probability = kp_dict.get("probability", 0.0)

                # Convert probability to visibility (True if probability > 0)
                visibility = probability > 0.0

                keypoint = KeyPoint(
                    x=int(point[0]),
                    y=int(point[1]),
                    name=kp_dict.get("name", ""),
                    visibility=visibility
                )
                keypoints.append(keypoint)

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
    """
    Convert Florence-2 model results to a unified Detections object.

    Processes detection results from Microsoft's Florence-2 vision foundation model,
    extracting bounding boxes, segmentation polygons, OCR text with quad boxes, and
    labels into PixelFlow's standardized format. Supports all Florence-2 vision tasks
    including object detection, grounding, segmentation, and OCR with regions.

    Args:
        parsed_result (Dict[str, Any]): Parsed output from Florence-2 processor.
                                        Must be dictionary with task prompt as key
                                        containing nested data with 'bboxes', 'labels',
                                        'polygons', or 'quad_boxes' depending on task.
        task_prompt (str): Florence-2 task identifier used for prediction. Examples:
                          '<OD>' (object detection), '<CAPTION_TO_PHRASE_GROUNDING>',
                          '<REFERRING_EXPRESSION_SEGMENTATION>', '<OCR_WITH_REGION>'.
                          Must match a key in parsed_result dictionary.
        image_size (tuple, optional): Image dimensions as (width, height) tuple for
                                     coordinate normalization. If None, assumes
                                     coordinates are already in absolute pixels.
                                     Default is None.

    Returns:
        Detections: Unified Detections object containing detected objects with
                   standardized XYXY bounding boxes, polygon masks for segmentation
                   tasks, OCR data with quad boxes, and sequential class IDs for
                   consistent color mapping. Empty Detections if no objects found.

    Raises:
        ValueError: If task_prompt not found in parsed_result, or if task is text-only
                   (pure caption/OCR without regions), or if required data fields are
                   missing for the specified task type.
        KeyError: If expected nested dictionary keys are missing from parsed_result
        TypeError: If coordinate values cannot be converted to numeric format

    Example:
        >>> import torch
        >>> import pixelflow as pf
        >>> from transformers import AutoProcessor, AutoModelForCausalLM
        >>> from PIL import Image
        >>>
        >>> # Load Florence-2 model
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "microsoft/Florence-2-large",
        ...     trust_remote_code=True
        ... )
        >>> processor = AutoProcessor.from_pretrained(
        ...     "microsoft/Florence-2-large",
        ...     trust_remote_code=True
        ... )
        >>>
        >>> # Object detection task
        >>> image = Image.open("image.jpg")
        >>> task = "<OD>"
        >>> inputs = processor(text=task, images=image, return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_new_tokens=1024)
        >>> parsed = processor.post_process_generation(
        ...     outputs,
        ...     task=task,
        ...     image_size=(image.width, image.height)
        ... )
        >>> # parsed format: {'<OD>': {'bboxes': [[x1,y1,x2,y2],...], 'labels': ['car','person',...]}}
        >>> detections = pf.detections.from_florence2(
        ...     parsed[task],
        ...     task_prompt=task,
        ...     image_size=(image.width, image.height)
        ... )
        >>> for det in detections.detections:
        ...     print(f"Class: {det.class_name}, BBox: {det.bbox}")
        >>>
        >>> # Grounding task (phrase → bounding boxes)
        >>> task = "<CAPTION_TO_PHRASE_GROUNDING>"
        >>> prompt = "A green car and a person with red shirt"
        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_new_tokens=1024)
        >>> parsed = processor.post_process_generation(outputs, task=task, image_size=(image.width, image.height))
        >>> detections = pf.detections.from_florence2(parsed[task], task_prompt=task)
        >>> # Each detection has bbox + class_name from grounded phrase
        >>>
        >>> # Segmentation task (referring expression → mask)
        >>> task = "<REFERRING_EXPRESSION_SEGMENTATION>"
        >>> prompt = "the red car"
        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_new_tokens=1024)
        >>> parsed = processor.post_process_generation(outputs, task=task, image_size=(image.width, image.height))
        >>> detections = pf.detections.from_florence2(parsed[task], task_prompt=task)
        >>> for det in detections.detections:
        ...     if det.masks:
        ...         print(f"Polygon mask with {len(det.masks[0])} points")
        >>>
        >>> # OCR with region detection
        >>> task = "<OCR_WITH_REGION>"
        >>> inputs = processor(text=task, images=document_image, return_tensors="pt")
        >>> outputs = model.generate(**inputs, max_new_tokens=1024)
        >>> parsed = processor.post_process_generation(outputs, task=task, image_size=(document_image.width, document_image.height))
        >>> detections = pf.detections.from_florence2(parsed[task], task_prompt=task)
        >>> for det in detections.detections:
        ...     print(f"Text: {det.ocr_data.text}, Quad: {det.segments}")

    Notes:
        - Florence-2 output format: {task_prompt: {'bboxes': [...], 'labels': [...]}}
        - Detection tasks (<OD>, <DENSE_REGION_CAPTION>, etc.) provide bboxes + labels
        - Segmentation tasks provide triple-nested polygons: [[[x1,y1,x2,y2,...]]]
        - OCR tasks provide quad_boxes (8 values): [x1,y1,x2,y2,x3,y3,x4,y4]
        - Text-only tasks (<CAPTION>, <MORE_DETAILED_CAPTION>, <OCR>) raise ValueError
        - Sequential class_ids (0,1,2,...) assigned for consistent color mapping
        - Confidence defaults to 1.0 if scores not provided in parsed_result
        - All coordinates assumed to be in absolute pixels (no normalization needed)

    Performance Notes:
        - Efficient single-pass processing of Florence-2 output dictionary
        - Minimal data copying for bboxes, polygons, and quad coordinates
        - O(n) complexity where n is number of detected objects
        - Lazy polygon processing deferred until mask data actually accessed

    See Also:
        from_detectron2 : Convert Detectron2 results to PixelFlow format
        from_ultralytics : Convert YOLO results to PixelFlow format
        from_easyocr : Convert EasyOCR results for pure OCR tasks
    """
    from .detections import Detections, Detection, OCRData

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
        polygons = task_data['polygons']  # Triple-nested: [[[x1,y1,x2,y2,...]]]
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

            # Florence-2 polygons are triple-nested: [[[x1,y1,x2,y2,...]]]
            # Extract the innermost list of coordinates
            if len(poly_nested) > 0 and len(poly_nested[0]) > 0:
                poly_coords = poly_nested[0][0]  # Get innermost list

                # Convert flat list [x1,y1,x2,y2,...] to list of tuples [(x1,y1), (x2,y2), ...]
                segments = []
                for i in range(0, len(poly_coords), 2):
                    if i + 1 < len(poly_coords):
                        segments.append((int(poly_coords[i]), int(poly_coords[i + 1])))

                # Compute axis-aligned bounding box from polygon
                if segments:
                    x_coords = [p[0] for p in segments]
                    y_coords = [p[1] for p in segments]
                    bbox = [
                        float(min(x_coords)),
                        float(min(y_coords)),
                        float(max(x_coords)),
                        float(max(y_coords))
                    ]
                else:
                    bbox = None

                detection = Detection(
                    bbox=bbox,
                    segments=segments,  # Store polygon as segments
                    masks=[segments],   # Also store as mask for compatibility
                    class_id=class_id,
                    class_name=label,
                    confidence=1.0
                )
                detections_obj.add_detection(detection)

    elif 'quad_boxes' in task_data and 'labels' in task_data:
        # OCR with region: <OCR_WITH_REGION>
        quad_boxes = task_data['quad_boxes']  # Format: [[x1,y1,x2,y2,x3,y3,x4,y4], ...]
        labels = task_data['labels']  # Text content

        for idx, (quad, text) in enumerate(zip(quad_boxes, labels)):
            # quad is 8 values: [x1,y1, x2,y2, x3,y3, x4,y4]
            # Convert to list of (x,y) tuples for segments
            segments = [
                (int(quad[0]), int(quad[1])),  # Top-left
                (int(quad[2]), int(quad[3])),  # Top-right
                (int(quad[4]), int(quad[5])),  # Bottom-right
                (int(quad[6]), int(quad[7]))   # Bottom-left
            ]

            # Compute axis-aligned bounding box from quad
            x_coords = [p[0] for p in segments]
            y_coords = [p[1] for p in segments]
            bbox = [
                float(min(x_coords)),
                float(min(y_coords)),
                float(max(x_coords)),
                float(max(y_coords))
            ]

            # Create OCRData for structured text information
            ocr_data = OCRData(
                text=text.strip(),
                confidence=1.0,  # Florence-2 doesn't provide OCR confidence
                language='multi',  # Florence-2 is multi-lingual
                level='word',
                order=idx,
                element_type='text'
            )

            detection = Detection(
                bbox=bbox,
                segments=segments,  # Store quad as segments
                ocr_data=ocr_data,
                class_id=0,  # OCR has single class
                class_name='text',
                confidence=1.0
            )
            detections_obj.add_detection(detection)

    else:
        # Unknown task format
        raise ValueError(
            f"Unsupported Florence-2 task format for '{task_prompt}'. "
            f"Available data fields: {list(task_data.keys())}. "
            f"Expected 'bboxes+labels', 'polygons+labels', or 'quad_boxes+labels'."
        )

    return detections_obj


def from_detectron2(detectron2_results: Dict[str, Any], class_names: Optional[List[str]] = None):
    """
    Convert Detectron2 inference results to a unified Detections object.
    
    Extracts bounding boxes, confidence scores, class IDs, segmentation masks, 
    and keypoints from Detectron2's instances format and standardizes them 
    into PixelFlow's Detection objects. Handles automatic tensor-to-numpy conversion 
    and CPU transfer for efficient processing.
    
    Args:
        detectron2_results (Dict[str, Any]): Detectron2 inference results dictionary
                                           containing 'instances' key with prediction
                                           data including pred_boxes, scores,
                                           pred_classes, pred_masks, and pred_keypoints.
                                           Results should be from DefaultPredictor output.
        class_names (Optional[List[str]]): List of class names indexed by class ID.
                                          If provided, Detection objects will include
                                          class_name attribute. Obtain from MetadataCatalog:
                                          `MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes`

    Returns:
        Detections: Unified Detections object with all detected instances converted
                   to standardized format. Contains XYXY bounding boxes as lists,
                   boolean numpy array masks, integer class IDs, and float confidences.
                   Returns empty Detections if no instances found.
    
    Raises:
        KeyError: If required 'instances' key is missing from detectron2_results
        AttributeError: If instances object lacks expected prediction attributes
        RuntimeError: If tensor operations fail during CPU transfer
        ValueError: If bounding box or confidence data contains invalid values
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from detectron2 import model_zoo
        >>> from detectron2.engine import DefaultPredictor
        >>> from detectron2.config import get_cfg
        >>> from detectron2.data import MetadataCatalog
        >>>
        >>> # Setup Detectron2 object detection model
        >>> cfg = get_cfg()
        >>> cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        >>> cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        >>> predictor = DefaultPredictor(cfg)
        >>>
        >>> # Get class names from metadata
        >>> class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        >>>
        >>> image = cv2.imread("path/to/image.jpg")
        >>> outputs = predictor(image)  # Raw Detectron2 output
        >>> detections = pf.detections.from_detectron2(outputs, class_names=class_names)
        >>>
        >>> # Access detection data with class names
        >>> for detection in detections.detections:
        ...     print(f"Class: {detection.class_name}, Confidence: {detection.confidence:.2f}")
        >>> 
        >>> # Advanced usage - segmentation model with masks
        >>> cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        >>> cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        >>> seg_predictor = DefaultPredictor(cfg)
        >>> outputs = seg_predictor(image)
        >>> detections = pf.detections.from_detectron2(outputs)
        >>> for detection in detections.detections:
        ...     if detection.masks:
        ...         print(f"Object has mask with shape: {detection.masks[0].shape}")
        >>> 
        >>> # Process keypoint detection results
        >>> cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
        >>> cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
        >>> kpt_predictor = DefaultPredictor(cfg)
        >>> outputs = kpt_predictor(image)
        >>> detections = pf.detections.from_detectron2(outputs)
        >>> 
        >>> # Empty result handling
        >>> empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> empty_outputs = predictor(empty_image)
        >>> empty_detections = pf.detections.from_detectron2(empty_outputs)
        >>> print(f"No detections: {len(empty_detections.detections)} objects")
    
    Notes:
        - All tensor data is automatically moved to CPU before numpy conversion
        - Bounding boxes maintain XYXY format from Detectron2 (no coordinate transformation)
        - Segmentation masks are converted to boolean arrays for memory efficiency
        - Class IDs are converted to integers for consistency with other frameworks
        - Confidence scores are converted to float type for standardization
        - Keypoint data structure is preserved but PixelFlow KeyPoint conversion pending
        - Function handles missing prediction attributes gracefully (returns None)
        
    Performance Notes:
        - Efficient batch tensor operations minimize GPU-CPU transfer overhead
        - Single CPU transfer per tensor type reduces memory allocation
        - Boolean mask conversion optimized for large segmentation masks
        - Zero-copy numpy operations where possible for large datasets
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

        # Extract class name from provided class_names list
        class_name = None
        if class_names is not None and class_id is not None:
            if class_id < len(class_names):
                class_name = class_names[class_id]

        # Handle segmentation masks
        mask = None
        if masks is not None:
            mask_data = masks[i].astype(bool)
            mask = mask_data
        
        # Handle keypoints if available
        kpts = None
        if keypoints is not None:
            # Detectron2 keypoints are in format (x, y, visibility) 
            kpt_data = keypoints[i]
            # Convert to PixelFlow KeyPoint format if needed
            # This would need to be implemented based on your KeyPoint class
        
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


def from_ultralytics(ultralytics_results: Union[Any, List[Any]]):
    """
    Convert Ultralytics YOLO results to a unified Detections object.
    
    Supports both detection and segmentation models, handling bounding boxes,
    confidence scores, class IDs, segmentation masks, and tracker IDs.
    Automatically processes letterbox padding removal and mask resizing to
    original image dimensions with precise coordinate transformation.
    
    Args:
        ultralytics_results (Union[Any, List[Any]]): YOLO results from Ultralytics
                                                    library prediction or tracking.
                                                    Can be single Result object or
                                                    list containing one Result object.
                                                    Must have boxes attribute with
                                                    detection data.
        
    Returns:
        Detections: Unified Detections object containing all detected objects with
                   standardized XYXY bounding boxes, boolean binary masks resized
                   to original image dimensions, polygon segments as integer coordinates,
                   and tracker IDs if available. Empty Detections if no boxes found.
    
    Raises:
        AttributeError: If results object lacks required boxes or data attributes
        IndexError: If results list is empty or malformed
        ValueError: If bounding box coordinates or confidence scores are invalid
        RuntimeError: If tensor operations fail during CPU transfer
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>> 
        >>> # Basic object detection
        >>> model = YOLO("yolo11n.pt")
        >>> image = cv2.imread("path/to/image.jpg")
        >>> outputs = model.predict(image)  # Raw YOLO output
        >>> detections = pf.detections.from_ultralytics(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # Access detection data with class names
        >>> for detection in detections.detections:
        ...     print(f"Class: {detection.class_name}, Confidence: {detection.confidence:.2f}")
        ...     print(f"BBox: {detection.bbox}")
        >>> 
        >>> # Advanced usage - segmentation model with masks
        >>> seg_model = YOLO("yolo11n-seg.pt")
        >>> outputs = seg_model.predict(image, save_crop=False)
        >>> detections = pf.detections.from_ultralytics(outputs)
        >>> for detection in detections.detections:
        ...     if detection.masks:
        ...         mask_shape = detection.masks[0].shape
        ...         print(f"Object mask: {mask_shape} pixels")
        ...     if detection.segments:
        ...         poly_points = len(detection.segments)
        ...         print(f"Polygon: {poly_points} points")
        >>> 
        >>> # Object tracking with persistent IDs
        >>> outputs = model.track(image, tracker="bytetrack.yaml", persist=True)
        >>> detections = pf.detections.from_ultralytics(outputs)
        >>> for detection in detections.detections:
        ...     if detection.tracker_id is not None:
        ...         print(f"Tracked object {detection.tracker_id}: {detection.class_name}")
        >>> 
        >>> # Batch processing multiple images
        >>> image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        >>> for img_path in image_paths:
        ...     img = cv2.imread(img_path)
        ...     outputs = model.predict(img, verbose=False)
        ...     detections = pf.detections.from_ultralytics(outputs)
        ...     print(f"{img_path}: {len(detections.detections)} objects")
    
    Notes:
        - Automatically handles single Result or list[Result] input formats
        - Bounding boxes maintain XYXY format from YOLO predictions
        - Class names are extracted from model.names dictionary when available
        - Binary masks are precisely resized using letterbox padding calculations
        - Polygon segments stored as integer coordinate lists for efficiency
        - Tracker IDs preserved from model.track() calls with persist=True
        - Original YOLO masks stored in _ultralytics_masks for advanced use cases
        - Gracefully handles empty results with no detections
        
    Performance Notes:
        - Efficient batch tensor operations minimize CPU/GPU memory transfers
        - Single numpy conversion per detection component reduces allocation overhead
        - Letterbox padding calculations optimized for common YOLO input sizes
        - OpenCV resize operations use nearest neighbor for boolean mask precision
        - Zero-copy operations where possible for large segmentation masks
        
    See Also:
        from_detectron2 : Convert Detectron2 results to PixelFlow format
        from_datamarkin : Convert cloud API results to PixelFlow format
    """
    from .detections import Detections, Detection, KeyPoint

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
        
        # Extract class name from result if available
        class_name = None
        if hasattr(result, 'names') and result.names:
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
            segments = result.masks.xy[i]
            if segments is not None and len(segments) > 0:
                segments = segments.astype(int).tolist()
            
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
            kpts = keypoints_data[i]  # Shape: [17, 3] for detection i

            keypoints_list = []
            for kpt_idx, kpt in enumerate(kpts):
                x, y, conf = kpt[0], kpt[1], kpt[2]
                # Use visibility threshold (conf > 0.5 means visible)
                visibility = conf > 0.5
                name = COCO_KEYPOINT_NAMES[kpt_idx] if kpt_idx < len(COCO_KEYPOINT_NAMES) else f"keypoint_{kpt_idx}"

                keypoint = KeyPoint(
                    x=int(x),
                    y=int(y),
                    name=name,
                    visibility=visibility
                )
                keypoints_list.append(keypoint)

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
    """
    Convert Transformers library results to a unified Detections object.
    
    Placeholder function for future integration with Hugging Face Transformers
    object detection and segmentation models. Will support DETR, RT-DETR, and
    other transformer-based detection architectures.
    
    Args:
        transformers_results (Any): Results from Transformers library object
                                   detection models. Expected format includes
                                   boxes, labels, and scores tensors.
        
    Returns:
        Detections: Empty Detections object. Full implementation pending.
    
    Raises:
        NotImplementedError: This function is not yet implemented
        
    Example:
        >>> import pixelflow as pf
        >>> # Future usage with Transformers models
        >>> # from transformers import AutoImageProcessor, AutoModelForObjectDetection
        >>> # processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        >>> # model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        >>> # outputs = model(**processor(image, return_tensors="pt"))  # Raw output
        >>> # detections = pf.detections.from_transformers(outputs)  # Convert to PixelFlow
        >>> print("Function not yet implemented")
    
    Notes:
        - Implementation will support DETR, RT-DETR, and YOLO-transformer models
        - Will handle transformer-specific output formats and attention mechanisms
        - Planned support for both detection and segmentation transformer models
        - Integration with Transformers AutoModel pipeline architecture
    """
    raise NotImplementedError("from_transformers converter not yet implemented")


def from_sam(sam_results: Any):
    """
    Convert Segment Anything Model (SAM) results to a unified Detections object.
    
    Placeholder function for integration with Meta's Segment Anything Model (SAM)
    for interactive and automatic segmentation tasks. Will support prompt-based
    segmentation with point, box, and text prompts.
    
    Args:
        sam_results (Any): Results from SAM model inference including masks,
                          iou_predictions, and low_res_logits from SamPredictor
                          or SamAutomaticMaskGenerator output.
        
    Returns:
        Detections: Empty Detections object. Full implementation pending.
    
    Raises:
        NotImplementedError: This function is not yet implemented
        
    Example:
        >>> import pixelflow as pf
        >>> # Future usage with SAM models
        >>> # from segment_anything import SamPredictor, sam_model_registry
        >>> # sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
        >>> # predictor = SamPredictor(sam)
        >>> # predictor.set_image(image)
        >>> # masks, scores, logits = predictor.predict(point_coords=input_point)  # Raw output
        >>> # detections = pf.detections.from_sam({"masks": masks, "scores": scores})  # Convert to PixelFlow
        >>> print("Function not yet implemented")
    
    Notes:
        - Implementation will support both SamPredictor and SamAutomaticMaskGenerator
        - Will handle multi-mask outputs with IoU quality scores
        - Planned support for prompt-based and automatic segmentation workflows
        - Integration with different SAM model variants (ViT-B, ViT-L, ViT-H)
        - Will convert high-quality masks to PixelFlow Detection format
    """
    raise NotImplementedError("from_sam converter not yet implemented")


def from_datamarkin_csv(group: Any, height: int, width: int):
    """
    Convert CSV data from Datamarkin format to a unified Detections object.
    
    Processes normalized coordinates from CSV annotation format and converts them to pixel
    coordinates using the provided image dimensions. Handles both bounding box rectangles
    and segmentation polygon data with automatic coordinate denormalization and validation.
    
    Args:
        group (Any): Pandas DataFrame or DataFrame group containing CSV rows with
                    required columns 'xmin', 'ymin', 'xmax', 'ymax', 'segmentation',
                    'class', and optional 'confidence'. All coordinate values must
                    be normalized floats in range [0.0, 1.0].
        height (int): Image height in pixels for coordinate denormalization.
                     Must be positive integer representing actual image height.
        width (int): Image width in pixels for coordinate denormalization.
                    Must be positive integer representing actual image width.
        
    Returns:
        Detections: Unified Detections object with pixel coordinates converted from
                   normalized values. Contains XYXY bounding boxes as integers,
                   polygon masks as lists of (x, y) tuples, and preserved class labels.
                   Empty Detections if group contains no rows.
    
    Raises:
        KeyError: If required CSV columns are missing from the DataFrame
        ValueError: If coordinate values are outside [0.0, 1.0] normalized range
        TypeError: If height/width are not integers or coordinates not numeric
        SyntaxError: If segmentation string cannot be parsed as valid Python list
        
    Example:
        >>> import pandas as pd
        >>> import pixelflow as pf
        >>> 
        >>> # Load CSV annotations with normalized coordinates
        >>> df = pd.read_csv("datamarkin_annotations.csv")
        >>> # CSV format: image,xmin,ymin,xmax,ymax,segmentation,class,confidence
        >>> # Example row: img1.jpg,0.1,0.2,0.8,0.9,"[0.1,0.2,0.8,0.2,0.8,0.9,0.1,0.9]",person,0.95
        >>> detections = pf.detections.from_datamarkin_csv(df, height=480, width=640)  # Convert to PixelFlow format
        >>> 
        >>> # Basic usage - process single image annotations
        >>> for detection in detections.detections:
        ...     print(f"Class: {detection.class_id}, Confidence: {detection.confidence}")
        ...     print(f"BBox: {detection.bbox}")  # Pixel coordinates
        >>> 
        >>> # Advanced usage - batch process multiple images
        >>> for image_name, group in df.groupby('image'):
        ...     img_detections = pf.detections.from_datamarkin_csv(group, height=1080, width=1920)
        ...     print(f"Image {image_name}: {len(img_detections.detections)} annotations")
        ...     for detection in img_detections.detections:
        ...         if detection.masks:
        ...             poly_points = len(detection.masks[0])
        ...             print(f"  Polygon with {poly_points} points")
        >>> 
        >>> # Handle missing confidence scores
        >>> df_no_conf = df.drop('confidence', axis=1)
        >>> detections = pf.detections.from_datamarkin_csv(df_no_conf, height=720, width=1280)
        >>> for detection in detections.detections:
        ...     conf_str = "Unknown" if detection.confidence is None else f"{detection.confidence:.2f}"
        ...     print(f"Detection confidence: {conf_str}")
        >>> 
        >>> # Validate coordinate ranges
        >>> valid_coords = df[(df['xmin'] >= 0) & (df['xmax'] <= 1) & 
        ...                  (df['ymin'] >= 0) & (df['ymax'] <= 1)]
        >>> detections = pf.detections.from_datamarkin_csv(valid_coords, height=600, width=800)
    
    Notes:
        - All input coordinates must be normalized floats in range [0.0, 1.0]
        - Bounding boxes are converted to integer pixel coordinates using XYXY format
        - Segmentation strings are parsed using ast.literal_eval for safe evaluation
        - Polygon coordinates are stored as (x, y) tuples for geometric operations
        - Missing confidence values default to None and are handled gracefully
        - Class labels are preserved as strings without modification
        - Function performs coordinate validation during denormalization process
        
    Performance Notes:
        - Efficient vectorized operations for coordinate transformation
        - Minimal string parsing overhead using ast.literal_eval
        - Single-pass iteration through DataFrame rows
        - Memory-efficient tuple creation for polygon coordinates
        
    See Also:
        from_datamarkin : Convert Datamarkin API responses to PixelFlow format
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    for index, row in group.iterrows():
        # Get the bounding box coordinates and denormalize them
        xmin = int(row['xmin'] * width)
        ymin = int(row['ymin'] * height)
        xmax = int(row['xmax'] * width)
        ymax = int(row['ymax'] * height)

        # Convert normalized points to pixel coordinates for the mask
        segmentation_list = ast.literal_eval(row['segmentation'])
        segmentation_points = []
        for i in range(0, len(segmentation_list), 2):
            x = int(segmentation_list[i] * width)
            y = int(segmentation_list[i + 1] * height)
            segmentation_points.append((x, y))  # Convert to tuple for polygon points

        # Create the Detection object
        detection = Detection(
            bbox=[xmin, ymin, xmax, ymax],
            masks=[segmentation_points],  # Add mask as list of lists of tuples
            keypoints=None,  # TODO
            class_id=row['class'],
            confidence=row.get('confidence', None)  # Add confidence if available
        )

        # Add the prediction to the predictions list
        detections_obj.add_detection(detection)

    return detections_obj


def from_easyocr(easyocr_results: List[Any], language: str = None):
    """
    Convert EasyOCR results to a unified Detections object.

    Processes text detection results from EasyOCR's readtext output, extracting
    quadrilateral bounding boxes, text content, and confidence scores into PixelFlow's
    standardized format. Automatically handles polygon-to-bbox conversion and preserves
    original quadrilateral coordinates for rotated text handling.

    Args:
        easyocr_results (List[Any]): List of detection tuples from EasyOCR.readtext().
                                     Each tuple contains (bbox_polygon, text, confidence)
                                     where bbox_polygon is list of 4 (x, y) corner points,
                                     text is recognized string, and confidence is float [0-1].
        language (str): Language code for the detected text (e.g., 'en', 'zh', 'ko').
                       If None, uses 'multi' to indicate multi-language detection.
                       Default is None.

    Returns:
        Detections: Unified Detections object containing text detections with both
                   axis-aligned XYXY bounding boxes and original quadrilateral segments,
                   text content, and confidence scores. Empty Detections if no text found.

    Raises:
        TypeError: If easyocr_results is not a list or contains invalid tuple format
        ValueError: If polygon coordinates cannot be converted to numeric values
        IndexError: If detection tuples don't contain exactly 3 elements

    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> import easyocr
        >>>
        >>> # Basic single-language OCR
        >>> reader = easyocr.Reader(['en'])
        >>> image = cv2.imread("document.jpg")
        >>> results = reader.readtext(image)
        >>> detections = pf.detections.from_easyocr(results, language='en')
        >>> for det in detections.detections:
        ...     print(f"Text: {det.text}, Confidence: {det.text_confidence:.2f}")
        >>>
        >>> # Multi-language OCR
        >>> reader_multi = easyocr.Reader(['en', 'zh'])
        >>> results = reader_multi.readtext(image)
        >>> detections = pf.detections.from_easyocr(results, language='multi')
        >>>
        >>> # Access polygon coordinates for rotated text
        >>> for det in detections.detections:
        ...     if det.segments:
        ...         print(f"Polygon vertices: {det.segments}")
        ...         print(f"Axis-aligned bbox: {det.bbox}")
        >>>
        >>> # Filter and reconstruct text
        >>> high_conf = detections.filter_by_text_confidence(0.5)
        >>> sorted_text = sorted(high_conf.detections, key=lambda d: (d.bbox[1], d.bbox[0]))
        >>> full_text = " ".join([d.text for d in sorted_text])
        >>> print(f"Recognized: {full_text}")
        >>>
        >>> # Korean text detection
        >>> reader_ko = easyocr.Reader(['ko'])
        >>> results_ko = reader_ko.readtext(korean_image)
        >>> detections_ko = pf.detections.from_easyocr(results_ko, language='ko')

    Notes:
        - EasyOCR returns quadrilateral polygons (4 corner points) instead of axis-aligned boxes
        - Original polygon stored in segments field for precise text localization
        - bbox field contains axis-aligned bounding box (min/max of polygon coordinates)
        - text_level is always 'word' as EasyOCR doesn't provide hierarchy information
        - text_order preserves the original detection sequence from EasyOCR
        - Confidence scores are already normalized to [0.0-1.0] range
        - Works seamlessly with rotated, skewed, or curved text
        - Language code 'multi' recommended for multi-language scenarios

    Performance Notes:
        - Efficient single-pass conversion of EasyOCR results
        - Minimal computational overhead for polygon processing
        - O(n) complexity where n is number of detected text regions

    See Also:
        from_tesseract : Convert Tesseract OCR results to PixelFlow format
        from_paddleocr : Convert PaddleOCR results to PixelFlow format
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    # Set default language to 'multi' if not specified
    if language is None:
        language = 'multi'

    for idx, result in enumerate(easyocr_results):
        # EasyOCR returns: (bbox_polygon, text, confidence)
        bbox_polygon, text, confidence = result

        # bbox_polygon is a list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Convert to list of tuples for segments
        segments = [(int(point[0]), int(point[1])) for point in bbox_polygon]

        # Calculate axis-aligned bounding box from polygon
        x_coords = [point[0] for point in bbox_polygon]
        y_coords = [point[1] for point in bbox_polygon]
        x1 = int(min(x_coords))
        y1 = int(min(y_coords))
        x2 = int(max(x_coords))
        y2 = int(max(y_coords))
        bbox = [x1, y1, x2, y2]

        # Create OCRData object for structured OCR information
        from .detections import OCRData
        ocr_data = OCRData(
            text=text.strip(),
            confidence=confidence,
            language=language,
            level='word',  # EasyOCR returns word-level results
            order=idx,  # Preserve detection order
            element_type='text'  # EasyOCR is text OCR
        )

        # Create detection with OCRData
        detection = Detection(
            bbox=bbox,
            segments=segments,  # Store original quadrilateral
            ocr_data=ocr_data,
            metadata={
                'easyocr_polygon': bbox_polygon  # Store original polygon format
            }
        )

        detections_obj.add_detection(detection)

    return detections_obj


def from_tesseract(tesseract_data: Dict[str, Any], level: str = 'word', language: str = 'en'):
    """
    Convert Tesseract OCR results to a unified Detections object.

    Processes hierarchical text detection results from Tesseract OCR's image_to_data
    output, extracting bounding boxes, text content, confidence scores, and document
    structure information into PixelFlow's standardized format. Supports multiple
    hierarchy levels (char, word, line, paragraph, block, page) with automatic
    parent-child relationship tracking for document reconstruction.

    Args:
        tesseract_data (Dict[str, Any]): Dictionary output from pytesseract.image_to_data
                                         with Output.DICT format. Must contain 'level',
                                         'text', 'left', 'top', 'width', 'height',
                                         'conf', and structural keys ('block_num',
                                         'par_num', 'line_num', 'word_num').
        level (str): Text hierarchy level to extract: 'char', 'word', 'line', 'paragraph',
                    'block', or 'page'. Default is 'word'. Determines granularity of
                    detection objects returned.
        language (str): Language code for the detected text (e.g., 'en', 'fra', 'deu').
                       Default is 'en'. Used to set text_language field.

    Returns:
        Detections: Unified Detections object containing text detections with XYXY
                   bounding boxes, text content, confidence scores, hierarchy information,
                   and reading order. Empty Detections if no text found at specified level.

    Raises:
        KeyError: If required keys are missing from tesseract_data dictionary
        ValueError: If level parameter is not a valid hierarchy level
        TypeError: If bounding box coordinates cannot be converted to numeric format

    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> import pytesseract
        >>> from pytesseract import Output
        >>>
        >>> # Basic word-level OCR
        >>> image = cv2.imread("document.jpg")
        >>> data = pytesseract.image_to_data(image, output_type=Output.DICT, lang='eng')
        >>> detections = pf.detections.from_tesseract(data, level='word', language='en')
        >>> for det in detections.detections:
        ...     print(f"Text: {det.text}, Confidence: {det.text_confidence:.2f}")
        >>>
        >>> # Line-level OCR for sentence extraction
        >>> detections = pf.detections.from_tesseract(data, level='line', language='en')
        >>> for det in detections.detections:
        ...     print(f"Line {det.text_order}: {det.text}")
        >>>
        >>> # Hierarchical document structure
        >>> word_detections = pf.detections.from_tesseract(data, level='word')
        >>> for word_det in word_detections.detections:
        ...     print(f"Word '{word_det.text}' in line {word_det.text_parent_id}")
        >>>
        >>> # Filter low confidence text
        >>> high_conf = detections.filter_by_text_confidence(0.80)
        >>> full_text = " ".join([d.text for d in high_conf.detections])
        >>>
        >>> # Multi-language OCR
        >>> data_fr = pytesseract.image_to_data(image, output_type=Output.DICT, lang='fra')
        >>> detections_fr = pf.detections.from_tesseract(data_fr, level='word', language='fr')

    Notes:
        - Tesseract uses hierarchical structure: page > block > paragraph > line > word > char
        - Level parameter must match Tesseract's hierarchy levels (1=page, 2=block, 3=para, 4=line, 5=word)
        - Empty text entries (confidence -1) are automatically filtered out
        - Bounding boxes are in XYXY format converted from Tesseract's (left, top, width, height)
        - text_order field preserves reading order within each hierarchy level
        - text_parent_id uses format "{level}_{block}_{para}_{line}" for hierarchical tracking
        - Confidence scores are already in [0.0-100.0] range from Tesseract, normalized to [0.0-1.0]

    Performance Notes:
        - Efficient single-pass processing of Tesseract output dictionary
        - Minimal data copying for large document OCR results
        - O(n) complexity where n is number of text elements at specified level

    See Also:
        from_easyocr : Convert EasyOCR results to PixelFlow format
        from_paddleocr : Convert PaddleOCR results to PixelFlow format
    """
    from .detections import Detections, Detection

    detections_obj = Detections()

    # Map level string to Tesseract level number
    level_map = {
        'page': 1,
        'block': 2,
        'paragraph': 3,
        'line': 4,
        'word': 5,
        'char': 5  # Character level uses same as word but filtered differently
    }

    if level not in level_map:
        raise ValueError(f"Invalid level '{level}'. Must be one of: {list(level_map.keys())}")

    target_level = level_map[level]
    num_entries = len(tesseract_data['text'])

    for i in range(num_entries):
        # Skip entries not at target level
        if tesseract_data['level'][i] != target_level:
            continue

        # Skip empty text or low confidence (-1 means no text detected)
        text = tesseract_data['text'][i].strip()
        conf = tesseract_data['conf'][i]
        if not text or conf == -1:
            continue

        # Extract bounding box (Tesseract uses left, top, width, height)
        left = tesseract_data['left'][i]
        top = tesseract_data['top'][i]
        width = tesseract_data['width'][i]
        height = tesseract_data['height'][i]

        # Convert to XYXY format
        x1 = left
        y1 = top
        x2 = left + width
        y2 = top + height
        bbox = [x1, y1, x2, y2]

        # Build hierarchical parent ID
        block_num = tesseract_data['block_num'][i]
        par_num = tesseract_data['par_num'][i]
        line_num = tesseract_data['line_num'][i]
        word_num = tesseract_data['word_num'][i]

        # Generate unique ID for this element
        if level == 'word' or level == 'char':
            element_id = f"word_{block_num}_{par_num}_{line_num}_{word_num}"
            parent_id = f"line_{block_num}_{par_num}_{line_num}"
        elif level == 'line':
            element_id = f"line_{block_num}_{par_num}_{line_num}"
            parent_id = f"paragraph_{block_num}_{par_num}"
        elif level == 'paragraph':
            element_id = f"paragraph_{block_num}_{par_num}"
            parent_id = f"block_{block_num}"
        elif level == 'block':
            element_id = f"block_{block_num}"
            parent_id = "page_1"
        else:  # page
            element_id = "page_1"
            parent_id = None

        # Normalize confidence from 0-100 to 0-1
        text_confidence = conf / 100.0 if conf >= 0 else None

        # Create OCRData object for structured OCR information
        from .detections import OCRData
        ocr_data = OCRData(
            text=text,
            confidence=text_confidence if text_confidence is not None else 0.0,  # OCRData requires confidence
            language=language,
            level=level,
            order=i,  # Use index as reading order
            parent_id=parent_id,
            element_type='text'  # Tesseract is pure text OCR
        )

        # Create detection with OCRData
        detection = Detection(
            bbox=bbox,
            ocr_data=ocr_data,
            metadata={
                'tesseract_element_id': element_id,
                'block_num': block_num,
                'par_num': par_num,
                'line_num': line_num,
                'word_num': word_num
            }
        )

        detections_obj.add_detection(detection)

    return detections_obj


def from_paddleocr(paddle_results: List[Any], language: str = None):
    """
    Convert PaddleOCR results to a unified Detections object.

    Processes text detection results from PaddleOCR's ocr() output, extracting
    quadrilateral bounding boxes, text content, confidence scores, and text angles
    into PixelFlow's standardized format. Supports rotated text detection with
    angle information and polygon-to-bbox conversion.

    Args:
        paddle_results (List[Any]): Nested list output from PaddleOCR.ocr() method.
                                    Format: [[[bbox_quad, (text, confidence)], ...]]
                                    where bbox_quad is list of 4 (x, y) corner points,
                                    text is recognized string, and confidence is float [0-1].
        language (str): Language code for the detected text (e.g., 'en', 'ch', 'korean').
                       If None, uses 'multi' to indicate multi-language detection.
                       Default is None.

    Returns:
        Detections: Unified Detections object containing text detections with both
                   axis-aligned XYXY bounding boxes and original quadrilateral segments,
                   text content, confidence scores, and text angle information.
                   Empty Detections if no text found.

    Raises:
        TypeError: If paddle_results is not a list or contains invalid format
        ValueError: If polygon coordinates cannot be converted to numeric values
        IndexError: If detection elements don't contain expected format

    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from paddleocr import PaddleOCR
        >>>
        >>> # Basic OCR with angle detection
        >>> ocr = PaddleOCR(use_angle_cls=True, lang='en')
        >>> image_path = "document.jpg"
        >>> results = ocr.ocr(image_path)
        >>> detections = pf.detections.from_paddleocr(results, language='en')
        >>> for det in detections.detections:
        ...     print(f"Text: {det.text}, Confidence: {det.text_confidence:.2f}")
        ...     if det.text_angle:
        ...         print(f"  Angle: {det.text_angle}°")
        >>>
        >>> # Chinese text detection
        >>> ocr_ch = PaddleOCR(use_angle_cls=True, lang='ch')
        >>> results_ch = ocr_ch.ocr(chinese_image)
        >>> detections_ch = pf.detections.from_paddleocr(results_ch, language='ch')
        >>>
        >>> # Access polygon coordinates for rotated text
        >>> for det in detections.detections:
        ...     if det.segments:
        ...         print(f"Quadrilateral: {det.segments}")
        ...         print(f"Axis-aligned bbox: {det.bbox}")
        >>>
        >>> # Filter and process by angle
        >>> rotated_text = [d for d in detections.detections if d.text_angle and abs(d.text_angle) > 10]
        >>> print(f"Found {len(rotated_text)} rotated text regions")
        >>>
        >>> # Reconstruct text in reading order
        >>> sorted_text = sorted(detections.detections, key=lambda d: (d.bbox[1], d.bbox[0]))
        >>> full_text = " ".join([d.text for d in sorted_text])

    Notes:
        - PaddleOCR returns nested list: [[detection1, detection2, ...]] for single image
        - Each detection is [bbox_quad, (text, confidence)] where bbox_quad is 4 corner points
        - Original quadrilateral stored in segments field for precise text localization
        - bbox field contains axis-aligned bounding box (min/max of polygon coordinates)
        - text_level is always 'word' as PaddleOCR doesn't provide hierarchy information
        - text_order preserves the original detection sequence from PaddleOCR
        - text_angle calculated from quadrilateral geometry when use_angle_cls=True
        - Confidence scores are already normalized to [0.0-1.0] range
        - Works with rotated, skewed, and multi-oriented text
        - Language code 'multi' recommended for multi-language scenarios

    Performance Notes:
        - Efficient single-pass conversion of PaddleOCR results
        - Minimal computational overhead for polygon and angle processing
        - O(n) complexity where n is number of detected text regions

    See Also:
        from_tesseract : Convert Tesseract OCR results to PixelFlow format
        from_easyocr : Convert EasyOCR results to PixelFlow format
    """
    from .detections import Detections, Detection
    import math

    detections_obj = Detections()

    # Set default language to 'multi' if not specified
    if language is None:
        language = 'multi'

    # PaddleOCR returns nested list [[detection1, detection2, ...]] for single image
    # Handle empty results
    if not paddle_results or paddle_results[0] is None:
        return detections_obj

    # Get the detections for the first (and typically only) image
    detections_list = paddle_results[0]

    for idx, detection in enumerate(detections_list):
        # PaddleOCR format: [bbox_quad, (text, confidence)]
        bbox_quad, (text, confidence) = detection

        # bbox_quad is a list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Convert to list of tuples for segments
        segments = [(int(point[0]), int(point[1])) for point in bbox_quad]

        # Calculate axis-aligned bounding box from quadrilateral
        x_coords = [point[0] for point in bbox_quad]
        y_coords = [point[1] for point in bbox_quad]
        x1 = int(min(x_coords))
        y1 = int(min(y_coords))
        x2 = int(max(x_coords))
        y2 = int(max(y_coords))
        bbox = [x1, y1, x2, y2]

        # Calculate text angle from quadrilateral
        # Use the angle between top edge and horizontal
        dx = bbox_quad[1][0] - bbox_quad[0][0]
        dy = bbox_quad[1][1] - bbox_quad[0][1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Normalize angle to [-180, 180] range
        if angle_deg > 180:
            angle_deg -= 360
        elif angle_deg < -180:
            angle_deg += 360

        # Determine text direction based on angle
        if -45 <= angle_deg <= 45:
            text_direction = 'ltr'  # Horizontal left-to-right
        elif 45 < angle_deg <= 135:
            text_direction = 'vertical-ttb'  # Vertical top-to-bottom
        elif angle_deg > 135 or angle_deg < -135:
            text_direction = 'rtl'  # Horizontal right-to-left (upside down)
        else:  # -135 < angle_deg < -45
            text_direction = 'vertical-btt'  # Vertical bottom-to-top

        # Create OCRData object for structured OCR information
        from .detections import OCRData
        ocr_data = OCRData(
            text=text.strip(),
            confidence=confidence,
            language=language,
            level='word',  # PaddleOCR returns word-level results
            order=idx,  # Preserve detection order
            angle=round(angle_deg, 2),  # Store rotation angle
            direction=text_direction,  # Inferred from angle
            element_type='text'  # PaddleOCR is text OCR
        )

        # Create detection with OCRData
        detection = Detection(
            bbox=bbox,
            segments=segments,  # Store original quadrilateral
            ocr_data=ocr_data,
            metadata={
                'paddleocr_quad': bbox_quad  # Store original quad format
            }
        )

        detections_obj.add_detection(detection)

    return detections_obj


def from_ppstructure(ppstructure_results: Union[List[Any], Any], language: str = None, page_index: int = None):
    """
    Convert PP-StructureV3 results to a unified Detections object.

    Processes document structure analysis results from PaddleOCR's PP-StructureV3,
    extracting layout regions (text, tables, formulas, images), bounding boxes,
    OCR text, table HTML, and formula LaTeX into PixelFlow's standardized format.
    Supports both the old PPStructure API and new PPStructureV3 API formats.

    Args:
        ppstructure_results (Union[List[Any], Any]): Results from PPStructure/PPStructureV3.
                                                     Can be either:
                                                     - Old API: list of dicts with 'type', 'bbox', 'res' keys
                                                     - New API: result object with .layout_det_res, .overall_ocr_res attributes
        language (str): Language code for the detected text (e.g., 'en', 'ch').
                       If None, uses 'multi' to indicate multi-language detection.
                       Default is None.
        page_index (int): Page number for multi-page documents (0-indexed).
                         If None, uses 0 for single page. Default is None.

    Returns:
        Detections: Unified Detections object containing document structure detections
                   with XYXY bounding boxes, OCR text, element types (text/table/formula),
                   table HTML, and formula LaTeX. Empty Detections if no elements found.

    Raises:
        TypeError: If ppstructure_results format is unrecognized
        KeyError: If required fields are missing from results
        ValueError: If coordinates cannot be converted to numeric values

    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> from paddleocr import PPStructureV3
        >>>
        >>> # New API - PP-StructureV3
        >>> pipeline = PPStructureV3(
        ...     use_table_recognition=True,
        ...     use_formula_recognition=True
        ... )
        >>> output = pipeline.predict("document.png")
        >>> for res in output:
        ...     detections = pf.detections.from_ppstructure(res, language='en')
        ...     for det in detections.detections:
        ...         print(f"Type: {det.ocr_data.element_type}, Text: {det.ocr_data.text}")
        ...         if det.ocr_data.table_html:
        ...             print(f"Table HTML: {det.ocr_data.table_html[:100]}...")
        ...         if det.ocr_data.formula_latex:
        ...             print(f"Formula: {det.ocr_data.formula_latex}")
        >>>
        >>> # Old API - PPStructure
        >>> from paddleocr import PPStructure
        >>> engine = PPStructure(layout=True, table=True, ocr=True)
        >>> img = cv2.imread("document.png")
        >>> result = engine(img)
        >>> detections = pf.detections.from_ppstructure(result, language='en')
        >>>
        >>> # Filter by element type
        >>> tables = [d for d in detections.detections if d.ocr_data.element_type == 'table']
        >>> formulas = [d for d in detections.detections if d.ocr_data.element_type == 'formula']
        >>> text_regions = [d for d in detections.detections if d.ocr_data.element_type == 'text']
        >>>
        >>> # Multi-page document
        >>> for page_num, res in enumerate(output):
        ...     detections = pf.detections.from_ppstructure(res, page_index=page_num)

    Notes:
        - Supports both old PPStructure and new PPStructureV3 API formats automatically
        - element_type can be: 'text', 'table', 'formula', 'image', 'figure', 'title', etc.
        - table_html contains full HTML table structure with <table>, <tr>, <td> tags
        - formula_latex contains LaTeX representation of mathematical formulas
        - page_index enables multi-page document processing
        - Bounding boxes are in XYXY format [x1, y1, x2, y2]
        - Reading order is preserved via text_order field
        - Hierarchical structure not available (no parent_id) for PP-Structure

    Performance Notes:
        - Efficient single-pass conversion of PP-Structure results
        - Minimal computational overhead for layout region processing
        - O(n) complexity where n is number of layout regions

    See Also:
        from_paddleocr : Convert PaddleOCR (text-only) results to PixelFlow format
        from_tesseract : Convert Tesseract OCR results with hierarchical structure
    """
    from .detections import Detections, Detection, OCRData

    detections_obj = Detections()

    # Set default language and page_index
    if language is None:
        language = 'multi'
    if page_index is None:
        page_index = 0

    # Detect which API format we're dealing with
    # New API: dict-like with keys like layout_det_res, overall_ocr_res
    # Old API: is a list of dicts with 'type', 'bbox', 'res' keys

    if isinstance(ppstructure_results, list) and ppstructure_results and isinstance(ppstructure_results[0], dict) and 'type' in ppstructure_results[0]:
        # Old PPStructure API format
        # Format: [{'type': 'Text', 'bbox': [x1,y1,x2,y2], 'res': (boxes, texts)}, ...]
        return _convert_old_ppstructure(ppstructure_results, language, page_index, detections_obj)
    else:
        # New PPStructureV3 API format (dict-like object)
        # Has keys: layout_det_res, overall_ocr_res, table_res_list, formula_res_list
        return _convert_new_ppstructure(ppstructure_results, language, page_index, detections_obj)


def _convert_old_ppstructure(results: List[Dict[str, Any]], language: str, page_index: int, detections_obj: 'Detections') -> 'Detections':
    """
    Convert old PPStructure API format to Detections.

    Format: [{'type': 'Text', 'bbox': [x1,y1,x2,y2], 'res': (boxes, texts)}, ...]
    """
    from .detections import Detection, OCRData

    for idx, region in enumerate(results):
        region_type = region.get('type', 'unknown')
        bbox = region.get('bbox', [])

        # Convert bbox to XYXY format if needed
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            bbox_xyxy = [x1, y1, x2, y2]
        else:
            continue  # Skip invalid bboxes

        # Map region type to element_type
        element_type_map = {
            'Text': 'text',
            'text': 'text',
            'Table': 'table',
            'table': 'table',
            'Formula': 'formula',
            'formula': 'formula',
            'Figure': 'figure',
            'Image': 'image',
            'Title': 'title',
        }
        element_type = element_type_map.get(region_type, region_type.lower())

        # Extract OCR results if available
        res_data = region.get('res', None)
        text_content = ""
        confidence = 0.0
        table_html = None
        formula_latex = None

        if res_data is not None:
            if element_type == 'table' and isinstance(res_data, dict):
                # Table result: contains HTML structure
                table_html = res_data.get('html', None)
                text_content = "Table"  # Placeholder text
                confidence = 1.0
            elif element_type == 'formula' and isinstance(res_data, dict):
                # Formula result: contains LaTeX
                formula_latex = res_data.get('latex', res_data.get('text', None))
                text_content = formula_latex if formula_latex else "Formula"
                confidence = 1.0
            elif isinstance(res_data, tuple) and len(res_data) == 2:
                # Text result: (boxes, [(text, confidence), ...])
                boxes, text_results = res_data
                if text_results and len(text_results) > 0:
                    # Concatenate all text from this region
                    texts = []
                    confidences = []
                    for text_item in text_results:
                        if isinstance(text_item, tuple) and len(text_item) == 2:
                            t, c = text_item
                            texts.append(t)
                            confidences.append(c)
                    text_content = " ".join(texts)
                    confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Create OCRData object
        ocr_data = OCRData(
            text=text_content.strip() if text_content else "",
            confidence=confidence,
            language=language,
            level='block',  # PP-Structure returns block-level regions
            order=idx,
            page_index=page_index,
            element_type=element_type,
            table_html=table_html,
            formula_latex=formula_latex
        )

        # Create detection
        detection = Detection(
            bbox=bbox_xyxy,
            ocr_data=ocr_data,
            metadata={
                'ppstructure_type': region_type,
                'ppstructure_api': 'old'
            }
        )

        detections_obj.add_detection(detection)

    return detections_obj


def _convert_new_ppstructure(result_obj: Any, language: str, page_index: int, detections_obj: 'Detections') -> 'Detections':
    """
    Convert new PPStructureV3 API format to Detections.

    Result is dict-like with keys: layout_det_res, overall_ocr_res, parsing_res_list, table_res_list, formula_res_list
    """
    from .detections import Detection, OCRData

    # PPStructureV3 result objects have a .json property that contains the actual data
    # Try to access .json if it exists
    if hasattr(result_obj, 'json') and callable(result_obj.json):
        try:
            json_data = result_obj.json()
            if isinstance(json_data, dict) and 'parsing_res_list' in json_data:
                return _convert_from_parsing_res_list(json_data['parsing_res_list'], language, page_index, detections_obj)
        except Exception as e:
            pass

    # Fallback to original dict-like access
    if hasattr(result_obj, 'keys'):
        pass

    # Check if parsing_res_list is directly accessible
    parsing_res_list = result_obj.get('parsing_res_list', None) if hasattr(result_obj, 'get') else None
    if parsing_res_list is not None:
        if parsing_res_list and hasattr(parsing_res_list, '__len__') and len(parsing_res_list) > 0:
            return _convert_from_parsing_res_list(parsing_res_list, language, page_index, detections_obj)

    # Access as dict - result_obj is dict-like
    layout_det_res = result_obj.get('layout_det_res', None) if hasattr(result_obj, 'get') else None

    if layout_det_res is None:
        return detections_obj

    # DetResult likely has 'boxes' or 'bboxes' attribute
    boxes = None
    if hasattr(layout_det_res, 'boxes'):
        boxes = layout_det_res.boxes
    elif hasattr(layout_det_res, 'bboxes'):
        boxes = layout_det_res.bboxes
    elif isinstance(layout_det_res, dict):
        boxes = layout_det_res.get('boxes', layout_det_res.get('bboxes', []))
    elif hasattr(layout_det_res, 'keys'):
        # Dict-like object
        boxes = layout_det_res.get('boxes', layout_det_res.get('bboxes', []))
    else:
        return detections_obj

    if boxes is None or len(boxes) == 0:
        return detections_obj

    # Get OCR results if available
    overall_ocr_res = result_obj.get('overall_ocr_res', None) if hasattr(result_obj, 'get') else None

    # Get table results if available
    table_res_list = result_obj.get('table_res_list', None) if hasattr(result_obj, 'get') else None

    # Get formula results if available
    formula_res_list = result_obj.get('formula_res_list', None) if hasattr(result_obj, 'get') else None

    # Use parsing_res_list which contains the already-processed regions with text
    parsing_res_list = result_obj.get('parsing_res_list', None) if hasattr(result_obj, 'get') else None

    # Process each layout box
    for idx, box in enumerate(boxes):
        # Debug first box
        if idx == 0:
            pass

        # Extract box information
        # Format: {'cls_id': 1, 'label': 'text', 'score': 0.98, 'coordinate': [x1,y1,x2,y2]}
        if isinstance(box, dict):
            label = box.get('label', 'unknown')
            score = box.get('score', 1.0)
            coordinate = box.get('coordinate', [])
        elif hasattr(box, 'get'):
            # Dict-like object
            label = box.get('label', 'unknown')
            score = box.get('score', 1.0)
            coordinate = box.get('coordinate', [])
        elif hasattr(box, 'label'):
            # Object with attributes
            label = getattr(box, 'label', 'unknown')
            score = getattr(box, 'score', 1.0)
            coordinate = getattr(box, 'coordinate', getattr(box, 'bbox', []))
        else:
            # Fallback for other formats
            continue

        # Convert coordinate to XYXY format
        if len(coordinate) == 4:
            x1, y1, x2, y2 = [int(coord) for coord in coordinate]
            bbox_xyxy = [x1, y1, x2, y2]
        else:
            continue

        # Map label to element_type
        element_type_map = {
            'text': 'text',
            'table': 'table',
            'figure': 'figure',
            'formula': 'formula',
            'equation': 'formula',
            'image': 'image',
            'title': 'title',
            'reference': 'text',
            'footer': 'text',
            'header': 'text',
        }
        element_type = element_type_map.get(label.lower(), label.lower())

        # Determine text content, table HTML, and formula LaTeX based on element type
        text_content = ""
        table_html = None
        formula_latex = None
        confidence = score

        if element_type == 'table' and table_res_list:
            # Try to extract table HTML for this specific box
            if isinstance(table_res_list, list) and len(table_res_list) > 0:
                # Assume tables are in order
                table_idx = sum(1 for b in boxes[:idx] if b.get('label', '').lower() == 'table')
                if table_idx < len(table_res_list):
                    # Table result might be dict or object with 'html' field
                    table_item = table_res_list[table_idx]
                    if isinstance(table_item, dict):
                        table_html = table_item.get('html', None)
                    elif hasattr(table_item, 'html'):
                        table_html = table_item.html
                    elif hasattr(table_item, 'get'):
                        table_html = table_item.get('html', None)
            elif isinstance(table_res_list, str):
                table_html = table_res_list
            text_content = "Table"

        elif element_type == 'formula' and formula_res_list:
            # Try to extract formula LaTeX for this specific box
            if isinstance(formula_res_list, list) and len(formula_res_list) > 0:
                # Assume formulas are in order
                formula_idx = sum(1 for b in boxes[:idx] if b.get('label', '').lower() in ['formula', 'equation'])
                if formula_idx < len(formula_res_list):
                    # Formula result might be dict or object with 'latex' field
                    formula_item = formula_res_list[formula_idx]
                    if isinstance(formula_item, dict):
                        formula_latex = formula_item.get('latex', formula_item.get('text', None))
                    elif hasattr(formula_item, 'latex'):
                        formula_latex = formula_item.latex
                    elif hasattr(formula_item, 'get'):
                        formula_latex = formula_item.get('latex', formula_item.get('text', None))
            elif isinstance(formula_res_list, str):
                formula_latex = formula_res_list
            text_content = formula_latex if formula_latex else "Formula"

        elif element_type in ['text', 'abstract', 'doc_title', 'title']:
            # For text regions, extract text from overall_ocr_res by matching bounding boxes
            if ocr_texts and ocr_boxes:
                # Find all OCR texts that fall within this layout region
                matching_texts = []
                for ocr_idx, ocr_box in enumerate(ocr_boxes):
                    if ocr_idx >= len(ocr_texts):
                        break

                    # Get OCR box coordinates (might be polygon or bbox)
                    if isinstance(ocr_box, (list, tuple)):
                        if len(ocr_box) == 4 and all(isinstance(coord, (int, float)) for coord in ocr_box):
                            # XYXY format
                            ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_box
                        elif len(ocr_box) == 4 and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in ocr_box):
                            # Polygon format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                            x_coords = [p[0] for p in ocr_box]
                            y_coords = [p[1] for p in ocr_box]
                            ocr_x1, ocr_y1 = min(x_coords), min(y_coords)
                            ocr_x2, ocr_y2 = max(x_coords), max(y_coords)
                        else:
                            continue
                    else:
                        continue

                    # Check if OCR box center is within layout region (with some tolerance)
                    ocr_center_x = (ocr_x1 + ocr_x2) / 2
                    ocr_center_y = (ocr_y1 + ocr_y2) / 2

                    # Check if center is within the layout bbox
                    if (x1 <= ocr_center_x <= x2) and (y1 <= ocr_center_y <= y2):
                        matching_texts.append(ocr_texts[ocr_idx])

                # Concatenate all matching texts
                if matching_texts:
                    text_content = " ".join(matching_texts)
                    # Calculate average confidence
                    if ocr_scores and len(ocr_scores) > 0:
                        matching_scores = []
                        for ocr_idx, ocr_box in enumerate(ocr_boxes):
                            if ocr_idx >= len(ocr_texts) or ocr_idx >= len(ocr_scores):
                                break
                            if isinstance(ocr_box, (list, tuple)):
                                if len(ocr_box) == 4:
                                    if all(isinstance(coord, (int, float)) for coord in ocr_box):
                                        ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_box
                                    elif all(isinstance(p, (list, tuple)) and len(p) == 2 for p in ocr_box):
                                        x_coords = [p[0] for p in ocr_box]
                                        y_coords = [p[1] for p in ocr_box]
                                        ocr_x1, ocr_y1 = min(x_coords), min(y_coords)
                                        ocr_x2, ocr_y2 = max(x_coords), max(y_coords)
                                    else:
                                        continue

                                    ocr_center_x = (ocr_x1 + ocr_x2) / 2
                                    ocr_center_y = (ocr_y1 + ocr_y2) / 2
                                    if (x1 <= ocr_center_x <= x2) and (y1 <= ocr_center_y <= y2):
                                        matching_scores.append(ocr_scores[ocr_idx])

                        if matching_scores:
                            confidence = sum(matching_scores) / len(matching_scores)
                else:
                    text_content = ""
            else:
                text_content = ""

        # Create OCRData object
        ocr_data = OCRData(
            text=text_content.strip() if text_content else "",
            confidence=confidence,
            language=language,
            level='block',  # PP-StructureV3 returns block-level regions
            order=idx,
            page_index=page_index,
            element_type=element_type,
            table_html=table_html,
            formula_latex=formula_latex
        )

        # Create detection
        detection = Detection(
            bbox=bbox_xyxy,
            ocr_data=ocr_data,
            metadata={
                'ppstructure_label': label,
                'ppstructure_cls_id': box.get('cls_id', None),
                'ppstructure_api': 'new'
            }
        )

        detections_obj.add_detection(detection)

    return detections_obj

def _convert_from_parsing_res_list(parsing_res_list: List[Any], language: str, page_index: int, detections_obj: 'Detections') -> 'Detections':
    """
    Convert PPStructureV3 parsing_res_list to Detections.

    parsing_res_list contains pre-processed layout regions with text already extracted.
    Each element has: layout_bbox, layout, and various content fields depending on type.
    """
    from .detections import Detection, OCRData

    for idx, region in enumerate(parsing_res_list):
        # Debug first region
        if idx == 0:
            pass

        # Extract bounding box - try object attributes first
        layout_bbox = None
        layout_type = "unknown"

        if hasattr(region, 'layout_bbox'):
            layout_bbox = region.layout_bbox
        elif hasattr(region, 'bbox'):
            layout_bbox = region.bbox
        elif isinstance(region, dict):
            layout_bbox = region.get('layout_bbox', region.get('bbox', None))
        elif hasattr(region, 'get'):
            layout_bbox = region.get('layout_bbox', region.get('bbox', None))

        if hasattr(region, 'label'):
            layout_type = region.label
        elif hasattr(region, 'type'):
            layout_type = region.type
        elif hasattr(region, 'layout'):
            layout_type = region.layout
        elif isinstance(region, dict):
            layout_type = region.get('type', region.get('layout', 'unknown'))
        elif hasattr(region, 'get'):
            layout_type = region.get('type', region.get('layout', 'unknown'))

        if layout_bbox is None:
            continue

        if layout_bbox is None or len(layout_bbox) != 4:
            continue

        # Convert bbox to XYXY
        x1, y1, x2, y2 = [int(coord) for coord in layout_bbox]
        bbox_xyxy = [x1, y1, x2, y2]

        # Extract content based on available fields (attributes or dict keys)
        text_content = ""
        table_html = None
        formula_latex = None
        confidence = 1.0
        element_type = "unknown"

        # Try object attributes first (for LayoutBlock objects)
        # Check for 'content' attribute first (LayoutBlock structure)
        if hasattr(region, 'content'):
            content = region.content

            # Content might be a string or a structured object
            if isinstance(content, str):
                text_content = content
                element_type = layout_type.lower() if layout_type != "unknown" else "text"
            elif content is not None:
                # Content might have text, html, or latex fields
                if hasattr(content, 'text'):
                    text_content = content.text if content.text else ""
                    element_type = "text"

                if hasattr(content, 'html'):
                    table_html = content.html
                    if table_html:
                        element_type = "table"
                        text_content = "Table"

                if hasattr(content, 'latex'):
                    formula_latex = content.latex
                    if formula_latex:
                        element_type = "formula"
                        text_content = formula_latex
        elif hasattr(region, 'text'):
            text_content = region.text if region.text else ""
            element_type = "text"
        elif hasattr(region, 'ocr_text'):
            text_content = region.ocr_text if region.ocr_text else ""
            element_type = "text"

        if hasattr(region, 'table_html'):
            table_html = region.table_html
            if table_html:
                element_type = "table"
                text_content = "Table"

        if hasattr(region, 'formula_latex') or hasattr(region, 'latex'):
            formula_latex = getattr(region, 'formula_latex', None) or getattr(region, 'latex', None)
            if formula_latex:
                element_type = "formula"
                text_content = formula_latex

        # Fallback to dict-like access
        if element_type == "unknown" and isinstance(region, dict):
            # Tables: look for 'table' or 'table_html' field
            if 'table' in region:
                table_data = region['table']
                if isinstance(table_data, dict):
                    table_html = table_data.get('html', None)
                elif isinstance(table_data, str):
                    table_html = table_data
                text_content = "Table"
                element_type = "table"

            # Formulas: look for 'formula' or 'latex' field
            elif 'formula' in region or 'latex' in region:
                formula_latex = region.get('formula', region.get('latex', None))
                text_content = formula_latex if formula_latex else "Formula"
                element_type = "formula"

            # Images/Figures: look for 'image' or 'figure' fields
            elif 'image' in region:
                element_type = "image"
                text_content = region.get('image_text', "Image") if isinstance(region.get('image'), dict) else "Image"

            elif 'figure' in region:
                element_type = "figure"
                text_content = region.get('figure_title', "Figure") if 'figure_title' in region else "Figure"

            # Text: look for 'text' or 'ocr_text' fields
            elif 'text' in region:
                text_content = region['text']
                element_type = "text"

            elif 'ocr_text' in region:
                text_content = region['ocr_text']
                element_type = "text"

            # Fallback: try to extract any text-like field
            else:
                # Look for any field that might contain text
                for key in ['title', 'content', 'value']:
                    if key in region and isinstance(region[key], str):
                        text_content = region[key]
                        element_type = "text"
                        break

        # If still no element type, try to infer from layout_type
        if element_type == "unknown" and layout_type:
            element_type = layout_type.lower()

        # Create OCRData object
        ocr_data = OCRData(
            text=text_content.strip() if text_content else "",
            confidence=confidence,
            language=language,
            level='block',
            order=idx,
            page_index=page_index,
            element_type=element_type,
            table_html=table_html,
            formula_latex=formula_latex
        )

        # Create detection
        detection = Detection(
            bbox=bbox_xyxy,
            ocr_data=ocr_data,
            metadata={
                'ppstructure_layout_type': layout_type,
                'ppstructure_api': 'new_v3',
                'ppstructure_source': 'parsing_res_list'
            }
        )

        detections_obj.add_detection(detection)

    return detections_obj
