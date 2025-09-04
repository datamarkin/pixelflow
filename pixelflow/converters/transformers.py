"""
HuggingFace Transformers results converter for PixelFlow

Converts HuggingFace Transformers detection results to PixelFlow's unified Detections format.
"""



def from_transformers(transformers_results):
    """
    Converts HuggingFace Transformers object detection results to a Detections object.
    
    Supports models like DETR, YOLOS, OWLv2, etc.
    
    Args:
        transformers_results: Results from HuggingFace transformers object detection pipeline
        
    Returns:
        Detections: A unified Detections object containing detections.
    """
    from pixelflow.detections import Detections, Detection
    detections_obj = Detections()
    
    # Handle different result formats
    if isinstance(transformers_results, list):
        # Pipeline returns a list of detections
        for result in transformers_results:
            bbox = result.get('box', {})
            # Convert from format {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
            bbox_list = [bbox.get('xmin', 0), bbox.get('ymin', 0), 
                        bbox.get('xmax', 0), bbox.get('ymax', 0)]
            
            detection = Detection(
                bbox=bbox_list,
                class_name=result.get('label', ''),
                confidence=result.get('score', 0.0)
            )
            detections_obj.add_detection(detection)
    
    elif isinstance(transformers_results, dict):
        # Raw model output format
        if 'pred_boxes' in transformers_results and 'scores' in transformers_results:
            boxes = transformers_results['pred_boxes']
            scores = transformers_results['scores'] 
            labels = transformers_results.get('pred_classes', transformers_results.get('labels', []))
            
            for i, (box, score) in enumerate(zip(boxes, scores)):
                # Convert from center format to xyxy if needed
                if hasattr(box, 'tolist'):
                    bbox = box.tolist()
                else:
                    bbox = list(box)
                    
                class_id = labels[i] if i < len(labels) else None
                if hasattr(class_id, 'item'):
                    class_id = class_id.item()
                    
                detection = Detection(
                    bbox=bbox,
                    class_id=class_id,
                    confidence=float(score) if hasattr(score, 'item') else score
                )
                detections_obj.add_detection(detection)
    
    return detections_obj