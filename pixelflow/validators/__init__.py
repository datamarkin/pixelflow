def validate_bbox(bbox):
    """
    Ensure that bbox contains exactly 4 integer values.
    If bbox is not valid, return a default bbox.
    """
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            # Convert all elements to integers if they aren't already
            bbox = [int(x) for x in bbox]
            return bbox
        except (ValueError, TypeError):
            # If conversion fails, handle the error
            print(f"Invalid bbox values: {bbox}")
    else:
        print(f"Invalid bbox length: {bbox}")

    # Return a default bbox if validation fails
    return None
