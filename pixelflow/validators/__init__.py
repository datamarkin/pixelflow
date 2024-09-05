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


def validate_mask(mask: list) -> list:
    """
    Validates and converts the values in the mask data (list of lists) to integers.
    Converts floats to integers by rounding them to the nearest integer.

    Args:
        mask (list): The mask data to be validated and converted, expected to be a list of lists of numbers.

    Returns:
        list: The validated mask data with all values as integers.

    Raises:
        ValueError: If a value in the mask is not an integer or float.
    """
    validated_mask = []
    for sublist in mask:
        validated_sublist = []
        for value in sublist:
            # Check if the value is an integer or float
            if isinstance(value, int):
                validated_sublist.append(value)
            elif isinstance(value, float):
                # Round the float to the nearest integer
                validated_sublist.append(round(value))
            else:
                raise ValueError(f"Invalid type found: {value} (not int or float)")
        validated_mask.append(validated_sublist)

    return validated_mask


def round_to_decimal(value, decimals=3):
    """
    Rounds the given value to the specified number of decimal places.

    Args:
        value (float or None): The value to be rounded.
        decimals (int): The number of decimal places (default is 3).

    Returns:
        float or None: The rounded value or None if the input is None.
    """
    if value is not None:
        return round(float(value), decimals)
    return None
