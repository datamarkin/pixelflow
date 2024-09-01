def hex_to_bgr(hex_color):
    """
    Convert a HEX color to BGR format used by OpenCV.
    Example: hex_color = "#ff0000" -> BGR = (0, 0, 255)
    """
    hex_color = hex_color.lstrip('#')
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
    return bgr
