# RGB format — all color tuples are (R, G, B)
# Scientifically chosen for maximum perceptual distinction
# Ordered to maximize distinction between adjacent colors

DEFAULT_PALETTE = [
    (14, 127, 255),  # Vivid Orange
    (180, 119, 31),  # Strong Blue
    (44, 160, 44),  # Green
    (40, 39, 214),  # Vivid Red
    (189, 103, 148),  # Purple
    (75, 86, 140),  # Brown
    (194, 119, 227),  # Pink
    (127, 127, 127),  # Gray
    (34, 189, 188),  # Olive
    (207, 190, 23),  # Cyan
    (120, 187, 255),  # Light Orange
    (232, 199, 174),  # Light Blue
    (138, 223, 152),  # Light Green
    (150, 152, 255),  # Light Red
    (213, 176, 197),  # Light Purple
    (148, 156, 196),  # Light Brown
    (210, 182, 247),  # Light Pink
    (199, 199, 199),  # Light Gray
    (141, 219, 219),  # Light Olive
    (229, 218, 158),  # Light Cyan
]

# High contrast palette for better visibility
VIBRANT_PALETTE = [
    (255, 0, 0),  # Pure Red
    (0, 255, 0),  # Pure Green
    (0, 0, 255),  # Pure Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (0, 255, 128),  # Lime
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Maroon
    (0, 128, 128),  # Navy
    (208, 224, 64),  # Turquoise
    (114, 128, 250),  # Salmon
    (0, 215, 255),  # Gold
    (180, 105, 255),  # Hot Pink
    (255, 191, 0),  # Deep Sky Blue
    (50, 205, 50),  # Lime Green
    (147, 20, 255),  # Deep Pink
    (0, 140, 255),  # Dark Orange
]

# Soft pastel colors for subtle annotations
PASTEL_PALETTE = [
    (255, 195, 203),  # Pastel Red
    (203, 255, 195),  # Pastel Green
    (195, 203, 255),  # Pastel Blue
    (255, 255, 195),  # Pastel Yellow
    (255, 195, 255),  # Pastel Magenta
    (195, 255, 255),  # Pastel Cyan
    (255, 225, 195),  # Pastel Orange
    (225, 195, 255),  # Pastel Purple
    (195, 255, 225),  # Pastel Lime
    (248, 234, 214),  # Pastel Sky
    (216, 219, 250),  # Pastel Rose
    (208, 235, 253),  # Pastel Peach
    (210, 234, 222),  # Pastel Mint
    (255, 224, 239),  # Pastel Lavender
    (215, 245, 255),  # Pastel Cream
    (241, 244, 230),  # Pastel Teal
    (213, 239, 255),  # Pastel Apricot
    (252, 238, 241),  # Pastel Periwinkle
    (230, 250, 255),  # Pastel Beige
    (240, 255, 240),  # Pastel Honeydew
]

# Palette dictionary for easy selection
PALETTES = {
    'default': DEFAULT_PALETTE,
    'vibrant': VIBRANT_PALETTE,
    'pastel': PASTEL_PALETTE
}


def _get_color_for_prediction(prediction, colors_override=None, palette='default'):
    """
    Ultra-fast color assignment for predictions.

    Args:
        prediction: Object with class_id attribute
        colors_override: Optional list of RGB color tuples (highest priority)
        palette: Palette name ('default', 'vibrant', 'pastel') - ignored if colors_override is provided

    Returns:
        RGB color tuple for the prediction
    """
    if colors_override:
        return colors_override[prediction.class_id % len(colors_override)]

    active = PALETTES.get(palette, DEFAULT_PALETTE)
    return active[prediction.class_id % len(active)]
