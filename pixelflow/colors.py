# BGR format for OpenCV compatibility
# Scientifically chosen for maximum perceptual distinction
# Ordered to maximize distinction between adjacent colors

DEFAULT_PALETTE = [
    (255, 127, 14),  # Vivid Orange
    (31, 119, 180),  # Strong Blue
    (44, 160, 44),  # Green
    (214, 39, 40),  # Vivid Red
    (148, 103, 189),  # Purple
    (140, 86, 75),  # Brown
    (227, 119, 194),  # Pink
    (127, 127, 127),  # Gray
    (188, 189, 34),  # Olive
    (23, 190, 207),  # Cyan
    (255, 187, 120),  # Light Orange
    (174, 199, 232),  # Light Blue
    (152, 223, 138),  # Light Green
    (255, 152, 150),  # Light Red
    (197, 176, 213),  # Light Purple
    (196, 156, 148),  # Light Brown
    (247, 182, 210),  # Light Pink
    (199, 199, 199),  # Light Gray
    (219, 219, 141),  # Light Olive
    (158, 218, 229),  # Light Cyan
]

# High contrast palette for better visibility
VIBRANT_PALETTE = [
    (0, 0, 255),  # Pure Red
    (0, 255, 0),  # Pure Green
    (255, 0, 0),  # Pure Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (0, 128, 255),  # Orange
    (255, 0, 128),  # Purple
    (128, 255, 0),  # Lime
    (0, 128, 128),  # Olive
    (128, 0, 128),  # Maroon
    (128, 128, 0),  # Navy
    (64, 224, 208),  # Turquoise
    (250, 128, 114),  # Salmon
    (255, 215, 0),  # Gold
    (255, 105, 180),  # Hot Pink
    (0, 191, 255),  # Deep Sky Blue
    (50, 205, 50),  # Lime Green
    (255, 20, 147),  # Deep Pink
    (255, 140, 0),  # Dark Orange
]

# Soft pastel colors for subtle annotations
PASTEL_PALETTE = [
    (203, 195, 255),  # Pastel Red
    (195, 255, 203),  # Pastel Green
    (255, 203, 195),  # Pastel Blue
    (195, 255, 255),  # Pastel Yellow
    (255, 195, 255),  # Pastel Magenta
    (255, 255, 195),  # Pastel Cyan
    (195, 225, 255),  # Pastel Orange
    (255, 195, 225),  # Pastel Purple
    (225, 255, 195),  # Pastel Lime
    (214, 234, 248),  # Pastel Sky
    (250, 219, 216),  # Pastel Rose
    (253, 235, 208),  # Pastel Peach
    (222, 234, 210),  # Pastel Mint
    (239, 224, 255),  # Pastel Lavender
    (255, 245, 215),  # Pastel Cream
    (230, 244, 241),  # Pastel Teal
    (255, 239, 213),  # Pastel Apricot
    (241, 238, 252),  # Pastel Periwinkle
    (255, 250, 230),  # Pastel Beige
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
        colors_override: Optional list of BGR color tuples (highest priority)
        palette: Palette name ('default', 'vibrant', 'pastel') - ignored if colors_override is provided

    Returns:
        BGR color tuple for the prediction
    """
    if colors_override:
        return colors_override[prediction.class_id % len(colors_override)]

    active = PALETTES.get(palette, DEFAULT_PALETTE)
    return active[prediction.class_id % len(active)]
