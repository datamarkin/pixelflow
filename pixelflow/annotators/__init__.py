"""
PixelFlow Annotators Module

Provides high-level annotation functions for computer vision visualizations.
All functions follow a consistent API pattern: func(image, results, **kwargs) -> image

This module uses a functional approach for simplicity and performance,
allowing easy chaining of annotation operations.
"""

# Import implemented annotator functions
from .anchors import anchors
from .blur import blur
from .box import box
from .label import label
from .crossing import crossings
from .mask import mask
from .oval import oval
from .pixelate import pixelate
from .polygon import polygon
from .zones import zones

# TODO: Import these once implemented
from .filled_box import filled_box
# from .dot import dot
# from .fps_counter import fps_counter
from .grid_overlay import grid_overlay
# from .heatmap import heatmap
# from .keypoint import keypoint
# from .keypoint_skeleton import keypoint_skeleton
# from .motion_dots import motion_dots
# from .motion_trails import motion_trails
# from .scale_bar import scale_bar

# Import utility functions if needed externally
from .utils import _get_adaptive_params, ADAPTIVE_SCALE_MULTIPLIER


# Define public API
__all__ = [
    # Core annotators
    'anchors',
    'box',
    'label',
    'mask',
    
    # Privacy annotators
    'blur',
    'pixelate',
    
    # Shape annotators
    'oval',
    'polygon',
    
    # Zone annotators
    'zones',
    'crossings'
]