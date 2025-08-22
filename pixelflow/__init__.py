__version__ = "0.1.0"
__author__ = "Datamarkin"

# Import core modules
from . import draw
from . import results
from . import video
from . import annotate
from . import colors

# Import specific functions for top-level access
from .video import lazy_frame_generator

# Define the public API
__all__ = ["draw", "annotate", "results", "lazy_frame_generator", "__version__", "__author__"]
