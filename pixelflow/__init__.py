__version__ = "0.4.0"
__author__ = "Datamarkin"

# Import core modules
from . import media
from . import annotators as annotate
from . import transforms as transform
from . import colors
from . import slicer
from . import smoother
from . import timer

# Result types and their converters. Every model output PixelFlow understands becomes
# one of two things: Detections for anything that localises, Classifications for
# anything that only names. They are peers, not variants of each other.
#
# Converters are flat -- pf.from_ultralytics(...) -- and each subpackage owns its own
# export list, so a new one is declared once rather than re-typed here. Detection is
# the unmarked case because almost every model that produces a container localises
# something; only classification needs the suffix.
from .detections import *
from .detections import __all__ as _DETECTION_EXPORTS
from .classifications import *
from .classifications import __all__ as _CLASSIFICATION_EXPORTS

# Import specific functions for top-level access
from .media import (
    DisplayExit, VideoReader, CameraStream, VideoWriter,
    read_image, read_video, display_video, display_image, save_image,
    close_display, to_pil, from_pil,
)
from .zones import Zones
from .crossings import Crossings
from .slicer import SlicedInference, auto_slice_size
from .buffer import Buffer
from .smoother import smooth
from .timer import TimeTracker

# ByteTrack is the only thing in PixelFlow that needs SciPy, and SciPy is most of what importing
# PixelFlow costs: 436 ms of 631 ms, from scipy.linalg in the Kalman filter and scipy.optimize in
# the assignment step. Every caller that only draws boxes or converts a model's output paid it.
#
# So `tracker` loads on first use instead of at import. This is PEP 562, the language feature for
# exactly this, and it is what SciPy and NumPy do with their own subpackages -- `pf.tracker`,
# `from pixelflow import tracker` and `import pixelflow.tracker` all still work, and the module is
# cached in globals() after the first access so the cost is paid once.
#
# Nothing else in PixelFlow imports tracker, and it publishes no top-level names of its own, which
# is what makes this a one-line deferral rather than a lazy-name table.
_LAZY = {"tracker"}


def __getattr__(name):
    """Load a deferred subpackage on first access. See the note above."""
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Keep the deferred subpackages visible to `dir()` and to tab completion."""
    return sorted(set(globals()) | _LAZY)


# Define the public API
__all__ = [
    # Result types and converters, declared by the subpackages that own them
    *_DETECTION_EXPORTS,
    *_CLASSIFICATION_EXPORTS,

    # Visual components
    "annotate",
    "colors",

    # Image transformations
    "transform",

    # Media handling
    "media",
    "DisplayExit",
    "VideoReader",
    "CameraStream",
    "VideoWriter",
    "read_image",
    "read_video",
    "display_video",
    "display_image",
    "save_image",
    "close_display",
    "to_pil",
    "from_pil",

    # Spatial analysis
    "zones",
    "Zones",
    "crossings",
    "Crossings",

    # Processing utilities
    "slicer",
    "SlicedInference",
    "auto_slice_size",
    "Buffer",
    "smooth",

    # Performance & tracking
    "timer",
    "TimeTracker",
    "tracker",

    # Metadata
    "__version__",
    "__author__"
]
