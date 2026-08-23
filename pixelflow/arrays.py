"""Array plumbing shared by every converter.

Frameworks hand their outputs over as torch tensors, numpy arrays or plain lists,
and every converter has to flatten that to one shape before it can index anything.
One definition, so the detection and classification converters cannot disagree
about what a None or a CUDA tensor means.
"""

import numpy as np

__all__ = ["to_numpy"]


def to_numpy(array):
    """Return `array` as numpy, detaching torch tensors and passing None through."""
    if array is None:
        return None
    if hasattr(array, "detach"):
        array = array.detach().cpu()
    return np.asarray(array)
