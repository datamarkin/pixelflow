"""
Image-Only Transformations.

All image transformation functions that don't involve detections.
Includes both geometric operations and enhancement operations.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union

__all__ = [
    # Geometric operations
    'resize',
    'rotate',
    'flip_horizontal',
    'flip_vertical',
    'crop',

    # Enhancement operations
    'clahe',
    'to_grayscale',
    'auto_contrast',
    'normalize',
    'gamma_correction',
    'standardize'
]


# ============================================================================
# Geometric Operations
# ============================================================================

def resize(
    image: np.ndarray,
    *,
    width: Optional[int] = None,
    height: Optional[int] = None
) -> np.ndarray:
    """
    Scale an image to a width or a height, keeping its aspect ratio.

    Exactly one of ``width`` or ``height`` is required. The other follows from the
    aspect ratio, which is what makes this a scale rather than a reshape -- an image's
    shape cannot be changed by this function, only its size.

    Both are keyword-only. ``resize(image, 640)`` is a ``TypeError`` rather than a
    guess, because PixelFlow's readers and writers previously took a positional
    ``width=`` and a silently reinterpreted axis would be worse than an error.

    Args:
        image: Input image (H, W, 3) RGB format
        width: Target width in pixels. Height follows.
        height: Target height in pixels. Width follows.

    Returns:
        The scaled image, or the input array itself when it already matches.

    Raises:
        ValueError: If neither or both of width/height are given, or if the
            target is less than 1 pixel.

    Example:
        ```python
        import pixelflow as pf

        video = pf.VideoReader("input.mp4")
        for frame in video:
            frame = pf.transform.resize(frame, width=640)
        ```

    Note:
        - Interpolation is chosen, not offered: ``INTER_AREA`` when shrinking and
          ``INTER_LINEAR`` when growing. A caller has no way to know which is right.
        - When the image already matches, the input array is returned **unchanged and
          uncopied**. PixelFlow's annotators draw in place, so annotating that result
          also draws on the array you passed in. Copy first if you need both.
    """
    if (width is None) == (height is None):
        raise ValueError(
            "resize() takes exactly one of width= or height=; "
            f"got width={width!r}, height={height!r}"
        )

    have_height, have_width = image.shape[:2]

    if width is None:
        if height < 1:
            raise ValueError(f"height must be at least 1, got {height}")
        if height == have_height:
            return image
        width = max(1, round(have_width * height / have_height))
    else:
        if width < 1:
            raise ValueError(f"width must be at least 1, got {width}")
        if width == have_width:
            return image
        height = max(1, round(have_height * width / have_width))

    # Shrinking averages the pixels being discarded; growing interpolates between the
    # ones being kept. Using the wrong one is visible: INTER_LINEAR downscaling aliases.
    shrinking = width * height < have_width * have_height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def rotate(
    image: np.ndarray,
    angle: float,
    center: Optional[Tuple[float, float]] = None,
    fillcolor: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Rotate image around center point.

    Args:
        image: Input image (H, W, 3) RGB format
        angle: Rotation angle in degrees (positive = counter-clockwise)
        center: Rotation center (x, y). If None, uses image center.
        fillcolor: Fill color for areas outside original image. If None, uses edge pixels.

    Returns:
        Rotated image

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        rotated = pf.transform.rotate(image, angle=45)
        ```

    Note:
        - Uses OpenCV's getRotationMatrix2D and warpAffine
        - Expands image to fit rotated content
        - Edge pixels are replicated if fillcolor is None
    """
    h, w = image.shape[:2]

    # Use image center if center not provided
    if center is None:
        center = (w / 2.0, h / 2.0)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new image dimensions
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix for new dimensions
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Determine fill strategy
    if fillcolor is not None:
        border_mode = cv2.BORDER_CONSTANT
        border_value = fillcolor
    else:
        border_mode = cv2.BORDER_REPLICATE
        border_value = None

    # Perform rotation
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        borderMode=border_mode,
        borderValue=border_value
    )

    return rotated


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """
    Flip image horizontally (left-right).

    Args:
        image: Input image (H, W, 3) RGB format

    Returns:
        Horizontally flipped image

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        flipped = pf.transform.flip_horizontal(image)
        ```
    """
    return cv2.flip(image, 1)


def flip_vertical(image: np.ndarray) -> np.ndarray:
    """
    Flip image vertically (top-bottom).

    Args:
        image: Input image (H, W, 3) RGB format

    Returns:
        Vertically flipped image

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        flipped = pf.transform.flip_vertical(image)
        ```
    """
    return cv2.flip(image, 0)


def crop(image: np.ndarray, bbox: list) -> np.ndarray:
    """
    Crop image to bounding box.

    Args:
        image: Input image (H, W, 3) RGB format
        bbox: Crop region [x1, y1, x2, y2] in pixels

    Returns:
        Cropped image

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        cropped = pf.transform.crop(image, [100, 50, 500, 400])
        ```

    Note:
        - Coordinates are clipped to image boundaries
        - Returns empty array if bbox is invalid
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Clip to image boundaries
    x1 = int(max(0, min(x1, w)))
    y1 = int(max(0, min(y1, h)))
    x2 = int(max(0, min(x2, w)))
    y2 = int(max(0, min(y2, h)))

    # Crop
    return image[y1:y2, x1:x2]


# ============================================================================
# Enhancement Operations
# ============================================================================

def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) contrast enhancement.

    Args:
        image: Input image (H, W, 3) RGB format or (H, W) grayscale
        clip_limit: Threshold for contrast limiting (higher = more contrast)
        tile_size: Size of grid for histogram equalization

    Returns:
        Contrast-enhanced image in same format as input

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        enhanced = pf.transform.clahe(image, clip_limit=2.0, tile_size=(8, 8))
        ```

    Note:
        - For color images, CLAHE is applied to L channel in LAB space
        - For grayscale images, CLAHE is applied directly
    """
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    # Check if image is grayscale or color
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Grayscale
        return clahe_obj.apply(image)
    else:
        # Color - apply to L channel in LAB space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def to_grayscale(
    image: np.ndarray,
    keep_channels: bool = False
) -> np.ndarray:
    """
    Convert image to grayscale.

    Args:
        image: Input image (H, W, 3) RGB format
        keep_channels: If True, returns (H, W, 3) with same values in all channels.
                      If False, returns (H, W) single channel.

    Returns:
        Grayscale image

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")

        # Single channel grayscale
        gray = pf.transform.to_grayscale(image)

        # Three channel grayscale (for compatibility)
        gray_3ch = pf.transform.to_grayscale(image, keep_channels=True)
        ```
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if keep_channels:
        # Stack to 3 channels
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        return gray


def auto_contrast(
    image: np.ndarray,
    cutoff: float = 1.0
) -> np.ndarray:
    """
    Apply automatic contrast adjustment by stretching histogram.

    Args:
        image: Input image (H, W, 3) RGB format or (H, W) grayscale
        cutoff: Percentage of extreme pixels to ignore (0-100)

    Returns:
        Contrast-adjusted image

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        enhanced = pf.transform.auto_contrast(image, cutoff=1.0)
        ```

    Note:
        - Similar to Photoshop's Auto Contrast
        - cutoff=1.0 ignores 1% darkest and 1% brightest pixels
    """
    # Handle grayscale and color
    if len(image.shape) == 2:
        # Grayscale
        channels = [image]
    else:
        # Color - split channels
        channels = cv2.split(image)

    result_channels = []
    for channel in channels:
        # Calculate histogram
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])

        # Calculate cumulative distribution
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]

        # Find cutoff points
        low_percent = cutoff / 100.0
        high_percent = 1.0 - (cutoff / 100.0)

        low_val = np.searchsorted(cdf_normalized, low_percent)
        high_val = np.searchsorted(cdf_normalized, high_percent)

        # Stretch histogram
        if high_val > low_val:
            lut = np.clip((np.arange(256) - low_val) * 255.0 / (high_val - low_val), 0, 255).astype(np.uint8)
        else:
            lut = np.arange(256, dtype=np.uint8)

        result_channels.append(cv2.LUT(channel, lut))

    # Merge channels back
    if len(image.shape) == 2:
        return result_channels[0]
    else:
        return cv2.merge(result_channels)


def normalize(
    image: np.ndarray,
    mean: Union[float, Tuple[float, float, float]],
    std: Union[float, Tuple[float, float, float]]
) -> np.ndarray:
    """
    Normalize image for model input using mean and standard deviation.

    Args:
        image: Input image (H, W, 3) RGB format
        mean: Mean value(s) to subtract. Single float or tuple of 3 floats for each channel.
        std: Standard deviation value(s) to divide by. Single float or tuple of 3 floats.

    Returns:
        Normalized image (float32)

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")

        # ImageNet normalization
        normalized = pf.transform.normalize(
            image,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
        ```

    Note:
        - Returns float32 array
        - Typically used before feeding to neural networks
    """
    # Convert to float
    image_float = image.astype(np.float32) / 255.0

    # Convert mean and std to arrays
    if isinstance(mean, (int, float)):
        mean = np.array([mean, mean, mean], dtype=np.float32)
    else:
        mean = np.array(mean, dtype=np.float32)

    if isinstance(std, (int, float)):
        std = np.array([std, std, std], dtype=np.float32)
    else:
        std = np.array(std, dtype=np.float32)

    # Normalize
    normalized = (image_float - mean) / std

    return normalized


def gamma_correction(
    image: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Apply gamma correction to adjust image brightness.

    Args:
        image: Input image (H, W, 3) RGB format or (H, W) grayscale
        gamma: Gamma value. <1 brightens, >1 darkens, =1 no change.

    Returns:
        Gamma-corrected image

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")

        # Brighten
        brightened = pf.transform.gamma_correction(image, gamma=0.5)

        # Darken
        darkened = pf.transform.gamma_correction(image, gamma=2.0)
        ```

    Note:
        - Uses lookup table for efficiency
        - Preserves image dtype (uint8)
    """
    # Build lookup table
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)

    # Apply lookup table
    return cv2.LUT(image, lut)


def standardize(image: np.ndarray) -> np.ndarray:
    """
    Standardize image to zero mean and unit variance.

    Args:
        image: Input image (H, W, 3) RGB format or (H, W) grayscale

    Returns:
        Standardized image (float32)

    Example:
        ```python
        import pixelflow as pf
        import cv2

        image = cv2.imread("image.jpg")
        standardized = pf.transform.standardize(image)
        ```

    Note:
        - Returns float32 array
        - Calculated per image, not using dataset statistics
    """
    # Convert to float
    image_float = image.astype(np.float32)

    # Calculate mean and std
    mean = np.mean(image_float)
    std = np.std(image_float)

    # Avoid division by zero
    if std == 0:
        std = 1.0

    # Standardize
    standardized = (image_float - mean) / std

    return standardized
