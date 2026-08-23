from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..classifications import Classifications

import cv2
import numpy as np

from ..colors import _get_color_for_prediction
from .utils import _caption_with_score, _get_adaptive_params


def classification(image: np.ndarray,
                   classifications: 'Classifications',
                   top_k: int = 1,
                   position: str = 'top_left',
                   font_scale: Optional[float] = None,
                   text_color: Union[tuple, str] = (255, 255, 255),
                   bg_color: Optional[tuple] = None,
                   padding: Optional[int] = None) -> np.ndarray:
    """
    Draw a classification result as a stacked panel in a corner of the image.

    A classification describes the whole image rather than a region of it, so it has
    nowhere to anchor the way a box label does. The panel goes in a corner and lists
    the best answers top down, one row per candidate, each row carrying its class
    colour so several rows stay distinguishable at a glance.

    Args:
        image (np.ndarray): Image to annotate, modified in place.
        classifications (Classifications): PixelFlow classifications object.
        top_k (int): How many candidates to draw, best first. Default is 1.
        position (str): Corner to draw in. One of 'top_left', 'top_right',
                       'bottom_left', 'bottom_right'. Default is 'top_left'.
        font_scale (Optional[float]): Font scale. Computed from image size when None.
        text_color (Union[tuple, str]): RGB text color. Default is white.
        bg_color (Optional[tuple]): RGB background for every row. When None each row
                                   takes the color of its own class.
        padding (Optional[int]): Padding in pixels inside each row. Computed from image
                                size when None.

    Returns:
        np.ndarray: The annotated image.

    Example:
        >>> import pixelflow as pf
        >>> from ultralytics import YOLO
        >>>
        >>> model = YOLO("yolo11n-cls.pt")
        >>> result = pf.from_ultralytics_classification(model.predict(image))
        >>>
        >>> # Just the answer
        >>> annotated = pf.annotate.classification(image, result)
        >>>
        >>> # Top 3, bottom right
        >>> annotated = pf.annotate.classification(image, result, top_k=3,
        ...                                        position='bottom_right')

    Notes:
        - An empty result returns the image unchanged.
        - A row with no class_name falls back to its class_id, which is always
          meaningful even when no vocabulary was supplied.
        - Rows are ranked by confidence, so the panel reads best-first regardless of
          the order the converter produced.
    """
    rows = classifications.top_k(top_k)
    if len(rows) == 0:
        return image

    params = _get_adaptive_params(image)
    if font_scale is None:
        font_scale = params['font_scale']
    if padding is None:
        padding = params['padding']
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = params['font_thickness']

    # Measure every row first: the panel is one block, so its width is set by the
    # widest line rather than by each row independently.
    entries, widths, heights = [], [], []
    for row in rows:
        caption = row.class_name or (str(row.class_id) if row.class_id is not None else None)
        text = _caption_with_score(caption, row.confidence)
        (width, height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        entries.append((text, row))
        widths.append(width)
        heights.append(height)

    # One height for every row. Measuring each row's own baseline left the text
    # jittering up and down inside a panel whose rows are all the same height.
    text_height = max(heights)
    row_height = text_height + 2 * padding
    panel_width = max(widths) + 2 * padding
    panel_height = row_height * len(entries)

    image_height, image_width = image.shape[:2]
    margin = params['margin']
    x = margin if 'left' in position else image_width - panel_width - margin
    y = margin if position.startswith('top') else image_height - panel_height - margin
    x, y = max(0, x), max(0, y)

    for index, (text, row) in enumerate(entries):
        row_top = y + index * row_height
        cv2.rectangle(
            image,
            (x, row_top),
            (x + panel_width, row_top + row_height),
            bg_color if bg_color is not None else _get_color_for_prediction(row),
            -1
        )
        cv2.putText(
            image,
            text,
            (x + padding, row_top + padding + text_height),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )

    return image
