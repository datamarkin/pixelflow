"""
Unit tests for pixelflow.annotators.

Annotators draw, so an assertion that only checks the returned array's shape
passes whether or not anything was drawn -- and passes just as happily when a
parameter is ignored outright. These tests assert on pixels instead: that the
colour asked for is the colour that appears, that a thicker line marks more
pixels than a thin one, that drawing lands inside the box it belongs to.

The rule each parameter test follows: it must fail if the annotator quietly
dropped that parameter.
"""

import numpy as np
import pytest

import pixelflow as pf
from tests.helpers import black_canvas as canvas, make_detections as detections


# ============================================================================
# Helpers
# ============================================================================

def marked(before, after):
    """How many pixels the annotator changed."""
    return int(np.any(before != after, axis=2).sum())


def colors_drawn(before, after):
    """The distinct RGB values the annotator introduced."""
    changed = np.any(before != after, axis=2)
    return {tuple(int(c) for c in px) for px in after[changed]}


BOX = [80, 60, 200, 170]


# ============================================================================
# box
# ============================================================================

class TestBox:

    def test_draws_inside_the_box_and_nowhere_else(self):
        """Drawing lands on the detection, not somewhere arbitrary."""
        before = canvas()
        after = pf.annotate.box(before.copy(), detections([BOX]))
        changed = np.any(before != after, axis=2)
        ys, xs = np.nonzero(changed)
        assert changed.any()
        assert xs.min() >= BOX[0] - 4 and xs.max() <= BOX[2] + 4
        assert ys.min() >= BOX[1] - 4 and ys.max() <= BOX[3] + 4

    def test_honours_the_requested_colour(self):
        """Regression: this passed for years while `colors` was ignored."""
        before = canvas()
        after = pf.annotate.box(before.copy(), detections([BOX]), colors=[(255, 0, 0)])
        assert colors_drawn(before, after) == {(255, 0, 0)}

    def test_different_colours_produce_different_output(self):
        """Two colours must not render identically."""
        red = pf.annotate.box(canvas(), detections([BOX]), colors=[(255, 0, 0)])
        blue = pf.annotate.box(canvas(), detections([BOX]), colors=[(0, 0, 255)])
        assert not np.array_equal(red, blue)

    def test_thicker_lines_mark_more_pixels(self):
        """Regression: this passed while `thickness` was ignored."""
        before = canvas()
        thin = marked(before, pf.annotate.box(before.copy(), detections([BOX]), thickness=1))
        thick = marked(before, pf.annotate.box(before.copy(), detections([BOX]), thickness=6))
        assert thick > thin

    def test_two_detections_draw_two_boxes(self):
        """Every detection is drawn, not just the first."""
        before = canvas()
        one = marked(before, pf.annotate.box(before.copy(), detections([BOX])))
        two = marked(before, pf.annotate.box(before.copy(),
                                             detections([BOX, [10, 10, 60, 50]])))
        assert two > one

    def test_empty_detections_leave_the_image_untouched(self):
        before = canvas()
        assert np.array_equal(pf.annotate.box(before.copy(), pf.Detections()), before)

    def test_returns_the_same_array_it_was_given(self):
        """Annotators draw in place and hand the array back."""
        image = canvas()
        assert pf.annotate.box(image, detections([BOX])) is image


# ============================================================================
# label
# ============================================================================

class TestLabel:

    def test_draws_something_for_a_named_detection(self):
        before = canvas()
        after = pf.annotate.label(before.copy(), detections([BOX]), texts=["person"])
        assert marked(before, after) > 0

    def test_custom_text_changes_what_is_drawn(self):
        """A different string must not render identically."""
        short = pf.annotate.label(canvas(), detections([BOX]), texts=["a"])
        long = pf.annotate.label(canvas(), detections([BOX]), texts=["a much longer label"])
        assert marked(canvas(), long) > marked(canvas(), short)

    def test_position_moves_the_label(self):
        """`position` must actually place the label somewhere else."""
        top = pf.annotate.label(canvas(), detections([BOX]), texts=["x"], position="top_left")
        bottom = pf.annotate.label(canvas(), detections([BOX]), texts=["x"], position="bottom_left")
        assert not np.array_equal(top, bottom)

    def test_detection_without_class_name_still_labels_with_its_score(self):
        """A from_sam-style detection has no name but does have a confidence."""
        before = canvas()
        after = pf.annotate.label(before.copy(), detections([BOX]))
        assert marked(before, after) > 0

    def test_empty_detections_leave_the_image_untouched(self):
        before = canvas()
        assert np.array_equal(pf.annotate.label(before.copy(), pf.Detections()), before)


# ============================================================================
# mask / filled_box -- opacity blending
# ============================================================================

class TestMaskAndFilledBox:

    def _masked_detections(self):
        mask = np.zeros((240, 320), dtype=bool)
        mask[BOX[1]:BOX[3], BOX[0]:BOX[2]] = True
        result = detections([BOX])
        result[0].masks = [mask]
        return result

    def test_mask_fills_the_masked_region(self):
        before = canvas()
        after = pf.annotate.mask(before.copy(), self._masked_detections())
        assert marked(before, after) > 1000

    def test_mask_opacity_changes_intensity(self):
        """Opacity must blend, not be ignored."""
        faint = pf.annotate.mask(canvas(), self._masked_detections(), opacity=0.2)
        solid = pf.annotate.mask(canvas(), self._masked_detections(), opacity=0.9)
        assert solid.sum() > faint.sum()

    def test_mask_honours_the_requested_colour(self):
        before = canvas()
        after = pf.annotate.mask(before.copy(), self._masked_detections(),
                                 opacity=1.0, colors=[(255, 0, 0)])
        drawn = colors_drawn(before, after)
        assert drawn and all(r > g and r > b for r, g, b in drawn)

    def test_mask_without_masks_draws_nothing(self):
        before = canvas()
        after = pf.annotate.mask(before.copy(), detections([BOX]))
        assert np.array_equal(before, after)

    def test_filled_box_covers_more_than_an_outline(self):
        """A filled box is not an outline, whatever the thickness."""
        outline = marked(canvas(), pf.annotate.box(canvas(), detections([BOX]), thickness=2))
        filled = marked(canvas(), pf.annotate.filled_box(canvas(), detections([BOX]), opacity=1.0))
        assert filled > outline * 5

    def test_filled_box_opacity_changes_intensity(self):
        faint = pf.annotate.filled_box(canvas(), detections([BOX]), opacity=0.2)
        solid = pf.annotate.filled_box(canvas(), detections([BOX]), opacity=0.9)
        assert solid.sum() > faint.sum()


# ============================================================================
# blur / pixelate -- privacy annotators must actually destroy detail
# ============================================================================

class TestPrivacyAnnotators:

    def _textured(self):
        """An image with fine detail, so destroying it is measurable."""
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)

    def _detail(self, image, bbox):
        """Local variance inside `bbox` -- high for texture, low once smoothed."""
        x1, y1, x2, y2 = bbox
        region = image[y1:y2, x1:x2].astype(float)
        return float(np.abs(np.diff(region, axis=0)).mean())

    def test_blur_destroys_detail_inside_the_box(self):
        """The whole point: the region must be less distinct afterwards."""
        image = self._textured()
        before = self._detail(image, BOX)
        after = self._detail(pf.annotate.blur(image.copy(), detections([BOX])), BOX)
        assert after < before / 2

    def test_blur_leaves_the_rest_of_the_image_alone(self):
        image = self._textured()
        blurred = pf.annotate.blur(image.copy(), detections([BOX]))
        assert np.array_equal(image[:BOX[1] - 20], blurred[:BOX[1] - 20])

    def test_stronger_blur_destroys_more(self):
        """`kernel_size` must not be ignored."""
        image = self._textured()
        light = self._detail(pf.annotate.blur(image.copy(), detections([BOX]), kernel_size=5), BOX)
        heavy = self._detail(pf.annotate.blur(image.copy(), detections([BOX]), kernel_size=51), BOX)
        assert heavy < light

    def test_pixelate_destroys_detail_inside_the_box(self):
        image = self._textured()
        before = self._detail(image, BOX)
        after = self._detail(pf.annotate.pixelate(image.copy(), detections([BOX])), BOX)
        assert after < before / 2

    def test_larger_pixels_destroy_more(self):
        """`pixel_size` must not be ignored."""
        image = self._textured()
        fine = self._detail(pf.annotate.pixelate(image.copy(), detections([BOX]), pixel_size=4), BOX)
        coarse = self._detail(pf.annotate.pixelate(image.copy(), detections([BOX]), pixel_size=40), BOX)
        assert coarse < fine

    def test_empty_detections_leave_the_image_untouched(self):
        image = self._textured()
        assert np.array_equal(pf.annotate.blur(image.copy(), pf.Detections()), image)
        assert np.array_equal(pf.annotate.pixelate(image.copy(), pf.Detections()), image)


# ============================================================================
# polygon / oval / anchors
# ============================================================================

class TestShapeAnnotators:

    def _with_segments(self):
        result = detections([BOX])
        result[0].segments = [[90, 70], [190, 70], [190, 160], [90, 160]]
        return result

    def test_polygon_traces_the_segment(self):
        before = canvas()
        after = pf.annotate.polygon(before.copy(), self._with_segments())
        assert marked(before, after) > 0

    def test_polygon_honours_the_requested_colour(self):
        before = canvas()
        after = pf.annotate.polygon(before.copy(), self._with_segments(), colors=[(255, 0, 0)])
        assert colors_drawn(before, after) == {(255, 0, 0)}

    def test_polygon_thickness_marks_more_pixels(self):
        before = canvas()
        thin = marked(before, pf.annotate.polygon(before.copy(), self._with_segments(), thickness=1))
        thick = marked(before, pf.annotate.polygon(before.copy(), self._with_segments(), thickness=5))
        assert thick > thin

    def test_polygon_without_segments_draws_nothing(self):
        before = canvas()
        assert np.array_equal(pf.annotate.polygon(before.copy(), detections([BOX])), before)

    def test_oval_draws_near_the_box_bottom(self):
        """The oval is a ground-plane footprint, so it sits low in the box."""
        before = canvas()
        after = pf.annotate.oval(before.copy(), detections([BOX]))
        ys = np.nonzero(np.any(before != after, axis=2))[0]
        assert ys.size and ys.mean() > (BOX[1] + BOX[3]) / 2

    def test_oval_thickness_marks_more_pixels(self):
        before = canvas()
        thin = marked(before, pf.annotate.oval(before.copy(), detections([BOX]), thickness=1))
        thick = marked(before, pf.annotate.oval(before.copy(), detections([BOX]), thickness=5))
        assert thick > thin

    def test_anchors_radius_marks_more_pixels(self):
        before = canvas()
        small = marked(before, pf.annotate.anchors(before.copy(), detections([BOX]), radius=2))
        large = marked(before, pf.annotate.anchors(before.copy(), detections([BOX]), radius=10))
        assert large > small

    def test_anchors_strategy_moves_the_point(self):
        """Different anchor strategies must resolve to different positions."""
        center = pf.annotate.anchors(canvas(), detections([BOX]), strategy="center")
        bottom = pf.annotate.anchors(canvas(), detections([BOX]), strategy="bottom_center")
        assert not np.array_equal(center, bottom)


# ============================================================================
# keypoint annotators
# ============================================================================

class TestKeypointAnnotators:

    def _posed(self):
        result = detections([BOX])
        result[0].keypoints = [
            pf.KeyPoint(x=120, y=90, id=0, name="nose", confidence=0.9),
            pf.KeyPoint(x=100, y=130, id=1, name="left_shoulder", confidence=0.9),
            pf.KeyPoint(x=160, y=130, id=2, name="right_shoulder", confidence=0.9),
            pf.KeyPoint(x=110, y=200, id=3, name="left_hip", confidence=0.1),
        ]
        return result

    def test_keypoints_are_drawn(self):
        before = canvas()
        after = pf.annotate.keypoint(before.copy(), self._posed())
        assert marked(before, after) > 0

    def test_keypoint_radius_marks_more_pixels(self):
        before = canvas()
        small = marked(before, pf.annotate.keypoint(before.copy(), self._posed(), radius=2))
        large = marked(before, pf.annotate.keypoint(before.copy(), self._posed(), radius=8))
        assert large > small

    def test_min_confidence_hides_weak_keypoints(self):
        """The low-confidence hip should drop out at a high threshold."""
        before = canvas()
        permissive = marked(before, pf.annotate.keypoint(before.copy(), self._posed(),
                                                         min_confidence=0.0))
        strict = marked(before, pf.annotate.keypoint(before.copy(), self._posed(),
                                                     min_confidence=0.5))
        assert strict < permissive

    def test_detection_without_keypoints_draws_nothing(self):
        before = canvas()
        assert np.array_equal(pf.annotate.keypoint(before.copy(), detections([BOX])), before)

    def test_skeleton_connects_keypoints(self):
        """A skeleton draws lines, so it marks more than the joints alone."""
        before = canvas()
        joints = marked(before, pf.annotate.keypoint(before.copy(), self._posed(), radius=2))
        skeleton = marked(before, pf.annotate.keypoint_skeleton(
            before.copy(), self._posed(),
            connections=[("nose", "left_shoulder"), ("left_shoulder", "right_shoulder")]))
        assert skeleton > 0 and skeleton > joints / 2

    def test_skeleton_without_keypoints_draws_nothing(self):
        before = canvas()
        assert np.array_equal(
            pf.annotate.keypoint_skeleton(before.copy(), detections([BOX])), before)


# ============================================================================
# Overlays that ignore detections
# ============================================================================

class TestOverlays:

    def test_grid_overlay_draws_inside_the_detection(self):
        """The grid subdivides each detection's box, not the whole frame."""
        before = canvas()
        after = pf.annotate.grid_overlay(before.copy(), detections([BOX]))
        assert marked(before, after) > 0

    def test_grid_overlay_ignores_empty_detections(self):
        before = canvas()
        assert np.array_equal(
            pf.annotate.grid_overlay(before.copy(), pf.Detections()), before)

    def test_grid_size_changes_line_count(self):
        """A denser grid marks more pixels."""
        before = canvas()
        coarse = marked(before, pf.annotate.grid_overlay(before.copy(), detections([BOX]),
                                                         grid_size=(2, 2)))
        fine = marked(before, pf.annotate.grid_overlay(before.copy(), detections([BOX]),
                                                       grid_size=(10, 10)))
        assert fine > coarse

    def test_fps_counter_draws_text(self):
        before = canvas()
        after = pf.annotate.fps_counter(before.copy(), pf.Detections())
        assert marked(before, after) > 0


# ============================================================================
# crossings -- takes a Crossings manager, not Detections
# ============================================================================

class TestCrossings:

    def _lines(self, *segments):
        crossings = pf.Crossings()
        for i, (start, end) in enumerate(segments):
            crossings.add_crossing(start=start, end=end, line_id=f"line{i}")
        return crossings

    def test_draws_the_line_where_it_was_placed(self):
        """The line is drawn along its own coordinates, not somewhere else."""
        before = canvas()
        after = pf.annotate.crossings(before.copy(),
                                      self._lines(((40, 120), (280, 120))))
        ys, xs = np.nonzero(np.any(before != after, axis=2))
        assert xs.size
        assert 40 - 30 <= xs.mean() <= 280 + 30
        assert abs(ys.mean() - 120) < 40

    def test_two_lines_mark_more_than_one(self):
        before = canvas()
        one = marked(before, pf.annotate.crossings(
            before.copy(), self._lines(((40, 120), (280, 120)))))
        two = marked(before, pf.annotate.crossings(
            before.copy(), self._lines(((40, 120), (280, 120)),
                                       ((160, 30), (160, 210)))))
        assert two > one

    def test_no_crossings_leaves_the_image_untouched(self):
        before = canvas()
        assert np.array_equal(pf.annotate.crossings(before.copy(), pf.Crossings()), before)


# ============================================================================
# Composition and edges
# ============================================================================

class TestCompositionAndEdges:

    def test_layers_accumulate(self):
        """Each annotator adds to what the previous one drew."""
        before = canvas()
        boxed = pf.annotate.box(before.copy(), detections([BOX]))
        both = pf.annotate.label(boxed.copy(), detections([BOX]), texts=["person"])
        assert marked(before, both) > marked(before, boxed)

    def test_privacy_pipeline_hides_then_marks(self):
        """Blur destroys the region, the box then marks where it was."""
        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        result = pf.annotate.box(pf.annotate.blur(image.copy(), detections([BOX])),
                                 detections([BOX]))
        assert not np.array_equal(result, image)

    def test_box_larger_than_the_image_does_not_raise(self):
        """Detections can extend past the frame after a transform."""
        image = canvas(100, 100)
        pf.annotate.box(image, detections([[-50, -50, 500, 500]]))

    def test_tiny_image_does_not_raise(self):
        pf.annotate.box(canvas(10, 10), detections([[1, 1, 8, 8]]))

