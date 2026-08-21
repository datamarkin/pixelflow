# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `from_arrays` takes `texts=` and `segments=`, and `class_ids=` is now optional. Deployment
  code extracted from a research repository returns arrays rather than a framework container,
  which is what `from_arrays` exists for — but until now it could carry neither a read string
  nor a polygon, so a vendored OCR model had to build `Detections` by hand or borrow
  `from_easyocr`, a converter named after a container it does not use. `texts` fills `text`
  per detection (an empty read is kept as `""`; only None leaves the field unset), and
  `segments` fills one polygon per detection, so a text quadrilateral or an oriented box
  survives instead of being flattened into `bbox`. `class_ids` defaults to None for the models
  that locate without naming — OCR reads content, SAM segments what it was pointed at — which
  no existing caller notices, since every one of them passes it.
- `Detection.text` — the free-form string an instance carries: what an OCR engine read inside
  the box, or a region caption. It is a field rather than a `metadata` key because untyped
  metadata means every annotator, filter and consumer re-invents the key and none can rely on
  it. It is separate from `class_name` because the two answer different questions: `class_name`
  is which class out of a vocabulary the model was trained on, `text` is content the model
  produced that belongs to no vocabulary. Putting a read string in `class_name` — as
  supervision's `from_easyocr` does — makes that field mean two things, and a detection can
  legitimately have both.
- `from_easyocr(reader.readtext(...))`. The quadrilateral EasyOCR read is kept in `segments`,
  its axis-aligned hull in `bbox`, the string in `text`, and `class_id`/`class_name` stay None.
  Keeping the quad matters because real-world text is rotated: EasyOCR emits a genuine
  quadrilateral for any line off the horizontal, and collapsing it to `bbox` throws the
  orientation away. Since it lives in `segments`, transforms move all four corners and the
  polygon annotator draws it with no further work.
  Handles the shapes `readtext` actually returns: the default `(quad, text, confidence)` tuples,
  the 2-element items from `paragraph=True` (which carries no confidence, so `confidence` is
  None rather than a fabricated 0), `output_format='dict'`, and the lists the Arabic path
  produces. `detail=0` and `output_format='json'` return strings with no geometry and raise.

### Fixed
- **Breaking.** `from_florence2` no longer puts free-form strings in `class_name`.
  `<DENSE_REGION_CAPTION>` descriptions, `<CAPTION_TO_PHRASE_GROUNDING>` and
  `<REFERRING_EXPRESSION_SEGMENTATION>` phrases, and `<OCR_WITH_REGION>` reads now populate
  `text`, with `class_id` and `class_name` left None — the model picked nothing out of a
  vocabulary, so there is no class to report. `<OD>`, `<REGION_PROPOSAL>` and
  `<OPEN_VOCABULARY_DETECTION>` genuinely name a class and are unchanged, sequential
  `class_id`s included.
  The output shapes cannot tell these apart — `<OD>` and `<DENSE_REGION_CAPTION>` both return
  `{"bboxes", "labels"}` — so the routing keys off `task_prompt`, which the converter already
  required. Unrecognised tasks now default to `text`: Florence-2's region tasks emit prose by
  default, and a category name sitting in `text` is inert, where prose in `class_name` minted
  a `class_id` per unique caption and leaked into the crossings class-name map.
  Callers reading `det.class_name` from a captioning task must read `det.text` instead.
- `from_florence2` supports `<OCR_WITH_REGION>`, which previously raised. Its quadrilaterals
  are kept in `segments` with the hull in `bbox`, matching `from_easyocr`.
- `from_florence2` rejects `<REGION_TO_CATEGORY>`, `<REGION_TO_DESCRIPTION>` and
  `<REGION_TO_OCR>` with the same "text only" error as the other pure-text tasks, rather than
  the less obvious "unsupported data format".
- `label()` no longer draws the literal string `"None: 0.87"` for a detection with no
  `class_name`. It had been formatting None into the label, which anything from `from_sam` or a
  hand-built box hit. Such detections are now labelled with their confidence alone, so no
  information is lost.

### Changed
- `label()` auto-generated labels use `text` when the detection has one, falling back to
  `class_name`. `{text}` is available as a template placeholder.

## [0.2.0] - 2026-08-18

### Changed
- **Breaking.** `KeyPoint` is now `KeyPoint(x, y, id, name=None, confidence=None)`, matching how
  `Detection` carries its class as `class_id` plus `class_name` — an index that is always
  correct, and a name that may be absent. The field is `id` rather than `keypoint_id` because
  a KeyPoint has no other id to disambiguate it from. `id` is the landmark's index in the model's keypoint
  vocabulary, always present and always correct; `name` is metadata and is `None` when nothing
  supplied it.
- **Breaking.** `KeyPoint.visibility` is removed. It was `score > 0` (or `> 0.5` in
  `from_ultralytics`) — a threshold applied on the caller's behalf that collapsed 0.01 and 0.99
  to the same `True` and discarded the score. The model's score is now carried through as
  `confidence`, and `keypoint()` / `keypoint_skeleton()` take `min_confidence` so the threshold
  is chosen where the rendering decision is actually made.
- `KeyPoint.to_dict()` now emits `id`, `name` and `confidence` in place of `visibility`.

### Removed
- **Breaking.** `COCO_LABELS` and the `pixelflow.classes` module. A detection container has no
  business knowing one dataset's vocabulary: which id space a model emits is a property of the
  model, and what id 73 is called is a property of the weights. Bundling COCO's names is what
  made the mislabelling above possible — the constant was there, so it got used as a fallback.
  Callers pass their model's own class names.

### Fixed
- Keypoint names are no longer guessed. `from_arrays`, `from_detectron2`, `from_mayaku`,
  `from_ultralytics` and `from_datamarkin` all fell back to COCO's 17 human-pose names whenever
  the caller supplied no labels, so a 21-point hand model reported `nose`, `left_eye`,
  `left_shoulder`, then `keypoint_17` onward. Wrong, and authoritative-looking enough that
  nobody would check it. The `_COCO_KEYPOINT_NAMES` fallback is deleted rather than guarded, so
  it cannot come back.
- `from_datamarkin` no longer turns a missing keypoint name into `""`.

### Added
- `from_arrays(boxes, scores, class_ids, masks, keypoints, labels)` — the framework-free
  converter. Every other converter is named after some framework's output container; deployment
  code extracted from a research repository has no such container, because removing it is the
  point. Accepts numpy or torch.

## [0.1.2] - 2024-12-22

### Added
- Pose estimation support in `from_ultralytics` converter - now extracts all 17 COCO keypoints with visibility detection
- Classification model support in `from_ultralytics` converter - returns top-1 prediction with top-5 in metadata
- `COCO_KEYPOINT_NAMES` constant for standard pose keypoint labels

### Fixed
- `from_ultralytics` now correctly extracts keypoints from YOLO pose models (previously always returned `None`)
- `from_ultralytics` now handles classification models that have `probs` instead of `boxes` (previously returned empty results)

## [0.1.1] - 2024-09-22

### Added
- Enhanced package configuration with comprehensive metadata
- Improved README with detailed usage examples and documentation
- Added build system configuration in pyproject.toml
- Comprehensive .gitignore for Python projects
- Keywords and classifiers for better PyPI discoverability
- Project URLs for homepage, repository, issues, and documentation

### Changed
- Updated project description to be more comprehensive
- Enhanced pyproject.toml with modern Python packaging standards
- Improved README structure with quick start guide and examples
- Better organized documentation links and contributing guidelines

### Fixed
- Version consistency across all package files
- Build artifacts cleanup and proper .gitignore configuration
- Package metadata completeness for PyPI publishing

## [0.1.0] - 2024-09-22

### Added
- Initial release of PixelFlow computer vision library
- Core detection and results data structures (`Prediction`, `Results`, `KeyPoint`)
- Framework adapters for Detectron2, Ultralytics, and Datamarkin API
- Low-level drawing primitives using OpenCV (`draw.py`)
- High-level annotation functions (`annotate.py`)
- Video processing with lazy frame loading (`video.py`)
- Zone-based filtering system (`zones.py`)
- Data validation and polygon utilities (`validators.py`)
- Color management system (`colors.py`)
- Object tracking capabilities (`tracker/`)
- Python 3.9+ compatibility
- MIT license

### Features
- Flexible annotation tools (box, mask, keypoint, heatmap, blur, pixelate, motion trails)
- Efficient video processing with memory optimization
- Multi-framework support (Detectron2, YOLO, Ultralytics)
- High-performance OpenCV-based rendering
- Modular architecture with focused, single-purpose modules
- Zone-based spatial filtering for targeted analysis