# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-08-27

Media I/O is rebuilt around one idea: **size is discovered, rate is declared.** A
processing loop is allowed to change a frame's size -- resizing is the caller's
business, and so is resizing their detections to match -- so a sink takes its size
from the frames it is given. A loop cannot change the frame *rate*, so that has to
cross from source to sink, and it is now the only thing that does.

### Removed

- **`width=` on `VideoReader`, `CameraStream`, `VideoWriter`, `read_image` and
  `read_video`.** Resizing on the way through made a reader's own `.width` report the
  post-resize number while the true size hid in `_raw_width`, so anything asking how
  big a video was got the wrong answer; it also put the original frames permanently
  out of reach. It existed only because `pf.transform` had no `resize`. It does now.
  The parameter survives as a sentinel that raises with the replacement line, because
  a bare `TypeError` says what broke but not what to write instead.

  Migration: `pf.transform.resize(frame, width=640)` inside the loop.

- **`len(reader)`.** `len()` promises an indexable, re-iterable sequence whose length
  is exact. A decode-forward stream is none of those, and several container formats
  derive their frame count from duration x rate, so `len(list(reader))` could differ
  from `len(reader)`. Calling it now raises with an explanation rather than returning
  a number that might be wrong. Use `.frames` for a progress total.

- **`reader.frame_count`** is now **`reader.frames`**, named as the estimate it always
  was. The old name raises pointing at the new one.

- **`__del__` on the reader, stream and writer.** It ran at interpreter shutdown when
  `cv2` may already have been torn down. The context managers are the real path, and
  OpenCV releases its own captures.

### Added

- **`pf.transform.resize(image, *, width=..., height=...)`.** Exactly one axis;
  the other follows from the aspect ratio, so the function can change an image's size
  but never its shape. Both are keyword-only: the readers took a `width=` for the whole
  of 0.4, and a call that kept working while silently changing which axis it meant
  would be worse than one that stops. Interpolation is chosen, not offered --
  `INTER_AREA` shrinking, `INTER_LINEAR` growing -- because a caller has no way to know
  which is right. Returns the input array untouched when it already matches.

- **`stride=` on `VideoReader` and `CameraStream`.** Yields every Nth frame, skipping
  the rest with `grab()` rather than `read()`: inter-frame coding means a skipped frame
  still has to be walked, but it does not have to be retrieved or colour-converted.
  Measured at 1.43x on 720p at `stride=5`.

- **`VideoWriter(path, like=source)`.** Takes the frame rate from a source. This is the
  one value that must cross the reader/writer join and the one that is easy to get
  wrong: reading every 5th frame of a 25 fps file is a 5 fps sequence, and a writer
  handed 25 produces a file that plays five times too fast. `like=` carries the
  corrected rate, so that file is unconstructible rather than merely catchable.

- **`read_image` accepts encoded bytes and arrays, not only paths.** `str`/`Path`
  decodes a file; `bytes`/`bytearray`/`memoryview` decodes a buffer, which is how an
  HTTP upload, an S3 object or a database blob actually arrives; an `np.ndarray` is
  checked and returned unchanged. Channel order is established at decode and
  invisible afterwards, so there has to be exactly one decoder -- otherwise every
  caller with bytes writes `cv2.imdecode` themselves and has to remember the
  BGR-to-RGB step, which is the mistake this function exists to make unmakeable.

  Both decode paths use `IMREAD_COLOR`, deliberately and in step: `IMREAD_UNCHANGED`
  ignores EXIF orientation, so the same file would come back rotated differently
  depending on whether you passed the path or its bytes. A test asserts the two
  agree, because nothing else would notice if they stopped.

  An array is the one input whose channel order cannot be verified -- shape, dtype
  and channel count are checked, but RGB versus BGR is not recoverable from pixels.
  It is therefore *trusted*, and the docstring says so: an array from `cv2.imread`
  is BGR, and the fix is to pass the path instead.

  A URL is still not a source. Reading a file must not make a network request; see
  the 0.4.0 note on `pf.assets`.

- **`pf.encode_image(image, ".png") -> bytes`**, the mirror of `read_image`'s buffer
  branch. Accepting bytes in with nothing to hand bytes out is the asymmetry that
  makes every HTTP handler hand-roll `cv2.imencode` -- along with the RGB-to-BGR
  step that goes with it, which is the one that gets forgotten. An unsupported
  format raises `ValueError` rather than leaking a raw `cv2.error`.

  **`save_image` now encodes through it** rather than calling `cv2.imwrite` itself.
  A function whose docstring says there must be one encoder should not sit next to a
  second one: the RGB-to-BGR conversion and the format handling now exist once, and
  `save_image` keeps only what is genuinely its own -- the parent-directory and
  missing-extension checks, which are path concerns. Its failure-to-write exception
  is consequently `OSError` rather than `RuntimeError`; the `ValueError` for a bad
  image or an unwritable format is unchanged.

- **`is_live`** on both sources, and a shared fact surface -- `width`, `height`, `fps`,
  `frames`, `duration` -- with the same names and the same meanings on each. Swapping a
  file for a camera is one line, and `is_live` is the only thing worth branching on.
  `read()` is shared too, so the `while True: frame = source.read()` webcam idiom works
  for a file; both sources run the same iteration loop rather than a copy of it.

- **`start=`** on `VideoReader`, equivalent to `seek(start)` at open.

### Changed

- **`fps` and `frames` are `None` when unknown, and both are divided by the stride.**
  OpenCV reports `0.0` for a rate it does not know, which is routine for cameras and
  common for streams; `0.0` flowed straight into a writer and produced a file that
  would not play. `None` cannot be mistaken for a number, so it forces a caller to
  have a policy. `duration` is unaffected by the stride -- striding changes how many
  frames you look at, not how much time the source covers. `frames / fps` approximates
  it rather than equalling it: `frames` rounds up to count the frames you actually
  receive, so a source whose length is not a multiple of the stride reports up to one
  stride more time than it has. `duration` is the property to ask for a length.

- **Iteration no longer rewinds.** `__iter__` used to seek to frame 0 on every call,
  which made `seek()` unobservable -- `seek(1000)` then iterating returned frame 0 --
  and is impossible for any non-seekable source. Iteration now continues from the
  current position. Replay is `reader.seek(0)`, stated rather than implied.

- **`CameraStream` takes its size from a frame decoded at open.** A device's reported
  `FRAME_WIDTH`/`HEIGHT` is frequently the driver's default rather than what it hands
  over. The probe frame is the first one yielded, not a discard. `VideoReader` keeps
  reading its size from metadata -- OpenCV applies a container's rotation flag to both
  the reported size and the decoded array, verified on a file carrying `rotation=90`
  -- but now corrects itself against the first frame actually handed out, so the facts
  can never describe an array nobody received.

- **`read_image` states what it does.** Always RGB `uint8` `HxWx3`: grayscale is
  expanded, which is lossless, and an alpha channel is dropped, which is not -- so the
  drop now warns. **EXIF orientation is applied**, matching what `VideoReader` does
  with a rotated video and what a model needs to produce upright coordinates. Worth
  knowing when debugging a coordinate mismatch: annotations made by a tool that ignores
  EXIF will not line up with this array.

- **`display_video` and `display_image` keep `width=`.** A display is the end of the
  line, so shrinking a 4K frame to fit a laptop screen cannot affect anything
  downstream. The array passed in is not modified.

### Fixed

- **`VideoWriter` no longer silently truncates a file.** `cv2.VideoWriter` drops a
  frame whose size differs from the first and reports nothing, so a run ended with a
  short file, an accurate-looking `frames_written`, and no error anywhere. Writing 10
  frames could produce 5. A mismatch now raises.

- **`VideoWriter.write` and `save_image` no longer encode non-RGB arrays as colour
  nonsense.** `cv2.cvtColor` accepts a 2D grayscale or 4-channel array for `RGB2BGR`
  and quietly returns `HxWx3`, so those were written with no complaint -- a plausible
  file full of wrong colours. Both now check for `uint8` `HxWx3` first, through one
  shared guard, because the contract belongs to the library and not to one class.

- **A bad frame rate fails at construction.** `fps` of `0`, `None`, `NaN`, `inf` or a
  negative number produced a writer that failed to open on the first `write()`, hours
  into a run, with a message that blamed the output path.

- **Undecodable image data raises `ValueError`, not `RuntimeError`.** Bytes that are
  not an image, or a file that exists but is not one, are a bad *value* rather than a
  runtime failure -- which is what lets a service turn them into a 400 without
  widening what it catches. A missing file is still `FileNotFoundError` and a
  directory is still `IsADirectoryError`.

  **`VideoReader`, `CameraStream` and `VideoWriter` were converted too**, so the rule
  holds for the whole module rather than for its image half: a file that exists and
  is not what it claims to be raises `ValueError` whether it was meant to be an image
  or a video, and one `except ValueError` covers both. `RuntimeError` no longer
  appears anywhere in `media.py`. The rule itself is written down in the module
  docstring rather than left implicit in one function's `Raises:` block.

  Two honest edges are recorded there rather than papered over. A live source that
  will not open is also `ValueError`, because `cv2.VideoCapture` reports failure as a
  bare `False` and cannot say whether the URL was wrong or the device was merely
  busy. And `CameraStream` cannot raise `FileNotFoundError` for a missing path the
  way `VideoReader` does, because its source may be a device index or a URL, so it
  has no path to resolve.

- **A malformed codec fails at construction.** A FourCC that is not four characters
  reached `cv2.VideoWriter_fourcc` at the first `write()` and surfaced as a bare
  `TypeError` about argument counts -- hours into a run, from a line that never
  mentions the codec. `VideoWriter` now checks it on the line that set it, as it
  already did for `fps`. A well-formed but unavailable codec still cannot be detected
  until the lazy open, and now says so.

- **`save_image` names the cause.** A missing parent directory raises
  `NotADirectoryError` instead of "Failed to write image", which named only the
  symptom; a missing or unwritable extension raises `ValueError` instead of leaking a
  raw `cv2.error` through a PixelFlow API.

- **`_resolve_path` chains its exceptions** with `raise ... from`.

### Tests

The suite drove a single synthetic 640x480 landscape file, which is why none of the
above could fail a test. It now covers portrait video, EXIF-rotated JPEGs, grayscale
and RGBA images, and frame-identity assertions for seeking and striding that compare
against decoded ground truth rather than authored pixel values -- mp4v is lossy, so a
frame written as solid 50 comes back as 46. `CameraStream` is driven by a file path,
which `cv2.VideoCapture` accepts wherever it accepts a device, so the whole live
contract is testable without hardware. `pixelflow/media.py` coverage: 91%.

### Not done, deliberately

- **Reconnect-on-drop and a frame-drop policy for `CameraStream`.** A live source that
  blocks when its consumer falls behind accumulates latency until the device's own
  buffer overflows, so the frames being processed are minutes old. That policy belongs
  in the stream, not the consumer -- but there is no live consumer yet to design
  against, so the seam is marked and left empty rather than guessed at.

- **A folder-of-images reader.** A dataset directory is a `for path in sorted(...)`
  that callers already know how to write, and absorbing it would drag per-frame
  provenance and non-homogeneous sizes into a contract that does not need them.

- **A `process_video(source, target, callback)` helper.** The loop is where the
  caller's program lives -- tracker state, zone counters, timers, an early `break`.
  Taking it away means every loop-level concern returns as a parameter.

## [0.4.0] - 2026-08-27

### Removed
- **`pf.assets` and the implicit download behind every file read.** `read_image` and
  `VideoReader` resolved a path by checking the disk and then, for anything not found,
  asking `dtmfiles.com` for it. A mistyped filename was therefore not an error but a
  network request -- one that could block for the full 30-second socket timeout, and one
  whose success wrote a file into the caller's working directory under `./dtmfiles/`.
  A read had a write as a side effect, and the only way to get a prompt "no such file"
  was to be offline.

  It also cost every user 30 ms of the 166 ms `import pixelflow` takes -- 18%, spent on
  `urllib.request` and `hashlib` for a convenience that existed to shorten a quick-start
  snippet. Import now measures 136 ms. The module had exactly one caller in the library,
  and that caller is gone.

  Migration: fetch the file yourself and pass a path that exists.

### Changed
- **A path that is not there says so, and says where it looked.** `FileNotFoundError` now
  carries the resolved absolute path, which is the part that reveals a wrong *relative*
  path -- the common case, and the one the old message could not distinguish from a failed
  download.
- **A directory raises `IsADirectoryError`.** Passing one previously reached `cv2.imread`,
  which returns `None`, which surfaced as "Failed to decode image" -- a decoding message
  for something that was never a decoding problem.

### Fixed
- `_resolve_path` no longer catches every `Exception` and re-raises `FileNotFoundError`
  without chaining, which discarded the reason a download failed. There is no download
  left to fail.

## [0.3.2] - 2026-08-23

### Fixed
- **`smooth()` no longer emits the same `tracker_id` twice in one frame.** A tracker missing
  from the frame being smoothed but present either side of it was claimed by both halves of the
  function: the smoothing pass, which happily works without a current-frame detection, and the
  interpolation pass, which exists precisely for that case. One object came back as two rows
  sharing an id, which breaks the one-id-one-instance-per-frame guarantee everything downstream
  keys on -- zones and crossings would have counted the object twice for exactly the frames it
  was occluded. Both halves now split on a single definition of "present in this frame", so they
  cannot disagree about who owns a tracker.
- **`smooth()` warns instead of silently dropping untracked detections.** Smoothing matches a
  detection to itself across frames, which only a `tracker_id` can establish, so detections
  without one grouped into nothing and the function returned empty. Callers who forgot to run
  the tracker lost every detection with no error and no warning. It now warns and returns the
  frame unsmoothed, matching how `crossings.update()` already reports the same mistake.

### Added
- Unit and scenario tests for `tracker` and `smoother`, the two least covered modules in the
  library (`bytetrack.py` 12%, `matching.py` 15%, `smoother.py` 73%). The tracker tests drive
  whole sequences and assert on identity rather than internals -- occlusion, crossing objects,
  re-entry, crowding, and the second association stage -- because a tracker can pass every unit
  test and still switch identities the moment two objects cross.
  One test is marked `xfail`: association is IoU-only, so once frame-to-frame displacement
  exceeds roughly 0.6x the box width the first match never happens, velocity is never learned,
  and no track is ever confirmed. That is the camera-motion-compensation gap, recorded rather
  than hidden.

## [0.3.1] - 2026-08-23

### Added
- **Continuous integration.** The suite runs on push to `main` and on every pull request across
  Python 3.9 through 3.12. The workflow installs `libgl1` before anything else: `opencv-python`
  links against libGL, which the runner image does not carry, so `import cv2` fails before a
  single test runs without it.
- **Tag-driven PyPI publishing.** Releasing is `git tag vX.Y.Z && git push --tags`. A version can
  never be re-uploaded to PyPI, so everything that could reject a release runs first -- the test
  suite, a check that the tag matches `__version__`, and `twine check` on the built artifacts.
  Upload goes through Trusted Publishing, which mints a short-lived OIDC token per run rather
  than keeping an API token in repository secrets.

### Fixed
- The README badge advertised Python 3.8 while `pyproject.toml` has required 3.9 since 0.2.0.

## [0.3.0] - 2026-08-23

### Added
- **`Classifications`, a result type alongside `Detections`.** A detection row is an independent
  instance — three rows mean three objects. A classification row is a competing hypothesis about
  one image — three rows mean three candidate answers for the same picture. Most of `Detections`
  is wrong for the second kind: `filter_by_size`, `filter_by_position`, `filter_overlapping`,
  `remove_duplicates` and `update_zones` all read geometry a classifier never produced, and
  `from_arrays` requires `boxes`, which would force a whole-image rectangle that reads to every
  downstream consumer as a localisation the model never performed.
  `Classification` carries `class_id`, `class_name`, `confidence` and `metadata`, and no geometry
  of any kind. `Classifications` implements the same container protocols as `Detections` and adds
  the two operations the type actually needs: `top1`/`top_k(n)` for ranking, and
  `filter_by_confidence(t)` for thresholding. `top1` is defined as the highest-scoring row rather
  than the first, so row order is never load-bearing.
  Confidence rounds through `round_to_decimal` in `pixelflow.validators` — the same policy
  `Detection` uses, imported rather than copied. A test pins the two together directly, because a
  private copy of that policy drifting out of step is exactly the failure this shares it to avoid.
  Unlike `Detection.confidence`, the value is not treated as `[0, 1]`: CLIP cosine similarity is
  legitimately negative and raw logits are unbounded.
- **`from_scores(scores, class_ids=None, labels=None)`.** The framework-free entry point, and the
  one every classifier can reach — a CLIP zero-shot run is a matrix multiply and a list of
  prompts, a vendored ResNet is a forward pass and a tensor, and neither has a container to name a
  converter after. `class_ids` defaults to each score's own position, which is what a full score
  vector means: `scores[i]` is the score for class `i`. Pass them explicitly for a partial vector,
  such as a top-5 slice the caller already extracted.
  Scores pass through untouched. Nothing normalises, re-softmaxes, or assumes they sum to 1,
  because both conventions are real and the arrays look identical — an ImageNet head emits a
  softmax over a fixed class list, CLIP zero-shot emits an independent similarity per prompt. Only
  the caller knows which they have, so only the caller may transform it.
- **`from_ultralytics_classification(results, labels=None, top_k=None)`.** The counterpart to
  `from_ultralytics` for `-cls` checkpoints. Every class the model scored is returned by default —
  ~1000 rows for an ImageNet head — with `top_k` as the escape hatch on per-frame paths. Passing
  more than one image's results raises rather than silently reporting only the first; the
  one-element list `model.predict()` returns for a single image is the normal case and is
  unwrapped.
- **`pf.annotate.classification(image, classifications, top_k=1, position='top_left')`.** A
  classification describes the whole image and has nowhere to anchor the way a box label does, so
  it draws as a stacked panel in a corner, each row in its own class colour. A row with no
  `class_name` falls back to its `class_id`, which is meaningful even when no vocabulary was
  supplied.
- **`pixelflow.arrays.to_numpy`.** Torch-tensor and list flattening, shared by both converter
  packages. It began as a private copy in each, on the reasoning that detaching a tensor is
  plumbing rather than policy and duplicating six lines was cheaper than coupling them. The two
  copies disagreed about `None` within two days, which is the same failure the shared rounding
  policy exists to prevent, so there is now one definition.
- **`pixelflow.labels.get_label_info`.** Class-name resolution, moved out of
  `detections/converters.py` so both result types share one rule. The three label formats a caller
  may supply — `List[str]`, `Dict[int, str]`, `List[dict]` — are the same either way, and so is the
  behaviour when a name cannot be found: the id is carried and `class_name` stays None. A name is
  never invented.

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
- **Breaking.** Converters and result types are top-level: `pf.from_ultralytics(...)` rather than
  `pf.detections.from_ultralytics(...)`, and `pf.Detections` rather than `pf.detections.Detections`.
  The middle segment carried no information at the call site — of course a YOLO detector produces
  detections — and it stuttered against the type it contained. The `from_` prefix stays leading so
  the whole family groups under `pf.from_<tab>`.
  Detection is the unmarked case because it genuinely is the majority: segmentation, pose, OBB,
  SAM, RF-DETR, Florence-2, EasyOCR, Detectron2 and supervision all produce `Detections`.
  Classification is the one real fork, so it is the one name that takes a suffix. `from_scores` is
  named for what it takes rather than suffixed, because it and `from_arrays` share no input shape
  at all — one requires boxes, the other has no geometry by definition.
- **Breaking.** `from_ultralytics` raises on a classification `Result` instead of returning one
  boxless `Detection` with the remaining predictions buried in `metadata['top5_*']`. That shape
  failed quietly in the place it mattered most: `label()` skips a detection with no `bbox`, so
  annotating a classification returned an unchanged image with no error. `len(result) == 1` also
  misreported a thousand hypotheses as one object. The error names
  `pf.from_ultralytics_classification`.
- Filters are no longer exported as free functions from `pixelflow.detections`. Every one of them
  is attached to `Detections` as a method, and two ways to call the same thing is one too many.
- `Detection.confidence` is documented as rounding to the shared `CONFIDENCE_DECIMALS` precision.
  It had said "4 decimal places" while the constant was 3 — a docstring disagreeing with the
  policy is how a private copy of that policy starts.
- Docstring examples across the library referenced `pf.results.from_ultralytics`, a namespace that
  had not existed for some time. They now use the flat names.
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