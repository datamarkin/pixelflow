# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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