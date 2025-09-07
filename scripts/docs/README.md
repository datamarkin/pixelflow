# PixelFlow Documentation Helpers

Ultra-simple unified documentation system that replaces all previous documentation scripts.

## Key Principle

**1 Python file = 1 MDX file**

The system mirrors your exact code structure:
```
pixelflow/buffer.py              → docs/buffer.mdx
pixelflow/annotators/box.py      → docs/annotators/box.mdx
pixelflow/detections/filters.py  → docs/detections/filters.mdx
```

## Files

- **`generate_docs.py`** - Main script that does everything
- **`docstring_parser.py`** - Extracts structured data from Python docstrings
- **`mdx_generator.py`** - Converts parsed data to Mintlify MDX format
- **`README.md`** - This file

## Usage

### Generate All Documentation
```bash
python docs_helpers/generate_docs.py
```

### Generate Specific Module Only
```bash
python docs_helpers/generate_docs.py --module annotators
python docs_helpers/generate_docs.py --module detections
```

### Clean and Regenerate
```bash
python docs_helpers/generate_docs.py --clean
```

### Preview What Will Be Generated (Dry Run)
```bash
python docs_helpers/generate_docs.py --dry-run
```

## How It Works

1. **Discovery**: Scans `pixelflow/` directory for Python files
2. **Export Detection**: Reads `__all__` from `__init__.py` files to know what's public
3. **Parsing**: Extracts docstrings and parses them into structured data
4. **Generation**: Creates corresponding MDX files with same folder structure
5. **Filtering**: Only documents items that are exported in `__all__`

## What Gets Documented

- ✅ Functions and classes listed in `__all__`
- ✅ Files that don't start with `_`
- ❌ Private functions (starting with `_`)
- ❌ Files not exported in `__all__`

## Output Structure

```
docs/
├── buffer.mdx                    # from pixelflow/buffer.py
├── annotators/
│   ├── box.mdx                   # from pixelflow/annotators/box.py
│   ├── blur.mdx                  # from pixelflow/annotators/blur.py
│   └── ...
└── detections/
    ├── detections.mdx            # from pixelflow/detections/detections.py
    ├── filters.mdx               # from pixelflow/detections/filters.py
    └── converters.mdx            # from pixelflow/detections/converters.py
```

## Advantages Over Previous System

- **Single script** instead of 3+ separate generators
- **No hardcoded paths** - uses your file structure
- **No configuration files** needed
- **Automatic discovery** of modules and functions
- **Respects `__all__` exports** for public API
- **Same folder structure** as your code

## Migration from Old System

1. Run the new system: `python docs_helpers/generate_docs.py`
2. Compare output with existing docs
3. Once satisfied, delete old scripts:
   - `scripts/class_docstring_to_mdx.py`
   - `scripts/function_docstring_to_mdx.py` 
   - `scripts/docstring_to_mdx.py`
   - `scripts/generate_all_docs.py`

## Requirements

- Python 3.9+
- Standard library only (no external dependencies)
- Properly structured docstrings in your Python files
- `__all__` lists in your `__init__.py` files