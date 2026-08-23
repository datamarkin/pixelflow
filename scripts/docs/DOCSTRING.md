# DOCSTRING.md - MkDocs Documentation Standards for PixelFlow

This file provides comprehensive standards for creating MkDocs-ready documentation in PixelFlow functions and modules.

## Module-Level Documentation

**When to Include Module-Level Docstrings:**
- **Multi-function modules**: Modules containing multiple functions or classes
- **Complex modules**: Modules with intricate logic or multiple responsibilities
- **API entry points**: Main modules that serve as package interfaces

**Skip Module-Level Docstrings for:**
- **Single-function modules**: Simple modules containing only one function (like annotators)
- **Utility modules**: Simple helper modules where the function name clearly indicates purpose

For modules that need module-level documentation, start with:

```python
"""
[Module Name] for [primary purpose/use case].

[2-3 sentence description of the module's functionality, target use cases, 
and key differentiators. Focus on what problems it solves.]
"""

from typing import List, Optional, Union, Any
import [required imports]
from [internal imports]

__all__ = ["function1", "function2"]  # Export list for API docs
```

## Standard Template for Functions

Use this comprehensive template for complex functions (see Template Selection Guidelines below):

```python
def function_name(
    param1: Type1,
    param2: Optional[Type2] = None,
    param3: float = 0.05
) -> ReturnType:
    """
    [One-line summary of what the function does].
    
    [2-3 sentences providing more detailed explanation of the function's 
    purpose, approach, and key benefits. Include algorithmic approach if relevant.]
    
    Args:
        param1 (Type1): [Description with expected format/range/constraints]
        param2 (Optional[Type2]): [Description including default behavior].
                                  [Additional details on separate line if needed].
        param3 (float): [Description with range/constraints]. 
                       Range: [min-max]. Default is [value] ([meaning]).
        
    Returns:
        ReturnType: [Description of return value, including any side effects
                   like in-place modifications].
    
    Raises:
        ExceptionType1: [When this exception occurs]
        ExceptionType2: [When this exception occurs]
        
    Example:
        >>> import cv2
        >>> import pixelflow as pf
        >>> 
        >>> # [Comment describing the setup]
        >>> image = cv2.imread("path/to/image.jpg")
        >>> outputs = model.predict(image)  # Raw model outputs
        >>> results = pf.from_ultralytics(outputs)  # Convert to PixelFlow format
        >>> 
        >>> # [Comment describing basic usage]
        >>> result = pf.module.function_name(param1, results)
        >>> 
        >>> # [Comment describing advanced usage]
        >>> result = pf.module.function_name(param1, results, param2=custom_value)
        >>> 
        >>> # [Comment describing edge case or alternative usage]
        >>> result = pf.module.function_name(param1, results, param3=0.2)
    
    Notes:
        - [Implementation detail or behavior note]
        - [Memory/performance consideration]
        - [Limitation or constraint]
        - [Automatic behavior or adaptation]
        
    Performance Notes: (Optional - only for computationally intensive functions)
        - [Optimization details]
        - [Scaling characteristics]
        - [Efficiency considerations]
        
    See Also: (Optional - only when related functions exist)
        function_name : [Brief description of related function]
        module.other_function : [Brief description of alternative/complementary function]
    """
```

## Documentation Standards

### 1. **Args Section**
- Always include type annotations in both signature AND docstring
- Specify ranges, constraints, and default behaviors
- Use "Optional[Type]" for nullable parameters
- Include units where relevant (e.g., "in pixels", "as percentage")
- Explain what "None" means for optional parameters

### 2. **Examples Section**
- **Always show PixelFlow workflow** (for functions using results): `outputs -> pf.from_*() -> function`
- **Functions Template**: Provide 3-4 examples showing:
  - Basic usage with defaults
  - Advanced usage with custom parameters  
  - Edge case or alternative workflow
  - Different parameter combinations
  - Basic usage
  - Alternative usage (if applicable)
- Use realistic variable names and paths
- Add comments explaining each example's purpose

### 3. **Error Handling**
- Document all exceptions that can be raised
- Include both explicit raises and implicit errors (IndexError, AttributeError, etc.)
- Specify the conditions that trigger each exception

### 4. **Type Hints**
- Use complete type annotations for all parameters and returns
- Import typing modules: `from typing import List, Optional, Union, Any`
- Be specific: `List[Results]` instead of `List`
- Use `Optional[Type]` instead of `Union[Type, None]`
- Follow Python 3.9+ compatibility (use `Union[X, Y]` not `X | Y`)

### 5. **Cross-References** (Conditional)
- **Include "See Also" only when related functions actually exist**
- **Skip for isolated utilities** with no clear related functions
- Use format: `function_name : Description`
- Link to complementary, alternative, or prerequisite functions

### 7. **Implementation Notes**
- Document in-place modifications
- Explain automatic adaptations or validations
- Note any clamping, clipping, or constraint enforcement
- Explain default parameter selection logic

## PixelFlow-Specific Patterns

### Framework Adapter Pattern
Always show the two-step process in examples:
```python
>>> import pixelflow as pf
>>> from ultralytics import YOLO # or from detectron2 or transformers or other framework
>>> image = cv2.imread("image.jpg") # or from pillow or other library read image
>>> model = YOLO("yolo11l.pt")
>>> outputs = model.predict(image)  # Raw framework output
>>> results = pf.from_ultralytics(outputs)  # or from_detectron2() or from_transformers() Convert to PixelFlow unified
>>> processed = pf.annotators.function_name(image, results)
```

### Common Result Types
- Use `results: List` for detection results with `.bbox` attributes
- Document expected attributes: "Each result must have a 'bbox' attribute"
- Mention coordinate format: "(x1, y1, x2, y2) coordinates"

### Parameter Validation
Document automatic validation behaviors:
- Range clamping: "Automatically clamped to range [0.0, 1.0]"
- Format correction: "Automatically converted to odd number if even"
- Boundary checking: "Clipped to image boundaries"

## Quality Checklist

### All Functions (Standard & Lightweight)
- [ ] Module docstring explains purpose and use cases
- [ ] Function signature has complete type hints
- [ ] Args section includes types, ranges, and constraints
- [ ] Examples use realistic code and variable names
- [ ] Raises section covers relevant exceptions
- [ ] No unused imports in typing section
- [ ] Follows Python 3.9+ compatibility requirements
- [ ] Examples show PixelFlow workflow (`outputs -> pf.from_*()`) when applicable
- [ ] 3-4 examples covering different use cases
- [ ] Notes section explains implementation details
- [ ] Performance notes included if computationally intensive
- [ ] See Also references related functions if they exist

This flexible approach ensures appropriate documentation depth while maintaining quality standards for all function types.