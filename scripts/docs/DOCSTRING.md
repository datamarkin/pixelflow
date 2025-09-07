# DOCSTRING.md - MkDocs Documentation Standards for PixelFlow

This file provides comprehensive standards for creating MkDocs-ready documentation in PixelFlow functions and modules.

## Module-Level Documentation

Every module should start with:

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

## Standard Template for Complex Functions

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
        >>> results = pf.results.from_ultralytics(outputs)  # Convert to PixelFlow format
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

## Lightweight Template for Simple Functions

For simple utility functions, validators, and basic converters, use this streamlined template:

```python
def simple_function(param1: Type1, param2: Optional[Type2] = None) -> ReturnType:
    """
    [One-line summary of what the function does].
    
    Args:
        param1 (Type1): [Description with expected format/constraints]
        param2 (Optional[Type2]): [Description including default behavior]
        
    Returns:
        ReturnType: [Description of return value]
    
    Raises:
        ExceptionType: [When this exception occurs - only if relevant]
        
    Example:
        >>> import pixelflow as pf
        >>> result = pf.module.simple_function(value)
        >>> # Optional second example if needed
        >>> result = pf.module.simple_function(value, custom_param)
    """
```

## Template Selection Guidelines

**Use Standard Template for:**
- Complex algorithms (annotators, processors, ML integrations)
- Functions with 3+ parameters or complex parameter relationships
- Computationally intensive operations
- Functions requiring detailed performance considerations
- Multi-step workflows or extensive examples

**Use Lightweight Template for:**
- Simple validators (`is_valid_bbox`, `check_coordinates`)
- Basic converters (`to_numpy`, `from_list`) 
- Utility functions (`clamp_value`, `normalize_path`)
- Simple getters/setters
- Functions with 1-2 straightforward parameters

## Documentation Standards

### 1. **Args Section**
- Always include type annotations in both signature AND docstring
- Specify ranges, constraints, and default behaviors
- Use "Optional[Type]" for nullable parameters
- Include units where relevant (e.g., "in pixels", "as percentage")
- Explain what "None" means for optional parameters

### 2. **Examples Section**
- **Always show PixelFlow workflow** (for functions using results): `outputs -> pf.results.from_*() -> function`
- **Standard Template**: Provide 3-4 examples showing:
  - Basic usage with defaults
  - Advanced usage with custom parameters  
  - Edge case or alternative workflow
  - Different parameter combinations
- **Lightweight Template**: Provide 1-2 examples showing:
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

### 6. **Performance Documentation** (Conditional)
- **Include only for computationally intensive functions**: image processing, ML operations, large data operations
- **Skip for simple utilities**: validators, converters, basic getters/setters
- When included, mention optimization techniques used (OpenCV, vectorization, etc.)
- Note scaling characteristics and efficiency considerations

### 7. **Implementation Notes**
- Document in-place modifications
- Explain automatic adaptations or validations
- Note any clamping, clipping, or constraint enforcement
- Explain default parameter selection logic

## PixelFlow-Specific Patterns

### Framework Adapter Pattern
Always show the two-step process in examples:
```python
>>> outputs = model.predict(image)  # Raw framework output
>>> results = pf.results.from_ultralytics(outputs)  # Convert to PixelFlow
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

### Standard Template Additional Requirements
- [ ] Examples show PixelFlow workflow (`outputs -> pf.results.from_*()`) when applicable
- [ ] 3-4 examples covering different use cases
- [ ] Notes section explains implementation details
- [ ] Performance notes included if computationally intensive
- [ ] See Also references related functions if they exist

### Lightweight Template Requirements
- [ ] 1-2 clear, focused examples
- [ ] Simple, direct documentation
- [ ] Skip Performance Notes and See Also unless clearly applicable

This flexible approach ensures appropriate documentation depth while maintaining quality standards for all function types.