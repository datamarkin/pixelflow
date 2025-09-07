#!/usr/bin/env python3
"""
Simple Python docstring to Mintlify MDX documentation generator.

Usage: Set file_input and mdx_output at the top of this script and run it.
"""

import ast
import re
from pathlib import Path

# =============================================================================
# CONFIGURATION - EDIT THESE PATHS (ARRAYS MUST BE SAME LENGTH)
# =============================================================================

file_input = [
    "pixelflow/annotators/anchors.py",
    "pixelflow/annotators/blur.py",
    "pixelflow/annotators/box.py",
    "pixelflow/annotators/crossing.py",
    "pixelflow/annotators/filled_box.py",
    "pixelflow/annotators/fps_counter.py",
    "pixelflow/annotators/grid_overlay.py",
    "pixelflow/annotators/heatmap.py",
    "pixelflow/annotators/keypoint_skeleton.py",
    "pixelflow/annotators/label.py",
    "pixelflow/annotators/mask.py",
    "pixelflow/annotators/motion_dots.py",
    "pixelflow/annotators/motion_trails.py",
    "pixelflow/annotators/oval.py",
    "pixelflow/annotators/pixelate.py",
    "pixelflow/annotators/polygon.py",
    "pixelflow/annotators/scale_bar.py",
    "pixelflow/annotators/zones.py"
]

mdx_output = [
    "docs/annotators/anchors.mdx",
    "docs/annotators/blur.mdx",
    "docs/annotators/box.mdx",
    "docs/annotators/crossing.mdx",
    "docs/annotators/filled_box.mdx",
    "docs/annotators/fps_counter.mdx",
    "docs/annotators/grid_overlay.mdx",
    "docs/annotators/heatmap.mdx",
    "docs/annotators/keypoint_skeleton.mdx",
    "docs/annotators/label.mdx",
    "docs/annotators/mask.mdx",
    "docs/annotators/motion_dots.mdx",
    "docs/annotators/motion_trails.mdx",
    "docs/annotators/oval.mdx",
    "docs/annotators/pixelate.mdx",
    "docs/annotators/polygon.mdx",
    "docs/annotators/scale_bar.mdx",
    "docs/annotators/zones.mdx"
]

# =============================================================================
# SIMPLE MDX GENERATOR
# =============================================================================

def extract_function_info(file_path):
    """Extract function name, docstring, and parameters from Python file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    tree = ast.parse(content)
    
    # Find the first public function
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
            func_name = node.name
            docstring = ast.get_docstring(node)
            
            # Extract parameters with types and defaults
            params = []
            for arg in node.args.args:
                param_info = {'name': arg.arg, 'type': 'Any', 'optional': False, 'default': None}
                params.append(param_info)
            
            # Check for defaults
            if node.args.defaults:
                defaults_start = len(params) - len(node.args.defaults)
                for i, default in enumerate(node.args.defaults):
                    param_idx = defaults_start + i
                    params[param_idx]['optional'] = True
                    if isinstance(default, ast.Constant):
                        params[param_idx]['default'] = repr(default.value)
                    elif isinstance(default, ast.Name):
                        params[param_idx]['default'] = default.id
            
            return func_name, docstring, params
    
    return None, None, []

def parse_docstring(docstring):
    """Parse docstring into structured sections."""
    if not docstring:
        return {}
    
    lines = docstring.strip().split('\n')
    result = {
        'summary': lines[0].strip() if lines else '',
        'description': '',
        'args': [],
        'returns': {'type': 'np.ndarray', 'description': 'Annotated image'},
        'examples': []
    }
    
    # Extract description (lines before Args section)
    desc_lines = []
    i = 1
    while i < len(lines) and not lines[i].strip().startswith('Args:'):
        if lines[i].strip():
            desc_lines.append(lines[i].strip())
        i += 1
    result['description'] = ' '.join(desc_lines)
    
    # Parse Args section
    if i < len(lines) and lines[i].strip().startswith('Args:'):
        i += 1
        while i < len(lines) and not lines[i].strip().startswith(('Returns:', 'Examples:')):
            line = lines[i].strip()
            if line and not line.startswith(' '):
                # New parameter
                param_match = re.match(r'(\w+)\s*\(([^)]+)\):\s*(.+)', line)
                if param_match:
                    result['args'].append({
                        'name': param_match.group(1),
                        'type': param_match.group(2),
                        'description': param_match.group(3),
                        'optional': 'Optional' in param_match.group(2) or 'optional' in param_match.group(3).lower()
                    })
            elif line and result['args']:
                # Continue previous parameter description
                result['args'][-1]['description'] += ' ' + line
            i += 1
    
    # Parse Examples section
    example_code = []
    in_examples = False
    for line in lines:
        if line.strip().startswith('Examples:'):
            in_examples = True
            continue
        if in_examples and line.strip().startswith('>>>'):
            example_code.append(line.strip()[3:].strip())
    
    if example_code:
        result['examples'] = [{'title': 'Example', 'code': example_code}]
    
    return result

def generate_mdx(func_name, parsed_doc):
    """Generate MDX content from parsed docstring."""
    title = func_name.capitalize()
    
    mdx_content = f"""---
title: {title}
description: {parsed_doc.get('summary', '')}
---

## Overview

{parsed_doc.get('summary', '')}

{parsed_doc.get('description', '')}

## Function Signature

```python
{func_name}(
    image: np.ndarray,
    detections: Detections,"""
    
    # Add other parameters from docstring
    for arg in parsed_doc.get('args', []):
        if arg['name'] not in ['image', 'detections']:
            optional_marker = 'Optional[' if arg.get('optional') else ''
            closing_bracket = ']' if arg.get('optional') else ''
            default_val = f" = {arg.get('default', 'None')}" if arg.get('optional') else ''
            mdx_content += f"\n    {arg['name']}: {optional_marker}{arg['type']}{closing_bracket}{default_val},"
    
    mdx_content += f"""
) -> {parsed_doc.get('returns', {}).get('type', 'np.ndarray')}
```

## Parameters
"""
    
    # Add parameters
    for arg in parsed_doc.get('args', []):
        required_attr = 'optional' if arg.get('optional') else 'required'
        default_attr = f'default="{arg.get("default")}"' if arg.get('default') and arg.get('default') != 'None' else ''
        
        param_line = f'<ParamField path="{arg["name"]}" type="{arg["type"]}" {required_attr}'
        if default_attr:
            param_line += f' {default_attr}'
        param_line += '>'
        
        mdx_content += f"""
{param_line}
  {arg["description"]}
</ParamField>
"""
    
    # Add returns section
    returns_info = parsed_doc.get('returns', {})
    mdx_content += f"""
## Returns

<ResponseField name="result" type="{returns_info.get('type', 'np.ndarray')}">
  {returns_info.get('description', 'Annotated image')}
</ResponseField>

## Examples

<CodeGroup>

```python Example 1
import cv2
import pixelflow as pf
from pixelflow.strategies import TriggerStrategy

# Load image and get detections
image = cv2.imread("path/to/image.jpg")
detections = pf.results.from_ultralytics(model(image))

# Draw all main anchor points (default)
annotated = pf.annotators.{func_name}(image, detections)

# Draw bottom center points (useful for ground-based tracking)
annotated = pf.annotators.{func_name}(image, detections, strategy="bottom_center")

# Draw all four corners
corners = ["top_left", "top_right", "bottom_left", "bottom_right"]
annotated = pf.annotators.{func_name}(image, detections, strategy=corners)

# Use AnchorConfig for complex multi-anchor setups
from pixelflow.strategies import AnchorConfig
config = AnchorConfig(["center", "bottom_center"], mode="any")
annotated = pf.annotators.{func_name}(image, detections, strategy=config)

# Custom styling
annotated = pf.annotators.{func_name}(
    image, detections, 
    strategy="center", 
    radius=8, 
    thickness=2,
    colors=[(0, 255, 0)]  # Green circles
)
```

</CodeGroup>
"""
    
    return mdx_content

def main():
    """Generate MDX documentation from Python files."""
    project_root = Path(__file__).parent.parent
    
    # Check array lengths match
    if len(file_input) != len(mdx_output):
        print("Error: file_input and mdx_output arrays must be the same length")
        return
    
    # Process each file pair
    for input_file, output_file in zip(file_input, mdx_output):
        input_path = project_root / input_file
        output_path = project_root / output_file
        
        print(f"\nProcessing: {input_file}")
        
        if not input_path.exists():
            print(f"Error: Input file {input_path} not found")
            continue
        
        # Extract function information
        func_name, docstring, params = extract_function_info(input_path)
        
        if not func_name:
            print(f"Error: No public function found in {input_path}")
            continue
        
        # Parse docstring
        parsed_doc = parse_docstring(docstring)
        
        # Generate MDX
        mdx_content = generate_mdx(func_name, parsed_doc)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write MDX file
        with open(output_path, 'w') as f:
            f.write(mdx_content)
        
        print(f"Generated: {output_path}")
        print(f"Function: {func_name}")
    
    print(f"\nCompleted processing {len(file_input)} files")

if __name__ == "__main__":
    main()