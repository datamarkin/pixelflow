#!/usr/bin/env python3
"""
Convert Python docstrings to Mintlify MDX documentation.

This script parses Python files, extracts docstrings, and generates
MDX documentation files suitable for Mintlify.
"""

import ast
import re
import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json


class DocstringParser:
    """Parse Python docstrings into structured data."""
    
    def __init__(self):
        self.sections_pattern = re.compile(
            r'^(Args|Returns|Raises|Example|Examples|Notes?|'
            r'Performance Notes?|See Also|Yields|Warns):\s*$',
            re.MULTILINE | re.IGNORECASE
        )
    
    def parse(self, docstring: str) -> Dict[str, Any]:
        """Parse a docstring into sections."""
        if not docstring:
            return {}
        
        lines = docstring.strip().split('\n')
        result = {
            'summary': '',
            'description': '',
            'args': [],
            'returns': None,
            'raises': [],
            'examples': [],
            'notes': [],
            'performance_notes': [],
            'see_also': []
        }
        
        # Extract summary (first line)
        if lines:
            result['summary'] = lines[0].strip()
        
        # Extract description (lines before first section)
        desc_lines = []
        i = 1
        while i < len(lines) and not self.sections_pattern.match(lines[i].strip()):
            if lines[i].strip():
                desc_lines.append(lines[i].strip())
            i += 1
        result['description'] = ' '.join(desc_lines)
        
        # Parse sections
        current_section = None
        section_content = []
        
        for line in lines[i:]:
            section_match = self.sections_pattern.match(line.strip())
            if section_match:
                # Process previous section
                if current_section:
                    self._process_section(result, current_section, section_content)
                current_section = section_match.group(1).lower()
                section_content = []
            else:
                section_content.append(line)
        
        # Process last section
        if current_section:
            self._process_section(result, current_section, section_content)
        
        return result
    
    def _process_section(self, result: Dict, section: str, content: List[str]):
        """Process a specific docstring section."""
        if section == 'args':
            result['args'] = self._parse_args(content)
        elif section == 'returns':
            result['returns'] = self._parse_returns(content)
        elif section == 'raises':
            result['raises'] = self._parse_raises(content)
        elif section in ['example', 'examples']:
            result['examples'] = self._parse_examples(content)
        elif section in ['note', 'notes']:
            result['notes'] = self._parse_notes(content)
        elif section in ['performance note', 'performance notes']:
            result['performance_notes'] = self._parse_notes(content)
        elif section == 'see also':
            result['see_also'] = self._parse_see_also(content)
    
    def _parse_args(self, lines: List[str]) -> List[Dict]:
        """Parse Args section into structured format."""
        args = []
        current_arg = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a new parameter
            param_match = re.match(r'^(\w+)\s*\(([^)]+)\)\s*:\s*(.*)$', line)
            if param_match:
                if current_arg:
                    args.append(current_arg)
                current_arg = {
                    'name': param_match.group(1),
                    'type': param_match.group(2),
                    'description': param_match.group(3),
                    'optional': 'Optional' in param_match.group(2),
                    'default': None
                }
                # Extract default value
                default_match = re.search(r'Default is ([^.]+)', param_match.group(3))
                if default_match:
                    current_arg['default'] = default_match.group(1).strip()
            elif current_arg and line:
                # Continuation of previous parameter description
                current_arg['description'] += ' ' + line
        
        if current_arg:
            args.append(current_arg)
        
        return args
    
    def _parse_returns(self, lines: List[str]) -> Dict:
        """Parse Returns section."""
        text = ' '.join(line.strip() for line in lines if line.strip())
        type_match = re.match(r'^(\S+):\s*(.*)$', text)
        if type_match:
            return {
                'type': type_match.group(1),
                'description': type_match.group(2)
            }
        return {'type': 'Any', 'description': text}
    
    def _parse_raises(self, lines: List[str]) -> List[Dict]:
        """Parse Raises section."""
        raises = []
        current_raise = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a new exception
            exc_match = re.match(r'^(\w+(?:Error)?)\s*:\s*(.*)$', line)
            if exc_match:
                if current_raise:
                    raises.append(current_raise)
                current_raise = {
                    'exception': exc_match.group(1),
                    'description': exc_match.group(2)
                }
            elif current_raise:
                current_raise['description'] += ' ' + line
        
        if current_raise:
            raises.append(current_raise)
        
        return raises
    
    def _parse_examples(self, lines: List[str]) -> List[Dict]:
        """Parse Examples section."""
        examples = []
        current_example = {'title': '', 'code': [], 'description': ''}
        in_code = False
        
        for line in lines:
            if line.strip().startswith('>>>'):
                in_code = True
                current_example['code'].append(line.strip()[3:].strip())
            elif in_code and (not line.strip() or not line.strip().startswith('>>>')):
                # End of code block
                if current_example['code']:
                    examples.append(current_example)
                    current_example = {'title': '', 'code': [], 'description': ''}
                in_code = False
            elif not in_code and line.strip().startswith('#'):
                # Comment that might be a title
                current_example['title'] = line.strip()[1:].strip()
        
        # Add last example if exists
        if current_example['code']:
            examples.append(current_example)
        
        return examples
    
    def _parse_notes(self, lines: List[str]) -> List[str]:
        """Parse Notes section."""
        notes = []
        for line in lines:
            line = line.strip()
            if line.startswith('-'):
                notes.append(line[1:].strip())
            elif line and notes:
                notes[-1] += ' ' + line
            elif line:
                notes.append(line)
        return notes
    
    def _parse_see_also(self, lines: List[str]) -> List[Dict]:
        """Parse See Also section."""
        see_also = []
        for line in lines:
            line = line.strip()
            if ':' in line:
                name, desc = line.split(':', 1)
                see_also.append({
                    'name': name.strip(),
                    'description': desc.strip()
                })
        return see_also


class MDXGenerator:
    """Generate MDX documentation from parsed docstrings."""
    
    def __init__(self):
        self.icon_map = {
            'blur': 'eye-slash',
            'pixelate': 'grid',
            'box': 'square',
            'mask': 'masks-theater',
            'polygon': 'draw-polygon',
            'heatmap': 'fire',
            'keypoint': 'circle-dot',
            'label': 'tag',
            'dot': 'circle',
            'oval': 'circle',
            'motion_trails': 'route',
            'motion_dots': 'braille'
        }
    
    def generate(self, func_name: str, parsed_doc: Dict, module_name: str = '') -> str:
        """Generate MDX content from parsed docstring."""
        lines = []
        
        # Frontmatter
        lines.extend([
            '---',
            f'title: {func_name}',
            f'description: {parsed_doc.get("summary", "")}',
            f'icon: "{self.icon_map.get(func_name, "function")}"',
            '---',
            ''
        ])
        
        # Overview
        lines.extend([
            '## Overview',
            '',
            parsed_doc.get('summary', ''),
            ''
        ])
        
        if parsed_doc.get('description'):
            lines.append(parsed_doc['description'])
            lines.append('')
        
        # Add notes as Info blocks
        if parsed_doc.get('notes'):
            lines.append('<Info>')
            for note in parsed_doc['notes'][:2]:  # First two notes
                lines.append(f'  {note}')
            lines.append('</Info>')
            lines.append('')
        
        # Function Signature
        if parsed_doc.get('args'):
            lines.extend([
                '## Function Signature',
                '',
                '```python',
                f'{func_name}(',
            ])
            for i, arg in enumerate(parsed_doc['args']):
                comma = ',' if i < len(parsed_doc['args']) - 1 else ''
                default = f" = {arg['default']}" if arg.get('default') else ''
                lines.append(f"    {arg['name']}: {arg['type']}{default}{comma}")
            
            returns = parsed_doc.get('returns', {})
            lines.append(f") -> {returns.get('type', 'Any')}")
            lines.extend(['```', ''])
        
        # Parameters
        if parsed_doc.get('args'):
            lines.extend(['## Parameters', ''])
            for arg in parsed_doc['args']:
                required = 'optional' if arg.get('optional') else 'required'
                default = f'default="{arg.get("default")}"' if arg.get('default') else ''
                lines.append(f'<ParamField path="{arg["name"]}" type="{arg["type"]}" {required} {default}>'.replace('  ', ' '))
                lines.append(f'  {arg["description"]}')
                lines.append('</ParamField>')
                lines.append('')
        
        # Returns
        if parsed_doc.get('returns'):
            lines.extend([
                '## Returns',
                '',
                f'<ResponseField name="result" type="{parsed_doc["returns"]["type"]}">',
                f'  {parsed_doc["returns"]["description"]}',
                '</ResponseField>',
                ''
            ])
        
        # Examples
        if parsed_doc.get('examples'):
            lines.extend(['## Examples', '', '<CodeGroup>', ''])
            for i, example in enumerate(parsed_doc['examples']):
                title = example.get('title') or f'Example {i+1}'
                lines.append(f'```python {title}')
                for code_line in example['code']:
                    lines.append(code_line)
                lines.append('```')
                lines.append('')
            lines.extend(['</CodeGroup>', ''])
        
        # Error Handling
        if parsed_doc.get('raises'):
            lines.extend([
                '## Error Handling',
                '',
                '<Warning>',
                '  The function will raise the following exceptions:'
            ])
            for exc in parsed_doc['raises']:
                lines.append(f'  - **{exc["exception"]}**: {exc["description"]}')
            lines.extend(['</Warning>', ''])
        
        # Performance Notes
        if parsed_doc.get('performance_notes'):
            lines.extend([
                '## Performance Notes',
                '',
                '<Accordion title="Performance Characteristics">'
            ])
            for note in parsed_doc['performance_notes']:
                lines.append(f'  - {note}')
            lines.extend(['</Accordion>', ''])
        
        # Implementation Details
        remaining_notes = parsed_doc.get('notes', [])[2:]  # Notes after the first two
        if remaining_notes:
            lines.extend([
                '## Implementation Details',
                '',
                '<Note>'
            ])
            for note in remaining_notes:
                lines.append(f'  - {note}')
            lines.extend(['</Note>', ''])
        
        # Related Functions
        if parsed_doc.get('see_also'):
            lines.extend([
                '## Related Functions',
                '',
                '<CardGroup cols={2}>'
            ])
            for related in parsed_doc['see_also']:
                name = related['name'].replace('_', '-')
                icon = self.icon_map.get(related['name'].split('.')[-1], 'function')
                lines.extend([
                    f'  <Card title="{related["name"]}" icon="{icon}" href="/annotators/{name}">',
                    f'    {related["description"]}',
                    '  </Card>'
                ])
            lines.extend(['</CardGroup>', ''])
        
        return '\n'.join(lines)


def convert_file(filepath: Path, output_dir: Path):
    """Convert a Python file's docstrings to MDX."""
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    
    parser = DocstringParser()
    generator = MDXGenerator()
    
    # Get module docstring
    module_doc = ast.get_docstring(tree)
    
    # Process functions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            if docstring and not node.name.startswith('_'):
                parsed = parser.parse(docstring)
                mdx_content = generator.generate(
                    node.name, 
                    parsed,
                    module_name=filepath.stem
                )
                
                # Write MDX file
                output_file = output_dir / f"{node.name}.mdx"
                output_file.write_text(mdx_content)
                print(f"Generated: {output_file}")


def get_annotator_category(func_name: str) -> str:
    """Determine the category for an annotator function."""
    categories = {
        'basics': ['box', 'polygon', 'mask', 'label', 'dot', 'oval', 'filled_box'],
        'privacy': ['blur', 'pixelate'],
        'tracking': ['motion_trails', 'motion_dots'],
        'analysis': ['heatmap', 'zones', 'line_zone', 'line_zones'],
        'overlays': ['fps_counter', 'grid_overlay', 'scale_bar', 'keypoint', 'keypoint_skeleton']
    }
    
    for category, functions in categories.items():
        if func_name in functions:
            return category
    return 'basics'  # Default category


def main():
    """Main conversion function."""
    # Set up paths
    project_root = Path(__file__).parent.parent
    
    # Module mappings
    modules = {
        'annotators': project_root / "pixelflow" / "annotators",
        'tracker': project_root / "pixelflow" / "tracker",
        'core': project_root / "pixelflow",  # Core modules in root
    }
    
    # Process annotators with categories
    annotators_dir = modules['annotators']
    for py_file in annotators_dir.glob("*.py"):
        if py_file.name not in ['__init__.py', 'utils.py']:
            print(f"Processing annotator: {py_file.name}")
            with open(py_file, 'r') as f:
                tree = ast.parse(f.read())
            
            parser = DocstringParser()
            generator = MDXGenerator()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if docstring and not node.name.startswith('_'):
                        # Determine category
                        category = get_annotator_category(node.name)
                        output_dir = project_root / "docs" / "annotators" / category
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Generate MDX
                        parsed = parser.parse(docstring)
                        mdx_content = generator.generate(
                            node.name, 
                            parsed,
                            module_name=py_file.stem
                        )
                        
                        # Write file
                        output_file = output_dir / f"{node.name}.mdx"
                        output_file.write_text(mdx_content)
                        print(f"  Generated: {output_file}")
    
    # Process tracker module
    tracker_dir = modules['tracker']
    output_dir = project_root / "docs" / "tracker"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for py_file in tracker_dir.glob("*.py"):
        if py_file.name not in ['__init__.py']:
            print(f"Processing tracker: {py_file.name}")
            try:
                convert_file(py_file, output_dir)
            except Exception as e:
                print(f"Error processing {py_file.name}: {e}")
    
    # Process core modules
    core_modules = ['results', 'video', 'colors', 'validators', 'zones', 'smoother', 'buffer', 'draw']
    output_dirs = {
        'results': project_root / "docs" / "core",
        'video': project_root / "docs" / "core",
        'colors': project_root / "docs" / "core",
        'validators': project_root / "docs" / "core",
        'zones': project_root / "docs" / "utilities",
        'smoother': project_root / "docs" / "utilities",
        'buffer': project_root / "docs" / "utilities",
        'draw': project_root / "docs" / "utilities",
    }
    
    for module_name in core_modules:
        py_file = modules['core'] / f"{module_name}.py"
        if py_file.exists():
            output_dir = output_dirs[module_name]
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Processing core module: {module_name}.py")
            try:
                convert_file(py_file, output_dir)
            except Exception as e:
                print(f"Error processing {module_name}.py: {e}")


if __name__ == "__main__":
    main()