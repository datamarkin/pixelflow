#!/usr/bin/env python3
"""
MDX generator for PixelFlow documentation.

Converts parsed docstring data into Mintlify-compatible MDX format.
"""

from typing import Dict, List, Any
from pathlib import Path


class MDXGenerator:
    """Generate MDX documentation from parsed docstrings."""
    
    def __init__(self):
        self.icon_map = {
            # Annotators
            'blur': 'eye-slash',
            'pixelate': 'grid',
            'box': 'square',
            'filled_box': 'square-fill',
            'mask': 'masks-theater',
            'polygon': 'draw-polygon',
            'oval': 'circle',
            'label': 'tag',
            'anchors': 'anchor',
            'zones': 'map-pin',
            'crossings': 'arrows-cross',
            'grid_overlay': 'grid-3x3',
            
            # Core classes
            'Detection': 'crosshairs',
            'Detections': 'list',
            'KeyPoint': 'circle-dot',
            'Buffer': 'layers',
            'TimeTracker': 'stopwatch',
            'Zones': 'map',
            'Crossings': 'route',
            
            # Utilities
            'from_ultralytics': 'brand-pytorch',
            'from_detectron2': 'brand-facebook',
            'from_datamarkin_api': 'api',
            'filter_by_confidence': 'filter',
            'filter_by_class_id': 'category',
            'remove_duplicates': 'copy-minus',
            
            # Default
            'function': 'function',
            'class': 'cpu'
        }
    
    def generate_for_file(self, items: List[Dict], file_stem: str, module_path: str = '', module_doc: Dict = None) -> str:
        """Generate MDX content for a file containing multiple items."""
        if len(items) == 1:
            # Single item - generate direct documentation
            item = items[0]
            return self._generate_single_item(item, file_stem, module_doc)
        elif len(items) > 1:
            # Multiple items - generate overview with individual sections
            return self._generate_multi_item(items, file_stem, module_path, module_doc)
        else:
            return self._generate_empty_file(file_stem, module_doc)
    
    def _generate_single_item(self, item: Dict, file_stem: str, module_doc: Dict = None) -> str:
        """Generate MDX for a single function or class."""
        parsed_doc = item.get('parsed_doc', {})
        name = item['name']
        item_type = item['type']
        
        lines = []
        
        # Frontmatter
        icon = self.icon_map.get(name, self.icon_map.get(item_type, 'function'))
        title = name.capitalize()  # Capitalize first letter
        lines.extend([
            '---',
            f'title: {title}',
            f'description: {parsed_doc.get("summary", "")}',
            # f'icon: "{icon}"',
            '---',
            ''
        ])
        
        # Overview - only add if there's additional content beyond the summary
        if parsed_doc.get('description') or (module_doc and module_doc.get('summary')):
            lines.extend([
                '## Overview',
                ''
            ])
            
            # Add module context if available and different from item summary
            if module_doc and module_doc.get('summary'):
                lines.append(module_doc['summary'])
                lines.append('')
                
                if module_doc.get('description'):
                    lines.append(module_doc['description'])
                    lines.append('')
            
            # Add item description if different from summary
            if parsed_doc.get('description') and parsed_doc['description'] != parsed_doc.get('summary', ''):
                lines.append(parsed_doc['description'])
                lines.append('')
        
        # Function signature or class info
        if item_type == 'function':
            self._add_function_signature(lines, name, parsed_doc)
        else:
            self._add_class_info(lines, name, parsed_doc)
        
        # Parameters/Constructor args
        if parsed_doc.get('args'):
            lines.extend(['## Parameters', ''])
            for arg in parsed_doc['args']:
                self._add_parameter(lines, arg)
        
        # Returns
        if parsed_doc.get('returns'):
            self._add_returns_section(lines, parsed_doc['returns'])
        
        # Examples
        if parsed_doc.get('examples'):
            self._add_examples_section(lines, parsed_doc['examples'])
        
        # Error handling
        if parsed_doc.get('raises'):
            self._add_raises_section(lines, parsed_doc['raises'])
        
        # Notes
        if parsed_doc.get('notes'):
            self._add_notes_section(lines, parsed_doc['notes'])
        
        return '\n'.join(lines)
    
    def _generate_multi_item(self, items: List[Dict], file_stem: str, module_path: str, module_doc: Dict = None) -> str:
        """Generate MDX for multiple items in one file."""
        lines = []
        
        # Determine title and description
        title = file_stem.replace('_', ' ').title()
        
        # Count items
        functions = [item for item in items if item['type'] == 'function']
        classes = [item for item in items if item['type'] == 'class']
        
        # Use module docstring summary for description if available, otherwise generate
        if module_doc and module_doc.get('summary'):
            description = module_doc['summary']
        else:
            description = f"Contains {len(classes)} classes and {len(functions)} functions"
            if len(classes) > 0 and len(functions) == 0:
                description = f"Contains {len(classes)} classes for {file_stem}"
            elif len(functions) > 0 and len(classes) == 0:
                description = f"Contains {len(functions)} functions for {file_stem}"
        
        # Frontmatter
        icon = self.icon_map.get(file_stem, 'cpu')
        lines.extend([
            '---',
            f'title: {title}',
            f'description: {description}',
            # f'icon: "{icon}"',
            '---',
            ''
        ])
        
        # Module overview
        lines.extend([
            # f'# {title}',
            ''
        ])
        
        # Add module description if available and different from title
        if module_doc:
            if module_doc.get('summary'):
                # lines.append(module_doc['summary'])
                lines.append('')
            
            if module_doc.get('description'):
                lines.append(module_doc['description'])
                lines.append('')
        else:
            # Fallback to generated description
            lines.append(description + '.')
            lines.append('')
        
        # Add overview cards if multiple items
        if len(items) > 1:
            self._add_items_overview(lines, items)
        
        # Generate documentation for each item
        for item in items:
            parsed_doc = item.get('parsed_doc', {})
            name = item['name']
            item_type = item['type']
            
            lines.extend([
                f'## {name}',
                ''
            ])
            
            if parsed_doc.get('summary'):
                lines.append(parsed_doc['summary'])
                lines.append('')
            
            if parsed_doc.get('description'):
                lines.append(parsed_doc['description'])
                lines.append('')
            
            # Function signature or class constructor
            if item_type == 'function':
                self._add_function_signature(lines, name, parsed_doc)
            
            # Parameters
            if parsed_doc.get('args'):
                lines.extend(['### Parameters', ''])
                for arg in parsed_doc['args']:
                    self._add_parameter(lines, arg)
            
            # Returns
            if parsed_doc.get('returns'):
                lines.extend(['### Returns', ''])
                returns = parsed_doc['returns']
                lines.extend([
                    f'<ResponseField name="result" type="{returns.get("type", "Any")}">',
                    f'  {returns.get("description", "")}',
                    '</ResponseField>',
                    ''
                ])
            
            # Examples (condensed)
            if parsed_doc.get('examples'):
                lines.extend(['### Example', ''])
                example = parsed_doc['examples'][0]  # Just first example
                lines.extend([
                    f'```python {example.get("title", "Usage")}',
                    *example.get('code', []),
                    '```',
                    ''
                ])
        
        return '\n'.join(lines)
    
    def _generate_empty_file(self, file_stem: str, module_doc: Dict = None) -> str:
        """Generate MDX for empty file."""
        title = file_stem.replace('_', ' ').title()
        
        # Use module docstring if available
        if module_doc and module_doc.get('summary'):
            description = module_doc['summary']
            content = module_doc.get('description', 'This module contains implementation details.')
        else:
            description = f"{title} module"
            content = "This module is currently empty or contains only private functions."
        
        return f'''---
title: {title}
description: {description}
---

# {title}

{content}
'''
    
    def _add_function_signature(self, lines: List[str], name: str, parsed_doc: Dict):
        """Add function signature section."""
        args = parsed_doc.get('args', [])
        returns = parsed_doc.get('returns', {})
        
        lines.extend([
            '## Function Signature',
            '',
            '```python',
            f'{name}(',
        ])
        
        for i, arg in enumerate(args):
            comma = ',' if i < len(args) - 1 else ''
            default = f' = {arg.get("default")}' if arg.get('default') else ''
            lines.append(f'    {arg["name"]}: {arg["type"]}{default}{comma}')
        
        lines.append(f') -> {returns.get("type", "Any")}')
        lines.extend(['```', ''])
    
    def _add_class_info(self, lines: List[str], name: str, parsed_doc: Dict):
        """Add class information section."""
        lines.extend([
            '## Class Overview',
            '',
            f'The `{name}` class provides structured data management for {name.lower()} operations.',
            ''
        ])
    
    def _add_parameter(self, lines: List[str], arg: Dict):
        """Add parameter documentation."""
        required_attr = 'optional' if arg.get('optional') else 'required'
        default_attr = f'default="{arg.get("default")}"' if arg.get('default') else ''
        
        param_line = f'<ParamField path="{arg["name"]}" type="{arg["type"]}" {required_attr}'
        if default_attr:
            param_line += f' {default_attr}'
        param_line += '>'
        
        lines.extend([
            param_line,
            f'  {arg["description"]}',
            '</ParamField>',
            ''
        ])
    
    def _add_returns_section(self, lines: List[str], returns: Dict):
        """Add returns section."""
        lines.extend([
            '## Returns',
            '',
            f'<ResponseField name="result" type="{returns.get("type", "Any")}">',
            f'  {returns.get("description", "")}',
            '</ResponseField>',
            ''
        ])
    
    def _add_examples_section(self, lines: List[str], examples: List[Dict]):
        """Add examples section."""
        lines.extend(['## Examples', '', '<CodeGroup>', ''])
        
        for example in examples:
            title = example.get('title', 'Example')
            lines.extend([
                f'```python {title}',
                *example.get('code', []),
                '```',
                ''
            ])
        
        lines.extend(['</CodeGroup>', ''])
    
    def _add_raises_section(self, lines: List[str], raises: List[Dict]):
        """Add error handling section."""
        lines.extend([
            '## Error Handling',
            '',
            '<Warning>',
            '  This function may raise the following exceptions:'
        ])
        
        for exc in raises:
            lines.append(f'  - **{exc["exception"]}**: {exc["description"]}')
        
        lines.extend(['</Warning>', ''])
    
    def _add_notes_section(self, lines: List[str], notes: List[str]):
        """Add notes section."""
        lines.extend([
            '## Notes',
            '',
            '<Note>'
        ])
        
        for note in notes:
            lines.append(f'  - {note}')
        
        lines.extend(['</Note>', ''])
    
    def _add_items_overview(self, lines: List[str], items: List[Dict]):
        """Add overview cards for multiple items."""
        functions = [item for item in items if item['type'] == 'function']
        classes = [item for item in items if item['type'] == 'class']
        
        if classes:
            lines.extend(['## Classes', ''])
            for cls in classes:
                parsed_doc = cls.get('parsed_doc', {})
                lines.append(f'- [`{cls["name"]}`](#{cls["name"].lower()}) - {parsed_doc.get("summary", "")}')
            lines.append('')
        
        if functions:
            lines.extend(['## Functions', ''])
            for func in functions:
                parsed_doc = func.get('parsed_doc', {})
                lines.append(f'- [`{func["name"]}`](#{func["name"].lower()}) - {parsed_doc.get("summary", "")}')
            lines.append('')