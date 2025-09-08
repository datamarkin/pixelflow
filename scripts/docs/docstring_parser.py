#!/usr/bin/env python3
"""
Ultra-simple docstring parser for PixelFlow documentation generation.

Extracts structured information from Python docstrings without complex configuration.
"""

import ast
import re
from typing import Dict, List, Optional, Any, Union


class DocstringParser:
    """Parse Python docstrings into structured data for MDX generation."""
    
    def __init__(self):
        self.sections_pattern = re.compile(
            r'^(Args|Returns|Raises|Example|Examples|Notes?|Performance Notes?|See Also):\s*$',
            re.MULTILINE | re.IGNORECASE
        )
    
    def parse(self, docstring: str, ast_node: ast.AST = None) -> Dict[str, Any]:
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
            'notes': []
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
                    self._process_section(result, current_section, section_content, ast_node)
                current_section = section_match.group(1).lower()
                section_content = []
            else:
                section_content.append(line)
        
        # Process last section
        if current_section:
            self._process_section(result, current_section, section_content, ast_node)
        
        return result
    
    def _process_section(self, result: Dict, section: str, content: List[str], ast_node: ast.AST = None):
        """Process a specific docstring section."""
        if section == 'args':
            result['args'] = self._parse_args(content, ast_node)
        elif section == 'returns':
            result['returns'] = self._parse_returns(content)
        elif section == 'raises':
            result['raises'] = self._parse_raises(content)
        elif section in ['example', 'examples']:
            result['examples'] = self._parse_examples(content)
        elif section in ['note', 'notes', 'performance notes']:
            result['notes'].extend(self._parse_notes(content))
    
    def _parse_args(self, lines: List[str], ast_node: ast.AST = None) -> List[Dict]:
        """Parse Args section into structured format."""
        args = []
        current_arg = None
        
        # Extract signature info from AST if available
        signature_info = {}
        if ast_node and isinstance(ast_node, ast.FunctionDef):
            for arg in ast_node.args.args:
                if arg.arg != 'self':  # Skip 'self' parameter
                    signature_info[arg.arg] = {'has_default': False, 'default_value': None}
            
            # Check for default values
            defaults_offset = len(ast_node.args.args) - len(ast_node.args.defaults)
            for i, default in enumerate(ast_node.args.defaults):
                arg_index = defaults_offset + i
                if arg_index < len(ast_node.args.args):
                    arg_name = ast_node.args.args[arg_index].arg
                    if arg_name != 'self':
                        signature_info[arg_name]['has_default'] = True
                        try:
                            if isinstance(default, ast.Constant):
                                signature_info[arg_name]['default_value'] = repr(default.value)
                            elif isinstance(default, ast.Name):
                                signature_info[arg_name]['default_value'] = default.id
                            else:
                                signature_info[arg_name]['default_value'] = 'default'
                        except:
                            signature_info[arg_name]['default_value'] = 'default'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a new parameter
            param_match = re.match(r'^(\w+)\s*\(([^)]+)\)\s*:\s*(.*)$', line)
            if param_match:
                if current_arg:
                    args.append(current_arg)
                
                param_name = param_match.group(1)
                param_type = param_match.group(2)
                param_desc = param_match.group(3)
                
                # Determine if parameter is optional
                is_optional = False
                default_val = None
                
                if param_name in signature_info:
                    is_optional = signature_info[param_name]['has_default']
                    default_val = signature_info[param_name]['default_value']
                else:
                    is_optional = ('optional' in param_desc.lower() or 'Optional' in param_type)
                
                current_arg = {
                    'name': param_name,
                    'type': param_type.strip(),
                    'description': param_desc,
                    'optional': is_optional,
                    'default': default_val
                }
                
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
            
            exc_match = re.match(r'^(\w+(?:Error)?):\s*(.*)$', line)
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
        current_example = {'title': 'Example', 'code': []}
        in_code = False
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('>>>'):
                in_code = True
                current_example['code'].append(stripped[3:].strip())
            elif in_code and (not stripped or stripped.startswith('#')):
                if stripped.startswith('#'):
                    current_example['code'].append(stripped)
                elif not stripped:
                    current_example['code'].append('')
            elif in_code and not stripped.startswith('>>>'):
                # End of code block
                if current_example['code']:
                    examples.append(current_example)
                    current_example = {'title': 'Example', 'code': []}
                in_code = False
        
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


def extract_items_from_file(file_path: str, exported_names: List[str] = None) -> Dict[str, Any]:
    """Extract functions, classes, and module docstring from a Python file.
    
    Returns:
        Dict containing:
            - 'items': List of functions and classes
            - 'module_docstring': Module-level docstring or None
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        items = []
        
        # Extract module-level docstring
        module_docstring = ast.get_docstring(tree)
        
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                # Only include if in exported names (if provided)
                if exported_names is None or node.name in exported_names:
                    docstring = ast.get_docstring(node)
                    items.append({
                        'name': node.name,
                        'type': 'function',
                        'docstring': docstring,
                        'ast_node': node
                    })
            
            elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                # Only include if in exported names (if provided)
                if exported_names is None or node.name in exported_names:
                    docstring = ast.get_docstring(node)
                    items.append({
                        'name': node.name,
                        'type': 'class',
                        'docstring': docstring,
                        'ast_node': node
                    })
        
        return {
            'items': items,
            'module_docstring': module_docstring
        }
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {'items': [], 'module_docstring': None}


def get_exports_from_init(init_file: str) -> List[str]:
    """Extract __all__ list from __init__.py file."""
    try:
        with open(init_file, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        if isinstance(node.value, ast.List):
                            return [elt.s if hasattr(elt, 's') else elt.value 
                                   for elt in node.value.elts 
                                   if hasattr(elt, 's') or hasattr(elt, 'value')]
        
        return []
    
    except Exception as e:
        print(f"Error reading exports from {init_file}: {e}")
        return []