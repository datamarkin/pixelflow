#!/usr/bin/env python3
"""
Unified Documentation Generator for PixelFlow

Ultra-simple system that mirrors your exact code structure:
- 1 Python file = 1 MDX file
- Same folder structure
- No configuration needed

Usage:
    python scripts/docs/generate_docs.py
    python scripts/docs/generate_docs.py --module annotators  # specific module only
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from docstring_parser import DocstringParser, extract_items_from_file, get_exports_from_init
from mdx_generator import MDXGenerator


class DocumentationGenerator:
    """Main documentation generation orchestrator."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pixelflow_dir = project_root / 'pixelflow'
        self.docs_dir = project_root / 'docs'
        self.parser = DocstringParser()
        self.generator = MDXGenerator()
    
    def generate_all_docs(self, specific_module: str = None):
        """Generate documentation for all modules or a specific module."""
        print("🚀 PixelFlow Documentation Generator")
        print("=" * 50)
        
        if specific_module:
            print(f"📋 Generating docs for module: {specific_module}")
            success = self._process_module(specific_module)
            if success:
                print(f"✅ Module '{specific_module}' documented successfully")
            else:
                print(f"❌ Failed to document module '{specific_module}'")
        else:
            print("📋 Discovering and processing all modules...")
            total_files = 0
            success_count = 0
            
            # Process root-level files first
            root_success, root_count = self._process_root_files()
            success_count += root_success
            total_files += root_count
            
            # Process module directories
            for module_dir in self._discover_modules():
                module_success, module_count = self._process_module_directory(module_dir)
                success_count += module_success
                total_files += module_count
            
            print(f"\n📊 Summary: {success_count}/{total_files} files documented successfully")
        
        print(f"📁 Output directory: {self.docs_dir}")
    
    def _discover_modules(self) -> List[Path]:
        """Discover all module directories in pixelflow."""
        modules = []
        for path in self.pixelflow_dir.iterdir():
            if path.is_dir() and not path.name.startswith('_') and not path.name.startswith('.'):
                # Check if it has an __init__.py
                if (path / '__init__.py').exists():
                    modules.append(path)
        return sorted(modules)
    
    def _process_root_files(self) -> tuple[int, int]:
        """Process Python files in the pixelflow root directory."""
        success_count = 0
        total_files = 0
        
        for py_file in self.pixelflow_dir.glob('*.py'):
            if py_file.name.startswith('_'):
                continue
            
            total_files += 1
            if self._process_single_file(py_file, self.docs_dir):
                success_count += 1
        
        return success_count, total_files
    
    def _process_module_directory(self, module_dir: Path) -> tuple[int, int]:
        """Process all Python files in a module directory."""
        module_name = module_dir.name
        print(f"\n🔧 Processing module: {module_name}")
        
        success_count = 0
        total_files = 0
        
        # Get exports from __init__.py
        init_file = module_dir / '__init__.py'
        exports = get_exports_from_init(str(init_file))
        print(f"   📤 Exports found: {len(exports)} items")
        
        # Process each Python file in the module
        output_dir = self.docs_dir / module_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for py_file in module_dir.glob('*.py'):
            if py_file.name.startswith('_'):
                continue
            
            total_files += 1
            if self._process_single_file(py_file, output_dir, exports):
                success_count += 1
        
        return success_count, total_files
    
    def _process_module(self, module_name: str) -> bool:
        """Process a specific module."""
        module_dir = self.pixelflow_dir / module_name
        if not module_dir.exists() or not module_dir.is_dir():
            print(f"❌ Module '{module_name}' not found")
            return False
        
        success_count, total_files = self._process_module_directory(module_dir)
        return success_count > 0
    
    def _process_single_file(self, py_file: Path, output_dir: Path, exports: List[str] = None) -> bool:
        """Process a single Python file and generate corresponding MDX."""
        try:
            print(f"   📄 Processing: {py_file.name}")
            
            # Extract items and module docstring from file
            extraction_result = extract_items_from_file(str(py_file), exports)
            items = extraction_result['items']
            module_docstring = extraction_result['module_docstring']
            
            # Filter to only exported items
            if exports:
                items = [item for item in items if item['name'] in exports]
            
            if not items:
                print(f"   ⚠️  No exportable items found in {py_file.name}")
                return False
            
            # Parse docstrings for each item
            for item in items:
                if item['docstring']:
                    item['parsed_doc'] = self.parser.parse(item['docstring'], item['ast_node'])
                else:
                    item['parsed_doc'] = {}
            
            # Parse module docstring if available
            parsed_module_doc = {}
            if module_docstring:
                parsed_module_doc = self.parser.parse(module_docstring)
            
            # Generate MDX content
            file_stem = py_file.stem
            relative_module_path = str(py_file.parent.relative_to(self.pixelflow_dir))
            
            mdx_content = self.generator.generate_for_file(items, file_stem, relative_module_path, parsed_module_doc)
            
            # Write MDX file
            output_file = output_dir / f"{file_stem}.mdx"
            output_file.write_text(mdx_content)
            
            print(f"   ✅ Generated: {output_file.relative_to(self.project_root)}")
            return True
            
        except Exception as e:
            print(f"   ❌ Error processing {py_file.name}: {e}")
            return False
    
    def clean_docs(self):
        """Clean the docs directory."""
        if self.docs_dir.exists():
            print(f"🧹 Cleaning docs directory: {self.docs_dir}")
            import shutil
            shutil.rmtree(self.docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate PixelFlow documentation from Python docstrings'
    )
    parser.add_argument(
        '--module', '-m',
        help='Generate docs for specific module only (e.g., annotators, detections)'
    )
    parser.add_argument(
        '--clean', '-c',
        action='store_true',
        help='Clean docs directory before generating'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without generating files'
    )
    
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent  # scripts/docs -> scripts -> project_root
    
    # Validate pixelflow directory exists
    pixelflow_dir = project_root / 'pixelflow'
    if not pixelflow_dir.exists():
        print(f"❌ Error: PixelFlow source directory not found at {pixelflow_dir}")
        sys.exit(1)
    
    # Initialize generator
    doc_gen = DocumentationGenerator(project_root)
    
    # Clean if requested
    if args.clean:
        doc_gen.clean_docs()
    
    # Dry run mode
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be generated")
        print("\nWould process:")
        
        # Show root files
        root_files = list(pixelflow_dir.glob('*.py'))
        root_files = [f for f in root_files if not f.name.startswith('_')]
        if root_files:
            print("  Root files:")
            for f in root_files:
                print(f"    📄 {f.name} → docs/{f.stem}.mdx")
        
        # Show module directories
        modules = doc_gen._discover_modules()
        for module_dir in modules:
            print(f"  {module_dir.name}/ module:")
            py_files = list(module_dir.glob('*.py'))
            py_files = [f for f in py_files if not f.name.startswith('_')]
            for f in py_files:
                print(f"    📄 {f.name} → docs/{module_dir.name}/{f.stem}.mdx")
        
        return
    
    # Generate documentation
    try:
        doc_gen.generate_all_docs(args.module)
        print("\n🎉 Documentation generation complete!")
        
    except KeyboardInterrupt:
        print("\n⏸️  Documentation generation interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()