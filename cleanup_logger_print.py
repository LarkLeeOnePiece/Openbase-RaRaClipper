#!/usr/bin/env python3
"""
Script to clean up logger and print calls from Python files.
"""

import re
import os
from pathlib import Path
from typing import Tuple, List

class LoggerPrintCleaner:
    def __init__(self):
        self.files_to_clean = [
            "d:/Projects/3DGS_Interaction/RaRaClipper/renderer/util.py",
            "d:/Projects/3DGS_Interaction/RaRaClipper/gaussian-splatting/scene/dataset_readers.py",
            "d:/Projects/3DGS_Interaction/RaRaClipper/gaussian-splatting/scene/gaussian_model.py",
            "d:/Projects/3DGS_Interaction/RaRaClipper/gaussian-splatting/scene/__init__.py",
            "d:/Projects/3DGS_Interaction/RaRaClipper/widgets/cam_widget.py",
            "d:/Projects/3DGS_Interaction/RaRaClipper/splatviz_utils/gui_utils/url.py",
            "d:/Projects/3DGS_Interaction/RaRaClipper/splatviz.py",
            "d:/Projects/3DGS_Interaction/RaRaClipper/renderer/attach_renderer.py",
            "d:/Projects/3DGS_Interaction/RaRaClipper/renderer/gaussian_decoder_renderer.py",
            "d:/Projects/3DGS_Interaction/RaRaClipper/gaussian-splatting/arguments/__init__.py",
            "d:/Projects/3DGS_Interaction/RaRaClipper/gaussian-splatting/utils/camera_utils.py",
            "d:/Projects/3DGS_Interaction/RaRaClipper/gaussian-splatting/scene/cameras.py",
            "d:/Projects/3DGS_Interaction/RaRaClipper/widgets/load_widget_ply.py",
        ]
        self.stats = {}

    def clean_file(self, filepath: str) -> Tuple[int, int, int]:
        """
        Clean a single file from logger and print calls.
        Returns: (lines_removed, logger_calls_removed, print_calls_removed)
        """
        if not os.path.exists(filepath):
            print(f"  WARNING: File not found: {filepath}")
            return 0, 0, 0

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            original_lines = content.count('\n')

        lines = content.split('\n')
        new_lines = []
        logger_removed = 0
        print_removed = 0
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            # Check if line should be removed
            should_remove = False

            # Pattern 1: logger.info() or logger.error() calls (not logger.add())
            if re.search(r'logger\.(info|error|debug|warning)\s*\(', line):
                if not re.search(r'logger\.add\s*\(', line):
                    should_remove = True
                    logger_removed += 1

            # Pattern 2: print() calls
            elif re.search(r'\bprint\s*\(', line):
                should_remove = True
                print_removed += 1

            # Pattern 3: Commented logger calls
            elif re.search(r'#\s*logger\.(info|error|debug|warning|add)\s*\(', line):
                should_remove = True
                logger_removed += 1

            # Pattern 4: Commented print calls
            elif re.search(r'#\s*print\s*\(', line):
                should_remove = True
                print_removed += 1

            if should_remove:
                # If this is a multi-line call, we need to find the closing parenthesis
                if line.count('(') > line.count(')'):
                    paren_count = line.count('(') - line.count(')')
                    j = i + 1
                    while j < len(lines) and paren_count > 0:
                        paren_count += lines[j].count('(') - lines[j].count(')')
                        j += 1
                    # Skip all lines until we find the closing parenthesis
                    i = j - 1
            else:
                new_lines.append(line)

            i += 1

        # Merge excessive blank lines
        merged_lines = []
        blank_count = 0
        for line in new_lines:
            if line.strip() == '':
                blank_count += 1
                if blank_count <= 2:  # Allow max 2 consecutive blank lines
                    merged_lines.append(line)
            else:
                blank_count = 0
                merged_lines.append(line)

        new_content = '\n'.join(merged_lines)
        new_lines_count = new_content.count('\n')

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        lines_removed = original_lines - new_lines_count
        return lines_removed, logger_removed, print_removed

    def run(self):
        """Run the cleanup on all files."""
        print("=" * 80)
        print("Logger and Print Call Cleanup Script")
        print("=" * 80)
        print()

        total_lines_removed = 0
        total_logger_removed = 0
        total_print_removed = 0
        successful_files = 0

        for filepath in self.files_to_clean:
            # Normalize path
            filepath = filepath.replace("\\", "/")
            
            print(f"Processing: {filepath}")
            
            try:
                lines_removed, logger_removed, print_removed = self.clean_file(filepath)
                
                if lines_removed > 0 or logger_removed > 0 or print_removed > 0:
                    print(f"  OK Cleaned successfully")
                    print(f"    - Lines removed: {lines_removed}")
                    print(f"    - Logger calls removed: {logger_removed}")
                    print(f"    - Print calls removed: {print_removed}")
                    total_lines_removed += lines_removed
                    total_logger_removed += logger_removed
                    total_print_removed += print_removed
                    successful_files += 1
                else:
                    print(f"  OK No cleanup needed (already clean)")
                    successful_files += 1
            except Exception as e:
                print(f"  ERROR: {e}")

            print()

        # Summary
        print("=" * 80)
        print("CLEANUP SUMMARY")
        print("=" * 80)
        print(f"Files processed: {len(self.files_to_clean)}")
        print(f"Files successfully cleaned: {successful_files}")
        print(f"Total lines removed: {total_lines_removed}")
        print(f"Total logger calls removed: {total_logger_removed}")
        print(f"Total print calls removed: {total_print_removed}")
        print()

        # Verify syntax of cleaned files
        print("=" * 80)
        print("SYNTAX VERIFICATION")
        print("=" * 80)
        
        import ast
        syntax_errors = 0
        
        for filepath in self.files_to_clean:
            if not os.path.exists(filepath):
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                ast.parse(code)
                print(f"OK {filepath}")
            except SyntaxError as e:
                print(f"ERROR {filepath}")
                print(f"  Error: {e}")
                syntax_errors += 1

        print()
        if syntax_errors == 0:
            print("All files have valid syntax!")
        else:
            print(f"WARNING: {syntax_errors} file(s) have syntax errors!")
        
        print()
        print("=" * 80)
        print("Cleanup process completed!")
        print("=" * 80)


if __name__ == "__main__":
    cleaner = LoggerPrintCleaner()
    cleaner.run()
