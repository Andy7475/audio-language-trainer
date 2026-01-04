"""Convert absolute `from src.xxx import y` imports to relative imports inside `src/`.

Usage:
  python scripts/convert_src_to_relative.py [--apply]

By default the script prints a dry-run of proposed changes. Pass `--apply` to overwrite files.

The script only modifies files under the `src/` package and only transforms `from src... import ...` lines.
It computes the correct number of leading dots for relative imports based on the file's package depth.
"""
import argparse
from pathlib import Path
import re


def compute_replacement(file_path: Path, module: str) -> str:
    # module is the dotted path after 'src.' e.g. 'audio.constants'
    module_parts = module.split('.') if module else []
    # file package parts (path segments between src/ and the file directory)
    rel = file_path.relative_to(Path.cwd() / 'src')
    file_pkg_parts = list(rel.parent.parts) if rel.parent.parts else []

    # find common prefix
    common = 0
    for a, b in zip(file_pkg_parts, module_parts):
        if a == b:
            common += 1
        else:
            break

    # If module is inside same package (or a submodule), use a single leading dot
    if common == len(file_pkg_parts):
        leading = '.'
        remaining = module_parts[common:]
    else:
        up_levels = len(file_pkg_parts) - common
        leading = '.' * (up_levels + 1)
        remaining = module_parts[common:]

    if remaining:
        return f"from {leading}{'.'.join(remaining)}"
    else:
        return f"from {leading}"


IMPORT_RE = re.compile(r"^(?P<prefix>\s*)from\s+src\.(?P<module>[A-Za-z0-9_.]+)\s+import\b(?P<rest>.*)$")


def process_file(path: Path, apply: bool=False):
    text = path.read_text(encoding='utf-8')
    new_lines = []
    changed = False
    for line in text.splitlines():
        m = IMPORT_RE.match(line)
        if m:
            prefix = m.group('prefix')
            module = m.group('module')
            rest = m.group('rest')
            replacement = compute_replacement(path, module)
            new_line = f"{prefix}{replacement} import{rest}"
            if new_line != line:
                new_lines.append(new_line)
                changed = True
                continue
        new_lines.append(line)

    if changed:
        print(f"--- {path}")
        print('\n'.join(new_lines))
        print()
        if apply:
            path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='Write changes to files')
    args = ap.parse_args()

    base = Path.cwd() / 'src'
    if not base.exists():
        print('src/ directory not found in current working directory')
        return

    py_files = list(base.rglob('*.py'))
    for p in py_files:
        process_file(p, apply=args.apply)

    if args.apply:
        print('Applied changes.')
    else:
        print('Dry-run complete. Rerun with --apply to modify files.')


if __name__ == '__main__':
    main()
