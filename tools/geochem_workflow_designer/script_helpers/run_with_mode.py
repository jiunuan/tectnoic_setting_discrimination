from __future__ import annotations

import argparse
import re
from pathlib import Path


def read_text_with_fallback(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode: {path}")


def patch_mode(source: str, mode: str) -> str:
    pattern = re.compile(r"(?m)^MODE\s*=\s*['\"](?:train|predict)['\"]")
    replacement = f'MODE = "{mode}"'
    if not pattern.search(source):
        raise RuntimeError("MODE assignment not found in target script.")
    return pattern.sub(replacement, source, count=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a hard-coded MODE script without editing the original file on disk.")
    parser.add_argument("--script", required=True, help="Target python file")
    parser.add_argument("--mode", required=True, choices=["train", "predict"], help="Desired MODE value")
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    source = read_text_with_fallback(script_path)
    patched = patch_mode(source, args.mode)
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(script_path),
    }
    exec(compile(patched, str(script_path), "exec"), globals_dict)


if __name__ == "__main__":
    main()
