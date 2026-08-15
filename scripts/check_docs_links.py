"""Fail CI on broken relative Markdown links."""

from __future__ import annotations

import re
from pathlib import Path

LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    broken: list[str] = []
    for document in ROOT.rglob("*.md"):
        if any(part in {".git", ".tmp", ".uv-cache", ".venv", "venv"} for part in document.parts):
            continue
        text = document.read_text(encoding="utf-8", errors="replace")
        # Link-like examples inside fenced code are not Markdown hyperlinks.
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        for raw in LINK.findall(text):
            target = raw.split("#", 1)[0].strip()
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue
            candidate = (document.parent / target).resolve()
            if not candidate.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {raw}")
    if broken:
        raise SystemExit("Broken Markdown links:\n" + "\n".join(broken))
    print("Documentation links are valid.")


if __name__ == "__main__":
    main()
