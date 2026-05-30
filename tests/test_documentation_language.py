"""Regression guard for the documentation-language pass (v0.4.9).

We translated every docstring, comment, README, and CHANGELOG from
Chinese to English in v0.4.9. This test makes sure the property holds
in any future commit: no CJK characters, no em-dashes, and no prose
``->`` arrows leak back in.

Greek letters and mathematical inequalities are intentionally allowed
because formulas in docstrings (e.g. ``mu * lambda_2 / L``, ``r_i^T
Sigma^{-1} r_i``, ``s_A``) often need them and they are not language
residue.
"""

from __future__ import annotations

import glob
import os
import re

import pytest

# CJK Unified Ideographs (the bulk of Chinese characters).
_CJK = re.compile(r"[\u4e00-\u9fff]")

# Em-dash and en-dash. The project style file (working-style.md) explicitly
# lists em-dashes as punctuation to avoid; en-dash is rare and easily
# confused with hyphen in plain-text source.
_DASHES = "\u2014\u2013"

# A non-math arrow appearing in prose. The math arrow inside paper
# formulas typically lives in LaTeX or Markdown math fences (which we do
# not have here), so any U+2192 in source is prose.
_PROSE_ARROW = "\u2192"

# Files to scan. We deliberately exclude:
#   - .git, _data, __pycache__, .pytest_cache, .ruff_cache, .mypy_cache
#   - this test file itself (it has to mention the characters by codepoint).
_INCLUDE_PATTERNS = [
    "*.py",
    "tests/*.py",
    "*.md",
    "*.toml",
    "*.cff",
    "*.txt",
    ".github/workflows/*.yml",
]
_EXCLUDE_FILES = {
    os.path.normpath("tests/test_documentation_language.py"),
}


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _collect_files() -> list[str]:
    root = _project_root()
    seen: set[str] = set()
    for pattern in _INCLUDE_PATTERNS:
        for path in glob.glob(os.path.join(root, pattern)):
            rel = os.path.normpath(os.path.relpath(path, root))
            if rel in _EXCLUDE_FILES:
                continue
            seen.add(rel)
    return sorted(seen)


def _read(rel_path: str) -> str:
    with open(os.path.join(_project_root(), rel_path), encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Hard guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path", _collect_files())
def test_no_cjk_characters(rel_path: str) -> None:
    """No file in the project should contain CJK ideographs."""
    text = _read(rel_path)
    matches = _CJK.findall(text)
    assert not matches, (
        f"{rel_path} contains {len(matches)} CJK character(s). "
        f"This project is English-only as of v0.4.9; please translate "
        f"the offending text to English."
    )


@pytest.mark.parametrize("rel_path", _collect_files())
def test_no_em_or_en_dashes(rel_path: str) -> None:
    """Em-dashes and en-dashes break copy and are project-style banned."""
    text = _read(rel_path)
    bad = [ch for ch in _DASHES if ch in text]
    assert not bad, (
        f"{rel_path} contains em-dash / en-dash characters "
        f"({[hex(ord(c)) for c in bad]}). Use ``--`` instead."
    )


@pytest.mark.parametrize("rel_path", _collect_files())
def test_no_prose_arrows(rel_path: str) -> None:
    """The Unicode rightwards arrow in prose should be ``->``."""
    text = _read(rel_path)
    assert _PROSE_ARROW not in text, (
        f"{rel_path} contains the Unicode rightwards arrow (U+2192). "
        f"Use ASCII ``->`` instead."
    )


# ---------------------------------------------------------------------------
# Self-check: the file collector does see the project, not an empty list.
# ---------------------------------------------------------------------------

def test_collector_finds_files() -> None:
    """If the parametrize argument list is empty, the three guards above
    silently pass without checking anything. This sanity test fails
    loudly in that case."""
    files = _collect_files()
    assert len(files) > 20, (
        f"_collect_files() returned only {len(files)} files; "
        "the include patterns are likely broken."
    )
    # Spot check: a few well-known files must be present.
    expected = {"main.py", "config.py", "README.md", "CHANGELOG.md"}
    missing = expected - {os.path.basename(f) for f in files}
    assert not missing, f"file collector missing well-known files: {missing}"
