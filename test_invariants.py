"""
Invariants for the ingest contract.

No key, no network, no model, no database, no spend.

Two habits are carried over from the previous project and are visible here.

  * The suite counts its own assertions and prints the number, because two
    counts were once stated from memory and both were wrong.

  * Every rule ships with a control that proves it can fail. A check that
    cannot fail is not a check, and this suite fails if a control stops
    failing.

Run:  python test_invariants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import make_fixtures
from ingest_contract import (
    ConversionFailed,
    Verdict,
    check_file,
    check_folder,
    detect_family,
    read_or_refuse,
)

HERE = Path(__file__).parent
GOOD = HERE / "fixtures" / "good"
BAD = HERE / "fixtures" / "bad"

RUN: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RUN.append(name)
    if condition:
        print(f"ok   {name}")
    else:
        FAILED.append(name)
        print(f"FAIL {name}" + (f"\n       {detail}" if detail else ""))


def control(name: str, condition_that_must_be_false: bool, detail: str = "") -> None:
    """A control passes when the thing it describes is NOT true.

    Used to prove a rule can fail. If the rule is silently removed, the
    control's condition becomes true and this goes red.
    """
    RUN.append(f"control: {name}")
    if not condition_that_must_be_false:
        print(f"ok   control: {name}")
    else:
        FAILED.append(f"control: {name}")
        print(f"FAIL control: {name}" + (f"\n       {detail}" if detail else ""))


# ---------------------------------------------------------------------------
make_fixtures.build()
# ---------------------------------------------------------------------------

# --- the bytes are read, not the name --------------------------------------

check("a real pdf is detected as pdf",
      detect_family(GOOD / "quarterly-review.pdf") == "pdf",
      f"got {detect_family(GOOD / 'quarterly-review.pdf')}")

check("a real docx is detected as ooxml-word",
      detect_family(GOOD / "meeting-notes.docx") == "ooxml-word",
      f"got {detect_family(GOOD / 'meeting-notes.docx')}")

check("markdown is detected as text",
      detect_family(GOOD / "team-handbook.md") == "text")

check("a pdf wearing a .docx name is still detected as pdf",
      detect_family(BAD / "meeting-notes-2025-01-08.docx") == "pdf",
      "the sniffer must not consult the extension at all")

check("a png wearing a .md name is not detected as text",
      detect_family(BAD / "diagram.md") != "text")

# --- what the contract accepts ---------------------------------------------

check("a correctly named pdf is accepted",
      check_file(GOOD / "quarterly-review.pdf").ok)

check("a correctly named docx is accepted",
      check_file(GOOD / "meeting-notes.docx").ok)

check("a correctly named markdown file is accepted",
      check_file(GOOD / "team-handbook.md").ok)

# --- what it rejects, and whether it says why ------------------------------

v_liar = check_file(BAD / "meeting-notes-2025-01-08.docx")
check("a pdf named .docx is rejected", not v_liar.ok)
check("the rejection names both what was claimed and what was found",
      "ooxml-word" in v_liar.reason and "pdf" in v_liar.reason,
      f"reason was: {v_liar.reason!r}")

check("the rule is not one-directional: a docx named .pdf is rejected",
      not check_file(BAD / "handbook.pdf").ok)

check("a binary named .md is rejected",
      not check_file(BAD / "diagram.md").ok)

check("an empty file is rejected",
      not check_file(BAD / "empty.txt").ok)

check("an extension the pipeline cannot read is rejected",
      not check_file(HERE / "make_fixtures.py").ok)

check("a path that does not exist is rejected rather than raising",
      not check_file(HERE / "nothing-here.pdf").ok)

# --- the control: prove the extension rule can fail ------------------------
# With the rule switched off, the liar must pass. If it still fails, then
# something other than the rule is rejecting it and the rule is unproven.

control("the pdf named .docx still fails with the extension rule off",
        not check_file(BAD / "meeting-notes-2025-01-08.docx",
                       enforce_extension=False).ok,
        "with the rule off this file must pass, or the rule is not what rejects it")

control("an empty file passes with the extension rule off",
        check_file(BAD / "empty.txt", enforce_extension=False).ok,
        "emptiness must be rejected by its own rule, not by the extension rule")

# --- deciding never returns content ----------------------------------------

check("the verdict carries no document text",
      not any(f in Verdict.__dataclass_fields__ for f in ("text", "content", "markdown")),
      f"fields are {tuple(Verdict.__dataclass_fields__)}")

# --- reading refuses instead of inventing ----------------------------------
# This is the behavioural form of the rule. It is the primary assertion,
# because a source-level check can pass while the call it guards is broken:
# that is exactly what happened in IA-148 on the previous project.


def _reader_that_explodes(path: Path) -> str:
    raise RuntimeError("the converter fell over")


def _reader_that_returns_nothing(path: Path) -> str:
    return "   "


def _reader_that_works(path: Path) -> str:
    return "real extracted text"


raised = False
try:
    read_or_refuse(GOOD / "quarterly-review.pdf", _reader_that_explodes)
except ConversionFailed:
    raised = True
check("a reader that raises produces a refusal, not a string", raised)

raised = False
try:
    read_or_refuse(GOOD / "quarterly-review.pdf", _reader_that_returns_nothing)
except ConversionFailed:
    raised = True
check("a reader that returns blank text produces a refusal", raised)

raised = False
try:
    read_or_refuse(BAD / "meeting-notes-2025-01-08.docx", _reader_that_works)
except ConversionFailed:
    raised = True
check("a file the contract rejects is never handed to the reader", raised)

check("a good file plus a working reader returns the reader's text",
      read_or_refuse(GOOD / "quarterly-review.pdf", _reader_that_works)
      == "real extracted text")

# The secondary, source-level form. Kept because the upstream defect was a
# bare `except` specifically, and labelled secondary because the assertion
# above is the one that would catch a regression in behaviour.
def _bare_excepts(source_path: Path) -> list[int]:
    """Line numbers carrying a bare `except:`, ignoring comments."""
    found = []
    for n, raw in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if line == "except:":
            found.append(n)
    return found


check("the contract contains no bare except (secondary, source-level)",
      _bare_excepts(HERE / "ingest_contract.py") == [],
      "a bare except is how an error became a document upstream")

control("the bare-except scanner finds nothing in a file that has one",
        _bare_excepts(HERE / "fixtures" / "has_bare_except.py") == [],
        "the scanner must find the planted one, or it proves nothing")

# --- folder scanning --------------------------------------------------------

good_verdicts = check_folder(GOOD)
check("every fixture in good/ is accepted",
      all(v.ok for v in good_verdicts),
      str([v.line() for v in good_verdicts if not v.ok]))

bad_verdicts = check_folder(BAD)
check("every fixture in bad/ is rejected",
      all(not v.ok for v in bad_verdicts),
      str([v.line() for v in bad_verdicts if v.ok]))

check("scanning is deterministic",
      [v.path.name for v in check_folder(GOOD)] == [v.path.name for v in good_verdicts])

check("the scan skips files the pipeline would never pick up",
      all(v.path.suffix.lower() != ".py" for v in check_folder(HERE)))

# ---------------------------------------------------------------------------

print()
if FAILED:
    print(f"{len(FAILED)} of {len(RUN)} invariants FAILED:")
    for name in FAILED:
        print(f"  - {name}")
    sys.exit(1)

print(f"all {len(RUN)} invariants hold")
