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

import chunker
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


# ===========================================================================
# the chunker
# ===========================================================================

SAMPLES = {
    "prose": (
        "Scaled Dot-Product Attention\n\n"
        "We call our particular attention \"Scaled Dot-Product Attention\". The "
        "input consists of queries and keys of dimension dk.\n\n"
        "We compute the dot products of the query with all keys, divide each by "
        "the square root of dk, and apply a softmax.\n"
    ),
    "one long word": "Supercalifragilisticexpialidocious" * 4,
    "no boundaries at all": "x" * 300,
    "single character": "x",
    "whitespace only": "   \n\n   \n",
    "unicode": "Índice de contenidos. Sección uno. Sección dos. Ñandú, café, año.\n" * 3,
}

for _tok in chunker.SUITE_TOKENIZERS:
    _n = _tok.name
    for _label, _text in SAMPLES.items():
        _chunks = chunker.chunk_text(_text, document_id="d", tokenizer=_tok,
                                     overlap_tokens=2)

        check(f"[{_n}] no chunk exceeds the budget, on {_label!r}",
              all(c.token_count <= _tok.budget for c in _chunks),
              f"max was {max((c.token_count for c in _chunks), default=0)} "
              f"against a budget of {_tok.budget}")

        check(f"[{_n}] the spans leave no gap, on {_label!r}",
              chunker.coverage_gaps(_text, _chunks) == [],
              f"gaps: {chunker.coverage_gaps(_text, _chunks)}")

        _rebuilt = "".join(_text[c.char_start:c.char_end]
                           for c in sorted(_chunks, key=lambda c: c.char_start))
        check(f"[{_n}] the spans rebuild the source exactly, on {_label!r}",
              _rebuilt == _text,
              "nothing may be dropped, and nothing may be truncated")

        check(f"[{_n}] every chunk moves forward, on {_label!r}",
              all(b.char_start > a.char_start for a, b in zip(_chunks, _chunks[1:])),
              "a chunk that does not advance is an infinite loop waiting to happen")

        check(f"[{_n}] every chunk records its tokenizer and budget, on {_label!r}",
              all(c.tokenizer == _tok.name and c.budget == _tok.budget
                  for c in _chunks))

        check(f"[{_n}] every split level is one the module declares, on {_label!r}",
              all(c.split_level in chunker.SPLIT_LEVELS for c in _chunks))

# --- the budget is what drives the packing ---------------------------------
# Control. With a budget large enough to hold everything, the splitter must
# produce exactly one chunk. If it still split, it is splitting on structure
# and the budget invariants above would pass without the budget doing any work.

_prose = SAMPLES["prose"]
_roomy = chunker.WordTokenizer(budget=10_000)
control("a budget that fits everything still produces more than one chunk",
        len(chunker.chunk_text(_prose, document_id="d", tokenizer=_roomy)) > 1,
        "the splitter must be driven by the budget, not by paragraph structure")

_tight = chunker.WordTokenizer(budget=4)
check("a tighter budget produces more chunks than a generous one",
      len(chunker.chunk_text(_prose, document_id="d", tokenizer=_tight))
      > len(chunker.chunk_text(_prose, document_id="d", tokenizer=chunker.WordTokenizer(20))))

# --- a word longer than the whole budget -----------------------------------

_giant = "a" * 500
_cg = chunker.CharGroupTokenizer(budget=4, chars_per_token=4)
_gc = chunker.chunk_text(_giant, document_id="d", tokenizer=_cg)
check("a single word longer than the budget is split, not dropped",
      "".join(_giant[c.char_start:c.char_end] for c in _gc) == _giant)
check("and the hard split is recorded rather than silent",
      "character" in {c.split_level for c in _gc},
      f"levels seen: {sorted({c.split_level for c in _gc})}")
check("the report counts how many chunks needed a hard split",
      chunker.report(_gc, _cg).by_level["character"] > 0,
      "a compromise nobody counts is a compromise nobody can argue with")

# --- empty input ------------------------------------------------------------

check("empty text produces no chunks at all, not one empty chunk",
      chunker.chunk_text("", document_id="d",
                         tokenizer=chunker.WordTokenizer(8)) == [])

# --- identity ---------------------------------------------------------------

_t = chunker.WordTokenizer(8)
_a = chunker.chunk_text(_prose, document_id="d", tokenizer=_t)
_b = chunker.chunk_text(_prose, document_id="d", tokenizer=_t)
check("chunking the same text twice gives the same ids",
      [c.chunk_id for c in _a] == [c.chunk_id for c in _b],
      "a fresh uuid per run is what both upstream projects do, and it removes "
      "the one property an explicit id exists to provide")

_appended = chunker.chunk_text(_prose + " and one more sentence here.",
                               document_id="d", tokenizer=_t)
check("appending text leaves the earlier chunks' ids untouched",
      [c.chunk_id for c in _appended][:len(_a) - 1] == [c.chunk_id for c in _a][:len(_a) - 1],
      "this is the property a content-derived id exists to give: a re-ingest "
      "updates in place instead of duplicating")
check("but the document as a whole gets a different set of ids",
      {c.chunk_id for c in _appended} != {c.chunk_id for c in _a})

_edited = chunker.chunk_text("Totally different opening. " + _prose,
                             document_id="d", tokenizer=_t)
check("changing the first chunk changes its id",
      _edited[0].chunk_id != _a[0].chunk_id)

check("the same text under a different tokenizer gives different ids",
      _a[0].chunk_id != chunker.chunk_text(
          _prose, document_id="d",
          tokenizer=chunker.SubwordishTokenizer(8))[0].chunk_id,
      "chunks built for one embedder must be distinguishable from another's")

check("a different document id gives different ids",
      _a[0].chunk_id != chunker.chunk_text(
          _prose, document_id="other", tokenizer=_t)[0].chunk_id)

# --- the fingerprint that catches a store built for another embedder --------

check("the same tokenizer and budget fingerprint the same",
      chunker.pack_fingerprint(chunker.WordTokenizer(8))
      == chunker.pack_fingerprint(chunker.WordTokenizer(8)))

check("a changed budget changes the fingerprint",
      chunker.pack_fingerprint(chunker.WordTokenizer(8))
      != chunker.pack_fingerprint(chunker.WordTokenizer(9)))

check("a changed tokenizer changes the fingerprint",
      chunker.pack_fingerprint(chunker.WordTokenizer(8))
      != chunker.pack_fingerprint(chunker.SubwordishTokenizer(8)))

# --- overlap ----------------------------------------------------------------

_ov = chunker.chunk_text(_prose, document_id="d", tokenizer=chunker.WordTokenizer(16),
                         overlap_tokens=3)
check("overlap does not push any chunk over budget",
      all(c.token_count <= 16 for c in _ov))
check("overlap does not disturb the tiling",
      chunker.coverage_gaps(_prose, _ov) == [])

_raised = False
try:
    chunker.chunk_text(_prose, document_id="d",
                       tokenizer=chunker.WordTokenizer(8), overlap_tokens=8)
except ValueError:
    _raised = True
check("an overlap as large as the budget is refused, not looped on", _raised,
      "it would never advance, and a hang is worse than an error")

_raised = False
try:
    chunker.chunk_text(_prose, document_id="d",
                       tokenizer=chunker.WordTokenizer(8), overlap_tokens=-1)
except ValueError:
    _raised = True
check("a negative overlap is refused", _raised)

# --- a budget nothing can fit ----------------------------------------------

class _ImpossibleTokenizer:
    """A usable budget, but every string costs more than it."""

    name = "impossible"
    budget = 1

    def count(self, text: str) -> int:
        return len(text) + 1


_raised = False
try:
    chunker.chunk_text("anything", document_id="d", tokenizer=_ImpossibleTokenizer())
except chunker.BudgetExceeded:
    _raised = True
check("a budget that cannot hold one character raises rather than truncating",
      _raised,
      "silently returning a shortened chunk is the upstream behaviour this "
      "module exists to make impossible")


class _ZeroBudgetTokenizer:
    name = "zero-budget"
    budget = 0

    def count(self, text: str) -> int:
        return len(text)


_raised = False
try:
    chunker.chunk_text("anything", document_id="d", tokenizer=_ZeroBudgetTokenizer())
except ValueError:
    _raised = True
check("a tokenizer reporting a budget below one is refused up front", _raised,
      "found by this suite: the overlap guard used to fire first and blame "
      "the overlap for a budget that was the real problem")

# --- the report -------------------------------------------------------------

_rep = chunker.report(_a, _t)
check("the report never shows a chunk over budget", _rep.over_budget == 0)
check("the report counts every chunk exactly once",
      sum(_rep.by_level.values()) == _rep.chunks)
check("the report carries the fingerprint of the packing",
      _rep.fingerprint == chunker.pack_fingerprint(_t))

# ---------------------------------------------------------------------------

print()
if FAILED:
    print(f"{len(FAILED)} of {len(RUN)} invariants FAILED:")
    for name in FAILED:
        print(f"  - {name}")
    sys.exit(1)

print(f"all {len(RUN)} invariants hold")
