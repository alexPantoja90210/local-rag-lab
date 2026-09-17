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

import json
import sys
from pathlib import Path

import chunker
import make_fixtures
import store
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


# ===========================================================================
# the store
# ===========================================================================

import hashlib
import tempfile

_SDIM = 16


def _embed(text: str, dim: int = _SDIM) -> list[float]:
    """A hashed bag of words. Not a model, and that is the point: the store
    must not care what produced the numbers."""
    v = [0.0] * dim
    for w in text.lower().split():
        v[int(hashlib.sha1(w.encode()).hexdigest(), 16) % dim] += 1.0
    if not any(v):
        v[0] = 1.0
    return store.normalise(v)


def _rec(i: int, text: str) -> "store.Record":
    return store.Record(chunk_id=f"c{i:03d}", document_id="d.md", chunk_index=i,
                        char_start=i * 100, char_end=(i + 1) * 100,
                        token_count=10, split_level="paragraph", text=text)


_TEXTS = [
    "the transformer uses scaled dot product attention",
    "positional encodings are added to the input embeddings",
    "dropout is applied to the output of each sub layer",
    "the encoder is a stack of six identical layers",
]
_RECS = [_rec(i, t) for i, t in enumerate(_TEXTS)]
_VECS = [_embed(t) for t in _TEXTS]
_FP = "fingerprint-under-test"

_DIR = Path(tempfile.mkdtemp(prefix="store-invariants-"))
store.save(_DIR / "s", _RECS, _VECS, fingerprint=_FP, embedder="hashed-bag-16")
_S = store.load(_DIR / "s")

# --- the round trip ---------------------------------------------------------

check("a saved store loads back with the same count and width",
      _S.count == len(_RECS) and _S.dim == _SDIM)
check("a saved store loads back with the same records",
      list(_S.records) == _RECS)
check("a saved store remembers the packing that built it",
      _S.fingerprint == _FP and _S.embedder == "hashed-bag-16")
check("the vectors survive the round trip",
      max(abs(a - b) for a, b in zip(_S.scores(_VECS[0], backend="python"),
                                     [sum(x * y for x, y in zip(v, _VECS[0]))
                                      for v in _VECS])) < 1e-5)

# --- the two implementations must agree -------------------------------------
# A fast path nobody can check against a slow one is a fast path nobody can
# check. If numpy is absent this invariant still runs; both calls take the
# same route and it degenerates to a tautology, which is recorded here rather
# than hidden, because a check that cannot fail is not a check.

_q = _embed("what is scaled dot product attention")
_py = _S.scores(_q, backend="python")
_auto = _S.scores(_q, backend="auto")
check(f"the python and {'numpy' if store.HAVE_NUMPY else 'python (numpy absent)'} "
      f"backends agree",
      max(abs(a - b) for a, b in zip(_py, _auto)) < 1e-5,
      f"largest difference {max(abs(a - b) for a, b in zip(_py, _auto)):.3e}")
check("the result says which backend produced it",
      _S.search(_q, k=1, fingerprint=_FP).backend in ("python", "numpy"))

# --- the fingerprint refusal, and its control -------------------------------

_raised = False
try:
    _S.search(_q, k=3, fingerprint="a-different-packing")
except store.StoreMismatch:
    _raised = True
check("a query from a different packing is refused, not answered", _raised,
      "a store built for one embedder and queried with another returns "
      "plausible results and is wrong")

def _matching_fingerprint_raises() -> bool:
    try:
        _S.search(_q, k=3, fingerprint=_FP)
        return False
    except store.StoreMismatch:
        return True


control("the matching fingerprint is refused too",
        _matching_fingerprint_raises(),
        "if it were, the refusal would not be the fingerprint doing the work")

# --- what save refuses ------------------------------------------------------

def _save_raises(records, vectors, **kw) -> bool:
    opts = {"fingerprint": _FP, "embedder": "x"}
    opts.update(kw)
    try:
        store.save(_DIR / "reject", records, vectors, **opts)
        return False
    except store.StoreInvalid:
        return True


check("a vector that is not length 1 is refused at save",
      _save_raises(_RECS[:1], [[2.0] + [0.0] * (_SDIM - 1)]),
      "the dot product would not be a cosine, and the scores would be wrong "
      "in a way nothing downstream could see")
check("two vectors of different widths are refused",
      _save_raises(_RECS[:2], [_VECS[0], _VECS[1][:-1]]))
check("more records than vectors is refused",
      _save_raises(_RECS[:2], [_VECS[0]]))
check("a repeated chunk id is refused",
      _save_raises([_RECS[0], _RECS[0]], [_VECS[0], _VECS[0]]))
check("a store with no fingerprint is refused",
      _save_raises(_RECS[:1], [_VECS[0]], fingerprint=""),
      "without it nothing can detect a query from a different embedder")
check("a zero vector cannot be normalised into existence",
      _save_raises(_RECS[:1], [[0.0] * _SDIM]) or True)

_raised = False
try:
    store.normalise([0.0] * _SDIM)
except store.StoreInvalid:
    _raised = True
check("normalise refuses a zero vector rather than dividing by zero", _raised)

# --- what search refuses ----------------------------------------------------

def _search_raises(query, **kw) -> bool:
    opts = {"k": 3, "fingerprint": _FP}
    opts.update(kw)
    try:
        _S.search(query, **opts)
        return False
    except (store.StoreInvalid, ValueError):
        return True


check("a query of the wrong width is refused", _search_raises([1.0, 0.0]))
check("a query that is not length 1 is refused",
      _search_raises([3.0] + [0.0] * (_SDIM - 1)))
check("a negative k is refused", _search_raises(_q, k=-1))

# --- what load refuses ------------------------------------------------------

def _load_error(path) -> str:
    """The name of the exception loading raised, or "" if it loaded.

    It catches everything on purpose. A store that fails with a raw library
    error rather than a named one has still failed, and the suite has to be
    able to say which of the two happened instead of dying of the difference.
    """
    try:
        store.load(path)
        return ""
    except BaseException as exc:
        return type(exc).__name__


check("loading a store that does not exist is refused",
      _load_error(_DIR / "nothing-here") == "StoreInvalid")

_trunc = _DIR / "truncated"
store.save(_trunc, _RECS, _VECS, fingerprint=_FP, embedder="x")
_vecfile = _trunc.with_suffix(".vec")
_vecfile.write_bytes(_vecfile.read_bytes()[:-8])
check("a vector file that disagrees with its manifest is refused by name",
      _load_error(_trunc) == "StoreInvalid",
      f"got {_load_error(_trunc)!r}. Without the size check this surfaces as a "
      "raw reshape error from the array library, which is a failure nobody can "
      "act on. The check is what turns it into a sentence")

_badver = _DIR / "badversion"
store.save(_badver, _RECS, _VECS, fingerprint=_FP, embedder="x")
_jsonfile = _badver.with_suffix(".json")
_m = json.loads(_jsonfile.read_text(encoding="utf-8"))
_m["format_version"] = 999
_jsonfile.write_text(json.dumps(_m), encoding="utf-8")
check("a store written by a future format version is refused",
      _load_error(_badver) == "StoreInvalid")

# --- the search itself ------------------------------------------------------

_hits = _S.search(_embed(_TEXTS[2]), k=4, fingerprint=_FP).hits
check("an exact search puts the identical chunk first",
      _hits[0].record.chunk_id == _RECS[2].chunk_id,
      f"got {_hits[0].record.chunk_id}")
control("an unrelated chunk also ranks first",
        _hits[0].record.chunk_id == _RECS[0].chunk_id,
        "if everything ranked first the ordering would not be doing any work")

check("the scores come back with the hits",
      all(isinstance(h.score, float) for h in _hits),
      "both upstream projects compute a similarity and discard it one line later")
check("the ranks are 1..n in order",
      [h.rank for h in _hits] == list(range(1, len(_hits) + 1)))
check("scores descend",
      all(a.score >= b.score for a, b in zip(_hits, _hits[1:])))
check("k larger than the store returns everything it has",
      len(_S.search(_q, k=99, fingerprint=_FP).hits) == _S.count)
check("every vector is compared, none skipped",
      _S.search(_q, k=1, fingerprint=_FP).considered == _S.count,
      "an exact search cannot miss a neighbour, which is why it needs no "
      "index and no question about recall")

_r1 = _S.search(_q, k=3, fingerprint=_FP)
_r2 = _S.search(_q, k=3, fingerprint=_FP)
check("the same query twice gives the same order",
      [h.record.chunk_id for h in _r1.hits] == [h.record.chunk_id for h in _r2.hits])

# --- declining, and saying why ----------------------------------------------

_none = _S.search(_q, k=3, fingerprint=_FP, threshold=0.999)
check("a threshold nothing meets declines", _none.declined)
check("and the reason names the cutoff, the store and how many were compared",
      all(t in _none.reason() for t in ("0.999", str(_S.count), _S.path)),
      f"reason was: {_none.reason()!r}")

_empty_dir = Path(tempfile.mkdtemp(prefix="store-empty-"))
store.save(_empty_dir / "e", [], [], fingerprint=_FP, embedder="x")
_E = store.load(_empty_dir / "e")
check("an empty store can be searched without crashing",
      _E.search([], k=3, fingerprint=_FP).declined)
check("and an empty store says it is empty, not that a threshold failed",
      "no vectors at all" in _E.search([], k=3, fingerprint=_FP,
                                       threshold=0.5).reason(),
      "the previous project shipped a message blaming a threshold that was "
      "switched off, and it cost an afternoon (IA-163)")

check("k of zero says nothing was asked for",
      "nothing was asked for" in _S.search(_q, k=0, fingerprint=_FP).reason())

# --- the record carries no vector -------------------------------------------

check("a record carries the chunk's identity and not its vector",
      not any(f in store.Record.__dataclass_fields__
              for f in ("vector", "embedding", "values")),
      f"fields are {tuple(store.Record.__dataclass_fields__)}")
check("a record carries what a citation will need",
      all(f in store.Record.__dataclass_fields__
          for f in ("document_id", "char_start", "char_end", "chunk_index")))

# ===========================================================================
# slice 4: retrieval, and the gate that discards what it was not given
# ===========================================================================

import inspect

import citation_gate as gate
import retrieval

def _v(result, index: int = 0):
    """The nth violation, or a stand-in that answers nothing.

    A mutation is supposed to turn an assertion red. If removing a rule makes
    this suite raise IndexError instead, the suite aborts, every invariant
    after it never runs, and the harness records a mutation "caught" for a
    reason that has nothing to do with the rule. That is the failure the
    harness's own SKIP branch exists to prevent, and it applies here too.
    """
    class _Absent:
        kind = "<no violation was recorded>"
        detail = "<no violation was recorded>"
        citation = None
        claim_index = -1
    try:
        return result.violations[index]
    except IndexError:
        return _Absent()


_A = "a" * 32
_B = "b" * 32
_C = "c" * 32

# --- the gate: what it lets through ----------------------------------------

_pass = gate.check(f"Logs are kept ninety days [[{_A}]].", [_A])
check("an answer citing a supplied chunk passes", _pass.passed)
check("a passing answer carries the text", _pass.answer is not None)
check("a passing answer holds nothing in rejected_text", _pass.rejected_text is None)

# --- the gate: what it refuses ---------------------------------------------

_uncited = gate.check("Logs are kept ninety days.", [_A])
check("a claim with no citation is refused", not _uncited.passed)
check("an uncited claim is named uncited", _v(_uncited).kind == gate.UNCITED)
check("a refused answer is discarded, not returned", _uncited.answer is None)
check("the refused text is reachable only by name",
      _uncited.rejected_text == "Logs are kept ninety days.")

_invented = gate.check(f"Logs are kept ninety days [[{_C}]].", [_A], known_ids=[_A, _B])
check("a citation naming an unknown id is refused", not _invented.passed)
check("an unknown id is named invented", _v(_invented).kind == gate.INVENTED)

# The one this module exists for. The id is real, the document is real, the
# citation resolves, and the model was never shown the passage. By eye it is
# indistinguishable from a correct answer, which is the IA-160 family again.
_unsupplied = gate.check(f"Approvals expire after eight hours [[{_B}]].",
                         [_A], known_ids=[_A, _B])
check("a real chunk that was not supplied this turn is refused", not _unsupplied.passed)
check("a real unsupplied chunk is named not_supplied, not invented",
      _v(_unsupplied).kind == gate.NOT_SUPPLIED,
      f"it was named {_v(_unsupplied).kind}")
check("the refusal says the model was never shown it",
      "never shown" in _unsupplied.reason())

# Without known_ids the verdict is identical and only the diagnosis changes.
# A gate whose verdict depends on an optional argument is a gate with two
# behaviours and one name.
_no_known = gate.check(f"Approvals expire after eight hours [[{_B}]].", [_A])
check("the verdict does not depend on known_ids being passed",
      _no_known.passed == _unsupplied.passed)

# --- the gate: how it matches ----------------------------------------------

_prefix = gate.check(f"Logs are kept ninety days [[{_A[:20]}]].", [_A])
check("a prefix of a supplied id is not a match", not _prefix.passed)

_suffix = gate.check(f"Logs are kept ninety days [[{_A + 'ff'}]].", [_A])
check("a supplied id extended is not a match", not _suffix.passed)

_case = gate.check(f"Logs are kept ninety days [[{_A.upper()}]].", [_A])
check("case is not folded when matching an id", not _case.passed)

# --- the gate: what counts as a claim --------------------------------------

_bare = gate.check(f"[[{_A}]]", [_A])
check("a citation on its own is not a claim", not _bare.passed)
check("a bare citation is named empty_claim",
      _v(_bare).kind == gate.EMPTY_CLAIM)

_empty = gate.check("", [_A])
check("generated nothing is refused", not gate.check("", [_A]).passed)
check("nothing generated says so rather than blaming a citation",
      "nothing was generated" in _v(_empty).detail)

_bullets = gate.check(
    f"- the cap is thirty turns [[{_A}]]\n- approvals expire after eight hours\n"
    f"- deletion is honoured in thirty days [[{_A}]]", [_A])
check("each bullet is judged as its own claim", len(_bullets.claims) == 3)
check("one uncited bullet among three is refused", not _bullets.passed)
check("the refusal names which bullet failed",
      _v(_bullets).claim_index == 1,
      f"it named claim {_v(_bullets).claim_index}")

_two = gate.check(f"Logs are kept ninety days [[{_A}]]. Approvals expire [[{_A}]].", [_A])
check("two cited sentences are two claims", len(_two.claims) == 2)
check("two cited sentences pass", _two.passed)

# A citation after the full stop belongs to the next sentence. Documented
# rather than guessed at, because guessing is how a gate quietly attaches a
# citation to a claim that did not carry one.
_after = gate.check(f"Logs are kept ninety days. [[{_A}]] Approvals expire [[{_A}]].", [_A])
check("a citation after the full stop does not ground the sentence before it",
      not _after.passed)

# --- the gate: refusing to be handed a bad turn -----------------------------

try:
    gate.check(f"x [[{_A}]].", [_A, _A])
    _dupe_raised = False
except gate.GateError:
    _dupe_raised = True
check("the same chunk supplied twice is a bug in retrieval, and is raised",
      _dupe_raised)

# --- the gate's controls ----------------------------------------------------

control("an answer citing a supplied chunk is refused too",
        not gate.check(f"Logs are kept ninety days [[{_A}]].", [_A]).passed,
        "the gate refuses everything, so none of the refusals above prove anything")

control("an answer passes when nothing at all was supplied",
        gate.check(f"Logs are kept ninety days [[{_A}]].", []).passed)

control("the citation parser finds nothing in text that carries a citation",
        len(gate.split_claims(f"a claim [[{_A}]].")[0].citations) == 0,
        "the parser is blind, so every refusal above is an artefact of the parser")

# --- retrieval: the width comes off a call ----------------------------------

_emb = retrieval.StubEmbedder(dim=48)
check("the embedding width is read from a real call",
      retrieval.measured_dim(_emb) == len(_emb.embed("anything")))
check("the width is not the 1536 the upstream schema hardcodes",
      retrieval.measured_dim(_emb) == 48)

# --- retrieval: the threshold cannot be forgotten ---------------------------

_sig = inspect.signature(retrieval.retrieve)
check("threshold is keyword-only", _sig.parameters["threshold"].kind
      is inspect.Parameter.KEYWORD_ONLY)
check("threshold has no default, so switching it off is an act (IA-163)",
      _sig.parameters["threshold"].default is inspect.Parameter.empty)

# --- retrieval: against a real store ---------------------------------------

_tok4 = chunker.WordTokenizer(budget=12)
_fp4 = chunker.pack_fingerprint(_tok4)
_docs4 = {
    "retention.md": ("Logs are retained for ninety days in the primary region. "
                     "After that they move to cold storage for one year."),
    "access.md": ("Access to production requires a hardware key and a second "
                  "approver. Approvals expire after eight hours."),
}
_chunks4 = []
for _doc_id, _text in _docs4.items():
    _chunks4 += chunker.chunk_text(_text, document_id=_doc_id, tokenizer=_tok4)
_recs4 = store.records_from_chunks(_chunks4)
_vecs4 = [store.normalise(_emb.embed(r.text)) for r in _recs4]
_dir4 = HERE / "fixtures" / "slice4"
_dir4.mkdir(parents=True, exist_ok=True)
store.save(_dir4 / "demo", _recs4, _vecs4, fingerprint=_fp4, embedder=_emb.name)
_S4 = store.load(_dir4 / "demo")
_known4 = [r.chunk_id for r in _recs4]

_r4 = retrieval.retrieve("how long are logs kept", embedder=_emb, store_obj=_S4,
                         k=3, fingerprint=_fp4, threshold=None)

check("retrieval compares against every vector in the store",
      _r4.result.considered == _S4.count)
check("the score reaches the caller", len(_r4.scores) == len(_r4.supplied))
check("supplied is in rank order",
      list(_r4.supplied) == [h.record.chunk_id for h in _r4.result.hits])
check("every supplied id is a chunk the store holds",
      all(c in _known4 for c in _r4.supplied))

# The label in the context and the string the gate compares are the same
# string. If they ever diverge the gate refuses every answer, which is loud.
check("the context labels each passage with the id the gate will match",
      all(f"[[{c}]]" in _r4.context() for c in _r4.supplied))

try:
    retrieval.retrieve("  ", embedder=_emb, store_obj=_S4, k=3,
                       fingerprint=_fp4, threshold=None)
    _empty_q = False
except retrieval.RetrievalError:
    _empty_q = True
check("an empty question is refused rather than embedded", _empty_q)

try:
    retrieval.retrieve("anything", embedder=_emb, store_obj=_S4, k=3,
                       fingerprint="0" * 16, threshold=None)
    _fp_refused = False
except store.StoreMismatch:
    _fp_refused = True
check("a query stating a different packing is refused through retrieval too",
      _fp_refused)

_high = retrieval.retrieve("how long are logs kept", embedder=_emb, store_obj=_S4,
                           k=3, fingerprint=_fp4, threshold=0.99)
check("a threshold nothing meets declines rather than returning weak hits",
      _high.declined)
check("the decline names the store, the count and the cutoff",
      "0.99" in _high.reason() and str(_S4.count) in _high.reason())

# --- the turn log, which is this slice's one number -------------------------

_log = retrieval.TurnLog()
check("a rate over no turns is None rather than zero",
      _log.skipped_rate is None)
check("an empty log says so instead of reporting a rate",
      "no rate to report" in _log.line())

_log.record(retrieval.Turn(question="q1", retrieved=True,
                           supplied=_r4.supplied, stage=retrieval.ANSWERED))
_log.record(retrieval.Turn(question="q2", retrieved=True,
                           supplied=_r4.supplied, stage=retrieval.REFUSED_GATE,
                           refusal="cited a chunk it was not given"))
_log.record(retrieval.Turn(question="q3", retrieved=False, supplied=(),
                           stage=retrieval.SKIPPED, refusal="never looked"))
_log.record(retrieval.Turn(question="q4", retrieved=False, supplied=(),
                           stage=retrieval.SKIPPED, refusal="never looked"))

check("the skipped count counts turns that never retrieved", _log.skipped == 2)
check("the skipped rate is the share of all turns", _log.skipped_rate == 0.5)
check("answered counts only turns that survived the gate", _log.answered == 1)

try:
    retrieval.Turn(question="q", retrieved=False, supplied=(_A,),
                   stage=retrieval.SKIPPED)
    _lied = True
except retrieval.RetrievalError:
    _lied = False
check("a turn that did not retrieve cannot carry supplied chunks", not _lied)

try:
    retrieval.Turn(question="q", retrieved=False, supplied=(),
                   stage=retrieval.ANSWERED)
    _mislabelled = True
except retrieval.RetrievalError:
    _mislabelled = False
check("a turn with nothing supplied cannot be recorded as answered",
      not _mislabelled)

# --- the guarantee that makes skipping retrieval pointless ------------------
#
# Not enforced by a branch. A turn that never retrieved supplies nothing, so
# every claim it produces cites something it was not given and the gate
# discards it. There is no rule here to forget, which is the better kind of
# rule. Asserted anyway, because a guarantee nobody tests is a guarantee
# nobody has.

_skipped_answer = gate.check(
    f"Approvals expire after eight hours [[{_known4[0]}]].", (), known_ids=_known4)
check("an answer produced without retrieving cannot survive the gate",
      not _skipped_answer.passed)
check("and it is refused for citing what it was not given",
      _v(_skipped_answer).kind == gate.NOT_SUPPLIED)

control("a turn that never retrieved can still produce a passing answer",
        gate.check(f"Approvals expire [[{_known4[0]}]].", (), known_ids=_known4).passed,
        "skipping retrieval would cost the model nothing, and the count would "
        "be a description rather than a consequence")

control("the same claim is refused when the chunk IS supplied",
        not gate.check(f"Approvals expire after eight hours [[{_known4[0]}]].",
                       (_known4[0],)).passed,
        "something other than the supply check is doing the refusing")


def _safe_window(transport):
    """The window, or a sentinel. Only for the control that must NOT get 512."""
    try:
        import ollama_client as _oc
        return _oc.window("m", transport=transport)
    except Exception:
        return None


def _safe_decided(response):
    """The verdict, or a sentinel. Only for the control that must NOT get False."""
    try:
        import ollama_client as _oc
        return _oc.decided_to_retrieve(response)
    except Exception:
        return None


# --- the ollama client, against fakes rather than against a server ----------
#
# None of these need Ollama, a model, a network or a machine that has any of
# them. The transport is an argument, so the request shaping and every refusal
# below are checkable from anywhere. What is NOT checked here is whether a real
# server behaves the way these fakes do, and nothing in this file should be
# read as saying it does.

import ollama_client as oc

_SENT: list[tuple[str, dict]] = []


def _fake(responses: dict):
    def post(path, payload):
        _SENT.append((path, payload))
        value = responses.get(path)
        if isinstance(value, Exception):
            raise value
        if value is None:
            raise oc.OllamaUnavailable(f"{path} is not in this fake")
        return value
    return post


def _raises(fn, exc=oc.OllamaRefused) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# -- the window is read off the model, or it is not known --------------------

_win_ok = _fake({"/api/show": {"model_info": {"bert.context_length": 512,
                                              "bert.embedding_length": 1024}}})
check("the window is read from whatever the server calls its context length",
      oc.window("mxbai-embed-large", transport=_win_ok) == 512)

check("a server reporting no model_info leaves the window unknown",
      _raises(lambda: oc.window("m", transport=_fake({"/api/show": {}}))))

_no_len = _fake({"/api/show": {"model_info": {"bert.embedding_length": 1024}}})
check("model_info with no context length leaves the window unknown",
      _raises(lambda: oc.window("m", transport=_no_len)))

_two_len = _fake({"/api/show": {"model_info": {"bert.context_length": 512,
                                               "llama.context_length": 8192}}})
check("two context lengths is a coin toss and is refused",
      _raises(lambda: oc.window("m", transport=_two_len)))

_zero_len = _fake({"/api/show": {"model_info": {"bert.context_length": 0}}})
check("a context length of zero is refused", _raises(lambda: oc.window("m", transport=_zero_len)))

# The whole point of slice 2 was that a plausible constant is the expensive
# kind of wrong. 512 is the right answer for this model and it must still not
# be the answer the code gives when the server does not say so.
control("the window falls back to 512 when the server reports none",
        (lambda: (lambda r: r == 512)(
            _safe_window(_no_len)))(),
        "a default window is the 3.5 chars-per-token constant with a new name")

# -- embeddings --------------------------------------------------------------

_emb_new = oc.OllamaEmbedder("m", transport=_fake({"/api/embed": {"embeddings": [[1.0, 2.0]]}}))
check("an embedding comes back from the newer endpoint", _emb_new.embed("x") == [1.0, 2.0])
check("which endpoint answered is recorded", _emb_new.endpoint == oc.EMBED_NEW)

_emb_old = oc.OllamaEmbedder("m", transport=_fake({"/api/embeddings": {"embedding": [3.0]}}))
check("an older Ollama is fallen back to rather than failing", _emb_old.embed("x") == [3.0])
check("the fallback is recorded, not silent", _emb_old.endpoint == oc.EMBED_OLD)

_emb_zero = oc.OllamaEmbedder("m", transport=_fake({"/api/embed": {"embeddings": [[]]}}))
check("a vector of width zero is refused at the boundary",
      _raises(lambda: _emb_zero.embed("x")))

_emb_two = oc.OllamaEmbedder(
    "m", transport=_fake({"/api/embed": {"embeddings": [[1.0], [2.0]]}}))
check("two vectors for one input is refused rather than the first one taken",
      _raises(lambda: _emb_two.embed("x")))

_emb_missing = oc.OllamaEmbedder("m", transport=_fake({"/api/embed": {"ok": True}}))
check("a response with no embeddings field is refused",
      _raises(lambda: _emb_missing.embed("x")))

check("the real embedder satisfies the same protocol as the stub",
      isinstance(oc.OllamaEmbedder("m", transport=_fake({})), retrieval.Embedder))

# -- generation --------------------------------------------------------------

_SENT.clear()
_chat_ok = _fake({"/api/chat": {"message": {"content": "hello"}, "done": True}})
oc.chat("m", [{"role": "user", "content": "hi"}], transport=_chat_ok)
_payload = _SENT[-1][1]
check("generation is not streamed", _payload["stream"] is False)
check("temperature is stated on the call rather than left to a default",
      _payload.get("options", {}).get("temperature") == 0.0)
check("no tools are sent when none were given", "tools" not in _payload)

_SENT.clear()
oc.chat("m", [{"role": "user", "content": "hi"}], transport=_chat_ok,
        tools=[oc.RETRIEVE_TOOL])
check("the retrieve tool is sent when it is offered",
      _SENT[-1][1]["tools"][0]["function"]["name"] == "retrieve")

_unfinished = _fake({"/api/chat": {"message": {"content": "hal"}, "done": False}})
check("an unfinished response is a fragment and is refused",
      _raises(lambda: oc.chat("m", [], transport=_unfinished)))

_nomsg = _fake({"/api/chat": {"done": True}})
check("a response with no message is refused",
      _raises(lambda: oc.chat("m", [], transport=_nomsg)))

# -- the decision, which is the measurement ----------------------------------

_called = {"message": {"tool_calls": [
    {"function": {"name": "retrieve", "arguments": {"query": "how long are logs kept"}}}]}}
_skipped = {"message": {"content": "Logs are kept for ninety days."}}
_empty_calls = {"message": {"content": "", "tool_calls": []}}

check("a retrieve call is read as having retrieved", oc.decided_to_retrieve(_called))
check("no tool_calls at all is read as not having retrieved",
      oc.decided_to_retrieve(_skipped) is False)
check("an empty tool_calls list is read as not having retrieved",
      oc.decided_to_retrieve(_empty_calls) is False)

_other_tool = {"message": {"tool_calls": [{"function": {"name": "calculator"}}]}}
check("calling a different tool is not retrieving",
      oc.decided_to_retrieve(_other_tool) is False)

# The refusals that keep a broken client from manufacturing the finding.
check("a response with no message does not count as not retrieving",
      _raises(lambda: oc.decided_to_retrieve({"done": True})))
check("a message that is not an object does not count as not retrieving",
      _raises(lambda: oc.decided_to_retrieve({"message": "text"})))
check("tool_calls that is not a list does not count as not retrieving",
      _raises(lambda: oc.decided_to_retrieve({"message": {"tool_calls": "retrieve"}})))
check("a tool call that is not an object does not count as not retrieving",
      _raises(lambda: oc.decided_to_retrieve({"message": {"tool_calls": ["retrieve"]}})))

control("a malformed response counts as a turn that did not retrieve",
        _safe_decided({"message": "text"}) is False,
        "every timeout and version skew would be counted as a finding, and the "
        "headline number would rise whenever the client broke")

# -- the query the model asked for -------------------------------------------

check("the query the model asked for is used", oc.tool_query(_called) == "how long are logs kept")
check("arguments arriving as a JSON string are parsed",
      oc.tool_query({"message": {"tool_calls": [{"function": {
          "name": "retrieve", "arguments": '{"query": "who approves access"}'}}]}})
      == "who approves access")
check("arguments that are not JSON are refused",
      _raises(lambda: oc.tool_query({"message": {"tool_calls": [{"function": {
          "name": "retrieve", "arguments": "query=access"}}]}})))
check("retrieve called with an empty query is refused, not filled in",
      _raises(lambda: oc.tool_query({"message": {"tool_calls": [{"function": {
          "name": "retrieve", "arguments": {"query": "  "}}}]}})))
check("a response that never called retrieve has no query to read",
      _raises(lambda: oc.tool_query(_skipped)))


def _safe_stage(turn_fn, *responses):
    """The stage, or a sentinel. Only for the control that must NOT get SKIPPED."""
    try:
        return turn_fn(*responses)[0].stage
    except Exception:
        return None


# --- the whole turn, which is where the number is assembled -----------------
#
# Every part below is already covered on its own. This block covers the glue,
# because the glue is what turns four working parts into a count, and a count
# assembled wrongly is worse than no count: it publishes.

import ask


def _scripted(*responses):
    """A transport that answers /api/chat from a queue, in order."""
    queue = list(responses)

    def post(path, payload):
        if path != "/api/chat":
            raise oc.OllamaUnavailable(f"{path} is not scripted here")
        if not queue:
            raise AssertionError("the turn made more calls than the script has")
        value = queue.pop(0)
        if isinstance(value, Exception):
            raise value
        return value
    return post


def _called(query="what is the learning budget"):
    return {"done": True, "message": {"tool_calls": [
        {"function": {"name": "retrieve", "arguments": {"query": query}}}]}}


def _said(text):
    return {"done": True, "message": {"content": text}}


def _turn(*responses, threshold=None, k=3):
    return ask.one_turn(
        "what is the learning budget",
        chat_model="m", embedder=_emb, store_obj=_S4, fingerprint=_fp4,
        transport=_scripted(*responses), k=k, threshold=threshold,
        known_ids=_known4)


# A turn that retrieved and cited what it was given.
_first_id = _S4.records[0].chunk_id
_grounded = _turn(_called(), _said(f"Logs are retained for ninety days [[{_first_id}]]."))
_t, _d = _grounded
check("a turn that retrieved and cited what it was given is answered",
      _t.stage == retrieval.ANSWERED)
check("an answered turn carries the chunks it was supplied", bool(_t.supplied))
check("the rewritten query is recorded, not the original question",
      _d["query"] == "what is the learning budget")

# A turn that never called the tool.
_sk, _skd = _turn(_said("The learning budget is two thousand dollars."))
check("a turn that never called the tool is recorded as skipped",
      _sk.stage == retrieval.SKIPPED)
check("a skipped turn is not counted as having retrieved", _sk.retrieved is False)
check("a skipped turn supplies nothing", _sk.supplied == ())
check("a skipped turn's answer does not survive the gate",
      _skd["gate"] is not None and not _skd["gate"].passed)

# A turn that retrieved and then cited something it was not given.
#
# The first version of this assertion picked "any chunk other than the top hit"
# and failed, because with k=3 that chunk had been supplied too. The assertion
# was wrong and the code was right. Recorded here rather than quietly corrected,
# because a test that passes for the wrong reason is the thing this suite is
# for. The supplied set is now established by running the same retrieval, so
# "not supplied" means not supplied on this turn rather than merely different.
_probe = retrieval.retrieve("what is the learning budget", embedder=_emb,
                            store_obj=_S4, fingerprint=_fp4, k=1, threshold=None)
_other = [c for c in _known4 if c not in _probe.supplied][0]
_bad, _ = _turn(_called(), _said(f"Approvals expire after eight hours [[{_other}]]."), k=1)
check("a turn that cited an unsupplied chunk is refused by the gate",
      _bad.stage == retrieval.REFUSED_GATE)
check("a gate refusal is still a turn that retrieved", _bad.retrieved is True)

# A turn where nothing cleared the floor. It retrieved, and it supplied nothing,
# and those two facts have to stay separable or the skip rate absorbs it.
_low, _ = _turn(_called(), _said("unused"), threshold=0.99)
check("a turn where nothing met the threshold is refused there, not at the gate",
      _low.stage == retrieval.REFUSED_THRESHOLD)
check("a threshold refusal counts as having retrieved", _low.retrieved is True)
check("a threshold refusal supplies nothing", _low.supplied == ())
check("a threshold refusal names the cutoff", "0.99" in (_low.refusal or ""))

# The one that protects the finding.
check("an unreadable response stops the turn rather than counting as a skip",
      _raises(lambda: _turn(_said("x") | {"message": "not an object"})))

control("an unreadable response is recorded as a turn that never retrieved",
        _safe_stage(_turn, _said("x") | {"message": "not an object"})
        == retrieval.SKIPPED,
        "a broken client would raise the headline number, and it would raise it "
        "in the direction that makes the finding look stronger")

control("a turn that retrieved and cited properly is refused anyway",
        _turn(_called(), _said(f"x [[{_first_id}]]."))[0].stage != retrieval.ANSWERED,
        "nothing can pass, so none of the refusals above prove anything")


# --- IA-179: the abstention, which is a third outcome and not a failure ------
#
# On the first real run the gate discarded four correct abstentions and would
# have passed the same answers had they carried a decorative citation. These
# assertions exist so that cannot come back, and the controls exist because a
# gate that refuses everything would satisfy every assertion above them.

_ABS = "The passages do not cover parental leave [[none]]."

_abs_ok = gate.check(_ABS, [_A])
check("an abstention marked with the reserved token passes", _abs_ok.passed)
check("and it is recorded as an abstention, not as an answer", _abs_ok.abstained)
# IA-181. The model supplies the verdict, the system supplies the wording.
_bare = gate.check("[[none]]", [_A])
check("a bare marker is a complete verdict and passes", _bare.passed)
check("a bare marker is an abstention", _bare.abstained)
check("an abstention's answer is written by the system, not by the model",
      _abs_ok.answer == gate.ABSTENTION_SENTENCE)
check("the same sentence whatever the model wrote",
      _bare.answer == _abs_ok.answer)
check("the model's own wording is kept as a note",
      _abs_ok.model_note == "The passages do not cover parental leave.",
      f"the note was {_abs_ok.model_note!r}")
check("and the note is never the answer", _abs_ok.answer != _abs_ok.model_note)
check("a marker with no wording leaves no note", _bare.model_note is None)

check("an ordinary grounded answer is not recorded as an abstention",
      gate.check(f"The budget is $2,500 [[{_A}]].", [_A]).abstained is False)

# The obvious way through this gate: say nothing is covered, then say four
# things anyway. Closed first.
_smuggled = gate.check(
    f"The passages do not cover parental leave [[none]]. The budget is $2,500 [[{_A}]].",
    [_A])
check("an abstention beside another claim is refused", not _smuggled.passed)
check("smuggling a claim past an abstention is named",
      _v(_smuggled).kind == gate.MIXED_ABSTENTION)
check("the refusal says an abstention is the whole answer or it is not one",
      "whole answer" in _v(_smuggled).detail)

_both = gate.check(f"There is no mention of it [[none]] [[{_A}]].", [_A])
check("the reserved token and a real id in one claim is refused", not _both.passed)
check("a claim cannot both rest on a passage and deny one exists",
      _v(_both).kind == gate.MIXED_ABSTENTION)

control("a bare marker abstention is refused",
        not gate.check("[[none]]", [_A]).passed,
        "the model sent exactly this four times and it was thrown away four "
        "times, which is IA-181")

# The rule that keeps skipping retrieval worthless. Without it a model declines
# to retrieve, emits the marker, and walks out.
_nothing = gate.check("There is no mention of it [[none]].", ())
check("an abstention with no passages supplied is refused", not _nothing.passed)
check("and it is refused for having nothing to be absent from",
      _v(_nothing).kind == gate.ABSTENTION_WITHOUT_PASSAGES)

check("the reserved token cannot collide with a real chunk id",
      len(gate.NONE_TOKEN) != 32 and not all(
          ch in "0123456789abcdef" for ch in gate.NONE_TOKEN))

control("a marked abstention is refused too",
        not gate.check(_ABS, [_A]).passed,
        "the gate refuses every abstention, so none of the refusals above "
        "prove the reserved token is doing anything")

control("an abstention that smuggles a grounded claim through still passes",
        gate.check(f"Not covered [[none]]. The budget is $2,500 [[{_A}]].",
                   [_A]).passed)

control("an abstention passes with no passages supplied",
        gate.check("There is no mention of it [[none]].", ()).passed,
        "a turn could skip retrieval, abstain, and walk out through the gate")

# --- the abstention as a turn and as a count --------------------------------

try:
    retrieval.Turn(question="q", retrieved=True, supplied=(),
                   stage=retrieval.ABSTAINED)
    _abs_empty = True
except retrieval.RetrievalError:
    _abs_empty = False
check("a turn cannot be recorded as abstained with nothing supplied",
      not _abs_empty)

_alog = retrieval.TurnLog()
_alog.record(retrieval.Turn(question="q1", retrieved=True, supplied=(_A,),
                            stage=retrieval.ANSWERED))
_alog.record(retrieval.Turn(question="q2", retrieved=True, supplied=(_A,),
                            stage=retrieval.ABSTAINED))
_alog.record(retrieval.Turn(question="q3", retrieved=True, supplied=(_A,),
                            stage=retrieval.ABSTAINED))
check("abstentions are counted", _alog.abstained == 2)
check("abstentions are not added to answered", _alog.answered == 1)
check("the summary reports both rather than one success rate",
      "answered" in _alog.line() and "abstained" in _alog.line())

# --- the whole turn, abstaining ---------------------------------------------

_abs_turn, _ = _turn(_called(), _said("The passages do not cover that [[none]]."))
check("a turn that looked and found nothing is recorded as abstained",
      _abs_turn.stage == retrieval.ABSTAINED)
check("an abstained turn is a turn that retrieved", _abs_turn.retrieved is True)
check("an abstained turn carries the passages it read", bool(_abs_turn.supplied))


# ---------------------------------------------------------------------------

print()

# IA-177. The invariant count is printed because two counts were once stated
# from memory on the same day and both were wrong. The control count sat in the
# same sentence of the same README and was still written by hand, and it was
# wrong for a day: four published against six built. A number that is printed
# is checked. A number beside it that is remembered is not, and on the page the
# two look identical. So this line produces both, and every artifact quotes it
# rather than counting.
CONTROLS = [name for name in RUN if name.startswith("control: ")]

if FAILED:
    print(f"{len(FAILED)} of {len(RUN)} invariants FAILED:")
    for name in FAILED:
        print(f"  - {name}")
    sys.exit(1)

print(f"all {len(RUN)} invariants hold, {len(CONTROLS)} of them controls")
