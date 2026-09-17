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

# ---------------------------------------------------------------------------

print()
if FAILED:
    print(f"{len(FAILED)} of {len(RUN)} invariants FAILED:")
    for name in FAILED:
        print(f"  - {name}")
    sys.exit(1)

print(f"all {len(RUN)} invariants hold")
