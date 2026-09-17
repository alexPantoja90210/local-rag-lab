"""
Chunking against the embedding model's real token window.

The rule this module exists for: **no chunk may exceed the budget the embedding
model can actually read.** Not approximately, not usually, not after truncation.

Where it comes from. The previous project sized its chunks by a rule of thumb,
896 characters at an assumed 3.5 characters per token, against a 256-token
window. On a real corpus 864 of 11,529 passages still went over, and the
pipeline truncated them silently at embed time. Truncation was recorded and
flagged, which was the right call at the time, but it is a compromise, not a
mechanism: text went into the store that the model never read, and the only
reason anyone knows is that someone thought to count.

Two things make that impossible here.

  1. **The budget comes from the tokenizer, not from a constant.** Ask the
     model how much it can read. A number in a config file is a claim about a
     model; a number read off the model is the model.

  2. **Nothing is ever truncated.** The spans of the chunks tile the document
     with no gaps, so every character reaches exactly one chunk. When a unit
     cannot fit, the splitter descends to a finer boundary and **records which
     one it had to use**. A hard split mid-word is legal, visible and counted,
     never silent.

No third-party dependencies. `Tokenizer` is a protocol with two methods, so the
real one and the test ones are the same shape and nothing here knows which it
has.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol, Sequence

# ---------------------------------------------------------------------------
# what a tokenizer has to be
# ---------------------------------------------------------------------------


class Tokenizer(Protocol):
    """Everything the chunker is allowed to know about a tokenizer."""

    @property
    def name(self) -> str:
        """Stable identifier, recorded on every chunk it produced."""

    @property
    def budget(self) -> int:
        """Tokens the model can read in one pass. Read from the model."""

    def count(self, text: str) -> int:
        """Tokens this text costs."""


class BudgetExceeded(Exception):
    """Raised when a chunk would go over budget. It is never returned."""


# ---------------------------------------------------------------------------
# what comes out
# ---------------------------------------------------------------------------

# Ordered coarsest to finest. The splitter descends only as far as it must,
# and every chunk records the level it stopped at.
SPLIT_LEVELS = ("paragraph", "sentence", "line", "word", "character")

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_LINE = re.compile(r"\n")
_WORD = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    split_level: str      # the finest boundary this chunk needed
    tokenizer: str        # which tokenizer's budget it was packed against
    budget: int

    @property
    def chunk_id(self) -> str:
        """Derived from content and identity, so a re-run produces the same id.

        Both upstream projects mint a fresh uuid4 per chunk, which removes the
        one property an explicit id exists to give you: re-adding identical
        content updates in place instead of duplicating.
        """
        h = hashlib.sha256()
        for part in (self.document_id, str(self.chunk_index), self.tokenizer,
                     str(self.budget), self.text):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:32]


def pack_fingerprint(tokenizer: Tokenizer) -> str:
    """Identifies the packing a store was built with.

    A store chunked for one embedder and queried with another is measuring
    nothing, and it looks exactly like a store that works. Recording this
    alongside the vectors is what lets a later slice notice.
    """
    return hashlib.sha256(
        f"{tokenizer.name}|{tokenizer.budget}".encode("utf-8")
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def _spans(text: str, pattern: re.Pattern[str] | None) -> list[tuple[int, int]]:
    """Spans of `text` at one boundary level, covering it with no gaps.

    Separators are kept with the piece that precedes them, so the spans tile
    the string exactly. `None` means split into single characters.
    """
    if not text:
        return []
    if pattern is None:
        return [(i, i + 1) for i in range(len(text))]

    spans: list[tuple[int, int]] = []
    cursor = 0
    for m in pattern.finditer(text):
        end = m.end()
        if end > cursor:
            spans.append((cursor, end))
            cursor = end
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return spans


_LEVEL_PATTERN: dict[str, re.Pattern[str] | None] = {
    "paragraph": _PARAGRAPH,
    "sentence": _SENTENCE,
    "line": _LINE,
    "word": _WORD,
    "character": None,
}


def _units(text: str, level_index: int) -> list[tuple[int, int]]:
    return _spans(text, _LEVEL_PATTERN[SPLIT_LEVELS[level_index]])


def _fit(text: str, tokenizer: Tokenizer, level_index: int) -> tuple[int, str]:
    """Longest prefix of `text` that fits the budget, and the level it needed.

    Descends through the boundary levels until something fits. Character level
    always terminates, because a single character either fits or the budget is
    unusable and the caller is told so rather than handed a silent truncation.
    """
    budget = tokenizer.budget

    for idx in range(level_index, len(SPLIT_LEVELS)):
        units = _units(text, idx)
        taken = 0
        for start, end in units:
            if tokenizer.count(text[:end]) > budget:
                break
            taken = end
        if taken > 0:
            return taken, SPLIT_LEVELS[idx]

    # Nothing fit, not even one character.
    raise BudgetExceeded(
        f"a budget of {budget} tokens cannot hold a single character of this "
        f"document under tokenizer {tokenizer.name!r}"
    )


def chunk_text(
    text: str,
    *,
    document_id: str,
    tokenizer: Tokenizer,
    overlap_tokens: int = 0,
) -> list[Chunk]:
    """Split `text` into chunks that each fit the tokenizer's budget.

    The spans tile `text`: every character lands in exactly one chunk's
    [char_start, char_end). Overlap, when asked for, repeats text from the
    previous chunk into the next one's `text` without changing that tiling, so
    coverage stays checkable and nothing is double-counted.
    """
    if tokenizer.budget < 1:
        raise ValueError(
            f"tokenizer {tokenizer.name!r} reports a budget of "
            f"{tokenizer.budget}, which cannot hold anything")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")
    if overlap_tokens >= tokenizer.budget:
        raise ValueError(
            f"overlap of {overlap_tokens} tokens does not fit a budget of "
            f"{tokenizer.budget}; it would never advance")
    if not text:
        return []

    chunks: list[Chunk] = []
    cursor = 0
    index = 0

    while cursor < len(text):
        remaining = text[cursor:]
        take, level = _fit(remaining, tokenizer, 0)
        body = remaining[:take]

        prefix = ""
        if overlap_tokens and chunks:
            prefix = _tail_within(chunks[-1].text, tokenizer, overlap_tokens)
            # The prefix is context, not coverage. If it pushes the chunk over
            # budget, it is shortened, never the body.
            while prefix and tokenizer.count(prefix + body) > tokenizer.budget:
                prefix = prefix[1:]

        chunk_text_value = prefix + body
        chunks.append(Chunk(
            document_id=document_id,
            chunk_index=index,
            text=chunk_text_value,
            char_start=cursor,
            char_end=cursor + take,
            token_count=tokenizer.count(chunk_text_value),
            split_level=level,
            tokenizer=tokenizer.name,
            budget=tokenizer.budget,
        ))

        cursor += take          # strictly positive, so this always terminates
        index += 1

    return chunks


def _tail_within(text: str, tokenizer: Tokenizer, tokens: int) -> str:
    """Longest suffix of `text` costing at most `tokens`, on a word boundary."""
    if tokens <= 0 or not text:
        return ""
    words = _spans(text, _WORD)
    best = ""
    for start, _ in reversed(words):
        candidate = text[start:]
        if tokenizer.count(candidate) > tokens:
            break
        best = candidate
    return best


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PackReport:
    chunks: int
    over_budget: int              # must always be 0
    by_level: dict[str, int]
    max_tokens: int
    tokenizer: str
    budget: int
    fingerprint: str

    def line(self) -> str:
        levels = " ".join(f"{k}={v}" for k, v in self.by_level.items() if v)
        return (f"{self.chunks:5d} chunks  max {self.max_tokens}/{self.budget} tok  "
                f"over budget {self.over_budget}  [{levels}]")


def report(chunks: Sequence[Chunk], tokenizer: Tokenizer) -> PackReport:
    by_level = {level: 0 for level in SPLIT_LEVELS}
    for c in chunks:
        by_level[c.split_level] += 1
    return PackReport(
        chunks=len(chunks),
        over_budget=sum(1 for c in chunks if c.token_count > c.budget),
        by_level=by_level,
        max_tokens=max((c.token_count for c in chunks), default=0),
        tokenizer=tokenizer.name,
        budget=tokenizer.budget,
        fingerprint=pack_fingerprint(tokenizer),
    )


def coverage_gaps(text: str, chunks: Sequence[Chunk]) -> list[tuple[int, int]]:
    """Stretches of `text` no chunk covers. Must always be empty.

    This is the assertion that makes silent truncation impossible to ship:
    it is a property of the output, checked against the input, and it does
    not care how the splitter reached its answer.
    """
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for c in sorted(chunks, key=lambda c: c.char_start):
        if c.char_start > cursor:
            gaps.append((cursor, c.char_start))
        cursor = max(cursor, c.char_end)
    if cursor < len(text):
        gaps.append((cursor, len(text)))
    return gaps


# ---------------------------------------------------------------------------
# tokenizers for the suite
# ---------------------------------------------------------------------------
# Three of them, structurally different on purpose. Every packing invariant
# runs against all three, so the logic cannot be tuned to one tokenizer's
# shape without the suite noticing.
#
# What this does NOT prove: none of them carries a real model's vocabulary, so
# none can produce a real model's numbers. That is what measure_window.py is
# for, and it needs the real tokenizer downloaded.


class WordTokenizer:
    """One token per whitespace-separated word."""

    def __init__(self, budget: int = 16) -> None:
        self._budget = budget

    @property
    def name(self) -> str:
        return f"stub-word-{self._budget}"

    @property
    def budget(self) -> int:
        return self._budget

    def count(self, text: str) -> int:
        return len(text.split())


class CharGroupTokenizer:
    """One token per fixed run of characters. Ignores word boundaries."""

    def __init__(self, budget: int = 8, chars_per_token: int = 4) -> None:
        self._budget = budget
        self._n = chars_per_token

    @property
    def name(self) -> str:
        return f"stub-chargroup-{self._n}-{self._budget}"

    @property
    def budget(self) -> int:
        return self._budget

    def count(self, text: str) -> int:
        return -(-len(text) // self._n) if text else 0


class SubwordishTokenizer:
    """Short words cost one token, long words cost several.

    Closer in shape to WordPiece than the other two: the cost of a word is not
    proportional to the text length, and a single long word can cost more than
    a short sentence.
    """

    def __init__(self, budget: int = 12, piece: int = 4) -> None:
        self._budget = budget
        self._piece = piece

    @property
    def name(self) -> str:
        return f"stub-subword-{self._piece}-{self._budget}"

    @property
    def budget(self) -> int:
        return self._budget

    def count(self, text: str) -> int:
        total = 0
        for word in text.split():
            total += max(1, -(-len(word) // self._piece))
        return total


SUITE_TOKENIZERS: tuple[Tokenizer, ...] = (
    WordTokenizer(16),
    CharGroupTokenizer(8, 4),
    SubwordishTokenizer(12, 4),
)
