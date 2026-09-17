"""
Retrieval: the score reaches the caller, and a turn that never looked is counted.

Slice 4 sits on top of the store and has two jobs. Turn a question into a query
the store can answer, carrying the score all the way back to whoever has to
decide. And record, per turn, whether the corpus was consulted at all.

The second job produces this slice's one number, and it is the number the
target architecture registered as Q2: **the share of answers produced without
ever retrieving.** Both upstream agents make retrieval a tool the model may
decline to call, and both hold it in place with a line of prompt telling it to
always search first. Nothing counts how often that instruction is ignored, so
nobody knows whether it works. Counting it is cheap and nothing else has to
work for the count to mean something.

Three things this module refuses to do
--------------------------------------

**It does not declare the embedding width.** The upstream schema hardcodes 1536
and a local model does not emit that. The width is whatever a real call returns,
`measured_dim()` is the only thing that says what it is, and the store compares
a query against the width it actually holds. A number in a schema is a claim; a
number off a call is a measurement.

**It does not default the threshold.** `threshold` has no default value, and
`None` has to be passed deliberately. On the previous project a message blamed
a threshold that was switched off and sent somebody looking in the wrong place
for an afternoon (IA-163). A cutoff that can be forgotten is a cutoff that will
be, and a retrieval that quietly has no floor looks identical to one that does
until the day it returns the whole corpus.

**It does not let a skipped retrieval become an answer.** A turn where the agent
never called retrieve supplies no chunks, so every claim in whatever it produces
cites something it was not given, and the citation gate discards it. That is not
a rule enforced here, it is a consequence of supplying nothing, which is the
better kind: there is no branch to forget. An invariant asserts it anyway,
because a guarantee nobody tests is a guarantee nobody has.

No dependencies. The stub embedder below is not a model and says so.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import store as _store


class RetrievalError(Exception):
    """Retrieval could not be attempted. Never the same thing as finding nothing."""


@runtime_checkable
class Embedder(Protocol):
    """What retrieval needs from an embedder, and nothing more.

    Deliberately no `dim`. Ask the thing for a vector and measure it.
    """

    @property
    def name(self) -> str: ...

    def embed(self, text: str) -> Sequence[float]: ...


def measured_dim(embedder: Embedder, probe: str = "dimension probe") -> int:
    """The width of this embedder's output, from one real call.

    Q1 of the target architecture, and the cheapest question in the project to
    answer honestly. It is answered by calling, never by reading a constant.
    """
    vector = embedder.embed(probe)
    n = len(vector)
    if n == 0:
        raise RetrievalError(
            f"{embedder.name} returned a vector of width 0, so there is nothing "
            "to compare against")
    return n


# ---------------------------------------------------------------------------
# what a turn produced
# ---------------------------------------------------------------------------

ANSWERED = "answered"

# IA-179. A third outcome, because the first two could not hold it honestly.
# On the first real run, four turns retrieved, read the passages, found the
# answer absent and said so. Counted as answered, the gate looks like it passed
# something it never checked. Counted as a gate refusal, which is what happened,
# the model looks like it failed on the four questions it handled best. An
# abstention is neither, and a run that folds it into either one is wrong in one
# direction or the other.
ABSTAINED = "abstained"

REFUSED_THRESHOLD = "refused_threshold"
REFUSED_GATE = "refused_gate"
REFUSED_EMPTY_STORE = "refused_empty_store"
SKIPPED = "skipped_retrieval"


@dataclass(frozen=True)
class Retrieval:
    """What the store returned, with the scores intact.

    Both upstream projects compute a similarity and discard it one line later,
    which is why neither can decline for a reason it can name. The score is
    carried here for the same reason it is carried in the store: a threshold
    that cannot see a number cannot be a threshold.
    """

    question: str
    result: _store.SearchResult
    embedder: str
    dim: int

    @property
    def supplied(self) -> tuple[str, ...]:
        """Exactly what will be shown to the model, in rank order.

        This tuple is what the citation gate is given. Anything not in it was
        not supplied this turn, whatever else is true of it.
        """
        return tuple(h.record.chunk_id for h in self.result.hits)

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(h.score for h in self.result.hits)

    @property
    def declined(self) -> bool:
        return self.result.declined

    def reason(self) -> str:
        return self.result.reason()

    def context(self) -> str:
        """The passages as the model will see them, each labelled with its id.

        The label is the id the model is asked to cite, and it is the same
        string the gate will compare against. If those two ever diverge the gate
        refuses every answer, which is loud, and that is the intended direction
        to fail in.
        """
        blocks = []
        for hit in self.result.hits:
            r = hit.record
            blocks.append(f"[[{r.chunk_id}]] ({r.document_id}, chunk {r.chunk_index})\n{r.text}")
        return "\n\n".join(blocks)


def retrieve(
    question: str,
    *,
    embedder: Embedder,
    store_obj: "_store.Store",
    fingerprint: str,
    k: int = 5,
    threshold: float | None,
) -> Retrieval:
    """Embed the question and search, or refuse in a way that says why.

    `threshold` is keyword-only and has no default. Passing None means there is
    no floor, and that has to be an act.
    """
    if not question.strip():
        raise RetrievalError("an empty question was passed to retrieval")

    raw = embedder.embed(question)
    if len(raw) == 0:
        raise RetrievalError(f"{embedder.name} returned a vector of width 0")

    query = _store.normalise(raw)
    result = store_obj.search(query, k=k, fingerprint=fingerprint, threshold=threshold)
    return Retrieval(question=question, result=result,
                     embedder=embedder.name, dim=len(query))


@dataclass(frozen=True)
class Turn:
    """One question, and everything that decides whether it produced an answer."""

    question: str
    retrieved: bool
    supplied: tuple[str, ...]
    stage: str
    refusal: str | None = None

    # IA-187. What the user actually saw. A conversation that replays its own
    # history has to replay what was shown, not what was generated: on an
    # abstention those are different strings, and on a refusal the generated
    # text was discarded on purpose. Reconstructing it later from the stage
    # would be a second source of truth for something already decided.
    answer: str | None = None

    def __post_init__(self) -> None:
        if not self.retrieved and self.supplied:
            raise RetrievalError(
                "a turn that did not retrieve cannot have supplied chunks. "
                "Something handed the model passages it did not ask for, and "
                "the skipped-retrieval count would be a lie")
        if self.stage in (ANSWERED, ABSTAINED) and self.answer is None:
            raise RetrievalError(
                f"a turn recorded as {self.stage} with no answer. Something was "
                "shown to the user and this turn does not know what, so the "
                "conversation cannot replay itself honestly")
        if self.stage not in (ANSWERED, ABSTAINED) and self.answer is not None:
            raise RetrievalError(
                f"a turn recorded as {self.stage} is carrying an answer. A "
                "refused turn's text was discarded, and keeping it here is how "
                "it comes back")
        if self.stage in (ANSWERED, ABSTAINED) and not self.supplied:
            raise RetrievalError(
                f"a turn recorded as {self.stage} with nothing supplied. An "
                "answer would cite what the model was not shown, and an "
                "abstention would be about passages that do not exist. The "
                "gate cannot have passed either, so this turn is mislabelled")


@dataclass
class TurnLog:
    """The slice's one number, and the counts it is built from.

    Kept as a list of turns rather than as running totals, because a rate with
    no rows behind it cannot be checked and cannot be broken down later.
    """

    turns: list[Turn] = field(default_factory=list)

    def record(self, turn: Turn) -> Turn:
        self.turns.append(turn)
        return turn

    @property
    def total(self) -> int:
        return len(self.turns)

    @property
    def skipped(self) -> int:
        return sum(1 for t in self.turns if not t.retrieved)

    @property
    def answered(self) -> int:
        return sum(1 for t in self.turns if t.stage == ANSWERED)

    @property
    def abstained(self) -> int:
        """Turns where the model looked, found nothing, and said so.

        Kept apart from `answered` deliberately. Adding them together would
        report a higher success rate and would stop distinguishing a model that
        answers from one that knows when it cannot, which is the distinction
        this whole project is about.
        """
        return sum(1 for t in self.turns if t.stage == ABSTAINED)

    @property
    def skipped_rate(self) -> float | None:
        """The share of turns where the corpus was never consulted.

        None on an empty log rather than 0.0. A rate of zero over no turns is
        the most confident possible way to say nothing, and it is exactly the
        shape of result this portfolio keeps finding: a number that reads as
        good news and is produced by an absence.
        """
        if not self.turns:
            return None
        return self.skipped / len(self.turns)

    def line(self) -> str:
        if not self.turns:
            return "no turns recorded, so there is no rate to report"
        pct = 100.0 * self.skipped_rate
        return (f"{self.skipped} of {self.total} turns never retrieved "
                f"({pct:.1f}%) · {self.answered} answered and survived the "
                f"gate · {self.abstained} abstained, having looked")


# ---------------------------------------------------------------------------
# a stand-in, which is not a model
# ---------------------------------------------------------------------------

class StubEmbedder:
    """Deterministic, offline, and carrying no knowledge of anything.

    It exists so the contract, the invariants and the mutations can be built and
    proved before Ollama is installed, exactly as the chunker was proved against
    three stand-in tokenizers before a real one was ever loaded. It hashes
    tokens into buckets. Similar text lands in similar buckets, which is enough
    to test plumbing and is not enough to test retrieval quality.

    **Nothing measured with this is a result about retrieval.** The suite uses
    it. The measurement will not.
    """

    def __init__(self, dim: int = 64, *, label: str = "stub") -> None:
        if dim <= 0:
            raise ValueError("an embedder with no dimensions embeds nothing")
        self._dim = dim
        self._label = label

    @property
    def name(self) -> str:
        return f"stub-{self._label}-{self._dim}"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for token in text.lower().split():
            h = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "big") % self._dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        if not any(vector):
            # A zero vector has no direction, and the store refuses to store one.
            # Giving it a fixed direction is honest as long as it is said: an
            # empty or unknown query lands in the same place every time.
            vector[0] = 1.0
        length = math.sqrt(sum(x * x for x in vector))
        return [x / length for x in vector]
