"""
The conversation: turns in sequence, a cap that refuses before anything is sent,
and the rule that a turn which produced no answer cannot be referred to.

Slice 4 answers one question at a time. Figure 2 of the target architecture draws
a conversation above that question and a refusal on it, and neither was ever
built. IA-186 recorded that the page said otherwise; this module is the half of
IA-187 that needs no model.

Three things live here.

**The cap, which refuses before the call rather than after.** Figure 2's first
refusal reads *nothing was sent*, and that is the whole point: the check happens
while it still costs nothing. At 90.9 seconds a turn, a request that was never
going to be worth sending is 90.9 seconds of somebody's afternoon.

**A ratio that is measured rather than assumed.** The cap is in tokens and this
repository cannot tokenize the generation model: Ollama serves it and exposes no
tokenizer. Writing down a characters-per-token constant here would be the exact
defect slice 2 exists to document, where 3.5 turned out to be 5.69. So the ratio
is **learned from the conversation's own completed turns**, because every Ollama
response reports `prompt_eval_count`, the token count the model itself assigned
to the prompt it just read. The caller states a seed for the first turn, with no
default, and from the second turn on the seed is replaced by measurement. The gap
between what was estimated and what the model actually counted is recorded rather
than discarded, so the estimator can be held to account by its own history.

**An abstention poisons the referent.** If the previous turn produced no answer,
a follow-up that refers to it has nothing to refer to, and the system says so
instead of letting the question be rewritten into a premise. Without this, a
model declines to answer, is asked *"how many weeks is that?"*, and rewrites the
absent thing into existence. That is a state comparison, not language
understanding: the previous turn's outcome is recorded.

What the referent rule is, precisely, and what it is not
--------------------------------------------------------

Detecting that a question refers to the previous turn needs a signal, and the
only one available without a second model is **lexical**: a closed list of
referring expressions. *that, it, they, those, this, them, the same*, and a
question so elliptical it has no subject of its own.

That list is English, hand-written, and it is the weakest part of this module.
It is stated here rather than buried: a question that refers back without using
any of these words is not caught, and the rule then allows a turn it should have
refused. The failure is in the permissive direction, which is the one that has to
be named out loud, and it is why the count this slice produces is a **lower
bound** on how often the model would have built on nothing.

No dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import retrieval

# Outcomes that leave nothing for a later turn to refer to.
NO_ANSWER_STAGES = frozenset({
    retrieval.ABSTAINED,
    retrieval.REFUSED_THRESHOLD,
    retrieval.REFUSED_GATE,
    retrieval.SKIPPED,
    retrieval.REFUSED_EMPTY_STORE,
})

# New stages this slice adds.
REFUSED_CAP = "refused_cap"
REFUSED_REFERENT = "refused_referent"
REFUSED_REWRITE = "refused_rewrite"

# The closed list. Hand-written, English, and the weakest part of this module.
REFERRING = frozenset({
    "that", "those", "it", "its", "they", "them", "their", "this", "these",
    "he", "she", "him", "her", "his", "hers", "same", "there", "then",
})

_WORD = re.compile(r"[a-z0-9']+")


class ConversationError(Exception):
    """The conversation cannot continue in a way that would still be honest."""


@dataclass(frozen=True)
class CapDecision:
    """Whether the next turn may be sent, decided before sending it."""

    fits: bool
    estimated_tokens: int
    budget: int
    ratio: float
    ratio_source: str
    chars: int

    def reason(self) -> str:
        if self.fits:
            return ""
        return (f"the next turn would be about {self.estimated_tokens} tokens "
                f"({self.chars} characters at {self.ratio:.2f} chars/token, "
                f"{self.ratio_source}) against a budget of {self.budget}. "
                f"Nothing was sent: the turn is refused, not the run")


@dataclass(frozen=True)
class RatioSample:
    """One completed turn's prompt, in characters and in the model's own tokens."""

    chars: int
    tokens: int
    estimated: int | None = None

    @property
    def ratio(self) -> float:
        return self.chars / self.tokens

    @property
    def error(self) -> int | None:
        """How far the estimate was out, in tokens. Positive means underestimated."""
        if self.estimated is None:
            return None
        return self.tokens - self.estimated


@dataclass(frozen=True)
class ReferentDecision:
    allowed: bool
    referring_words: tuple[str, ...]
    previous_stage: str | None

    def reason(self) -> str:
        if self.allowed:
            return ""
        words = ", ".join(self.referring_words)
        return (f"this question refers back ({words}) and the turn before it "
                f"ended in {self.previous_stage}, so there is nothing to refer "
                f"to. Answering would mean inventing what the reference points "
                f"at. Ask the question without the reference")


@dataclass
class Conversation:
    """Turns in order, and the two refusals that sit above them.

    `seed_ratio` has no default. The first turn has no measurement behind it, and
    a characters-per-token constant written into this file would be exactly what
    slice 2 exists to document. Stating it is an act, it is recorded as a seed,
    and it is replaced by measurement from the second turn onward.
    """

    token_budget: int
    seed_ratio: float
    turns: list[retrieval.Turn] = field(default_factory=list)
    samples: list[RatioSample] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.token_budget < 1:
            raise ConversationError(
                f"a token budget of {self.token_budget} cannot hold a question")
        if self.seed_ratio <= 0:
            raise ConversationError(
                f"a seed ratio of {self.seed_ratio} is not a ratio")

    # -- the measured ratio -------------------------------------------------

    @property
    def ratio(self) -> float:
        """Characters per token: measured if there is anything to measure."""
        if not self.samples:
            return self.seed_ratio
        return sum(s.chars for s in self.samples) / sum(s.tokens for s in self.samples)

    @property
    def ratio_source(self) -> str:
        if not self.samples:
            return "seed, nothing measured yet"
        n = len(self.samples)
        return f"measured over {n} turn{'s' if n != 1 else ''}"

    def record_prompt(self, chars: int, tokens: int,
                      estimated: int | None = None) -> RatioSample:
        """Record what the model said the prompt cost, in its own tokens.

        `prompt_eval_count` is the only place in this pipeline where the
        generation model's tokenizer is observable. Throwing it away would leave
        the estimator with nothing but the seed forever.
        """
        if chars < 0 or tokens < 1:
            raise ConversationError(
                f"a prompt of {chars} characters and {tokens} tokens is not a "
                "measurement")
        sample = RatioSample(chars=chars, tokens=tokens, estimated=estimated)
        self.samples.append(sample)
        return sample

    @property
    def worst_underestimate(self) -> int:
        """The largest number of tokens the estimator ever missed low by.

        Reported rather than smoothed, because an estimator that is usually right
        and occasionally very wrong is the one that lets a turn through.
        """
        errors = [s.error for s in self.samples if s.error is not None]
        return max([e for e in errors if e > 0], default=0)

    # -- the cap ------------------------------------------------------------

    def check_cap(self, prompt: str) -> CapDecision:
        """Does the next turn fit? Decided before anything is sent."""
        chars = len(prompt)
        ratio = self.ratio
        estimated = int(chars / ratio + 0.5)
        return CapDecision(
            fits=estimated <= self.token_budget,
            estimated_tokens=estimated,
            budget=self.token_budget,
            ratio=ratio,
            ratio_source=self.ratio_source,
            chars=chars,
        )

    # -- the referent -------------------------------------------------------

    @property
    def last(self) -> retrieval.Turn | None:
        return self.turns[-1] if self.turns else None

    def check_referent(self, question: str) -> ReferentDecision:
        """A turn that produced no answer cannot be referred to."""
        words = tuple(sorted({w for w in _WORD.findall(question.lower())
                              if w in REFERRING}))
        previous = self.last
        if previous is None or previous.stage not in NO_ANSWER_STAGES:
            return ReferentDecision(True, words,
                                    previous.stage if previous else None)
        if not words:
            # A fresh, self-contained question after an abstention is fine. The
            # abstention poisons the referent, not the conversation.
            return ReferentDecision(True, words, previous.stage)
        return ReferentDecision(False, words, previous.stage)

    # -- history ------------------------------------------------------------

    def record(self, turn: retrieval.Turn) -> retrieval.Turn:
        self.turns.append(turn)
        return turn

    def transcript(self) -> list[dict]:
        """The turns as chat messages, for the model to read.

        Only turns that produced something the user saw are carried. A refused
        turn is not part of the conversation's content: replaying it would hand
        the model the very premise the refusal existed to withhold.
        """
        messages: list[dict] = []
        for turn in self.turns:
            if turn.stage not in (retrieval.ANSWERED, retrieval.ABSTAINED):
                continue
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer or ""})
        return messages

    def line(self) -> str:
        refused = sum(1 for t in self.turns
                      if t.stage in (REFUSED_CAP, REFUSED_REFERENT, REFUSED_REWRITE))
        return (f"{len(self.turns)} turns · {refused} refused before the model "
                f"was asked · ratio {self.ratio:.2f} chars/token "
                f"({self.ratio_source})")
