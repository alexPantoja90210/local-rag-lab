"""
The citation gate: generated text that cites something it was not given is discarded.

Every tutorial in this family holds its grounding in place with a line of prompt.
*Always cite the provided context.* *Only answer from the knowledge base.* That is
an instruction with no mechanism, and an instruction with no mechanism is not a
rule. Nothing reads the answer afterwards, nothing compares what was cited
against what was supplied, and nothing can refuse. The model complies most of
the time, which is the worst possible failure rate: often enough to look
trustworthy, not often enough to be.

This module is the mechanism. It takes the text the model produced and the exact
set of chunk ids retrieval handed it on THIS turn, and it either returns the
answer or refuses it and says which claim failed and how.

Four things it refuses, and the third is the one that matters
-------------------------------------------------------------

1. **A claim with no citation at all.** The default failure. Easy to see.

2. **A citation naming an id that does not exist anywhere.** The model invented
   a plausible-looking identifier. Also easy to see, once something looks.

3. **A citation naming an id that is real, and was not supplied this turn.**
   This is the one this module exists for. It is indistinguishable from a
   correct answer by eye: the id resolves, the document is real, the citation
   renders as a working link. The only thing wrong with it is that the model
   was never shown that passage, so whatever it said about it came from its
   weights. It is the same shape as the store's fingerprint refusal in IA-175:
   a failure whose output looks exactly like success.

4. **A claim that is nothing but a citation.** `[[a3f...]]` on its own is a
   citation, not a claim, and a gate that counts it as satisfied can be passed
   by a model that emits citations and no content. A check that can be
   satisfied without doing the thing is not a check.

What it does not do, stated here rather than discovered later
-------------------------------------------------------------

**It does not check that the claim is faithful to the chunk it cites.** A
sentence may cite a supplied passage and misrepresent it completely, and this
module will pass it. Verbatim span matching is designed and deliberately not
built, and the target architecture says so. What is enforced is narrower and
worth naming precisely: *every claim points at something the model was actually
shown*. That is a real guarantee and it is not the guarantee a reader assumes
from the phrase "citation gate", which is why this paragraph is here.

Matching is exact
-----------------

No case folding, no whitespace stripping inside the id, no prefix matching. An
id that is a prefix of a supplied id is not supplied. Prefix matching is how a
check like this usually fails open: `if cited in supplied_text` passes on a
substring and nobody notices until the day it matters.

No dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

# The citation form the model is asked to produce. Deliberately not a shape that
# occurs in ordinary prose, so a false positive needs effort.
CITATION = re.compile(r"\[\[([^\[\]]*)\]\]")


def plain(text: str) -> str:
    """An answer's prose, with its citation markup removed.

    IA-197. What the user read was "$2,500". The `[[d7b771e5...]]` beside it is
    this system's own bookkeeping -- the proof carried alongside the sentence,
    not a word of it. Anything that reuses an answer as **content** has to drop
    the markup first, or the machine's accounting leaks into the next turn: a
    chunk id ended up inside a search query, and every cited id was being
    recorded as a word the conversation contained.

    One function, because the rule has more than one caller and a rule with two
    implementations has two ways to be forgotten.
    """
    return " ".join(CITATION.sub(" ", text).split())

# A claim ends at a sentence terminator or at a line break. Models in this
# family answer in both paragraphs and bullet lists, and a bullet without a full
# stop is still a claim. Splitting on sentences alone would let a whole bulleted
# answer count as one claim that is satisfied by one citation on the last line.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Leading list markers, so "- the retention is 90 days [[id]]" is judged on its
# content rather than on its bullet.
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

UNCITED = "uncited"
INVENTED = "invented"
NOT_SUPPLIED = "not_supplied"
EMPTY_CLAIM = "empty_claim"
MIXED_ABSTENTION = "mixed_abstention"
ABSTENTION_WITHOUT_PASSAGES = "abstention_without_passages"

# IA-179. The reserved citation, meaning: no passage supports this, and this
# answer is an abstention.
#
# The first version of this gate had no such thing, and on the first real run it
# discarded all four correct abstentions. The model had retrieved, read the
# passages, found nothing and said so, which is the best available behaviour on
# an unanswerable question, and the gate threw it away because the sentence
# carried no id.
#
# Worse than that. The prompt had told the model to abstain AND cite the closest
# passage, so had the model obeyed, it would have produced a sentence citing a
# passage that does not support it, attached purely to satisfy a format, and
# **the gate would have passed it**. On abstentions the sign was inverted: a
# decorative citation rewarded, an honest one punished.
#
# It is not a hex id, so it can never collide with a real chunk.
NONE_TOKEN = "none"

# IA-181. The sentence a reader sees when the answer is an abstention, written
# here and never by the model.
#
# The first fix asked the model for the marker AND a sentence. It sent the
# marker and no sentence, exactly as the version before it had sent a sentence
# and no marker: a two-part instruction obeyed by halves, a different half each
# time, refused both times.
#
# The two parts are not alike. The marker is a VERDICT and only the model can
# produce it, because only the model read the passages. The sentence is WORDING.
# It is not a claim, it rests on no passage, it states nothing about the corpus
# that could be wrong, and nothing downstream checks it. Requiring it of the
# model was a wish with no mechanism behind it, which is the thing this
# repository exists to point at. Generating it is a mechanism.
#
# It is also safer. The model's own phrasing can misdescribe what it looked at.
# This cannot.
ABSTENTION_SENTENCE = (
    "The passages retrieved for this question do not contain an answer to it.")


class GateError(Exception):
    """The gate could not be applied. Never the same thing as a refusal."""


@dataclass(frozen=True)
class Claim:
    """One sentence of the generated text, and what it cited."""

    index: int
    text: str
    citations: tuple[str, ...]

    @property
    def prose(self) -> str:
        """The claim with its citations removed, which is what it actually says.

        Removing a citation leaves the gap it occupied, so the whitespace is
        closed up afterwards. This text goes into the refusal message, and a
        refusal that quotes the claim back with "90 days ." in it reads like the
        gate is confused rather than the answer.
        """
        without = _LIST_MARKER.sub("", CITATION.sub("", self.text))
        without = re.sub(r"\s+", " ", without)
        without = re.sub(r"\s+([.,;:!?)\]])", r"\1", without)
        return without.strip()


@dataclass(frozen=True)
class Violation:
    kind: str
    claim_index: int
    citation: str | None
    detail: str


@dataclass(frozen=True)
class GateResult:
    """Passed, with the answer, or refused, with the text discarded.

    `answer` is None whenever `passed` is False, and that is not a formality.
    The architecture says the generated text is discarded on refusal, so this
    object does not carry it. A caller that wants to log the rejected text has
    to reach for `rejected_text` by name, which makes showing it to a user an
    explicit act rather than an accident of returning everything.
    """

    passed: bool
    answer: str | None
    claims: tuple[Claim, ...]
    supplied: tuple[str, ...]
    violations: tuple[Violation, ...]
    rejected_text: str | None = None
    abstained: bool = False
    model_note: str | None = None   # what the model wrote alongside an
                                    # abstention marker. Unchecked text, kept
                                    # for the record and never shown as the
                                    # answer.

    @property
    def cited(self) -> tuple[str, ...]:
        seen: list[str] = []
        for claim in self.claims:
            for c in claim.citations:
                if c not in seen:
                    seen.append(c)
        return tuple(seen)

    def reason(self) -> str:
        """Why it was refused, naming the claim and the id, never just 'failed'.

        The previous project shipped a message blaming a threshold that was
        switched off and it cost an afternoon (IA-163). A refusal that does not
        say which claim and which id sends the reader to the prompt, which is
        almost never where the problem is.
        """
        if self.passed:
            return ""
        if not self.violations:  # pragma: no cover - guarded by an invariant
            raise GateError("refused with no violation recorded, which is a bug "
                            "in the gate rather than a fact about the answer")

        lines = []
        for v in self.violations:
            claim = self.claims[v.claim_index] if 0 <= v.claim_index < len(self.claims) else None
            where = f"claim {v.claim_index + 1}"
            if claim is not None and claim.prose:
                excerpt = claim.prose if len(claim.prose) <= 70 else claim.prose[:67] + "..."
                where += f' ("{excerpt}")'
            lines.append(f"{where}: {v.detail}")

        supplied = ", ".join(self.supplied) if self.supplied else "nothing"
        return (f"the generated text was discarded. {len(self.violations)} of "
                f"{len(self.claims)} claims did not hold.\n  "
                + "\n  ".join(lines)
                + f"\n  supplied this turn: {supplied}")


def split_claims(text: str) -> tuple[Claim, ...]:
    """Split generated text into claims, keeping each one's citations with it.

    A citation belongs to the claim whose text contains it. That is the whole
    rule, and it is why the model is asked to put the citation inside the
    sentence rather than at the end of a paragraph: a citation after the full
    stop belongs to the next sentence, and the gate will say so rather than
    guessing what was meant.
    """
    claims: list[Claim] = []
    for line in text.splitlines():
        for part in _SENTENCE.split(line):
            if not part.strip():
                continue
            claims.append(Claim(
                index=len(claims),
                text=part.strip(),
                citations=tuple(m.group(1).strip() for m in CITATION.finditer(part)),
            ))
    return tuple(claims)


def check(
    text: str,
    supplied: Sequence[str] | Iterable[str],
    *,
    known_ids: Sequence[str] | Iterable[str] | None = None,
) -> GateResult:
    """Apply the gate.

    `supplied` is what retrieval handed the model on this turn. Not what is in
    the store, not what it was shown three turns ago. The distinction is the
    point of the module.

    `known_ids` is optional and changes no verdict. It only separates an
    invented id from a real one that was not supplied, because those two
    failures mean different things: the first is a model making things up, the
    second is a model answering from its weights about a passage it recognises.
    Passing it in is how the more interesting one gets counted.
    """
    supplied_t = tuple(supplied)
    known_t = tuple(known_ids) if known_ids is not None else None

    duplicates = [c for c in set(supplied_t) if supplied_t.count(c) > 1]
    if duplicates:
        raise GateError(
            "the same chunk id was supplied more than once this turn "
            f"({', '.join(sorted(duplicates))}). Retrieval returned the same "
            "passage twice, which inflates how grounded an answer looks")

    supplied_set = set(supplied_t)
    known_set = set(known_t) if known_t is not None else None

    claims = split_claims(text)
    violations: list[Violation] = []
    abstained = False

    if not claims:
        violations.append(Violation(
            kind=UNCITED, claim_index=-1, citation=None,
            detail="nothing was generated, so there is no claim to ground"))

    # IA-179. An abstention is handled before anything else, and it is all or
    # nothing. The reserved token may not sit beside a claim drawn from a
    # passage, because an answer that says "the passages do not cover this" and
    # then states three facts anyway has smuggled the three facts through under
    # the abstention. That is the obvious way to walk through this gate and it
    # is the first thing closed.
    marked = [c for c in claims if NONE_TOKEN in c.citations]
    if marked:
        claim = claims[0]
        if len(claims) != 1:
            violations.append(Violation(
                kind=MIXED_ABSTENTION, claim_index=marked[0].index,
                citation=NONE_TOKEN,
                detail=(f"{len(claims)} claims, one of them an abstention. An "
                        "abstention is the whole answer or it is not one")))
        elif set(claim.citations) != {NONE_TOKEN}:
            others = sorted(set(claim.citations) - {NONE_TOKEN})
            violations.append(Violation(
                kind=MIXED_ABSTENTION, claim_index=claim.index,
                citation=NONE_TOKEN,
                detail=(f"cited {NONE_TOKEN} and {', '.join(others)} in the same "
                        "claim. It cannot both rest on a passage and say no "
                        "passage supports it")))
        elif not supplied_t:
            # The rule that keeps skipping retrieval worthless. An abstention is
            # a statement ABOUT the passages, so with no passages there is
            # nothing to abstain about. Without this, a model could decline to
            # retrieve, emit the marker, and walk out through the gate, and the
            # guarantee that a skipped turn cannot produce an answer would be
            # gone.
            violations.append(Violation(
                kind=ABSTENTION_WITHOUT_PASSAGES, claim_index=claim.index,
                citation=NONE_TOKEN,
                detail=("nothing was supplied this turn, so there are no "
                        "passages for this answer to be absent from")))
        else:
            abstained = True

        passed = not violations
        return GateResult(
            passed=passed,
            answer=ABSTENTION_SENTENCE if passed else None,
            claims=claims, supplied=supplied_t, violations=tuple(violations),
            rejected_text=None if passed else text, abstained=abstained,
            model_note=claim.prose or None)

    for claim in claims:
        if not claim.citations:
            violations.append(Violation(
                kind=UNCITED, claim_index=claim.index, citation=None,
                detail="no citation at all"))
            continue

        if not claim.prose:
            violations.append(Violation(
                kind=EMPTY_CLAIM, claim_index=claim.index, citation=None,
                detail="a citation on its own is not a claim"))
            continue

        for cited in claim.citations:
            if cited in supplied_set:
                continue
            if known_set is not None and cited in known_set:
                violations.append(Violation(
                    kind=NOT_SUPPLIED, claim_index=claim.index, citation=cited,
                    detail=(f"cited {cited}, which is a real chunk that was NOT "
                            "supplied this turn. The model was never shown it")))
            else:
                violations.append(Violation(
                    kind=INVENTED, claim_index=claim.index, citation=cited,
                    detail=f"cited {cited}, which was not supplied and is not a known chunk"))

    passed = not violations
    return GateResult(
        passed=passed,
        answer=text if passed else None,
        claims=claims,
        supplied=supplied_t,
        violations=tuple(violations),
        rejected_text=None if passed else text,
        abstained=False,
    )
