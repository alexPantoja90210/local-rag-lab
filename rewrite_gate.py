"""
The rewritten question goes through the same gate as the answer.

The asymmetry this closes is the whole point, and it is not about understanding
sentences. **The rewrite is generated text.** The same model produces it, in the
same turn, and it carries assertions. Until now it was printed on screen and
nothing checked it, while the answer beside it was checked against every supplied
passage. There was no justification for that split.

The failure it catches, which the citation gate cannot
------------------------------------------------------

Turn 1 asks about a policy the corpus does not contain, and the system abstains,
correctly. Turn 2 asks *"how many weeks is that?"*, and the rewrite becomes
*"how many weeks of parental leave does the company offer?"*. **That sentence
asserts the policy exists.** The user never said it and the system said the
opposite one turn earlier.

Retrieval then runs on the drifted question and returns whatever the corpus has
about weeks, which in a business corpus is usually something. The model answers,
cites a passage it was genuinely supplied, and **the citation gate passes it**,
because every claim does point at something the model was shown. The system
satisfies its entire contract and the answer is about a different question.

That is the method's third sentence exactly: a system can be correct and still
lie to its user. The defect is upstream of everything the gate can see, so the
check has to be upstream too.

Two mechanisms, dividing the work
---------------------------------

**An introduced word.** The rewrite may not contain a content word that appears
in neither the user's own turns nor the passages supplied so far. This catches
invented entities and figures: *"and the second $2,500 tranche?"* introduces
*tranche*, which exists nowhere.

It does **not** catch the parental-leave case, because the user supplied that
phrase in turn 1. The two mechanisms are not alternatives.

**A cited presupposition.** The model is asked to state what its own rewritten
question presupposes, one claim per line, each citing a supplied chunk or marked
`[[none]]`. Those lines go through `citation_gate` unchanged. Same contract, same
controls, same mutations, and **no second model judging semantics**, which was
the trap: anything that understands is another unaudited model policing the
audited one.

The inversion, which is worth reading twice
-------------------------------------------

`[[none]]` means opposite things in the two texts, and that is deliberate rather
than an accident of reuse.

In an **answer**, `[[none]]` is an honest abstention and the gate lets it
through: the model looked and the passages do not cover it.

In a **rewrite's presuppositions**, `[[none]]` means this question rests on
something no supplied passage supports, and the turn is **refused before an
answer is generated**. Same marker, same mechanism, opposite consequence,
because the two texts play different roles: one reports a finding, the other
smuggles a premise.

On the ordering
---------------

The rewrite happens before retrieval, so there are no passages to cite yet. Two
phases, as the threshold already does: retrieve *with* the rewrite, then judge
its presuppositions against what came back. That looks circular, because
retrieval was steered by the drifted question, and it is not: the search returns
passages about weeks, the presupposition that the policy exists is grounded in
none of them, it comes back `[[none]]`, and the turn dies there.

What this does not cover
------------------------

A presupposition assembled from two real passages and a false relation between
them. *"The $2,500 budget renews quarterly"*: the figure is in one chunk,
"quarterly" in another, both words exist, both can be cited, and the relation is
invented. That is **faithfulness**, carried as designed and deliberately unbuilt
since the first day, and it is not being solved here by the back door.

The word list below is English and hand-written. A rewrite that introduces a
concept using only words already present is not caught, so the count this
produces is a **lower bound**.

No dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import citation_gate as gate
import ollama_client as oc

_WORD = re.compile(r"[a-z0-9$%]+")

# Function words carry no claim on their own. Deliberately short: every word left
# out of this list is a word the check will police, and over-policing shows up as
# a refused legitimate follow-up, which a control catches.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from
by with about into over after before under again further is are was were be been
being do does did doing have has had having can could will would shall should
may might must i you he she it we they me him her us them my your his its our
their what which who whom when where why how all any both each few more most
other some such no nor not only own same so too very s t just dont now as up
down out off there here also any
""".split())

PRESUPPOSITION_SYSTEM = (
    "You are checking a question, not answering it.\n\n"
    "List every fact the question below takes for granted. One per line, each a "
    "short statement. End each line with a citation [[id]] naming a passage "
    "below that supports it, or [[none]] if no passage below supports it.\n\n"
    "Do not answer the question. Do not add facts the question does not assume.\n\n"
    "Passages:\n\n{context}"
)


class RewriteUnreadable(Exception):
    """The model did not produce a presupposition list. Never a refusal.

    IA-189. The first real run refused the control on an answerable question,
    because asked what its question took for granted, the model answered the
    question instead and pasted the passage back. That text has no citations in
    it, the citation gate refused it correctly, and the turn was recorded as a
    premise the passages do not support.

    It was nothing of the kind. **The gate had read its own inability to parse as
    evidence of drift**, which is IA-180 one module over: a client that reads its
    own bugs as evidence produces a number that rises whenever the client breaks.
    That one was hardened in `decided_to_retrieve` the same morning, with three
    mutations behind it, and then built again here before the day was out.

    So an unparseable response raises. It is not allowed, it is not refused, and
    it is not counted as anything about the model's grounding. The turn stops
    either way, which is the safe direction, and the count stays honest.
    """


@dataclass(frozen=True)
class RewriteVerdict:
    """Whether the rewritten question may proceed to an answer."""

    allowed: bool
    readable: bool
    introduced: tuple[str, ...]
    presuppositions: str | None
    gate_result: "gate.GateResult | None"
    rewrite: str

    def reason(self) -> str:
        if self.allowed:
            return ""
        if self.introduced:
            return (f"the rewritten question introduced {', '.join(self.introduced)}, "
                    f"which appears neither in anything you said nor in any passage "
                    f"supplied so far. Nothing was answered: the question was "
                    f"rewritten into something you did not ask")
        if self.gate_result is not None and self.gate_result.abstained:
            return ("the rewritten question rests on something no supplied passage "
                    "supports, so answering it would mean accepting a premise "
                    "nobody established. Nothing was answered.\n  rewritten as: "
                    f"{self.rewrite}")
        if self.gate_result is not None:
            return ("the question's own presuppositions did not survive the "
                    f"citation gate.\n  {self.gate_result.reason()}")
        return "the rewritten question was refused"   # pragma: no cover


def _normalise(word: str) -> str:
    """A crude singular. Stated as crude: there is no lemmatiser here."""
    if len(word) > 3 and word.endswith("es") and not word.endswith("ees"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_words(text: str) -> set[str]:
    """Content words, lowercased, crudely singularised, stopwords dropped."""
    out = set()
    for raw in _WORD.findall(text.lower()):
        if raw in STOPWORDS:
            continue
        out.add(raw)
        out.add(_normalise(raw))
    return out


def introduced_words(rewrite: str, *, said: Sequence[str],
                     supplied_texts: Sequence[str]) -> tuple[str, ...]:
    """Content words in the rewrite that the conversation never contained."""
    known: set[str] = set()
    for text in list(said) + list(supplied_texts):
        known |= content_words(text)

    new = []
    for raw in _WORD.findall(rewrite.lower()):
        if raw in STOPWORDS:
            continue
        if raw in known or _normalise(raw) in known:
            continue
        if raw not in new:
            new.append(raw)
    return tuple(new)


def check_rewrite(
    rewrite: str,
    *,
    said: Sequence[str],
    supplied: Sequence[str],
    supplied_texts: Sequence[str],
    context: str,
    model: str,
    transport,
    known_ids: Sequence[str] | None = None,
) -> RewriteVerdict:
    """Refuse a rewritten question that introduced anything, before answering it.

    The cheap check runs first and costs nothing. The model is only asked about
    presuppositions when the words all check out, which keeps the extra call off
    the turns that are already refused.
    """
    introduced = introduced_words(rewrite, said=said, supplied_texts=supplied_texts)
    if introduced:
        return RewriteVerdict(False, True, introduced, None, None, rewrite)

    response = oc.chat(
        model,
        [{"role": "system", "content": PRESUPPOSITION_SYSTEM.format(context=context)},
         {"role": "user", "content": rewrite}],
        transport=transport)
    text = (response.get("message") or {}).get("content") or ""

    # IA-189. Is this a presupposition list at all? A list is claims, each
    # carrying a citation. Anything else, an answer to the question, a header
    # line, a paragraph of prose, is the model not doing the task, and that is
    # not a fact about whether the question rests on a supported premise.
    claims = gate.split_claims(text)
    if not claims:
        raise RewriteUnreadable(
            "the model returned nothing when asked what the question presupposes")
    uncited = [c for c in claims if not c.citations]
    if uncited:
        first = uncited[0].prose[:70]
        raise RewriteUnreadable(
            f"{len(uncited)} of {len(claims)} lines carry no citation, so this is "
            f"not a presupposition list. The first is {first!r}. The model did "
            "not do the task, which says nothing about the question's premises")

    verdict = gate.check(text, supplied, known_ids=known_ids)

    # The inversion. In an answer an abstention is honest; here it means the
    # question assumes something no passage supports, and the turn stops.
    allowed = verdict.passed and not verdict.abstained
    return RewriteVerdict(allowed, True, (), text, verdict, rewrite)
