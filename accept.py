"""
The stop condition, as something that can come back red.

A stop condition written in a ticket is prose. This repository's own first
sentence says what prose is worth: an instruction with no mechanism is not a
mechanism. IA-187's condition was prose, and prose is how it ended up written in
one direction.

What it used to say
-------------------

    if a legitimate follow-up on an answered turn is REFUSED, the mechanism is
    wrong and the slice re-scopes.

On 17 September the control turn was not refused. It was **answered, with a
valid citation, about a different question**: "can it be used for conferences?"
became "NeuralFlow AI for conferences", retrieval found a passage about
conference speaking, and every check passed. Read literally, that condition
approved it.

The condition watched for the system refusing too much. False acceptance --
the system answering when it should not, or answering something else -- is this
project's entire thesis, and the condition could not express it.

The rule that follows, and it is general
----------------------------------------

**A check that can only fail in one direction is not a check. It is an
expectation.** Every clause below therefore comes in pairs, and a conversation
passes only when both halves hold.

How "answered something else" is made checkable
-----------------------------------------------

Not by reading the sentence. By its citations: a follow-up on an answered turn
must cite **at least one chunk the earlier turn cited**. That is deterministic,
store-independent, and it is exactly what the 17 September failure violated --
turn 1 cited the budget passage, turn 2 cited the brand-positioning one.

It does not prove the answer is right. It proves the answer is about the same
thing that was asked about, which is the half that was missing.

The self-check, which is the same rule turned on this file
-----------------------------------------------------------

A harness that reports "held" for everything looks identical to a harness that
works. So before any model is asked, `judge` is shown three hand-made
transcripts and must sort them correctly:

* one that should pass
* one that fails by refusing what it should have answered
* one that fails by answering something else, which is the real 17 September
  transcript, written out

The run refuses to start unless all three land where they belong. It costs
nothing and needs no model.

Run:
    python accept.py --store store/demo --seed-ratio 4.0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import chat as chatmod
import chunker
import citation_gate as gate
import conversation as conv
import ollama_client as oc
import retrieval
import store as store_mod
from ask import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL
from measure_window import DEFAULT_MODEL, RealTokenizer


@dataclass(frozen=True)
class Step:
    """One turn, and what must be true of it in BOTH directions.

    `must` is the stage this turn has to end in. `cites_same_as` is the index of
    an earlier step whose citations this turn has to share at least one of.

    `must=None` means observed and not asserted. It exists for conversation C,
    whose mechanism has never fired in a real run (IA-193): a result from a
    mechanism that has never been seen to work is not evidence, and printing it
    as a pass would be the defect this file exists to prevent.
    """

    question: str
    must: str | None
    cites_same_as: int | None = None
    because: str = ""


@dataclass(frozen=True)
class Script:
    name: str
    watches: str
    steps: tuple[Step, ...]


CONVERSATIONS = (
    Script(
        "A — the control",
        "that the system answers what it is asked",
        (
            Step("What is the annual learning budget per employee?",
                 retrieval.ANSWERED,
                 because="an answerable question is answered"),
            Step("Can it be used for conferences?",
                 retrieval.ANSWERED, cites_same_as=0,
                 because="a legitimate follow-up is answered (not refused), "
                         "and it is about what it referred to (not something "
                         "else that happens to be in the corpus)"),
        ),
    ),
    Script(
        "B — the poisoned referent",
        "that an abstention poisons the referent and not the conversation",
        (
            Step("What is the parental leave policy at NeuralFlow AI?",
                 retrieval.ABSTAINED,
                 because="a question the corpus cannot answer is abstained on"),
            Step("How many weeks is that?",
                 conv.REFUSED_REFERENT,
                 because="a reference back to an abstention is refused (not "
                         "answered by inventing what it points at)"),
            Step("What is the annual learning budget per employee?",
                 retrieval.ANSWERED,
                 because="and the conversation still works afterwards (not "
                         "refused, which would mean the rule poisoned "
                         "everything rather than the referent)"),
        ),
    ),
    Script(
        "C — the invented premise",
        "nothing yet: see IA-193",
        (
            Step("What is the annual learning budget per employee?",
                 retrieval.ANSWERED),
            Step("And the second $2,500 tranche?", None,
                 because="the introduced-word check has never fired in a real "
                         "conversation, so this turn is observed and not "
                         "asserted. Whatever it does is not evidence"),
        ),
    ),
)


def cited(turn: retrieval.Turn) -> tuple[str, ...]:
    """The chunks the answer actually pointed at, not the ones it was given."""
    if not turn.answer:
        return ()
    return tuple(m.group(1).strip() for m in gate.CITATION.finditer(turn.answer))


@dataclass(frozen=True)
class Verdict:
    step: Step
    turn: retrieval.Turn
    held: bool | None      # None = observed, not asserted
    why: str


def judge(script: Script, turns: list[retrieval.Turn]) -> list[Verdict]:
    """Decide each step against both halves of its condition.

    Pure: no model, no store, no network. That is what lets the self-check
    below hand it hand-made transcripts.
    """
    out: list[Verdict] = []
    for i, (step, turn) in enumerate(zip(script.steps, turns)):
        if step.must is None:
            out.append(Verdict(step, turn, None,
                               f"observed: {turn.stage}"))
            continue
        if turn.stage != step.must:
            out.append(Verdict(step, turn, False,
                               f"ended in {turn.stage}, and the condition "
                               f"requires {step.must}"))
            continue
        if step.cites_same_as is not None:
            here = set(cited(turn))
            there = set(cited(turns[step.cites_same_as]))
            shared = here & there
            if not shared:
                out.append(Verdict(
                    step, turn, False,
                    f"answered, but cited {sorted(here) or 'nothing'} while the "
                    f"turn it refers to cited {sorted(there) or 'nothing'}. "
                    f"Nothing in common: this is an answer to a different "
                    f"question, which is the 17 September failure"))
                continue
            out.append(Verdict(step, turn, True,
                               f"{turn.stage}, sharing {sorted(shared)[0]}"))
            continue
        out.append(Verdict(step, turn, True, turn.stage))
    return out


# --------------------------------------------------------------------------
# The self-check. Three transcripts, and judge has to sort them.
# --------------------------------------------------------------------------

_A_ID = "1111111111111111111111111111111a"
_B_ID = "2222222222222222222222222222222b"


def _turn(stage, answer=None):
    retrieved = stage not in (conv.REFUSED_CAP, conv.REFUSED_REFERENT,
                              conv.REFUSED_DRIFT)
    return retrieval.Turn(
        question="q", retrieved=retrieved,
        supplied=(_A_ID,) if retrieved else (),
        stage=stage, answer=answer,
        refusal=None if answer is not None else "refused")


def self_check() -> tuple[bool, list[str]]:
    """Show that judge can say yes, and no in each direction, before it says
    anything about a real run."""
    control = CONVERSATIONS[0]
    lines = []
    ok = True

    good = [_turn(retrieval.ANSWERED, f"$2,500 [[{_A_ID}]]."),
            _turn(retrieval.ANSWERED, f"Yes, for conferences [[{_A_ID}]].")]
    held = all(v.held for v in judge(control, good))
    lines.append(f"  a correct transcript is called correct        "
                 f"{'yes' if held else 'NO'}")
    ok = ok and held

    too_strict = [_turn(retrieval.ANSWERED, f"$2,500 [[{_A_ID}]]."),
                  _turn(conv.REFUSED_DRIFT)]
    v = judge(control, too_strict)[1]
    caught = v.held is False and "requires" in v.why
    lines.append(f"  refusing what it should answer is caught      "
                 f"{'yes' if caught else 'NO'}")
    ok = ok and caught

    # The 17 September transcript, written out: answered, cited, wrong question.
    wrong_question = [_turn(retrieval.ANSWERED, f"$2,500 [[{_A_ID}]]."),
                      _turn(retrieval.ANSWERED,
                            f"NeuralFlow AI speaks at conferences [[{_B_ID}]].")]
    v = judge(control, wrong_question)[1]
    caught = v.held is False and "different question" in v.why
    lines.append(f"  answering something else is caught            "
                 f"{'yes' if caught else 'NO'}")
    ok = ok and caught

    return ok, lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="store/demo")
    ap.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=oc.DEFAULT_HOST)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--token-budget", type=int, default=4096)
    ap.add_argument("--seed-ratio", type=float, required=True)
    args = ap.parse_args(argv)

    ok, lines = self_check()
    print("self-check  judge is shown it can fail in both directions")
    for line in lines:
        print(line)
    if not ok:
        print("\nREFUSED  the judge cannot sort three hand-made transcripts, so "
              "nothing it says about a real run means anything.", file=sys.stderr)
        return 2
    print()

    transport = oc.http_transport(args.host)
    store_obj = store_mod.load(args.store)
    tok = RealTokenizer(args.tokenizer)
    fingerprint = chunker.pack_fingerprint(tok)
    embedder = oc.OllamaEmbedder(args.embed_model, transport=transport)
    known_ids = [r.chunk_id for r in store_obj.records]

    if fingerprint != store_obj.fingerprint:
        print(f"REFUSED  this store was built under packing "
              f"{store_obj.fingerprint}, these queries under {fingerprint}.",
              file=sys.stderr)
        return 1

    failed = []
    for script in CONVERSATIONS:
        print(f"{script.name}")
        print(f"  watches: {script.watches}")
        talk = conv.Conversation(token_budget=args.token_budget,
                                 seed_ratio=args.seed_ratio)
        turns = []
        for step in script.steps:
            turn, _ = chatmod.converse_turn(
                step.question, talk=talk, chat_model=args.chat_model,
                embedder=embedder, store_obj=store_obj, fingerprint=fingerprint,
                transport=transport, k=args.k, threshold=args.threshold,
                known_ids=known_ids)
            talk.record(turn)
            turns.append(turn)

        for i, v in enumerate(judge(script, turns)):
            mark = "observed" if v.held is None else ("held" if v.held else "FAILED")
            print(f"  [{mark:8}] {v.step.question}")
            print(f"             {v.why}")
            if v.step.because:
                print(f"             condition: {v.step.because}")
            if v.held is False:
                failed.append(f"{script.name}, step {i + 1}")
        print()

    print("-" * 70)
    if failed:
        print(f"{len(failed)} step(s) failed the stop condition:")
        for f in failed:
            print(f"  - {f}")
        print("\nThe slice re-scopes. It does not ship.")
        return 1
    print("every asserted step held, in both directions.")
    print("C is observed and not asserted: see IA-193.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
