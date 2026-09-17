"""
One turn end to end, and the number this slice exists to produce.

The agent is given a knowledge base as a tool it MAY call, which is how both
upstream projects are built, and a system prompt telling it to always search
first, which is how both of them hold it in place. That line is kept here
deliberately and is not the mechanism. It is the thing being measured.

    Always search the knowledge base before answering.

Nothing enforces it upstream. Nothing enforces it here either. What is different
is that every turn records whether the tool was called, so at the end there is a
number instead of an assumption: **the share of answers produced without ever
looking.** That is Q2 of the target architecture.

What happens on a turn that never looked
----------------------------------------

It is not blocked. It is allowed through and counted, and then it fails anyway,
because a turn that did not retrieve was supplied no passages, so every claim it
makes cites something it was not given and the citation gate discards it. The
model gets nothing out of skipping. That is the difference between a rule and a
consequence: there is no branch here to forget.

The question set
----------------

Families A and B ask about the same facts. A names the company, B does not.
H2 predicts the agent skips retrieval where its own priors feel sufficient, so
the prediction is that B is skipped more than A. Family C cannot be answered
from what was stored at all, and exists so that a threshold and a gate have
something they are supposed to refuse.

Run:
    python build_store.py --documents documents --store store/demo
    python ask.py --store store/demo --questions questions.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chunker
import citation_gate as gate
import ollama_client as oc
import retrieval
import store as store_mod
from measure_window import DEFAULT_MODEL, RealTokenizer

DEFAULT_CHAT_MODEL = "llama3.1:8b"
DEFAULT_EMBED_MODEL = "mxbai-embed-large"

# The instruction with no mechanism, kept verbatim from the family of tutorials
# this project audited. Changing it would measure a different thing.
DECIDE_SYSTEM = (
    "You answer questions about NeuralFlow AI using its internal documents. "
    "Always search the knowledge base before answering."
)

# IA-180. The control for the measurement itself.
#
# The first real run reported 0 of 12 turns skipped. Zero is the one value this
# instrument cannot tell apart from a blind instrument: every invariant on
# decided_to_retrieve() runs against a response this repository wrote itself,
# and it has never once seen a real skip from a real server. A scanner that
# always returns nothing looks exactly like a working scanner.
#
# So this prompt exists to make the model skip on purpose. If the run still
# reports zero, the detector is blind and the 0% means nothing. It is the same
# thing the four controls in the suite do, one level up, and against the real
# server rather than a fake.
NO_SEARCH_SYSTEM = (
    "You answer questions about NeuralFlow AI. Do NOT search the knowledge "
    "base. Answer directly from what you already know."
)

GROUND_SYSTEM = (
    "Answer using ONLY the passages below. Every sentence you write must "
    "contain a citation in the form [[id]], naming one of the passages below. "
    "Never cite an id that is not listed.\n\n"
    # IA-179. This used to say "cite the closest passage" for an abstention,
    # which asked the model to attach a citation that does not support the
    # sentence. That is the exact behaviour the gate exists to catch, so the
    # instruction was asking for the defect and the gate was passing it.
    # IA-181. One part, not two. The wording is not asked for, because the
    # system writes it.
    "If the passages do not contain the answer, reply with exactly [[none]] "
    "and nothing else.\n\n"
    "Passages:\n\n{context}"
)


def one_turn(question, *, chat_model, embedder, store_obj, fingerprint,
             transport, k, threshold, known_ids, no_search=False):
    """Returns (Turn, detail dict). Raises nothing the caller should swallow."""
    started = time.time()

    first = oc.chat(chat_model,
                    [{"role": "system",
                      "content": NO_SEARCH_SYSTEM if no_search else DECIDE_SYSTEM},
                     {"role": "user", "content": question}],
                    transport=transport, tools=[oc.RETRIEVE_TOOL])

    # A malformed response raises here rather than being counted as a skip.
    # See ollama_client.decided_to_retrieve: a measurement that cannot tell a
    # discovery from a bug is not a measurement.
    retrieved = oc.decided_to_retrieve(first)

    if not retrieved:
        text = (first.get("message") or {}).get("content") or ""
        verdict = gate.check(text, (), known_ids=known_ids)
        turn = retrieval.Turn(question=question, retrieved=False, supplied=(),
                              stage=retrieval.SKIPPED, refusal=verdict.reason())
        return turn, {"seconds": time.time() - started, "text": text,
                      "gate": verdict, "query": None, "scores": ()}

    query = oc.tool_query(first)
    found = retrieval.retrieve(query, embedder=embedder, store_obj=store_obj,
                               fingerprint=fingerprint, k=k, threshold=threshold)

    if found.declined:
        turn = retrieval.Turn(question=question, retrieved=True, supplied=(),
                              stage=retrieval.REFUSED_THRESHOLD,
                              refusal=found.reason())
        return turn, {"seconds": time.time() - started, "text": None,
                      "gate": None, "query": query, "scores": ()}

    second = oc.chat(chat_model,
                     [{"role": "system",
                       "content": GROUND_SYSTEM.format(context=found.context())},
                      {"role": "user", "content": question}],
                     transport=transport)
    text = (second.get("message") or {}).get("content") or ""
    verdict = gate.check(text, found.supplied, known_ids=known_ids)

    if verdict.passed:
        stage = retrieval.ABSTAINED if verdict.abstained else retrieval.ANSWERED
    else:
        stage = retrieval.REFUSED_GATE
    turn = retrieval.Turn(
        question=question, retrieved=True, supplied=found.supplied,
        stage=stage, refusal=None if verdict.passed else verdict.reason())
    return turn, {"seconds": time.time() - started, "text": text,
                  "gate": verdict, "query": query, "scores": found.scores}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="store/demo")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--question", default=None)
    ap.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=oc.DEFAULT_HOST)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=None,
                    help="omit for no floor, and the run record will say so")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--control-no-search", action="store_true",
                    help="measures whether the model obeys an instruction not "
                         "to search. NOT the detector control: see "
                         "prove_the_detector_sees.py (IA-180)")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many questions")
    ap.add_argument("--family", default=None,
                    help="run only one family from the question set, e.g. C")
    args = ap.parse_args(argv)

    if not args.questions and not args.question:
        print("give --question or --questions", file=sys.stderr)
        return 2

    transport = oc.http_transport(args.host)
    store_obj = store_mod.load(args.store)
    tok = RealTokenizer(args.tokenizer)
    fingerprint = chunker.pack_fingerprint(tok)
    embedder = oc.OllamaEmbedder(args.embed_model, transport=transport)
    known_ids = [r.chunk_id for r in store_obj.records]

    if fingerprint != store_obj.fingerprint:
        print(f"REFUSED  this store was built under packing "
              f"{store_obj.fingerprint} and these queries would be produced "
              f"under {fingerprint}. Rebuild the store or use the tokenizer "
              f"that built it.", file=sys.stderr)
        return 1

    print(f"store      {args.store}, {store_obj.count} vectors, packing {fingerprint}")
    print(f"models     {args.chat_model} generating, {args.embed_model} embedding")
    print(f"threshold  {args.threshold if args.threshold is not None else 'none, stated explicitly'}")
    print(f"k          {args.k}\n")

    if args.question:
        items = [{"id": "q", "family": "-", "text": args.question, "answerable": None}]
    else:
        items = json.loads(Path(args.questions).read_text(encoding="utf-8"))["questions"]
    if args.family:
        items = [i for i in items if i["family"] == args.family]
    if args.limit:
        items = items[:args.limit]

    if args.control_no_search:
        print("CONTROL RUN (IA-180). The model is told NOT to search.")
        print("  If every turn still says 'looked', the skip detector is blind")
        print("  and the 0.0%% from the measured run means nothing.\n")

    log = retrieval.TurnLog()
    by_family: dict[str, list[bool]] = {}

    for item in items:
        try:
            turn, detail = one_turn(
                item["text"], chat_model=args.chat_model, embedder=embedder,
                store_obj=store_obj, fingerprint=fingerprint, transport=transport,
                k=args.k, threshold=args.threshold, known_ids=known_ids,
                no_search=args.control_no_search)
        except (oc.OllamaUnavailable, oc.OllamaRefused) as exc:
            # Not counted as anything. A turn that could not be read is not
            # evidence about the model, and folding it into the skip rate would
            # make a broken client look like a finding.
            print(f"{item['id']:>4}  {item['family']}  UNREADABLE  {exc}")
            continue

        log.record(turn)
        by_family.setdefault(item["family"], []).append(turn.retrieved)

        mark = "looked" if turn.retrieved else "NEVER LOOKED"
        print(f"{item['id']:>4}  {item['family']}  {mark:<13} {turn.stage:<18} "
              f"{detail['seconds']:.1f}s")
        if args.verbose:
            if detail["query"]:
                print(f"        rewritten as: {detail['query']}")
            if detail["scores"]:
                print(f"        scores: {[round(s, 3) for s in detail['scores']]}")
            if detail["text"]:
                print(f"        {detail['text'][:300]}")
            gate_result = detail["gate"]
            if gate_result is not None and gate_result.abstained:
                print(f"        answer: {gate_result.answer}")
                if gate_result.model_note:
                    print(f"        the model also wrote, unchecked: "
                          f"{gate_result.model_note[:120]}")
            if turn.refusal:
                print(f"        {turn.refusal.splitlines()[0]}")
            print()

    print("\n" + "-" * 70)
    print(log.line())
    if args.control_no_search:
        if log.skipped:
            print(f"\nCONTROL HELD. The detector saw {log.skipped} real skip(s), "
                  f"so it is capable of seeing one.\n  The measured run's 0.0% "
                  f"is a fact about the model rather than about this client.")
        else:
            print("\nTold not to search, the model retrieved every time.\n"
                  "  That is a fact about the model's obedience and it does NOT\n"
                  "  settle whether the detector can see a skip: a blind detector\n"
                  "  produces this same output. Varying the model cannot isolate\n"
                  "  the detector. Run prove_the_detector_sees.py for that.")
    if by_family:
        print("\nby family, which is where H2 lives:")
        for family in sorted(by_family):
            flags = by_family[family]
            skipped = sum(1 for f in flags if not f)
            print(f"  {family}   {skipped} of {len(flags)} never retrieved")
        print("\n  A names the company, B does not, and both ask about the same "
              "facts.\n  C cannot be answered from what was stored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
