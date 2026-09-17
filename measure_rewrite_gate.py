"""
Can this model produce a presupposition list at all?

IA-189, step 2. The first real run of the conversation refused an answerable
question, because the rewrite gate could not tell "the model says this premise
is unsupported" from "the model did not do the task". Step 1 separated those.
This is the measurement that decides whether the mechanism is available at all.

It measures one thing and deliberately not two. **Not** whether the
presuppositions were grounded, which is the gate's job and the interesting
question. Only whether the response was a presupposition list: at least one
line, every line carrying a citation. A mechanism that needs a format the model
does not produce is an instruction, not a mechanism, and that is decided before
anything is built on top of it.

The self-check, which exists because of this morning
----------------------------------------------------

A rate of zero has two explanations: the model cannot do the task, or the
classifier cannot recognise a correct answer. Those are indistinguishable from
the output, and this project spent an afternoon on that exact problem with the
skip detector (IA-180).

So before any question is asked, a **hand-written, correctly formatted list is
put through the same code path**, and the run refuses to start unless it comes
back readable. The classifier is shown capable of saying yes before it is
allowed to say no twelve times.

That check costs nothing and needs no model.

Run:
    python measure_rewrite_gate.py --store store/demo --questions questions.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chunker
import ollama_client as oc
import retrieval
import rewrite_gate as rg
import store as store_mod
from ask import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL
from measure_window import DEFAULT_MODEL, RealTokenizer


def self_check(supplied_id: str) -> tuple[bool, str]:
    """Show the classifier can call a correct list readable. No model needed."""
    good = (f"An annual learning budget exists [[{supplied_id}]].\n"
            f"The budget has a stated amount [[{supplied_id}]].")

    def fake(path, payload):
        return {"done": True, "message": {"content": good}}

    try:
        verdict = rg.check_rewrite(
            "what is the annual learning budget",
            said=["what is the annual learning budget"],
            supplied=(supplied_id,), supplied_texts=["annual learning budget"],
            context="", model="self-check", transport=fake,
            known_ids=[supplied_id])
    except rg.RewriteUnreadable as exc:
        return False, f"a correctly formatted list was called unreadable: {exc}"
    if not verdict.readable:
        return False, "a correctly formatted list was not marked readable"
    return True, "a correctly formatted list is recognised"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="store/demo")
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_MODEL)
    ap.add_argument("--host", default=oc.DEFAULT_HOST)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    transport = oc.http_transport(args.host)
    store_obj = store_mod.load(args.store)
    tok = RealTokenizer(args.tokenizer)
    fingerprint = chunker.pack_fingerprint(tok)
    embedder = oc.OllamaEmbedder(args.embed_model, transport=transport)
    known_ids = [r.chunk_id for r in store_obj.records]

    if fingerprint != store_obj.fingerprint:
        print("REFUSED  the store was built under a different packing",
              file=sys.stderr)
        return 1

    held, note = self_check(known_ids[0])
    print(f"self-check  {'held' if held else 'FAILED'}: {note}")
    if not held:
        print("\nThe classifier cannot recognise a correct presupposition list, so "
              "a rate of zero would say nothing about the model. Nothing was run.",
              file=sys.stderr)
        return 1
    print("            so a rate of zero below is about the model, not about this "
          "script\n")

    items = json.loads(Path(args.questions).read_text(encoding="utf-8"))["questions"]
    if args.limit:
        items = items[:args.limit]

    readable = 0
    grounded = 0
    unreadable_reasons: list[str] = []
    started = time.time()

    for item in items:
        question = item["text"]
        found = retrieval.retrieve(question, embedder=embedder,
                                   store_obj=store_obj, fingerprint=fingerprint,
                                   k=args.k, threshold=None)
        if found.declined:
            print(f"{item['id']:>4}  nothing retrieved, skipped")
            continue

        texts = [h.record.text for h in found.result.hits]
        try:
            verdict = rg.check_rewrite(
                question, said=[question], supplied=found.supplied,
                supplied_texts=texts, context=found.context(),
                model=args.chat_model, transport=transport, known_ids=known_ids)
        except rg.RewriteUnreadable as exc:
            unreadable_reasons.append(f"{item['id']}: {exc}")
            print(f"{item['id']:>4}  UNREADABLE")
            if args.verbose:
                print(f"        {exc}")
            continue
        except (oc.OllamaUnavailable, oc.OllamaRefused) as exc:
            print(f"{item['id']:>4}  server did not answer: {exc}")
            continue

        readable += 1
        if verdict.allowed:
            grounded += 1
        print(f"{item['id']:>4}  readable, "
              f"{'all premises grounded' if verdict.allowed else 'a premise unsupported'}")
        if args.verbose and verdict.presuppositions:
            for line in verdict.presuppositions.splitlines()[:4]:
                if line.strip():
                    print(f"        {line.strip()[:100]}")

    total = len(items)
    print("\n" + "-" * 70)
    pct = 100.0 * readable / total if total else 0.0
    print(f"{readable} of {total} responses were a presupposition list at all "
          f"({pct:.0f}%)")
    print(f"{grounded} of those {readable} had every premise grounded")
    print(f"{time.time() - started:.0f} seconds")

    print("\nThe number that decides IA-189 is the first one. The mechanism is "
          "\nonly available if the model reliably produces the format it needs.")
    if unreadable_reasons and args.verbose:
        print("\nwhy each was unreadable:")
        for r in unreadable_reasons:
            print(f"  {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
