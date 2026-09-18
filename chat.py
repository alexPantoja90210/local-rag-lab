"""
The conversation, end to end, and the thing a person can actually sit in front of.

Slice 5. Four refusals now sit above an answer, and three of them happen before
the model is asked anything:

    the cap        -> nothing was sent
    the referent   -> the turn you are referring to produced no answer
    the drift      -> the rewrite dropped the thing you were referring to
    the rewrite    -> your question was rewritten into one you did not ask
    the gate       -> the answer cited something it was not given

The first three cost nothing. At 90.9 seconds a turn that matters more than it
sounds: a refusal that happens after the call has already spent the minute and a
half it was trying to save.

`converse_turn` is separated from the loop so the glue has invariants. The glue
is where four working parts become a count, and a count assembled wrongly is
worse than no count, because it publishes.

Run:
    python chat.py --store store/demo
"""

from __future__ import annotations

import argparse
import sys

import chunker
import citation_gate as gate
import conversation as conv
import ollama_client as oc
import retrieval
import rewrite_gate as rg
import store as store_mod
from ask import DECIDE_SYSTEM, GROUND_SYSTEM, DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL
from measure_window import DEFAULT_MODEL, RealTokenizer


def _q(query_obj):
    """One shape for the query detail, so the two paths cannot drift apart."""
    return {"query": query_obj.text, "query_source": query_obj.source,
            "carried": query_obj.carried}


def converse_turn(question, *, talk, chat_model, embedder, store_obj,
                  fingerprint, transport, k, threshold, known_ids,
                  presupposition_check=False):
    """One turn of a conversation. Returns (Turn, detail).

    Order matters and is the order of Figure 2: everything that can refuse for
    free refuses first.
    """
    history = talk.transcript()
    preview = "".join(m["content"] for m in history) + question + DECIDE_SYSTEM

    cap = talk.check_cap(preview)
    if not cap.fits:
        return retrieval.Turn(question=question, retrieved=False, supplied=(),
                              stage=conv.REFUSED_CAP, refusal=cap.reason()), {}

    referent = talk.check_referent(question)
    if not referent.allowed:
        return retrieval.Turn(question=question, retrieved=False, supplied=(),
                              stage=conv.REFUSED_REFERENT,
                              refusal=referent.reason()), {}

    first = oc.chat(chat_model,
                    [{"role": "system", "content": DECIDE_SYSTEM}] + history
                    + [{"role": "user", "content": question}],
                    transport=transport, tools=[oc.RETRIEVE_TOOL])
    if "prompt_eval_count" in first:
        talk.record_prompt(len(preview), int(first["prompt_eval_count"]),
                           estimated=cap.estimated_tokens)

    if not oc.decided_to_retrieve(first):
        text = (first.get("message") or {}).get("content") or ""
        verdict = gate.check(text, (), known_ids=known_ids)
        return retrieval.Turn(question=question, retrieved=False, supplied=(),
                              stage=retrieval.SKIPPED,
                              refusal=verdict.reason()), {"text": text}

    # IA-194. The query is composed here, not requested. When the question
    # leans on a reference, the model is not asked to resolve it: the turn it
    # refers to is appended in code. A question with no reference keeps the
    # model's rewrite, and the Query records which of the two happened.
    antecedent = talk.antecedent()
    query_obj = rg.compose_query(question, oc.tool_query(first),
                                 antecedent=antecedent)
    query = query_obj.text

    # IA-192, and the reason this sits here rather than below: it needs the
    # query and nothing else, so it refuses before the embedding call. Every
    # other check reads the query on its own; this one is the only one that
    # compares the question asked against the question sent.
    #
    # With a composed query it is satisfied by construction. It stays as the
    # guard: it is what goes red if the model's rewrite is ever routed back
    # into retrieval, and the harness mutates exactly that.
    survives = rg.check_referent_survives(question, query,
                                          antecedent=antecedent)
    if not survives.allowed:
        return retrieval.Turn(question=question, retrieved=False, supplied=(),
                              stage=conv.REFUSED_DRIFT,
                              refusal=survives.reason()), _q(query_obj)

    found = retrieval.retrieve(query, embedder=embedder, store_obj=store_obj,
                               fingerprint=fingerprint, k=k, threshold=threshold)
    if found.declined:
        return retrieval.Turn(question=question, retrieved=True, supplied=(),
                              stage=retrieval.REFUSED_THRESHOLD,
                              refusal=found.reason()), _q(query_obj)

    # Both sides of the conversation, with the citation markup dropped. The
    # rules are IA-194's and IA-197's and they live in `Conversation.said`,
    # not here: this file had its own copy of the first one and got the
    # second one wrong.
    said = talk.said(question)
    texts = [h.record.text for h in found.result.hits]

    # The deterministic half always runs: it asks the model for nothing.
    introduced = rg.introduced_words(query, said=said, supplied_texts=texts)
    if introduced:
        return retrieval.Turn(
            question=question, retrieved=True, supplied=found.supplied,
            stage=conv.REFUSED_REWRITE,
            refusal=rg.RewriteVerdict(False, True, introduced, None, None,
                                      query).reason()), _q(query_obj)

    # IA-189. The presupposition check is off by default. It is the only
    # mechanism here that needs the model to perform a task, and on the first
    # real run llama3.1:8b did not perform it: asked what its question took for
    # granted, it answered the question. Until a measurement says a model can
    # produce a presupposition list reliably, this stays opt-in, and what the
    # gate guarantees without it is stated rather than implied.
    if presupposition_check:
        try:
            rewrite = rg.check_rewrite(
                query, said=said, supplied=found.supplied, supplied_texts=texts,
                context=found.context(), model=chat_model, transport=transport,
                known_ids=known_ids)
        except rg.RewriteUnreadable as exc:
            return retrieval.Turn(
                question=question, retrieved=True, supplied=found.supplied,
                stage=conv.UNREADABLE_REWRITE,
                refusal=str(exc)), _q(query_obj)
        if not rewrite.allowed:
            return retrieval.Turn(
                question=question, retrieved=True, supplied=found.supplied,
                stage=conv.REFUSED_REWRITE,
                refusal=rewrite.reason()), _q(query_obj)

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
    return retrieval.Turn(
        question=question, retrieved=True, supplied=found.supplied, stage=stage,
        refusal=None if verdict.passed else verdict.reason(),
        answer=verdict.answer if verdict.passed else None,
    ), dict(_q(query_obj), gate=verdict)


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
    ap.add_argument("--presupposition-check", action="store_true",
                    help="IA-189: ask the model what its rewritten question "
                         "presupposes and check those against the passages. OFF "
                         "by default because the first real run showed the model "
                         "does not produce that list. Measure it with "
                         "measure_rewrite_gate.py before turning it on")
    ap.add_argument("--seed-ratio", type=float, required=True,
                    help="characters per token for the FIRST turn only. No "
                         "default: a constant nobody chose is what slice 2 is "
                         "about. It is replaced by measurement from turn 2")
    args = ap.parse_args(argv)

    transport = oc.http_transport(args.host)
    store_obj = store_mod.load(args.store)
    tok = RealTokenizer(args.tokenizer)
    fingerprint = chunker.pack_fingerprint(tok)
    embedder = oc.OllamaEmbedder(args.embed_model, transport=transport)
    known_ids = [r.chunk_id for r in store_obj.records]

    if fingerprint != store_obj.fingerprint:
        print(f"REFUSED  this store was built under packing "
              f"{store_obj.fingerprint} and these queries would be produced "
              f"under {fingerprint}.", file=sys.stderr)
        return 1

    talk = conv.Conversation(token_budget=args.token_budget,
                             seed_ratio=args.seed_ratio)

    print(f"store      {args.store}, {store_obj.count} vectors")
    print(f"models     {args.chat_model} generating, {args.embed_model} embedding")
    print(f"budget     {args.token_budget} tokens per turn, seed ratio "
          f"{args.seed_ratio} chars/token")
    print(f"threshold  {args.threshold if args.threshold is not None else 'none, stated explicitly'}")
    print(f"query      composed in code when the question refers back (IA-194); "
          f"the model's rewrite otherwise")
    print(f"rewrite    referent-survival and introduced-word checks always on; "
          f"presupposition check "
          f"{'ON' if args.presupposition_check else 'OFF (IA-189)'}")
    print("\nAsk a question. Empty line or Ctrl-C to stop.\n")

    while True:
        try:
            question = input("you  ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        try:
            turn, detail = converse_turn(
                question, talk=talk, chat_model=args.chat_model,
                embedder=embedder, store_obj=store_obj, fingerprint=fingerprint,
                transport=transport, k=args.k, threshold=args.threshold,
                known_ids=known_ids,
                presupposition_check=args.presupposition_check)
        except (oc.OllamaUnavailable, oc.OllamaRefused) as exc:
            # Not recorded as anything. A turn that could not be read is not
            # evidence about the model.
            print(f"\n     UNREADABLE  {exc}\n")
            continue

        talk.record(turn)
        if detail.get("query"):
            label = ("composed" if detail.get("query_source") == rg.QUERY_COMPOSED
                     else "model   ")
            print(f"     searched as [{label}]: {detail['query']}")
        if turn.answer is not None:
            print(f"\n{turn.answer}\n")
        else:
            print(f"\n     [{turn.stage}] {turn.refusal}\n")

    print("-" * 70)
    print(talk.line())
    if talk.samples:
        print(f"the estimator's worst underestimate: "
              f"{talk.worst_underestimate} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
