"""
Build the store: chunk with the embedder's own tokenizer, embed on the box, save.

This is the first script in the repository that needs a running model, so it is
also the first one whose output I cannot produce. It prints what it measured
rather than what it expected, and the numbers in any artifact come from the
operator's terminal.

The check this script exists for
--------------------------------

Two independent sources report the same number, and they are asked separately.

* **Ollama**, which serves the model, reports a context length for it.
* **The tokenizer**, loaded from the model's own repository, reports the window
  it packs against.

Slice 2 read the window off the tokenizer and that was already better than the
constant it replaced. It still left one thing unchecked: the tokenizer and the
server are two different copies of an idea about the same model, and nothing
compared them. If they disagree, the chunks fit a window the server does not
have, every chunk is silently truncated at generation time, and nothing in the
pipeline raises. That is the shape this portfolio keeps finding, so it is
checked here and the run stops when they differ.

They are expected to agree. A check that is expected to pass is still a check,
as long as it is capable of failing, and a mutation proves this one is.

Run:
    python build_store.py --documents documents --store store/demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chunker
import ollama_client as oc
import store as store_mod
from measure_window import DEFAULT_MODEL, RealTokenizer, readable_text_files

DEFAULT_EMBED_MODEL = "mxbai-embed-large"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents", default="documents")
    ap.add_argument("--store", default="store/demo")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                    help="the model Ollama serves")
    ap.add_argument("--tokenizer", default=DEFAULT_MODEL,
                    help="the same model's tokenizer, from its own repository")
    ap.add_argument("--host", default=oc.DEFAULT_HOST)
    args = ap.parse_args(argv)

    folder = Path(args.documents)
    if not folder.is_dir():
        print(f"no folder at {folder}", file=sys.stderr)
        return 2

    paths, skipped = readable_text_files(folder)
    if not paths:
        print(f"no plain-text documents in {folder}", file=sys.stderr)
        return 2

    transport = oc.http_transport(args.host)

    # -- the two windows, asked separately ---------------------------------
    try:
        served = oc.window(args.embed_model, transport=transport)
    except (oc.OllamaUnavailable, oc.OllamaRefused) as exc:
        print(f"the server could not tell us the window: {exc}", file=sys.stderr)
        return 3

    tok = RealTokenizer(args.tokenizer)

    print(f"window from Ollama      {served} tokens  ({args.embed_model})")
    print(f"window from tokenizer   {tok.budget} tokens  ({args.tokenizer})")

    if served != tok.budget:
        print(f"\nREFUSED  the server serves a {served}-token window and the "
              f"tokenizer packs to {tok.budget}.\n"
              f"         Chunks would be packed to a window the model does not "
              f"have, and nothing downstream would say so.\n"
              f"         Nothing was embedded and nothing was written.",
              file=sys.stderr)
        return 1
    print("                        they agree, so the packing is against a real window\n")

    # -- chunk --------------------------------------------------------------
    chunks = []
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8", errors="strict")
        made = chunker.chunk_text(text, document_id=path.name, tokenizer=tok)
        chunks.extend(made)
        print(f"  {path.name:<40} {len(text):>7} chars -> {len(made):>3} chunks")
    if skipped:
        print(f"\n  skipped {len(skipped)} file(s) this script cannot read: "
              f"{', '.join(skipped[:5])}")

    if not chunks:
        print("nothing to embed", file=sys.stderr)
        return 2

    # -- embed --------------------------------------------------------------
    embedder = oc.OllamaEmbedder(args.embed_model, transport=transport)
    vectors = []
    print(f"\nembedding {len(chunks)} chunks on this machine, no key, no network")
    for i, chunk in enumerate(chunks, 1):
        try:
            raw = embedder.embed(chunk.text)
        except (oc.OllamaUnavailable, oc.OllamaRefused) as exc:
            print(f"\nstopped at chunk {i} of {len(chunks)}: {exc}", file=sys.stderr)
            return 3
        vectors.append(store_mod.normalise(raw))
        if i % 25 == 0 or i == len(chunks):
            print(f"  {i}/{len(chunks)}")

    widths = {len(v) for v in vectors}
    if len(widths) != 1:
        print(f"\nREFUSED  the embedder returned vectors of {sorted(widths)} "
              f"different widths in one run", file=sys.stderr)
        return 1
    dim = widths.pop()

    records = store_mod.records_from_chunks(chunks)
    fingerprint = chunker.pack_fingerprint(tok)
    manifest = store_mod.save(args.store, records, vectors,
                              fingerprint=fingerprint, embedder=embedder.name)

    print(f"\nwrote      {args.store}.vec and {args.store}.json")
    print(f"vectors    {manifest['count']}")
    # IA-178. This said "tokens wide" on the first real run. It is a count of
    # vector dimensions, and a 512-token window is printed a few lines above it,
    # so the report put two different kinds of number under one unit. The line
    # was right about what it measured and wrong about what it said it measured,
    # which is the defect this whole repository is about.
    print(f"width      {dim} dimensions per vector, measured from the model's "
          f"own output")
    print(f"           not tokens: the window above is {served} tokens, and "
          f"these are different counts")
    print(f"           the upstream schema hardcodes 1536 dimensions")
    print(f"packing    {fingerprint}   ({tok.name}, budget {tok.budget})")
    print(f"endpoint   {embedder.endpoint}")
    print(f"\nthe store is built. A query produced under any other packing will "
          f"be refused rather than answered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
