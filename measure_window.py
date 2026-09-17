"""
How often does the rule of thumb overflow the model's window?

This is the one number slice 2 exists to produce, and it is the only script in
the repository that needs the real tokenizer downloaded. The invariant suite
proves the packing logic against three stand-in tokenizers and never touches
the network; this asks a real embedding model how much it can actually read,
and measures what the old approach would have fed it.

Two arms over the same text:

    A. character budget    chunk at budget x chars_per_token characters,
                           the rule of thumb the previous project used
                           (896 characters for a 256-token window)

    B. token aware         chunk with the model's own tokenizer, this repo

Both are then measured with the real tokenizer. Arm B must overflow zero
times, by construction. Arm A is the question.

It refuses rather than estimating. If the tokenizer cannot be loaded there is
no fallback, because a number produced by guessing the thing being measured is
worse than no number.

    pip install transformers
    python measure_window.py --documents documents/
    python measure_window.py --documents documents/ --model nomic-ai/nomic-embed-text-v1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chunker
from ingest_contract import EXTENSION_FAMILY, check_file, read_or_refuse

DEFAULT_MODEL = "mixedbread-ai/mxbai-embed-large-v1"

# The rule of thumb this measurement is about. Not a setting to tune: it is the
# number the previous project assumed, and changing it would change what is
# being measured.
CHARS_PER_TOKEN = 3.5


class RealTokenizer:
    """The embedding model's own tokenizer, with the budget read off the model."""

    def __init__(self, model_id: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "transformers is not installed, and this script will not "
                "estimate a token count instead.\n    pip install transformers"
            ) from exc

        try:
            self._tk = AutoTokenizer.from_pretrained(model_id)
        except Exception as exc:
            raise SystemExit(
                f"could not load the tokenizer for {model_id!r}: {exc}\n"
                "Nothing was measured. There is no fallback on purpose: a "
                "number produced by guessing the thing being measured would be "
                "worse than no number."
            ) from exc

        self._name = model_id
        budget = getattr(self._tk, "model_max_length", None)
        if not isinstance(budget, int) or budget < 1 or budget > 1_000_000:
            raise SystemExit(
                f"{model_id!r} reports model_max_length={budget!r}, which is "
                "not a usable window. Nothing was measured."
            )
        self._budget = budget

    @property
    def name(self) -> str:
        return self._name

    @property
    def budget(self) -> int:
        return self._budget

    def count(self, text: str) -> int:
        return len(self._tk.encode(text, add_special_tokens=True))


def character_budget_chunks(text: str, budget_tokens: int) -> list[str]:
    """Arm A. Fixed character windows, the way the rule of thumb does it."""
    size = int(budget_tokens * CHARS_PER_TOKEN)
    return [text[i:i + size] for i in range(0, len(text), size)] if text else []


def readable_text_files(folder: Path) -> tuple[list[Path], list[str]]:
    """Files this script can read without a document converter, plus the skips."""
    keep, skipped = [], []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSION_FAMILY:
            continue
        verdict = check_file(path)
        if not verdict.ok:
            skipped.append(f"{path.name}: rejected by the contract, {verdict.reason}")
        elif verdict.detected != "text":
            skipped.append(f"{path.name}: {verdict.detected}, needs a converter")
        else:
            keep.append(path)
    return keep, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--documents", default="documents")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--overlap-tokens", type=int, default=0)
    args = ap.parse_args(argv)

    folder = Path(args.documents)
    if not folder.is_dir():
        print(f"no such folder: {folder}", file=sys.stderr)
        return 2

    files, skipped = readable_text_files(folder)
    if not files:
        print(f"no plain-text documents in {folder}. Nothing was measured.")
        for s in skipped:
            print(f"  skipped  {s}")
        return 2

    tok = RealTokenizer(args.model)
    size_a = int(tok.budget * CHARS_PER_TOKEN)

    print(f"model      {tok.name}")
    print(f"window     {tok.budget} tokens, read from the model")
    print(f"arm A      {size_a} characters per chunk "
          f"({tok.budget} x {CHARS_PER_TOKEN} chars/token)")
    print("arm B      packed with the model's own tokenizer")
    print(f"documents  {len(files)} plain-text, {len(skipped)} skipped")
    print(f"packing    {chunker.pack_fingerprint(tok)}")
    print()

    a_total = a_over = a_worst = 0
    b_total = b_over = b_worst = 0
    b_hard_splits = 0
    a_overflow_tokens = 0
    a_tokens = b_tokens = 0

    print(f"{'document':<38} {'A chunks':>9} {'A over':>7} {'B chunks':>9} {'B over':>7}")
    for path in files:
        text = read_or_refuse(path, lambda p: p.read_text(encoding="utf-8"))

        a_chunks = character_budget_chunks(text, tok.budget)
        a_counts = [tok.count(c) for c in a_chunks]
        a_bad = [n for n in a_counts if n > tok.budget]

        b_chunks = chunker.chunk_text(text, document_id=path.name, tokenizer=tok,
                                      overlap_tokens=args.overlap_tokens)
        b_bad = [c for c in b_chunks if c.token_count > tok.budget]
        b_hard_splits += sum(1 for c in b_chunks if c.split_level == "character")

        a_total += len(a_chunks); a_over += len(a_bad)
        b_total += len(b_chunks); b_over += len(b_bad)
        a_worst = max([a_worst] + a_counts)
        b_worst = max([b_worst] + [c.token_count for c in b_chunks])
        a_overflow_tokens += sum(n - tok.budget for n in a_bad)
        a_tokens += sum(a_counts)
        b_tokens += sum(c.token_count for c in b_chunks)

        print(f"{path.name:<38} {len(a_chunks):>9} {len(a_bad):>7} "
              f"{len(b_chunks):>9} {len(b_bad):>7}")

    print()
    pct = (100.0 * a_over / a_total) if a_total else 0.0
    a_mean = (a_tokens / a_total) if a_total else 0.0
    b_mean = (b_tokens / b_total) if b_total else 0.0
    a_use = 100.0 * a_mean / tok.budget
    b_use = 100.0 * b_mean / tok.budget
    print("arm A, the rule of thumb")
    print(f"  chunks                 {a_total}")
    print(f"  over the window        {a_over}  ({pct:.1f}%)")
    print(f"  largest chunk          {a_worst} tokens against a window of {tok.budget}")
    print(f"  tokens past the window {a_overflow_tokens}  "
          f"(text the model would never have read)")
    print(f"  window used            {a_use:.0f}% on average  "
          f"(mean {a_mean:.0f} of {tok.budget} tokens)")
    print()
    print("arm B, packed to the real window")
    print(f"  chunks                 {b_total}")
    print(f"  over the window        {b_over}")
    print(f"  largest chunk          {b_worst} tokens against a window of {tok.budget}")
    print(f"  needed a hard split    {b_hard_splits}  (recorded, not silent)")
    print(f"  window used            {b_use:.0f}% on average  "
          f"(mean {b_mean:.0f} of {tok.budget} tokens)")

    if skipped:
        print()
        print("skipped, and why:")
        for s in skipped:
            print(f"  {s}")

    print()
    if b_over:
        print("FAIL: arm B went over the window. That is a defect in the chunker, "
              "not a finding.")
        return 1
    if a_over:
        print(f"Arm A would have silently truncated {a_over} of {a_total} chunks, "
              f"{a_overflow_tokens} tokens of text the model never sees. Arm B "
              "truncated nothing.")
    else:
        print("Arm A did not overflow on this corpus. On this text the rule of "
              "thumb held, and that is a fact about this text, not about the rule.")

    print(f"Either way it left {100 - a_use:.0f}% of the window unused against "
          f"arm B's {100 - b_use:.0f}%, across {a_total} chunks versus {b_total}. "
          "A fixed character budget cannot adapt, so it has to be conservative "
          "enough for the worst text and is therefore wasteful on ordinary text, "
          "while still overflowing on the atypical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
