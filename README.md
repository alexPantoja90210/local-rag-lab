# local-rag-lab

A grounded RAG that runs on one laptop and refuses what it cannot cite.

**Status: slices 1 and 2 of the target architecture.** What is in this
repository today decides whether a document may enter the knowledge base, and
splits the ones that may into chunks the embedding model can actually read.
There is no retrieval, no model and no database yet, and the target
architecture says which parts are still missing and why each one is there.

---

## What is here

Slice 1 decides whether a file may enter the knowledge base. Slice 2 splits
what gets in, against the embedding model's real token window.

```
python make_fixtures.py                      # build the fixture corpus
python test_invariants.py                    # 160 invariants, no key, no network, no spend
python prove_it_can_fail.py                  # break the rules on purpose, check the suite notices
python ingest_contract.py --documents docs/  # check a real folder
python measure_window.py --documents docs/   # the one script that needs a download
```

Everything except the last line costs nothing and reaches no network. There is
**no `requirements.txt`**: the standard library is the whole dependency list,
and `measure_window.py` needs `transformers` only because it asks a real model
how much it can read.

## The two rules, and where they came from

Both come from defects found in upstream repositories before a line of this was
written. Neither upstream has anything that enforces either one.

### 1. A file is what its bytes say, not what its name says

The sample corpus this project will use ships two files named `.docx` that are
PDFs:

```
meeting-notes-2025-01-08.docx:  PDF document, version 1.4, 6 page(s)
meeting-notes-2025-01-15.docx:  PDF document, version 1.4, 5 page(s)
```

The upstream pipeline routes on the extension alone. Run the contract over that
folder as shipped:

```
REJECT  meeting-notes-2025-01-08.docx   the name says ooxml-word, the bytes say pdf
REJECT  meeting-notes-2025-01-15.docx   the name says ooxml-word, the bytes say pdf

checked  13
accepted 11
rejected 2
```

Exit code 1, and nothing is ingested. A corpus that half-loaded must not look
like one that loaded. Rename those two to `.pdf` and all 13 pass.

The sniffing is signature-based and written out in full in `ingest_contract.py`
rather than delegated to a library, for the same reason the previous project
kept a hashed bag-of-words embedding: a mechanism you can read is a mechanism
you can argue with.

### 2. Deciding and reading are separate, and the deciding one never returns text

Upstream, when a conversion fails the reader falls back to a raw byte read, and
when that fails too a bare `except` returns this **as the document's content**:

```python
return f"[Error: Could not read file {os.path.basename(file_path)}]"
```

Fifty-eight characters that are then chunked, embedded and stored. The knowledge
base ends up holding a chunk whose entire content is an error message, the agent
can retrieve and cite it, and the ingest reports success.

Here, `check_file` returns a `Verdict` that has no field capable of holding
document text, and `read_or_refuse` raises `ConversionFailed` rather than
substituting anything. A function that cannot return content cannot fabricate it.

## Slice 2: chunking to the window the model actually has

The previous project sized chunks by a rule of thumb: 896 characters, at an
assumed 3.5 characters per token, against a 256-token window. On a real corpus
**864 of 11,529 passages still went over**, and the pipeline truncated them
silently at embed time. That was recorded and flagged, which was the right call
at the time, but it is a compromise rather than a mechanism. Text went into the
store that the model never read.

Two properties make that impossible here.

**The budget comes from the tokenizer, not from a constant.** `Tokenizer` is a
protocol with a `budget` the chunker reads off the model. A number in a config
file is a claim about a model; a number read from the model is the model.

**Nothing is truncated, ever.** The chunks' spans tile the document, so every
character lands in exactly one chunk. When a unit will not fit, the splitter
descends through paragraph, sentence, line, word and finally character, and
**records the level it had to use**. A hard split mid-word is legal, visible
and counted. The invariant that enforces this compares the output against the
input and does not care how the splitter got there:

```python
coverage_gaps(text, chunks) == []          # no gap, on every sample, every tokenizer
"".join(text[c.char_start:c.char_end] ...) == text
```

Chunk ids are derived from content and identity rather than minted fresh, so
re-chunking unchanged text produces the same ids and a re-ingest updates in
place. Both upstream projects use a new `uuid4` per chunk, which removes the
one property an explicit id exists to provide.

`pack_fingerprint()` identifies the tokenizer and budget a store was built
with. A store chunked for one embedder and queried with another measures
nothing and looks exactly like a store that works.

### Three tokenizers in the suite, and what that does not prove

Every packing invariant runs against three stand-in tokenizers that behave
structurally differently: one token per word, one per fixed run of characters,
and one where long words cost several tokens. The logic therefore cannot be
tuned to one tokenizer's shape without the suite noticing.

**None of them carries a real model's vocabulary**, so none can produce a real
model's numbers. That is what `measure_window.py` is for, and it is the only
script here that needs a download.

### The measurement, which you have to run yourself

```
pip install transformers
python measure_window.py --documents documents/
```

It asks the model for its window, chunks the same text two ways, and measures
both with the real tokenizer:

- **arm A**, the rule of thumb: fixed character windows
- **arm B**, this chunker: packed with the model's own tokenizer

Arm B must overflow zero times by construction; arm A is the question.

If the tokenizer cannot be loaded the script stops and says so. There is no
fallback estimate, because a number produced by guessing the thing being
measured is worse than no number.

### What it measured

`mxbai-embed-large-v1`, window **512 tokens read from the model**, over the
four plain-text documents in the sample corpus. The nine others were skipped
and listed: seven need a converter, two were rejected by the ingest contract.

|  | arm A, rule of thumb | arm B, token aware |
| --- | --- | --- |
| chunks | 16 | **12** |
| over the window | 0 | 0 |
| largest chunk | 398 / 512 | 510 / 512 |
| window used, mean | **62%** | **82%** |

**Arm A did not overflow.** On business prose the rule of thumb held, and that
is a fact about this text rather than about the rule: the same constant
overflowed 864 of 11,529 passages of regulatory text on the previous project.
One constant cannot be right for two corpora.

**What it did instead is the finding.** 1792 characters came to a mean of 315
tokens, so the real ratio on this text is **5.69 characters per token against
the 3.5 assumed, off by 63%** — and off in the safe direction, which is why
nothing ever failed. No error, no warning, no red test. It quietly left 38% of
the window unused, forever.

In product terms, at `k=5` over the same corpus: arm A delivers 1,575 tokens of
context, arm B delivers 2,090. **A third more usable context, from the same
documents and the same k**, for no extra retrieval and no extra spend.

The ratio line is in the script's output, so the next corpus reports its own.

## The suite counts itself, and proves it can fail

```
$ python test_invariants.py
...
all 160 invariants hold
```

The count is printed rather than left to be counted by hand, because on the
previous project two counts of assertions were stated from memory on the same
day and both were wrong.

Passing is not the claim. `prove_it_can_fail.py` copies the tree, breaks the
code seven ways, and checks that the suite goes red **and that the invariants
which should catch each break are the ones that do**. A mutation that reddens
the suite for some unrelated reason would prove nothing, so each one names the
invariants it expects:

```
baseline: all 160 invariants hold  (exit 0)

mutation: the extension rule removed                          in ingest_contract.py
mutation: a failed read returned as content, the upstream defect
mutation: a bare except reintroduced
mutation: the token budget ignored while packing              in chunker.py
mutation: the tail of the document dropped, the truncation this module forbids
mutation: a fresh id per run, the upstream defect
mutation: the split level no longer recorded

all 7 mutations were caught. The suite can fail.
```

Three of the 160 are **controls**: they pass only when something is *not* true.
Switching the extension rule off must make the mislabelled file pass, otherwise
something else is rejecting it. A budget large enough to hold a whole document
must still produce more than one chunk, otherwise the splitter is cutting on
structure and the budget invariants would pass without the budget doing any
work. And the bare-except scanner must find a planted one, otherwise a scanner
that always returns nothing would look identical to a working one.

## What this does not do yet

- It does not read a document. It decides whether one may be read, and how a
  document's text would be split once something else has read it.
- It does not retrieve, generate, cite or refuse an answer. Those are later
  slices, drawn in the target architecture.
- It does not verify that a well-formed file says anything true.
- The bare-except scanner is a source-level check and is labelled secondary in
  the suite, because on the previous project a source-level assertion passed
  while the call it guarded raised on every invocation. The behavioural
  assertion beside it is the one that matters.

## Provenance and licences

This repository is original code. Two upstream projects were read and audited,
and the findings above are about them.

- `coleam00/ottomator-agents/docling-rag-agent` — **MIT**, Copyright (c) 2024
  Cole Medin. The architecture this project takes forward, and the source of the
  sample corpus it will be measured on. See `NOTICE.md`.
- `ThomasJanssen-tech/Local-RAG-with-Ollama` — **no licence file**. Read only.
  Nothing from it is copied or redistributed here.

Neither upstream corpus is committed to this repository.
