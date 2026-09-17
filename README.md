# local-rag-lab

A grounded RAG that runs on one laptop and refuses what it cannot cite.

**Status: slice 1 of the target architecture.** What is in this repository today
is the ingest contract and its suite. There is no retrieval, no model and no
database yet, and the target architecture says which parts are still missing
and why each one is there.

---

## What slice 1 does

It decides whether a file may enter the knowledge base, and says why not when
it may not.

```
python make_fixtures.py                      # build the fixture corpus
python test_invariants.py                    # 28 invariants, no key, no network, no spend
python prove_it_can_fail.py                  # break the rules on purpose, check the suite notices
python ingest_contract.py --documents docs/  # check a real folder
```

Nothing above costs anything or reaches the network. There is **no
`requirements.txt`**: the standard library is the whole dependency list.

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

## The suite counts itself, and proves it can fail

```
$ python test_invariants.py
...
all 28 invariants hold
```

The count is printed rather than left to be counted by hand, because on the
previous project two counts of assertions were stated from memory on the same
day and both were wrong.

Passing is not the claim. `prove_it_can_fail.py` copies the tree, breaks the
contract three ways, and checks that the suite goes red **and that the
invariants which should catch each break are the ones that do**:

```
mutation: the extension rule removed
  suite exit code      1   (must be non-zero)
  invariants gone red  6
  RESULT  caught, by the invariants that should catch it

mutation: a failed read returned as content, the upstream defect
  RESULT  caught, by the invariants that should catch it

mutation: a bare except reintroduced
  RESULT  caught, by the invariants that should catch it

all 3 mutations were caught. The suite can fail.
```

Two of the invariants are **controls**: they pass only when something is *not*
true. Switching the extension rule off must make the mislabelled file pass,
otherwise something other than that rule is rejecting it and the rule is
unproven.

## What this does not do yet

- It does not read a document. It decides whether one may be read.
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
