# local-rag-lab

A grounded RAG that runs on one laptop and refuses what it cannot cite.

**Status: four slices of four.** This repository decides whether a document may
enter the knowledge base, splits what gets in against the embedding model's real
token window, keeps the vectors in two files that refuse a query built by a
different embedder, retrieves with the score reaching the caller, and discards
generated text that cites anything it was not given.

It runs on one laptop with Ollama. No key, no account, no database, no Docker,
and nothing leaves the machine.

**There is no database, and that is a decision with numbers behind it.** See
*The store* below.

---

## What is here

Slice 1 decides whether a file may enter the knowledge base. Slice 2 splits what
gets in, against the embedding model's real token window. Slice 3 keeps the
vectors, and refuses a query that was not built by the same packing. Slice 4
retrieves, generates, and throws away an answer that cites something it was
never shown.

```
python make_fixtures.py                          # build the fixture corpus
python test_invariants.py                        # counts itself, no key, no network, no spend
python prove_it_can_fail.py                      # break the code, check the suite notices
python ingest_contract.py --documents documents  # check a real folder

# these three need Ollama running on this machine
python build_store.py  --documents documents --store store/demo
python ask.py          --store store/demo --questions questions.json --verbose
python prove_the_detector_sees.py                # the control on the measurement itself
```

The first four cost nothing and reach no network. There is **no
`requirements.txt`**: the standard library is the whole dependency list.
`measure_window.py` and `build_store.py` need `transformers`, only because they
ask a real model how much it can read.

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

## Slice 3: the store

A vector store's job here is small. Keep N vectors and the identity of the chunk
each came from; given a query, return the closest k with their scores. That is a
dot product and a sort.

**Two files.** `store.vec` holds raw little-endian float32, written with the
standard library's `array`. `store.json` holds the manifest and the records, so
the store can be inspected without this module. No server, no account, no Docker.

### Why not pgvector, and why not Chroma

Version 2 of the target architecture used Postgres. Version 3 does not, and the
reasons are measured rather than preferred.

| | |
| --- | --- |
| 400 chunks, compared against **all** of them | 1.6 MB, **0.21 ms** |
| 11,529 passages, the previous project's second corpus | 45 MB, **1.6 ms** |
| `chromadb` added to a repository that requires nothing | **79 packages** |

An approximate index exists so you do not have to compare against everything. At
this size comparing against everything is free, and it **cannot miss a
neighbour**. The previous project already measured that: `check_index.py` found
identical top-5 lists on 21 of 21 queries at 48 vectors, with a largest score gap
of 2.384e-07, which is float32 precision.

Among Chroma's 79 packages are a Kubernetes client, an ASGI web server and an
ONNX runtime. Postgres needs Docker, which on a 16 GB laptop takes one to two
gigabytes from a budget the local model already claims most of.

So **a database is the sophisticated option that has to earn its place**, the
same call as keeping the hashed bag-of-words embedding in the previous project:
hold the crude floor, so there is something to measure the expensive thing
against. When the corpus stops fitting in memory, `check_index.py` is what says
so.

### What the store refuses

**A query built by a different packing.** This is the reason the module exists
rather than a dictionary:

```python
store.search(query, k=5, fingerprint="...")   # required, not a convenience
# StoreMismatch: the store was built under packing c3e3d917... by mxbai-...,
# and the query was produced under 0000... Nothing was searched: the scores
# would have been meaningless and would have looked fine.
```

A store built for one embedder and queried with another returns results that look
entirely normal. Nothing downstream can tell.

**A vector that is not length 1**, at save time, because the dot product would
not be a cosine and every score would be quietly wrong. **Two files that disagree**
about how many vectors they hold, by name as `StoreInvalid` rather than as a raw
reshape error. **A query of the wrong width.** **A repeated chunk id**, because a
content-derived id repeating means the same chunk was added twice.

### Saying why there is nothing

An empty result is never just empty:

```
4 vectors were compared and none scored above 0.99; the store at
/path/to/corpus holds 4 vectors under packing c3e3d91703a20b0d
```

and, for a store with nothing in it, the reason says **the store is empty**
rather than blaming the threshold. The previous project shipped a message that
blamed a threshold which was switched off, and it sent someone looking in the
wrong place for an afternoon.

### Two implementations, and they have to agree

The search is written twice: once in plain Python, readable and slow, and once
through numpy when it is installed. An invariant asserts they agree to within
1e-5 on every query. **A fast path nobody can check against a slow one is a fast
path nobody can check.** numpy is never required.

## Slice 4: retrieval, and the gate that discards what it was not given

Every tutorial in this family holds its grounding in place with a line of
prompt. *Always search the knowledge base.* *Only answer from the provided
context.* Nothing reads the answer afterwards, nothing compares what was cited
against what was supplied, and nothing can refuse. The model complies most of
the time, which is the worst possible failure rate: often enough to look
trustworthy, not often enough to be.

### The score reaches the caller

Both upstream agents compute a similarity and discard it one line later. pyflakes
says so out loud in one of them: `local variable 'similarity' is assigned to but
never used`. A score that never reaches the caller is a threshold that cannot
exist and an abstention that cannot be justified, so `threshold` here is
keyword-only and **has no default value**. Passing `None` means there is no
floor and has to be an act, because on the previous project a message blamed a
cutoff that was switched off and sent somebody looking in the wrong place for an
afternoon.

### Four things the gate refuses, and the third is why it exists

1. **A claim with no citation.** The obvious one.
2. **A citation naming an id that does not exist.** Also obvious, once something
   looks.
3. **A citation naming a real chunk that was not supplied on this turn.** This
   is the one. The id resolves, the document is real, the citation renders as a
   working reference, and the model was never shown that passage, so whatever it
   said about it came out of its weights. By eye it is indistinguishable from a
   correct answer. It is the same shape as the store's fingerprint refusal: a
   failure whose output looks exactly like success.
4. **A claim that is nothing but a citation.** A gate that counts `[[a3f...]]`
   as satisfied can be passed by a model that emits citations and no content.

Matching is exact. No case folding, no prefix matching. `if cited in
supplied_text` passes on a substring, and that is how a check like this usually
fails open.

### The abstention, and a lesson about two-part instructions

The first version of this gate discarded **correct** abstentions. Asked an
unanswerable question, the model retrieved, read the passages, found nothing and
said so, and the gate threw it away for carrying no citation. Worse: the prompt
had asked it to abstain *and* cite the closest passage, so had the model obeyed,
it would have produced a sentence citing a passage that does not support it, and
**the gate would have passed it.** On abstentions the sign was inverted, and a
decorative citation scored better than an honest one.

The fix was a reserved citation, `[[none]]`, and it failed too, in the mirror
image. Told to send the marker *and* a sentence, the model sent the marker and no
sentence, exactly as it had sent a sentence and no marker before. It obeyed half
of a two-part instruction both times, a different half each time, and both times
a correct abstention was discarded.

The two halves were never alike. The marker is a **verdict** and only the model
can produce it, because only the model read the passages. The sentence is
**wording**: not a claim, resting on no passage, stating nothing about the corpus
that could be wrong. So the sentence is not asked for. The system writes it.

    The passages retrieved for this question do not contain an answer to it.

An abstention is its own outcome, counted apart from an answer. Folding the two
together would report a higher success rate and would stop distinguishing a
model that answers from one that knows when it cannot, which is the distinction
this repository is about.

An abstention is all or nothing, it may not sit beside a grounded claim, and it
is **refused outright when nothing was supplied**. That last rule is what keeps
skipping retrieval worthless: without it, a model declines to retrieve, emits
the marker, and walks out through the gate.

### The measurement, and the control on the measurement

Retrieval is a tool the model may decline to call. The question is how often it
declines, and that number is this slice's result.

```
0 of 12 turns never retrieved (0.0%) · 8 answered and survived the gate · 4 abstained, having looked

  A   0 of 4 never retrieved      names the company
  B   0 of 4 never retrieved      same facts, does not name the company
  C   0 of 4 never retrieved      cannot be answered from what was stored
```

`llama3.1:8b`, `mxbai-embed-large`, k of 5, no threshold, twelve questions, one
run. **The prediction registered before the run was that the agent would skip
retrieval where its own priors felt sufficient, and that family B would be
skipped more than family A. It was not supported.** Nothing was skipped
anywhere. That publishes as it stands.

The eight answerable questions were answered correctly and cited a supplied
passage every time. The four unanswerable ones abstained, having looked.

Two things about that table are worth saying out loud.

**It is one run and not two spliced together.** An earlier run had these same
eight answers under a different generation prompt, and the prompt changed while
fixing the abstention. Reusing those eight beside four fresh ones would have
been two runs reported as one, and the differences would almost certainly have
been nil, which is not the same as measured.

**Latency is the cost that is not zero**, and the target architecture said so
before any of this ran: **90.9 seconds mean per question**, two model calls
each, 65.5 to 119.1, 18.2 minutes for twelve questions on a 16 GB Windows
laptop. Money is not the budget here. Waiting is.

**Zero is the one value this instrument could not tell apart from a blind
instrument.** Every assertion about the skip detector runs against a response
this repository wrote itself, so the negative path had never been walked with a
real server behind it, and a scanner that always returns nothing looks identical
to a working scanner.

`prove_the_detector_sees.py` settles it, and the shape of that control matters
more than its result. The first attempt told the model *not* to search and
watched for skips. The model searched anyway, so the run ended with the two
explanations still stuck together: a model that ignores the instruction and a
detector that cannot see a skip produce identical output. **A control that
varies the thing under test cannot isolate it.**

The control that works does not ask the model for anything. It **removes the
tool**. With no tool in the request the response can only be plain content,
which is exactly the shape a real skip has, and it arrives from the real server:

```
arm 1  the tool is offered      message keys ['content', 'role', 'tool_calls']  -> True
arm 2  NO tool is offered       message keys ['content', 'role']                -> False
```

Two API facts fell out of it, previously assumed and now observed: a real
no-tool-call response **omits** `tool_calls` rather than sending an empty list,
and when a tool *is* called, `content` is the empty string.

### What the control caught while looking for something else

Arm 2 gave the model no tool, no passages and no corpus. It answered:

> *According to the NeuralFlow AI Knowledge Base, the company has a dedicated
> "Annual Learning..."*

It attributed its answer, by name, in the voice of a citation, to a source it
had never read. Not a model getting a fact wrong, which is ordinary. A model
manufacturing **the appearance of grounding** while ungrounded, which is the
failure that survives review because it looks like diligence.

No instruction would have prevented that sentence. The gate discards it, because
prose attribution is not a supplied chunk id and the gate does not accept prose
as a citation.

### What a threshold cannot do at this size

The unanswerable questions did not score lower than the answerable ones.

| | top score |
| --- | --- |
| c3, severance, not in the corpus | **0.732** |
| a3, the 24-hour agenda rule, in the corpus | 0.686 |
| b1, the 401(k) match, in the corpus | 0.545 |

The best-scoring unanswerable question beat **five of the eight** answerable
ones. Any floor that refuses c3 also refuses those five; any floor that admits
all eight admits all four unanswerable ones. **At this corpus size a similarity
threshold does not separate a question the corpus can answer from one it
cannot.** The citation gate, not the threshold, is what ends those turns, and
the threshold in Figure 2 of the target architecture does not do the job the
drawing gives it here.

Measured on twelve questions and four documents. Not claimed to generalise.

## The suite counts itself, and proves it can fail

```
$ python test_invariants.py
...
all 329 invariants hold, 19 of them controls
```

Both counts are printed rather than left to be counted by hand, because on the
previous project two counts of assertions were stated from memory on the same
day and both were wrong.

The control count is printed for a narrower reason, and it is the better story.
The invariant count has been printed since the first commit and has never been
wrong since. The control count sat in the same sentence of this same README and
was still written by hand, and it was wrong for a day: this file said four while
the suite held six. **A number that is printed is checked. A number beside it
that is remembered is not, and on the page the two look identical.** So the
suite produces both, and everything that quotes them quotes a line it printed.

Passing is not the claim. `prove_it_can_fail.py` copies the tree, breaks the
code twelve ways, and checks that the suite goes red **and that the invariants
which should catch each break are the ones that do**. A mutation that reddens
the suite for some unrelated reason would prove nothing, so each one names the
invariants it expects:

```
baseline: all 329 invariants hold, 19 of them controls  (exit 0)

mutation: the extension rule removed                          in ingest_contract.py
mutation: a failed read returned as content, the upstream defect
mutation: the token budget ignored while packing              in chunker.py
mutation: the tail of the document dropped, the truncation this module forbids
mutation: the packing fingerprint no longer checked            in store.py
mutation: the supply check removed: every citation accepted    in citation_gate.py
mutation: prefix matching, the way a check like this usually fails open
mutation: an abstention allowed with no passages supplied, so skipping pays again
mutation: a malformed tool_calls read as 'did not retrieve'    in ollama_client.py
...
all 28 mutations were caught. The suite can fail.
```

Twice now the harness has refused a mutation of mine rather than counting it.
Once because an anchor had moved, so a rule was going untested. Once with
`wrong invariants caught it`, because the mutation I wrote broke the code in a
way that reddened the suite for an unrelated reason. Both times the mechanism
caught it and not my attention, which is the entire argument for having the
mechanism.

Nineteen of the 329 are **controls**: they pass only when something is *not* true.
Switching the extension rule off must make the mislabelled file pass, otherwise
something else is rejecting it. A budget large enough to hold a whole document
must still produce more than one chunk, otherwise the splitter is cutting on
structure and the budget invariants would pass without the budget doing any
work. And the bare-except scanner must find a planted one, otherwise a scanner
that always returns nothing would look identical to a working one.

## Walkthrough: one ungrounded sentence, and how you know the refusal works

The claim this repository makes is not *the model hallucinates less*. That is unverifiable: there is no experiment you can run to establish it, so it is asserted, hoped for, and discovered in production.

The claim is narrower and it is checkable: **an ungrounded answer is detectable and refusable.** This section is the whole chain, because a claim about mechanism should be followable end to end by a stranger.

### Step 1. The sentence

The model retrieved, was given three passages, and produced this:

```
Approvals expire after eight hours [[be5798a9...]].
```

It looks correct. The id is well formed, it resolves, the document is real, and the fact is true of the corpus. **It was not one of the three passages supplied on this turn.** The model recognised the chunk id and wrote about a passage it was not shown, which means the sentence came out of its weights.

Nothing about the output distinguishes this from a correct answer. That is the point of the example.

### Step 2. What refuses it

Four lines in `citation_gate.py`:

```python
for cited in claim.citations:
    if cited in supplied_set:
        continue
    if known_set is not None and cited in known_set:
        violations.append(Violation(kind=NOT_SUPPLIED, ...))
```

`supplied_set` is what retrieval handed the model **on this turn**. Not what is in the store, not what it saw three turns ago. The refusal names the claim and the id, and the generated text is discarded rather than returned.

That is the mechanism. Everything after this step exists to establish that those four lines actually run and actually decide.

### Step 3. How you know the line runs

```python
check("a real chunk that was not supplied this turn is refused", not _unsupplied.passed)
check("a real unsupplied chunk is named not_supplied, not invented",
      _v(_unsupplied).kind == gate.NOT_SUPPLIED)
```

The suite asserts it, prints `ok`, and counts itself. **This is the weakest link in the chain**, and most projects stop here. An assertion that has never been observed failing tells you nothing about whether it is capable of failing.

### Step 4. How you know the assertion can fail

`prove_it_can_fail.py` copies the tree, removes the rule, and requires the suite to go red:

```python
("citation_gate.py",
 "the supply check removed: every citation accepted",
 "            if cited in supplied_set:\n                continue",
 "            if True:\n                continue",
 [...the invariants that must go red...])
```

Real output:

```
mutation: the supply check removed: every citation accepted
  in                   citation_gate.py
  suite exit code      1   (must be non-zero)
  invariants gone red  13
          - a real chunk that was not supplied this turn is refused
          - a citation naming an unknown id is refused
          - a prefix of a supplied id is not a match
          - case is not folded when matching an id
          - an answer produced without retrieving cannot survive the gate
  RESULT  caught, by the invariants that should catch it
```

Break the rule on purpose, and thirteen assertions go red.

### Step 5. How you know the mutation proved *that* rule

A mutation that reddens the suite for an unrelated reason proves nothing. So each one **names the invariants it expects**, and the harness checks that those are the ones that fired. Anything else prints `RESULT wrong invariants caught it` and exits non-zero.

That branch is not decoration. **It has fired twice for real.** Once when a mutation's anchor text had moved, so a rule was silently going untested, and once today on a mutation of mine that broke the code in a way that reddened the suite for a reason unrelated to the rule it claimed to test. Both times the harness caught it and the author did not.

### Step 6. How you know it is not simply refusing everything

A gate that refuses every answer would satisfy every assertion in step 3. So four of the nineteen controls exist for this rule alone, and they pass only when something is **not** true:

```
control: an answer citing a supplied chunk is refused too
control: an answer passes when nothing at all was supplied
control: the citation parser finds nothing in text that carries a citation
control: a turn that never retrieved can still produce a passing answer
```

If the gate started refusing everything, the first goes red. If the parser went blind, the third goes red and every refusal above it becomes an artefact of the parser rather than evidence about the answer.

### What this does not prove

It does not prove the model hallucinates rarely. It proves that **when it does, the output does not reach a user with the appearance of a citation.** Those are different claims, and only one of them can be checked by running something.

It also does not prove faithfulness. A claim citing a supplied passage can still misrepresent it, and this chain will pass it. That gap is named in *What this does not do yet* rather than covered.

### The contrast, stated plainly

Ask what experiment would verify *"it hallucinates less."* Less than what, measured how, on which corpus, against what baseline, and observable by whom? There is no answer that a reader could run.

Now ask what experiment verifies *"an ungrounded claim is refused."* Delete the check, run the suite, and watch thirteen assertions go red. **That is the difference, and it is the whole argument for the six steps above.**

## What this does not do yet

- **It does not check that a claim is faithful to the passage it cites.** A
  sentence can cite a supplied passage and misrepresent it completely, and the
  gate will pass it. Verbatim span matching is designed and deliberately not
  built. What is enforced is narrower and worth naming precisely: every claim
  points at something the model was actually shown. That is a real guarantee and
  it is not the guarantee a reader assumes from the words "citation gate".
- It does not measure **over-abstention**, a model declining a question it could
  have answered. The gate's job is grounding, and abstaining too readily is
  unhelpful rather than ungrounded. The question set can measure it and nothing
  here does.
- **There is no conversation.** Every call sends one system message and one user
  message: no history, no follow-ups, no turn cap, and no interface a person can
  sit in front of. Figure 2 of the target architecture draws three refusals and
  two are built. Named as IA-186 and scoped as IA-187, rather than left as an
  implied promise.
- It does not read PDFs, Word files or audio. Only `.md` and `.txt` are stored,
  so four of the sample corpus's thirteen documents reach the store. Docling is
  a later slice, and "unanswerable" in the question set means unanswerable from
  what was actually stored.
- It does not make the model forget. The knowledge is in the weights. This makes
  using it without support detectable and refusable, which is detection and
  incentive, not amnesia.
- It holds everything in memory when it searches. At 45 MB that is not a
  problem. At 50 GB it would be, and that is when a real database earns its
  place.
- There is no concurrency. One process writes, one process reads.
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
