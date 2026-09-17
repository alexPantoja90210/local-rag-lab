"""
Prove the suite can go red.

A suite that has only ever been seen passing is indistinguishable from one that
cannot fail. This script breaks the code on purpose, several ways, in a
throwaway copy of the tree, and asserts that the suite notices each time and
names the right invariants.

It changes nothing in the working tree.

Run:  python prove_it_can_fail.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
# Each mutation names the file it breaks, a (find, replace) pair, and the
# invariants that must go red as a result. Naming the invariants matters: a
# mutation that reddens the suite for some unrelated reason proves nothing.
MUTATIONS = [
    (
        "ingest_contract.py",
        "the extension rule removed",
        "    if detected != declared:",
        "    if False:",
        ["a pdf named .docx is rejected",
         "the rule is not one-directional: a docx named .pdf is rejected",
         "every fixture in bad/ is rejected"],
    ),
    (
        "ingest_contract.py",
        "a failed read returned as content, the upstream defect",
        "    except Exception as exc:  # re-raised, never swallowed, never substituted\n"
        "        raise ConversionFailed(f\"{path.name}: the reader failed: {exc!r}\") from exc",
        "    except Exception:\n"
        "        return f\"[Error: Could not read file {path.name}]\"",
        ["a reader that raises produces a refusal, not a string"],
    ),
    (
        "ingest_contract.py",
        "a bare except reintroduced",
        "    except (zipfile.BadZipFile, OSError):",
        "    except:",
        ["the contract contains no bare except (secondary, source-level)"],
    ),
    (
        "chunker.py",
        "the token budget ignored while packing",
        "            if tokenizer.count(text[:end]) > budget:\n                break",
        "            if False:\n                break",
        ["[stub-word-16] no chunk exceeds the budget, on 'prose'",
         "[stub-subword-4-12] no chunk exceeds the budget, on 'prose'"],
    ),
    (
        "chunker.py",
        "the tail of the document dropped, the truncation this module forbids",
        "        cursor += take          # strictly positive, so this always terminates",
        "        cursor += take\n        if len(chunks) >= 2:\n            break",
        ["[stub-word-16] the spans leave no gap, on 'prose'",
         "[stub-word-16] the spans rebuild the source exactly, on 'prose'"],
    ),
    (
        "chunker.py",
        "a fresh id per run, the upstream defect",
        "        h = hashlib.sha256()",
        "        import uuid\n        return uuid.uuid4().hex[:32]\n        h = hashlib.sha256()",
        ["chunking the same text twice gives the same ids"],
    ),
    (
        "chunker.py",
        "the split level no longer recorded",
        "            return taken, SPLIT_LEVELS[idx]",
        "            return taken, SPLIT_LEVELS[0]",
        ["and the hard split is recorded rather than silent",
         "the report counts how many chunks needed a hard split"],
    ),
    (
        "store.py",
        "the packing fingerprint no longer checked",
        "        if fingerprint != self.fingerprint:",
        "        if False:",
        ["a query from a different packing is refused, not answered"],
    ),
    (
        "store.py",
        "an unnormalised vector accepted, so the scores stop being cosines",
        "        n = _norm(vector)\n        if abs(n - 1.0) > NORM_TOLERANCE:",
        "        n = _norm(vector)\n        if False:",
        ["a vector that is not length 1 is refused at save"],
    ),
    (
        "store.py",
        "results returned unsorted, so the closest is no longer first",
        "        kept.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))",
        "        pass",
        ["an exact search puts the identical chunk first"],
    ),
    (
        "store.py",
        "the two files allowed to disagree about how many vectors there are",
        "    if actual_bytes != expected_bytes:",
        "    if False:",
        ["a vector file that disagrees with its manifest is refused by name"],
    ),
    (
        "store.py",
        "an empty store blamed on the threshold, the IA-163 message",
        "        if self.considered == 0:",
        "        if False:",
        ["and an empty store says it is empty, not that a threshold failed"],
    ),
    (
        "citation_gate.py",
        "the supply check removed: every citation accepted",
        "            if cited in supplied_set:\n                continue",
        "            if True:\n                continue",
        ["a real chunk that was not supplied this turn is refused",
         "a citation naming an unknown id is refused",
         "a prefix of a supplied id is not a match",
         "case is not folded when matching an id",
         "an answer produced without retrieving cannot survive the gate"],
    ),
    (
        "citation_gate.py",
        "prefix matching, the way a check like this usually fails open",
        "            if cited in supplied_set:\n                continue",
        "            if any(_s.startswith(cited) for _s in supplied_set):\n                continue",
        ["a prefix of a supplied id is not a match"],
    ),
    (
        "citation_gate.py",
        "a claim with no citation allowed through",
        "        if not claim.citations:",
        "        if False:",
        ["a claim with no citation is refused",
         "one uncited bullet among three is refused",
         "a citation after the full stop does not ground the sentence before it"],
    ),
    (
        "citation_gate.py",
        "a bare citation counted as a grounded claim",
        "        if not claim.prose:",
        "        if False:",
        ["a citation on its own is not a claim",
         "a bare citation is named empty_claim"],
    ),
    (
        "retrieval.py",
        "a skipped turn allowed to carry supplied chunks, so the count lies",
        "        if not self.retrieved and self.supplied:",
        "        if False:",
        ["a turn that did not retrieve cannot carry supplied chunks"],
    ),
    (
        "ollama_client.py",
        "the window defaulted to 512 when the server does not report one",
        "    if not found:\n        raise OllamaRefused(",
        "    if not found:\n        return 512\n    if False:\n        raise OllamaRefused(",
        ["model_info with no context length leaves the window unknown"],
    ),
    (
        "ollama_client.py",
        "a malformed tool_calls read as 'did not retrieve', which fakes the finding",
        "    if not isinstance(calls, list):\n        raise OllamaRefused(",
        "    if not isinstance(calls, list):\n        return False\n    if False:\n        raise OllamaRefused(",
        ["tool_calls that is not a list does not count as not retrieving"],
    ),
    (
        "ollama_client.py",
        "a response with no message read as 'did not retrieve'",
        "    if not isinstance(response, dict) or \"message\" not in response:\n        raise OllamaRefused(",
        "    if not isinstance(response, dict) or \"message\" not in response:\n        return False\n    if False:\n        raise OllamaRefused(",
        ["a response with no message does not count as not retrieving"],
    ),
    (
        "ollama_client.py",
        "generation streamed, so a turn can be read from a fragment",
        "        \"stream\": False,",
        "        \"stream\": True,",
        ["generation is not streamed"],
    ),
    (
        "ollama_client.py",
        "temperature left to whatever the server defaults to",
        "        \"options\": {\"temperature\": temperature},",
        "        \"options\": {},",
        ["temperature is stated on the call rather than left to a default"],
    ),
    (
        "ollama_client.py",
        "an unfinished response accepted as an answer",
        "    if data.get(\"done\") is False:",
        "    if False:",
        ["an unfinished response is a fragment and is refused"],
    ),
    (
        "ollama_client.py",
        "the endpoint fallback made silent",
        # The first attempt at this mutation only disabled the assignment, which
        # left `elif self.endpoint != endpoint` comparing None against the
        # endpoint and raising on the very first call. The suite went red for a
        # reason that had nothing to do with recording the fallback, and the
        # harness said so: "wrong invariants caught it". Second time that branch
        # has fired for real. Silence has to be modelled as silence, so both
        # arms go.
        "        if self.endpoint is None:\n            self.endpoint = endpoint\n"
        "        elif self.endpoint != endpoint:",
        "        if False:\n            self.endpoint = endpoint\n"
        "        elif False:",
        ["which endpoint answered is recorded",
         "the fallback is recorded, not silent"],
    ),
    (
        "citation_gate.py",
        "an abstention allowed to sit beside a grounded claim, IA-179",
        "        if len(claims) != 1:",
        "        if False:",
        ["an abstention beside another claim is refused",
         "smuggling a claim past an abstention is named"],
    ),
    (
        "citation_gate.py",
        "an abstention allowed with no passages supplied, so skipping pays again",
        "        elif not supplied_t:",
        "        elif False:",
        ["an abstention with no passages supplied is refused",
         "and it is refused for having nothing to be absent from"],
    ),
    (
        "citation_gate.py",
        "the reserved token treated as an ordinary id, the pre-IA-179 gate",
        "    marked = [c for c in claims if NONE_TOKEN in c.citations]",
        "    marked = []",
        ["an abstention marked with the reserved token passes",
         "and it is recorded as an abstention, not as an answer",
         "a turn that looked and found nothing is recorded as abstained"],
    ),
(
        "citation_gate.py",
        "the abstention answer taken from the model instead of written here",
        "            answer=ABSTENTION_SENTENCE if passed else None,",
        "            answer=text if passed else None,",
        ["an abstention's answer is written by the system, not by the model",
         "the same sentence whatever the model wrote"],
    ),
]





def run_suite(cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "test_invariants.py"],
        cwd=cwd, capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    code, out = run_suite(HERE)
    if code != 0:
        print("The suite is already red on the working tree. Fix that first.")
        print(out)
        return 2
    print(f"baseline: {out.strip().splitlines()[-1]}  (exit {code})\n")

    problems: list[str] = []

    for target, name, find, replace, must_fail in MUTATIONS:
        with tempfile.TemporaryDirectory() as tmp:
            lab = Path(tmp) / "lab"
            shutil.copytree(HERE, lab, ignore=shutil.ignore_patterns(
                "__pycache__", ".git", "fixtures"))

            source = (lab / target).read_text(encoding="utf-8")
            if find not in source:
                problems.append(f"{name}: the text to mutate is no longer in {target}")
                print(f"SKIP  {name}\n        the anchor has moved, so this proves nothing\n")
                continue
            (lab / target).write_text(source.replace(find, replace, 1), encoding="utf-8")

            code, out = run_suite(lab)
            reddened = [line[5:] for line in out.splitlines() if line.startswith("FAIL ")]

            print(f"mutation: {name}")
            print(f"  in                   {target}")
            print(f"  suite exit code      {code}   (must be non-zero)")
            print(f"  invariants gone red  {len(reddened)}")

            if code == 0:
                problems.append(f"{name}: the suite stayed green")
                print("  RESULT  the suite did not notice. The rule is unproven.")
            else:
                missing = [inv for inv in must_fail if inv not in reddened]
                if missing:
                    problems.append(f"{name}: expected red but green: {missing}")
                    print(f"  RESULT  wrong invariants caught it, missing: {missing}")
                else:
                    for inv in must_fail:
                        print(f"          - {inv}")
                    print("  RESULT  caught, by the invariants that should catch it")
            print()

    if problems:
        print(f"{len(problems)} mutation(s) not caught:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print(f"all {len(MUTATIONS)} mutations were caught. The suite can fail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
