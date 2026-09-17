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
