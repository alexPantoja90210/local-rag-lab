"""
Prove the suite can go red.

A suite that has only ever been seen passing is indistinguishable from one that
cannot fail. This script breaks the contract on purpose, three ways, in a
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
TARGET = "ingest_contract.py"

# Each mutation is a (find, replace) pair applied to the contract, plus the
# invariants that must go red as a result. Naming them matters: a mutation
# that reddens the suite for some unrelated reason proves nothing.
MUTATIONS = [
    (
        "the extension rule removed",
        "    if detected != declared:",
        "    if False:",
        ["a pdf named .docx is rejected",
         "the rule is not one-directional: a docx named .pdf is rejected",
         "every fixture in bad/ is rejected"],
    ),
    (
        "a failed read returned as content, the upstream defect",
        "    except Exception as exc:  # re-raised, never swallowed, never substituted\n"
        "        raise ConversionFailed(f\"{path.name}: the reader failed: {exc!r}\") from exc",
        "    except Exception:\n"
        "        return f\"[Error: Could not read file {path.name}]\"",
        ["a reader that raises produces a refusal, not a string"],
    ),
    (
        "a bare except reintroduced",
        "    except (zipfile.BadZipFile, OSError):",
        "    except:",
        ["the contract contains no bare except (secondary, source-level)"],
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

    for name, find, replace, must_fail in MUTATIONS:
        with tempfile.TemporaryDirectory() as tmp:
            lab = Path(tmp) / "lab"
            shutil.copytree(HERE, lab, ignore=shutil.ignore_patterns(
                "__pycache__", ".git", "fixtures"))

            source = (lab / TARGET).read_text(encoding="utf-8")
            if find not in source:
                problems.append(f"{name}: the text to mutate is no longer in {TARGET}")
                print(f"SKIP  {name}\n        the anchor has moved, so this proves nothing\n")
                continue
            (lab / TARGET).write_text(source.replace(find, replace, 1), encoding="utf-8")

            code, out = run_suite(lab)
            reddened = [line[5:] for line in out.splitlines() if line.startswith("FAIL ")]

            print(f"mutation: {name}")
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
