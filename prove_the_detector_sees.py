"""
Prove the skip detector can see a skip, against the real server.

IA-180. The measured run reported 0 of 12 turns skipped, and zero is the one
value this instrument could not tell apart from a blind instrument. Every
invariant on `decided_to_retrieve()` runs against a response this repository
wrote itself, so the False path had never been walked with a real Ollama
behind it.

The first attempt at this control was badly designed and is kept here as the
reason this one is shaped the way it is. It told the model *not* to search and
watched for skips. The model retrieved anyway, four times out of four, so the
run ended with the two explanations still stuck together: a model that ignores
the instruction, and a detector that cannot see a skip, produce the same output.
**A control that varies the thing under test cannot isolate it.** That control
was measuring the model's obedience, which is not what was in doubt.

This one does not ask the model for anything. It removes the tool.

    arm 1  the tool is offered      the model calls it       must read as True
    arm 2  NO tool is offered       it cannot call anything  must read as False

Arm 2 is the point. With no tool in the request the response can only be plain
content, which is exactly the shape a real skip has, and it arrives from the
real server rather than from a fake this repository built. If
`decided_to_retrieve()` returns False there, without raising, the False path is
proven against reality and the 0.0% is a fact about the model. If it raises, or
returns True, the detector never could have reported a skip and the 0.0% meant
nothing, which is what this project would rather find out here than in a README.

Arm 1 needs no defending: twelve real turns already returned True. It is run
anyway, because a control that only exercises one arm proves half a thing.

The raw keys of each response are printed. The fakes in the suite were written
from the API shape as this repository understood it, and understanding is not
evidence. Whether a real no-tool-call response carries `tool_calls: []` or omits
the key is a fact, and until this runs nobody here has seen it.

Run:  python prove_the_detector_sees.py
"""

from __future__ import annotations

import argparse
import sys

import ollama_client as oc

QUESTION = "What is the annual learning budget for each employee at NeuralFlow AI?"
SYSTEM = ("You answer questions about NeuralFlow AI using its internal "
          "documents. Always search the knowledge base before answering.")


def arm(label, *, model, transport, with_tool):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": QUESTION}]
    tools = [oc.RETRIEVE_TOOL] if with_tool else None

    print(f"\n{label}")
    print(f"  tools sent           {'retrieve' if with_tool else 'NONE'}")

    response = oc.chat(model, messages, transport=transport, tools=tools)
    message = response.get("message", {})
    keys = sorted(message) if isinstance(message, dict) else type(message).__name__
    print(f"  message keys         {keys}")
    if isinstance(message, dict):
        calls = message.get("tool_calls", "<the key is absent>")
        print(f"  tool_calls           {calls if calls == '<the key is absent>' else repr(calls)[:100]}")
        content = (message.get("content") or "").strip()
        print(f"  content              {content[:90]!r}")

    try:
        verdict = oc.decided_to_retrieve(response)
        raised = None
    except oc.OllamaRefused as exc:
        verdict, raised = None, exc
    print(f"  decided_to_retrieve  {verdict if raised is None else f'RAISED {raised}'}")
    return verdict, raised


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chat-model", default="llama3.1:8b")
    ap.add_argument("--host", default=oc.DEFAULT_HOST)
    args = ap.parse_args(argv)

    transport = oc.http_transport(args.host)

    print("IA-180. Two arms, and the second one is the control.")
    print("The model is not asked to behave differently. The tool is removed.")

    try:
        true_arm, true_raised = arm(
            "arm 1: the tool is offered, so a call is possible",
            model=args.chat_model, transport=transport, with_tool=True)
        false_arm, false_raised = arm(
            "arm 2: NO tool is offered, so a call is impossible",
            model=args.chat_model, transport=transport, with_tool=False)
    except (oc.OllamaUnavailable, oc.OllamaRefused) as exc:
        print(f"\nthe server did not answer: {exc}", file=sys.stderr)
        return 3

    print("\n" + "-" * 70)
    problems = []
    if true_raised is not None:
        problems.append("arm 1 raised rather than reporting a tool call")
    elif true_arm is not True:
        problems.append(f"arm 1 read a real tool call as {true_arm}")

    if false_raised is not None:
        problems.append(
            f"arm 2 RAISED on a real no-tool-call response: {false_raised}. "
            "A real skip would have stopped the run instead of being counted")
    elif false_arm is not False:
        problems.append(
            f"arm 2 read a response that CANNOT contain a tool call as "
            f"{false_arm}. The detector cannot report a skip, so 0.0% was "
            "never capable of being anything else")

    if problems:
        print("CONTROL FAILED. The 0.0% skip rate is not publishable.")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("CONTROL HELD. Both arms, against the real server.")
    print("  A real tool call reads as retrieved, and a response that cannot")
    print("  contain one reads as not retrieved, without raising.")
    print("  The detector is capable of reporting a skip, so 0 of 12 is a fact")
    print("  about the model and not an artefact of this client.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
