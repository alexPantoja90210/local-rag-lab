"""
The only part of this repository that talks to a server, and the only part I
could not run.

Ollama listens on the machine that owns the models. This module is written from
the cloud side of a bridge that cannot reach it: `localhost:11434` does not
answer there and egress to the host is blocked. So every number slice 4
produces comes out of the operator's own terminal, which is the arrangement the
run documentation standard has been asking for since the first runbook. It is
recorded here because a constraint that is only in somebody's head is a
constraint that gets forgotten.

That leaves a problem worth solving rather than apologising for. A client nobody
can exercise is a client nobody has tested, and "it worked when I ran it" is the
weakest evidence in the portfolio. So the transport is an argument. Every
function here takes a callable that turns a path and a payload into a response,
the real one speaks HTTP over the standard library, and the suite passes fakes
that return exactly what a server would. The request shaping, the response
parsing and every refusal below are covered by invariants that need no server,
no model and no network.

Three refusals, and the third is the one that protects the measurement
----------------------------------------------------------------------

**The window is read off the model or it is not known.** `window()` looks for
the context length the server reports and raises when it is absent. It does not
fall back to 512. A default here would be the 3.5 characters-per-token constant
all over again: a number that is right often enough to never be questioned, in a
project whose one finding so far is what that costs.

**A vector of width zero is not an embedding.** Checked at the boundary, because
by the time it reaches the store the error is about a reshape.

**A response that cannot be parsed is not evidence of anything.** This is the
one that matters. The number slice 4 exists to produce is the share of turns
where the model never retrieved. If a malformed or unexpected response were
quietly read as "no tool call", every timeout, every version skew and every
parsing bug would be counted as a finding, and the headline number would grow
whenever the client broke. `decided_to_retrieve()` raises instead. **A
measurement that cannot tell a discovery from a bug is not a measurement.**

No dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Sequence

DEFAULT_HOST = "http://localhost:11434"

# (path, payload) -> decoded json
Transport = Callable[[str, dict], Any]


class OllamaUnavailable(Exception):
    """The server could not be reached or answered with an error."""


class OllamaRefused(Exception):
    """The server answered, and the answer cannot be used. Never guessed at."""


def http_transport(host: str = DEFAULT_HOST, timeout: float = 300.0) -> Transport:
    """The real transport. Standard library only, no streaming."""

    def post(path: str, payload: dict) -> Any:
        url = f"{host.rstrip('/')}{path}"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise OllamaUnavailable(
                f"{url} answered {exc.code}. If this is 404 the endpoint does "
                f"not exist in this Ollama version; run `ollama --version` and "
                f"say which it is rather than working around it") from exc
        except OSError as exc:
            raise OllamaUnavailable(
                f"{url} could not be reached: {exc}. Ollama has to be running "
                f"on the machine this script runs on") from exc
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaRefused(
                f"{url} answered with something that is not JSON, first 200 "
                f"bytes: {raw[:200]!r}") from exc

    return post


# ---------------------------------------------------------------------------
# what the model says about itself
# ---------------------------------------------------------------------------

def show(model: str, *, transport: Transport) -> dict:
    data = transport("/api/show", {"model": model})
    if not isinstance(data, dict):
        raise OllamaRefused(f"/api/show returned {type(data).__name__}, not an object")
    return data


def window(model: str, *, transport: Transport) -> int:
    """The context length the server reports for this model.

    The key is namespaced by architecture, `bert.context_length` for one model
    and something else for the next, so it is found by suffix rather than by a
    name written here. A key written here would be a constant again.
    """
    info = show(model, transport=transport).get("model_info")
    if not isinstance(info, dict):
        raise OllamaRefused(
            f"the server reported no model_info for {model}, so its window is "
            "not known. It is not assumed")

    found = {k: v for k, v in info.items()
             if k.endswith(".context_length") and isinstance(v, int)}
    if not found:
        raise OllamaRefused(
            f"the server reported model_info for {model} with no "
            f"*.context_length in it, so its window is not known. It is not "
            f"assumed, and 512 is not a safe guess: the keys present are "
            f"{sorted(info)[:12]}")
    if len(found) > 1:
        raise OllamaRefused(
            f"{model} reports more than one context length {found}, and "
            "picking one would be a coin toss recorded as a measurement")

    value = next(iter(found.values()))
    if value < 1:
        raise OllamaRefused(f"{model} reports a context length of {value}")
    return value


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

EMBED_NEW = "/api/embed"
EMBED_OLD = "/api/embeddings"


class OllamaEmbedder:
    """A real embedder, satisfying the same protocol as the suite's stub.

    Two endpoints exist across Ollama versions and this tries the newer one
    first. The fallback is **recorded and reported**, not silent: which endpoint
    answered goes into the run record, because a fallback nobody sees is how a
    client keeps working for two years against an API that changed underneath it.
    """

    def __init__(self, model: str, *, transport: Transport) -> None:
        self.model = model
        self._transport = transport
        self.endpoint: str | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def embed(self, text: str) -> list[float]:
        vector, endpoint = self._embed_once(text)
        if self.endpoint is None:
            self.endpoint = endpoint
        elif self.endpoint != endpoint:
            raise OllamaRefused(
                f"this embedder answered on {self.endpoint} and then on "
                f"{endpoint}. Two endpoints in one run means two code paths "
                "produced the vectors in one store")
        if not vector:
            raise OllamaRefused(
                f"{self.name} returned a vector of width 0 for {text[:40]!r}")
        return [float(x) for x in vector]

    def _embed_once(self, text: str) -> tuple[Sequence[float], str]:
        try:
            data = self._transport(EMBED_NEW, {"model": self.model, "input": text})
            return self._unwrap_new(data), EMBED_NEW
        except OllamaUnavailable:
            data = self._transport(EMBED_OLD, {"model": self.model, "prompt": text})
            return self._unwrap_old(data), EMBED_OLD

    @staticmethod
    def _unwrap_new(data: Any) -> Sequence[float]:
        if not isinstance(data, dict) or "embeddings" not in data:
            raise OllamaRefused(f"{EMBED_NEW} returned no embeddings field")
        rows = data["embeddings"]
        if not isinstance(rows, list) or len(rows) != 1:
            raise OllamaRefused(
                f"{EMBED_NEW} returned {len(rows) if isinstance(rows, list) else '?'} "
                "vectors for one input")
        return rows[0]

    @staticmethod
    def _unwrap_old(data: Any) -> Sequence[float]:
        if not isinstance(data, dict) or "embedding" not in data:
            raise OllamaRefused(f"{EMBED_OLD} returned no embedding field")
        return data["embedding"]


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def chat(
    model: str,
    messages: Sequence[dict],
    *,
    transport: Transport,
    tools: Sequence[dict] | None = None,
    temperature: float = 0.0,
) -> dict:
    """One turn, not streamed, temperature 0 unless something says otherwise.

    Temperature is stated on every call rather than left to the server's
    default, because a run whose temperature came from a config file somewhere
    is a run that cannot be repeated by somebody reading the script.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": False,
        "options": {"temperature": temperature},
    }
    if tools:
        payload["tools"] = list(tools)

    data = transport("/api/chat", payload)
    if not isinstance(data, dict) or "message" not in data:
        raise OllamaRefused(
            f"/api/chat returned no message field, keys were "
            f"{sorted(data) if isinstance(data, dict) else type(data).__name__}")
    if data.get("done") is False:
        raise OllamaRefused(
            "/api/chat returned an unfinished response, so this turn is a "
            "fragment. It is not read as an answer")
    return data


RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "Search the knowledge base and return passages with their ids.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, as a standalone question.",
                },
            },
            "required": ["query"],
        },
    },
}


def decided_to_retrieve(response: dict) -> bool:
    """Did the model call the retrieve tool on this turn? Or refuse to say.

    This is the one function whose failure mode would corrupt the finding. The
    quantity being measured is how often the model answers without looking, so
    anything unreadable must NOT come back as False. A client that reads its own
    bugs as evidence produces a number that rises when the client breaks, and it
    would rise in the direction that makes the finding look stronger.
    """
    if not isinstance(response, dict) or "message" not in response:
        raise OllamaRefused(
            "no message in the response, so whether it retrieved is unknown. "
            "Unknown is not the same as no")
    message = response["message"]
    if not isinstance(message, dict):
        raise OllamaRefused(
            f"the message is a {type(message).__name__}, so whether it "
            "retrieved is unknown. Unknown is not the same as no")

    calls = message.get("tool_calls")
    if calls is None:
        return False
    if not isinstance(calls, list):
        raise OllamaRefused(
            f"tool_calls is a {type(calls).__name__}, not a list. Unknown is "
            "not the same as no")
    if not calls:
        return False

    for call in calls:
        if not isinstance(call, dict):
            raise OllamaRefused("a tool call is not an object. Unknown is not the same as no")
        name = (call.get("function") or {}).get("name")
        if name == "retrieve":
            return True
    return False


def tool_query(response: dict) -> str:
    """The query the model asked for, or a refusal. Never a silent substitution.

    Falling back to the user's original question here would hide the case where
    the model calls the tool with nothing, which is a different behaviour from
    calling it properly and would be averaged into the same number.
    """
    for call in response.get("message", {}).get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") != "retrieve":
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise OllamaRefused(
                    f"the model called retrieve with arguments that are not "
                    f"JSON: {arguments[:120]!r}") from exc
        if not isinstance(arguments, dict) or not str(arguments.get("query", "")).strip():
            raise OllamaRefused(
                "the model called retrieve with no query. That is not the same "
                "as not calling it, and it is not the same as asking the "
                "original question")
        return str(arguments["query"])
    raise OllamaRefused("no retrieve call in this response")
