"""
The store: two files, and a search that compares against everything.

What a vector store has to do here is small. Keep N vectors and the identity of
the chunk each one came from; given a query vector, return the closest k with
their scores. That is a dot product and a sort.

Version 2 of the target architecture put this in Postgres with pgvector, and
version 3 does not. The reasons are on that page and they are measured, not
preferred: at this corpus size an exact comparison against every vector takes
under a millisecond, the previous project already measured that its approximate
index returned identical top-5 lists on 21 of 21 queries, and Chroma resolves to
79 packages in a repository that requires none.

So a database is the sophisticated option that has to earn its place, the same
call as keeping the hashed bag-of-words embedding in the previous project. When
the corpus stops fitting in memory, that is what says so.

Three things are kept from the pgvector design, because none of them was ever
about Postgres.

    1. **The score reaches the caller.** Both upstream projects compute a
       similarity and discard it one line later, which is why neither can
       decline for a reason it can name.

    2. **The chunk's identity travels with its vector.** Not a row id: the
       document, the character span, the token count and the split level.

    3. **The packing fingerprint sits beside the data, and a query that does
       not match it is REFUSED.** A store built with one embedder and queried
       with another returns plausible results and is wrong, and it looks
       exactly like a store that works. That is the IA-160 family and it is
       the reason this module exists rather than a dict.

No required dependencies. Vectors are written as raw float32 with the standard
library's `array`. If numpy is installed the search uses it and is faster; the
pure-Python path stays, and an invariant asserts the two agree. A fast path
nobody can check against a slow one is a fast path nobody can check.
"""

from __future__ import annotations

import array
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

FORMAT_VERSION = 1

# A vector whose length is not 1 makes the dot product something other than a
# cosine, quietly. Float32 round-tripping moves the norm a little, so the check
# has a tolerance rather than demanding exactness.
NORM_TOLERANCE = 1e-4

try:  # optional, and the suite runs without it
    import numpy as _np
except ImportError:  # pragma: no cover - exercised by whichever machine lacks it
    _np = None

HAVE_NUMPY = _np is not None


class StoreInvalid(Exception):
    """The store on disk cannot be trusted. Raised, never worked around."""


class StoreMismatch(Exception):
    """The query was built by a different packing than the store. Refused."""


# ---------------------------------------------------------------------------
# what is kept per vector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Record:
    """The identity of one chunk. Deliberately carries no vector.

    Everything here is what a citation will eventually need. A row id would be
    enough to retrieve the text and useless for saying where it came from.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    split_level: str
    text: str


@dataclass(frozen=True)
class Hit:
    record: Record
    score: float
    rank: int


@dataclass(frozen=True)
class SearchResult:
    """The outcome of a search, including why it is empty when it is empty."""

    hits: tuple[Hit, ...]
    considered: int          # vectors compared, always the whole store
    threshold: float | None
    above_threshold: int
    store_path: str
    fingerprint: str
    backend: str

    @property
    def declined(self) -> bool:
        return not self.hits

    def reason(self) -> str:
        """Why there is nothing, in terms a person can act on.

        The previous project shipped a message blaming a threshold that was
        switched off, and it sent someone looking in the wrong place for an
        afternoon (IA-163). So this never guesses: it reports what the store
        holds, what was compared and what the cutoff was.
        """
        if self.hits:
            return ""
        if self.considered == 0:
            return (f"the store at {self.store_path} holds no vectors at all, "
                    f"so nothing was compared")
        if self.threshold is None:
            return (f"{self.considered} vectors were compared and k was 0, "
                    f"so nothing was asked for")
        return (f"{self.considered} vectors were compared and none scored above "
                f"{self.threshold}; the store at {self.store_path} holds "
                f"{self.considered} vectors under packing {self.fingerprint}")


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vector))


def save(
    path: Path | str,
    records: Sequence[Record],
    vectors: Sequence[Sequence[float]],
    *,
    fingerprint: str,
    embedder: str,
) -> dict:
    """Write the store as two files, or refuse.

    `<path>.vec`   raw little-endian float32, count x dim
    `<path>.json`  the manifest and the records

    Two files rather than one because the vectors are the only part that has to
    be read fast, and keeping the metadata as readable JSON means the store can
    be inspected without this module.
    """
    path = Path(path)
    if len(records) != len(vectors):
        raise StoreInvalid(
            f"{len(records)} records against {len(vectors)} vectors; "
            "every vector must carry the identity of its chunk")
    if not fingerprint:
        raise StoreInvalid(
            "a store must record the packing that built it, or nothing can "
            "detect a query from a different embedder")

    dim = len(vectors[0]) if vectors else 0
    if dim == 0 and vectors:
        raise StoreInvalid("a vector of width 0 cannot be searched")

    flat = array.array("f")
    for i, vector in enumerate(vectors):
        if len(vector) != dim:
            raise StoreInvalid(
                f"vector {i} has width {len(vector)}, the first has {dim}; "
                "a store cannot hold two widths")
        n = _norm(vector)
        if abs(n - 1.0) > NORM_TOLERANCE:
            raise StoreInvalid(
                f"vector {i} for chunk {records[i].chunk_id!r} has length "
                f"{n:.6f}, not 1. The dot product would not be a cosine, and "
                "the scores would be wrong in a way nothing downstream could see")
        flat.extend(float(x) for x in vector)

    seen: set[str] = set()
    for r in records:
        if r.chunk_id in seen:
            raise StoreInvalid(
                f"chunk id {r.chunk_id!r} appears twice; a content-derived id "
                "repeating means the same chunk was added twice")
        seen.add(r.chunk_id)

    manifest = {
        "format_version": FORMAT_VERSION,
        "embedder": embedder,
        "fingerprint": fingerprint,
        "dim": dim,
        "count": len(records),
        "records": [asdict(r) for r in records],
    }

    vec_path, json_path = _paths(path)
    vec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(vec_path, "wb") as f:
        flat.tofile(f)
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    return manifest


def _paths(path: Path) -> tuple[Path, Path]:
    path = Path(path)
    return path.with_suffix(".vec"), path.with_suffix(".json")


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def load(path: Path | str) -> "Store":
    """Read a store, or refuse. It never returns a partially-loaded one."""
    vec_path, json_path = _paths(Path(path))
    if not json_path.is_file():
        raise StoreInvalid(f"no manifest at {json_path}")
    if not vec_path.is_file():
        raise StoreInvalid(f"no vectors at {vec_path}")

    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise StoreInvalid(
            f"{json_path} is format version {manifest.get('format_version')!r}, "
            f"this module reads {FORMAT_VERSION}")

    dim = int(manifest["dim"])
    count = int(manifest["count"])
    records = tuple(Record(**r) for r in manifest["records"])

    if len(records) != count:
        raise StoreInvalid(
            f"the manifest says {count} records and carries {len(records)}")

    expected_bytes = count * dim * 4
    actual_bytes = vec_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise StoreInvalid(
            f"{vec_path} is {actual_bytes} bytes; {count} vectors of width "
            f"{dim} would be {expected_bytes}. The two files disagree, so "
            "neither can be trusted")

    raw = vec_path.read_bytes()
    return Store(path=str(Path(path)), records=records, raw=raw, dim=dim,
                 fingerprint=manifest["fingerprint"], embedder=manifest["embedder"])


class Store:
    """An opened store. Searching it compares against every vector it holds."""

    def __init__(self, *, path: str, records: tuple[Record, ...], raw: bytes,
                 dim: int, fingerprint: str, embedder: str) -> None:
        self.path = path
        self.records = records
        self.dim = dim
        self.fingerprint = fingerprint
        self.embedder = embedder
        self._raw = raw
        self._np_matrix = None
        if HAVE_NUMPY and records:
            self._np_matrix = _np.frombuffer(raw, dtype="<f4").reshape(len(records), dim)

    @property
    def count(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (f"<Store {self.path} {self.count} vectors dim {self.dim} "
                f"packing {self.fingerprint}>")

    # -- scoring, twice -----------------------------------------------------

    def _scores_python(self, query: Sequence[float]) -> list[float]:
        """The readable one. No dependencies, and it is the reference."""
        flat = array.array("f")
        flat.frombytes(self._raw)
        dim = self.dim
        out = []
        for i in range(self.count):
            base = i * dim
            total = 0.0
            for j in range(dim):
                total += flat[base + j] * query[j]
            out.append(total)
        return out

    def _scores_numpy(self, query: Sequence[float]) -> list[float]:
        """The fast one. Same arithmetic, one call."""
        q = _np.asarray(query, dtype="<f4")
        return (self._np_matrix @ q).tolist()

    def scores(self, query: Sequence[float], *, backend: str = "auto") -> list[float]:
        if self.count == 0:
            return []
        if backend == "auto":
            backend = "numpy" if self._np_matrix is not None else "python"
        if backend == "numpy":
            if self._np_matrix is None:
                raise StoreInvalid("numpy is not available in this environment")
            return self._scores_numpy(query)
        if backend == "python":
            return self._scores_python(query)
        raise ValueError(f"unknown backend {backend!r}")

    # -- the search ---------------------------------------------------------

    def search(
        self,
        query: Sequence[float],
        *,
        k: int = 5,
        fingerprint: str,
        threshold: float | None = None,
        backend: str = "auto",
    ) -> SearchResult:
        """Return the closest k, with scores, or say why there are none.

        `fingerprint` is required and is not a convenience. The caller has to
        state the packing its query vector was produced with, and a mismatch
        is refused rather than answered. A store built for one embedder and
        queried with another returns results that look entirely normal.
        """
        if fingerprint != self.fingerprint:
            raise StoreMismatch(
                f"the store at {self.path} was built under packing "
                f"{self.fingerprint} by {self.embedder}, and the query was "
                f"produced under {fingerprint}. Nothing was searched: the "
                "scores would have been meaningless and would have looked fine")
        if k < 0:
            raise ValueError("k cannot be negative")
        if self.count and len(query) != self.dim:
            raise StoreInvalid(
                f"the query has width {len(query)} and the store holds vectors "
                f"of width {self.dim}")
        if self.count:
            n = _norm(query)
            if abs(n - 1.0) > NORM_TOLERANCE:
                raise StoreInvalid(
                    f"the query vector has length {n:.6f}, not 1, so the "
                    "scores would not be cosines")

        used = backend
        if used == "auto":
            used = "numpy" if self._np_matrix is not None else "python"

        raw_scores = self.scores(query, backend=backend)
        pairs = list(zip(raw_scores, self.records))
        if threshold is not None:
            kept = [(s, r) for s, r in pairs if s >= threshold]
        else:
            kept = pairs

        # Ties break on chunk id, so the same store and query always produce
        # the same list. A search whose order depends on insertion is a search
        # whose results cannot be compared between runs.
        kept.sort(key=lambda pair: (-pair[0], pair[1].chunk_id))

        hits = tuple(
            Hit(record=r, score=float(s), rank=i + 1)
            for i, (s, r) in enumerate(kept[:k])
        )
        return SearchResult(
            hits=hits,
            considered=self.count,
            threshold=threshold,
            above_threshold=len(kept),
            store_path=self.path,
            fingerprint=self.fingerprint,
            backend=used,
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def normalise(vector: Sequence[float]) -> list[float]:
    """Scale a vector to length 1, or refuse. Never returns a zero vector."""
    n = _norm(vector)
    if n == 0.0:
        raise StoreInvalid("a zero vector has no direction and cannot be stored")
    return [float(x) / n for x in vector]


def records_from_chunks(chunks: Iterable) -> list[Record]:
    """Turn chunker.Chunk objects into store records, keeping the identity."""
    out = []
    for c in chunks:
        out.append(Record(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            chunk_index=c.chunk_index,
            char_start=c.char_start,
            char_end=c.char_end,
            token_count=c.token_count,
            split_level=c.split_level,
            text=c.text,
        ))
    return out
