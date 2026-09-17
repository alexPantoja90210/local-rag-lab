"""
The ingest contract.

One job: decide whether a file may enter the knowledge base, and say why not
when it may not.

Two rules shape this module, and both come from defects found in the upstream
repositories before a line of this was written.

    1. A file is what its bytes say, not what its name says.
       Upstream ships two files named .docx that are PDFs, and routes by
       extension alone.

    2. Deciding and reading are separate functions, and the deciding one
       never returns text.
       Upstream, a conversion failure falls back to a raw read, and when that
       fails too a bare `except` returns the string
       "[Error: Could not read file <name>]" AS THE DOCUMENT'S CONTENT. It is
       then chunked, embedded and retrievable, and the ingest reports success.
       A function that cannot return content cannot fabricate it, so this one
       returns a verdict and nothing else.

No third-party dependencies. The sniffing is visible on this page on purpose:
the same reason the hashed-bag-of-words backend stayed in the previous project.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

# --------------------------------------------------------------------------
# what a kind is
# --------------------------------------------------------------------------

# The families the pipeline can actually read. A file is accepted only when
# the family its bytes belong to is the family its extension promised.
EXTENSION_FAMILY = {
    ".pdf": "pdf",
    ".docx": "ooxml-word",
    ".pptx": "ooxml-presentation",
    ".xlsx": "ooxml-sheet",
    ".doc": "ole2",
    ".ppt": "ole2",
    ".xls": "ole2",
    ".html": "html",
    ".htm": "html",
    ".md": "text",
    ".markdown": "text",
    ".txt": "text",
    ".mp3": "mp3",
    ".wav": "wav",
    ".flac": "flac",
    ".m4a": "m4a",
}

SUPPORTED_EXTENSIONS = tuple(sorted(EXTENSION_FAMILY))

MIN_BYTES = 1


class ConversionFailed(Exception):
    """A reader could not produce text. It is raised, never returned."""


@dataclass(frozen=True)
class Verdict:
    """The result of the contract. Deliberately carries no document text."""

    path: Path
    declared: str          # family the extension promised, or "unknown"
    detected: str          # family the bytes belong to, or "unknown"
    ok: bool
    reason: str            # empty when ok

    def line(self) -> str:
        mark = "ok    " if self.ok else "REJECT"
        return f"{mark}  {self.path.name:<42s} {self.reason}"


# --------------------------------------------------------------------------
# sniffing: what the bytes say
# --------------------------------------------------------------------------

OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def _ooxml_family(path: Path) -> str:
    """Distinguish docx from pptx from xlsx by what the zip actually holds."""
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
    except (zipfile.BadZipFile, OSError):
        return "zip-unreadable"
    if any(n.startswith("word/") for n in names):
        return "ooxml-word"
    if any(n.startswith("ppt/") for n in names):
        return "ooxml-presentation"
    if any(n.startswith("xl/") for n in names):
        return "ooxml-sheet"
    return "zip-other"


def _looks_like_text(head: bytes) -> bool:
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        # A multi-byte character may straddle the cut, so retry one short.
        try:
            head[:-3].decode("utf-8")
        except UnicodeDecodeError:
            return False
    return True


def detect_family(path: Path) -> str:
    """Return the family the file's bytes belong to, or 'unknown'.

    Signatures only, no guessing from the name. A file this cannot place is
    'unknown', which the contract treats as a rejection rather than as a maybe.
    """
    with open(path, "rb") as f:
        head = f.read(4096)

    if len(head) < MIN_BYTES:
        return "empty"

    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        return _ooxml_family(path)
    if head.startswith(OLE2_MAGIC):
        return "ole2"
    if head.startswith(b"fLaC"):
        return "flac"
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return "wav"
    if head[4:8] == b"ftyp":
        return "m4a"
    if head.startswith(b"ID3") or (head[0:1] == b"\xff" and (head[1] & 0xE0) == 0xE0):
        return "mp3"

    if _looks_like_text(head):
        stripped = head.lstrip()[:512].lower()
        if stripped.startswith(b"<!doctype html") or stripped.startswith(b"<html"):
            return "html"
        return "text"

    return "unknown"


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------

def check_file(path: Path, *, enforce_extension: bool = True) -> Verdict:
    """Decide whether one file may enter the knowledge base.

    `enforce_extension=False` exists for one reason: the suite turns the rule
    off to prove the check can fail. A check that cannot fail is not a check.
    Nothing else may call it with False.
    """
    path = Path(path)
    declared = EXTENSION_FAMILY.get(path.suffix.lower(), "unknown")

    if declared == "unknown":
        return Verdict(path, declared, "unread", False,
                       f"the extension '{path.suffix}' is not one this pipeline reads")

    if not path.is_file():
        return Verdict(path, declared, "unread", False, "not a file")

    if path.stat().st_size < MIN_BYTES:
        return Verdict(path, declared, "empty", False, "the file is empty")

    detected = detect_family(path)

    if detected == "empty":
        return Verdict(path, declared, detected, False, "the file is empty")

    if not enforce_extension:
        return Verdict(path, declared, detected, True, "")

    if detected != declared:
        return Verdict(
            path, declared, detected, False,
            f"the name says {declared}, the bytes say {detected}",
        )

    return Verdict(path, declared, detected, True, "")


def check_folder(folder: Path, *, enforce_extension: bool = True) -> list[Verdict]:
    """Check every file the pipeline would pick up, in a stable order."""
    folder = Path(folder)
    paths: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTENSION_FAMILY:
            paths.append(p)
    return [check_file(p, enforce_extension=enforce_extension) for p in paths]


# --------------------------------------------------------------------------
# reading, kept apart from deciding
# --------------------------------------------------------------------------

def read_or_refuse(path: Path, reader: Callable[[Path], str]) -> str:
    """Run `reader`, or raise. It never invents a stand-in for the text.

    This is the whole of rule 2. There is no `except` here that returns a
    string, because the moment one exists, an unreadable file becomes a
    document whose content is the sentence describing why it could not be read.
    """
    path = Path(path)
    verdict = check_file(path)
    if not verdict.ok:
        raise ConversionFailed(f"{path.name}: {verdict.reason}")

    try:
        text = reader(path)
    except Exception as exc:  # re-raised, never swallowed, never substituted
        raise ConversionFailed(f"{path.name}: the reader failed: {exc!r}") from exc

    if not isinstance(text, str) or not text.strip():
        raise ConversionFailed(f"{path.name}: the reader returned nothing usable")

    return text


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def report(verdicts: Iterable[Verdict]) -> int:
    verdicts = list(verdicts)
    rejected = [v for v in verdicts if not v.ok]

    for v in verdicts:
        print(v.line())

    print()
    print(f"checked  {len(verdicts)}")
    print(f"accepted {len(verdicts) - len(rejected)}")
    print(f"rejected {len(rejected)}")

    if rejected:
        print()
        print("Nothing was ingested. A corpus that half-loaded must not look like one that loaded.")
        for v in rejected:
            print(f"  - {v.path.name}: {v.reason}")
    return 1 if rejected else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Refuse any document whose extension lies about its bytes.")
    parser.add_argument("--documents", default="documents",
                        help="folder to check (default: documents)")
    args = parser.parse_args(argv)

    folder = Path(args.documents)
    if not folder.is_dir():
        print(f"no such folder: {folder}", file=sys.stderr)
        return 2

    print(f"ingest contract, over {folder}")
    print(f"readable extensions: {' '.join(SUPPORTED_EXTENSIONS)}")
    print()
    return report(check_folder(folder))


if __name__ == "__main__":
    raise SystemExit(main())
