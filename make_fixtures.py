"""
Build the fixture corpus the suite runs against.

The fixtures are generated rather than committed as binaries so that anyone can
see exactly what makes each one good or bad. Two of them are deliberately
wrong, and they reproduce a defect found in a real upstream corpus: a file
named .docx whose bytes are a PDF.

Run:  python make_fixtures.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def minimal_pdf(title: str) -> bytes:
    """A small but genuinely valid one-page PDF, with real xref offsets."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        None,  # content stream, built below
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = f"BT /F1 12 Tf 20 50 Td ({title}) Tj ET".encode("ascii")
    objects[3] = b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n").encode()
    return bytes(out)


def minimal_docx(paragraph: str) -> bytes:
    """A small but genuinely valid .docx: an OOXML package with a word/ part."""
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                   '</Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" '
                   'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
                   'Target="word/document.xml"/></Relationships>')
        z.writestr("word/document.xml",
                   '<?xml version="1.0" encoding="UTF-8"?>'
                   '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                   f'<w:body><w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p></w:body></w:document>')
    return buf.getvalue()


def build() -> list[Path]:
    good = FIXTURES / "good"
    bad = FIXTURES / "bad"
    for d in (good, bad):
        d.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    def write(path: Path, data: bytes) -> None:
        path.write_bytes(data)
        written.append(path)

    # --- files that must pass -------------------------------------------
    write(good / "quarterly-review.pdf", minimal_pdf("Quarterly review"))
    write(good / "meeting-notes.docx", minimal_docx("Attendees: two."))
    write(good / "team-handbook.md",
          b"# Team handbook\n\nOne short paragraph, so the file is not empty.\n")

    # --- files that must be rejected -------------------------------------
    # The reproduction. A real PDF, saved under a .docx name. This is the
    # shape of the two mislabelled files in the upstream sample corpus.
    write(bad / "meeting-notes-2025-01-08.docx", minimal_pdf("Notes, 8 Jan"))

    # The mirror of it, so the rule is not accidentally one-directional.
    write(bad / "handbook.pdf", minimal_docx("Not a PDF at all."))

    # A binary wearing a text extension.
    write(bad / "diagram.md", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01")

    # Nothing at all. An empty file must not read as a document with no content.
    write(bad / "empty.txt", b"")

    # --- not a document at all -------------------------------------------
    # The control subject for the bare-except scanner. It sits in fixtures/
    # rather than in bad/ because it is not a corpus file: the contract would
    # never pick up a .py, and the suite would then have to explain why this
    # one was accepted.
    write(FIXTURES / "has_bare_except.py",
          b"def read(path):\n"
          b"    try:\n"
          b"        return open(path).read()\n"
          b"    except:\n"
          b"        return f'[Error: Could not read file {path}]'\n")

    return written


if __name__ == "__main__":
    paths = build()
    for p in sorted(paths):
        print(f"{p.stat().st_size:7d} B  {p.relative_to(FIXTURES.parent)}")
    print(f"\n{len(paths)} fixtures written")
