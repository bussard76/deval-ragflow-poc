import base64
import json
from pathlib import Path

import fitz

from deval_ragflow.pdf import extract_pdf, extract_pdf_bytes


FIXTURE = Path(__file__).parent / "fixtures" / "golden.pdf"


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")


def make_pdf(*pages, encrypted=False):
    doc = fitz.open()
    for page_items in pages:
        page = doc.new_page(width=300, height=200)
        for kind, value in page_items:
            if kind == "text":
                page.insert_text((30, 50 + 25 * value[0]), value[1])
            elif kind == "image":
                page.insert_image(fitz.Rect(20, 20, 100, 100), stream=PNG)
    if encrypted:
        return doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="password")
    return doc.tobytes(garbage=4, deflate=False)


def test_golden_provenance_is_exact():
    extraction = extract_pdf(FIXTURE)
    expected = json.loads((FIXTURE.parent / "golden.expected.json").read_text())
    actual = extraction.as_dict()
    for key in ("byte_size", "canonical_text", "classification", "document_uid", "page_count", "sha256", "source_basename", "version_uid"):
        assert actual[key] == expected[key]
    assert [
        {key: passage[key] for key in ("bbox", "char_end", "char_start", "page_number", "paragraph_number", "section_path", "text")}
        for passage in actual["passages"]
    ] == expected["passages"]


def test_identity_ignores_source_path_but_parser_policy_changes_version(tmp_path):
    raw = FIXTURE.read_bytes()
    first = tmp_path / "one.pdf"
    second = tmp_path / "nested" / "two.pdf"
    second.parent.mkdir()
    first.write_bytes(raw)
    second.write_bytes(raw)
    a = extract_pdf(first)
    b = extract_pdf(second)
    assert a.document_uid == b.document_uid
    assert a.version_uid == b.version_uid
    assert a.source_basename == "one.pdf"
    assert b.source_basename == "two.pdf"
    assert extract_pdf_bytes(raw, parser_config={"chunk_token_num": 99}).version_uid != a.version_uid
    assert extract_pdf_bytes(raw + b"x").document_uid != a.document_uid


def test_classifies_malformed_empty_scanned_mixed_and_encrypted():
    assert extract_pdf_bytes(b"not a pdf").classification == "malformed"
    assert extract_pdf_bytes(make_pdf([("empty", None)])).classification == "empty"
    assert extract_pdf_bytes(make_pdf([("image", None)])).classification == "scanned"
    assert extract_pdf_bytes(make_pdf([("text", (0, "text")), ("image", None)])).classification == "mixed"
    assert extract_pdf_bytes(make_pdf([("text", (0, "secret"))], encrypted=True)).classification == "encrypted"


def test_limits_are_applied_before_remote_use():
    result = extract_pdf(FIXTURE, max_bytes=1)
    assert result.classification == "too_large"
    result = extract_pdf(FIXTURE, max_pages=1)
    assert result.classification == "too_many_pages"
    result = extract_pdf(FIXTURE, max_text_chars=2)
    assert result.classification == "too_large_text"
