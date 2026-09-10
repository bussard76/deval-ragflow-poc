"""Deterministic, local-only PDF validation and provenance extraction."""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

from .errors import DependencyError, ExtractionError
from .models import EXTRACTOR_VERSION, PROVENANCE_SCHEMA, DocumentExtraction, Passage

logger = logging.getLogger(__name__)

try:  # Keep --help usable when optional runtime wheels have not been installed yet.
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - exercised only in an uninstalled environment
    fitz = None


PathLike = Union[str, bytes, Path]


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def document_uid_for(raw: bytes) -> str:
    return sha256_bytes(raw)


def version_uid_for(document_uid: str, parser_config: dict[str, Any]) -> str:
    identity = {
        "document_uid": document_uid,
        "provenance_schema": PROVENANCE_SCHEMA,
        "extractor_version": EXTRACTOR_VERSION,
        "parser_config": parser_config,
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _normalise_block_text(value: str) -> str:
    value = (
        unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    )
    # PyMuPDF text blocks have a trailing newline. Trim only line-edge noise;
    # internal whitespace remains part of the provenance text and offsets.
    return "\n".join(line.rstrip() for line in value.split("\n")).strip()


def _round_bbox(block: Sequence[Any]) -> tuple[float, float, float, float]:
    try:
        values = tuple(round(float(block[i]), 2) for i in range(4))
    except (IndexError, TypeError, ValueError) as exc:
        raise ExtractionError("PDF text block has an invalid bounding box") from exc
    return tuple(0.0 if value == 0 else value for value in values)  # type: ignore


def _page_outline_paths(doc: Any, page_count: int) -> list[tuple[str, ...] | None]:
    try:
        toc = doc.get_toc(simple=True) or []
    except (RuntimeError, TypeError, ValueError):
        toc = []
    entries = []
    for item in toc:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        try:
            level, title, page = int(item[0]), str(item[1]).strip(), int(item[2])
        except (TypeError, ValueError):
            continue
        if level > 0 and title and page > 0:
            entries.append((level, title, page))
    paths: list[tuple[str, ...] | None] = []
    active: list[str] = []
    cursor = 0
    for page_number in range(1, page_count + 1):
        while cursor < len(entries) and entries[cursor][2] <= page_number:
            level, title, _ = entries[cursor]
            active = active[: max(0, level - 1)]
            active.append(title)
            cursor += 1
        paths.append(tuple(active) if active else None)
    return paths


def _error_extraction(
    document_uid: str,
    version_uid: str,
    sha256: str,
    source_basename: str,
    byte_size: int,
    parser_config: dict[str, Any],
    classification: str,
    message: str,
    page_count: int = 0,
) -> DocumentExtraction:
    return DocumentExtraction(
        document_uid=document_uid,
        version_uid=version_uid,
        sha256=sha256,
        source_basename=source_basename,
        classification=classification,
        page_count=page_count,
        byte_size=byte_size,
        canonical_text="",
        parser_config=parser_config,
        errors=[message],
    )


def extract_pdf_bytes(
    raw: bytes,
    source_name: str = "document.pdf",
    parser_config: dict[str, Any] | None = None,
    max_pages: int = 2000,
    max_bytes: int = 50 * 1024 * 1024,
    max_text_chars: int = 20 * 1000 * 1000,
) -> DocumentExtraction:
    """Extract text blocks and coordinates without retaining a source path."""
    if fitz is None:
        raise DependencyError(
            "PyMuPDF is required for PDF extraction; install the project dependencies"
        )
    if not isinstance(raw, bytes):
        raise TypeError("raw PDF data must be bytes")
    config = dict(parser_config or {})
    digest = sha256_bytes(raw)
    uid = document_uid_for(raw)
    version_uid = version_uid_for(uid, config)
    basename = Path(source_name).name or "document.pdf"
    if len(raw) > max_bytes:
        return _error_extraction(
            uid,
            version_uid,
            digest,
            basename,
            len(raw),
            config,
            "too_large",
            "PDF exceeds byte limit",
        )

    doc = None
    try:
        doc = fitz.open(stream=raw, filetype="pdf")
        is_encrypted = getattr(doc, "is_encrypted", False)
        needs_pass = getattr(doc, "needs_pass", False)
        if callable(is_encrypted):
            is_encrypted = is_encrypted()
        if callable(needs_pass):
            needs_pass = needs_pass()
        encrypted = bool(is_encrypted or needs_pass)
        if encrypted:
            return _error_extraction(
                uid,
                version_uid,
                digest,
                basename,
                len(raw),
                config,
                "encrypted",
                "PDF is encrypted or password-protected",
            )
        page_count = int(doc.page_count)
        if page_count > max_pages:
            return _error_extraction(
                uid,
                version_uid,
                digest,
                basename,
                len(raw),
                config,
                "too_many_pages",
                "PDF exceeds page limit",
                page_count,
            )
        outline = _page_outline_paths(doc, page_count)
        page_texts: list[str] = []
        page_classes: list[str] = []
        passages: list[Passage] = []
        for page_index in range(page_count):
            page = doc.load_page(page_index)
            blocks = page.get_text("blocks", sort=False) or []
            text_blocks = []
            for block_index, block in enumerate(blocks):
                if len(block) < 5:
                    continue
                block_type = (
                    int(block[6])
                    if len(block) > 6 and isinstance(block[6], (int, float))
                    else 0
                )
                if block_type != 0:
                    continue
                text = _normalise_block_text(str(block[4]))
                if text:
                    text_blocks.append((block_index, block, text))
            text_blocks.sort(
                key=lambda item: (float(item[1][1]), float(item[1][0]), item[0])
            )
            has_image = bool(page.get_images(full=True))
            if text_blocks:
                page_classes.append("mixed" if has_image else "text")
            else:
                page_classes.append("scanned" if has_image else "empty")
            page_text = "\n".join(item[2] for item in text_blocks)
            page_texts.append(page_text)

            page_offset = sum(len(previous) for previous in page_texts[:-1]) + max(
                0, page_index
            )  # form-feed per prior page
            running = page_offset
            for paragraph_number, (_, block, text) in enumerate(text_blocks, 1):
                start = running
                end = start + len(text)
                passage_identity = (
                    f"{uid}:{version_uid}:{page_index + 1}:{paragraph_number}:{start}"
                )
                passage_uid = hashlib.sha256(
                    passage_identity.encode("utf-8")
                ).hexdigest()
                passages.append(
                    Passage(
                        passage_uid=passage_uid,
                        document_uid=uid,
                        version_uid=version_uid,
                        page_number=page_index + 1,
                        paragraph_number=paragraph_number,
                        section_path=outline[page_index]
                        if page_index < len(outline)
                        else None,
                        bbox=_round_bbox(block),
                        char_start=start,
                        char_end=end,
                        text=text,
                    )
                )
                running = end + 1
        canonical_text = "\f".join(page_texts)
        if len(canonical_text) > max_text_chars:
            return DocumentExtraction(
                document_uid=uid,
                version_uid=version_uid,
                sha256=digest,
                source_basename=basename,
                classification="too_large_text",
                page_count=page_count,
                byte_size=len(raw),
                canonical_text=canonical_text,
                passages=passages,
                parser_config=config,
                errors=["PDF text exceeds character limit"],
                page_classifications=page_classes,
                raw_bytes=raw,
            )
        text_pages = sum(
            1 for value in page_classes if value == "text" or value == "mixed"
        )
        scanned_pages = sum(1 for value in page_classes if value == "scanned")
        if text_pages == 0:
            classification = "scanned" if scanned_pages else "empty"
        elif scanned_pages or "mixed" in page_classes:
            classification = "mixed"
        else:
            classification = "extracted"
        return DocumentExtraction(
            document_uid=uid,
            version_uid=version_uid,
            sha256=digest,
            source_basename=basename,
            classification=classification,
            page_count=page_count,
            byte_size=len(raw),
            canonical_text=canonical_text,
            passages=passages,
            parser_config=config,
            page_classifications=page_classes,
            raw_bytes=raw,
        )
    except (ExtractionError, DependencyError):
        raise
    except Exception as exc:  # noqa: BLE001 - PyMuPDF may raise implementation-specific errors
        return _error_extraction(
            uid,
            version_uid,
            digest,
            basename,
            len(raw),
            config,
            "malformed",
            f"unable to parse PDF: {exc}",
        )
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception as exc:  # noqa: BLE001 - preserve the parse result if cleanup fails
                logger.debug("failed to close PDF document: %s", exc)


def extract_pdf(
    path: PathLike,
    parser_config: dict[str, Any] | None = None,
    max_pages: int = 2000,
    max_bytes: int = 50 * 1024 * 1024,
    max_text_chars: int = 20 * 1000 * 1000,
) -> DocumentExtraction:
    if isinstance(path, bytes):
        raw = path
        source_name = "document.pdf"
    else:
        source_path = Path(path)
        source_name = source_path.name
        try:
            raw = source_path.read_bytes()
        except OSError as exc:
            # Keep diagnostics useful without echoing the absolute input path.
            detail = getattr(exc, "strerror", None)
            if not detail:
                detail = exc.__class__.__name__
            raise ExtractionError(f"cannot read PDF {source_name}: {detail}")
    return extract_pdf_bytes(
        raw, source_name, parser_config, max_pages, max_bytes, max_text_chars
    )


# Explicit aliases for callers that prefer validation/identity terminology.
validate_pdf = extract_pdf
extract_pdf_provenance = extract_pdf
compute_sha256 = sha256_bytes
compute_document_uid = document_uid_for
compute_version_uid = version_uid_for
