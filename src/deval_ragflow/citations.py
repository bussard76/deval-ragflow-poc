"""Deterministic citation resolution; no LLM or filename guessing."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, cast

from .models import Citation, Passage

_whitespace = re.compile(r"\s+", re.UNICODE)
_tokens = re.compile(r"\w+", re.UNICODE)


def normalize_text(value: Any) -> str:
    return _whitespace.sub(
        " ", unicodedata.normalize("NFKC", str(value or "")).casefold()
    ).strip()


def token_dice(left: str, right: str) -> float:
    left_tokens = _tokens.findall(normalize_text(left))
    right_tokens = _tokens.findall(normalize_text(right))
    if not left_tokens or not right_tokens:
        return 0.0
    a, b = Counter(left_tokens), Counter(right_tokens)
    common = sum((a & b).values())
    return (2.0 * common) / (len(left_tokens) + len(right_tokens))


def _remote_id(chunk: dict[str, Any]) -> str | None:
    for key in ("document_id", "doc_id", "documentId"):
        value = chunk.get(key)
        if value:
            return str(value)
    return None


def _position_values(chunk: dict[str, Any]) -> list[Sequence[Any]]:
    positions = (
        chunk.get("_pdf_positions")
        or chunk.get("positions")
        or chunk.get("position_int")
    )
    if isinstance(positions, dict):
        positions = (
            positions.get("positions")
            or positions.get("position")
            or positions.get("position_int")
            or positions.get("_pdf_positions")
        )
    if isinstance(positions, (list, tuple)):
        if positions and isinstance(positions[0], (list, tuple)):
            return [item for item in positions if isinstance(item, (list, tuple))]
        if positions and all(
            not isinstance(item, (list, tuple, dict)) for item in positions
        ):
            return [positions]
    if all(
        chunk.get(key) is not None
        for key in ("page_number", "x0", "x1", "top", "bottom")
    ):
        return [[chunk[key] for key in ("page_number", "x0", "x1", "top", "bottom")]]
    return []


class CitationResolver:
    def __init__(self, registry: Any, threshold: float = 0.60):
        self.registry = registry
        self.threshold = threshold

    def _unresolved(
        self, remote_id: str | None, reason: str, matched_text: str = ""
    ) -> Citation:
        return Citation(
            passage_uid=None,
            document_uid=None,
            version_uid=None,
            page_number=None,
            paragraph_number=None,
            char_start=None,
            char_end=None,
            method="unresolved",
            confidence=0.0,
            matched_text=matched_text,
            remote_document_id=remote_id,
            reason=reason,
        )

    @staticmethod
    def _from_passage(
        passage: Passage, method: str, confidence: float, remote_id: str, text: str = ""
    ) -> Citation:
        return Citation(
            passage_uid=passage.passage_uid,
            document_uid=passage.document_uid,
            version_uid=passage.version_uid,
            page_number=passage.page_number,
            paragraph_number=passage.paragraph_number,
            char_start=passage.char_start,
            char_end=passage.char_end,
            method=method,
            confidence=round(confidence, 6),
            matched_text=text or passage.text,
            remote_document_id=remote_id,
            section_path=passage.section_path,
            bbox=passage.bbox,
        )

    def _mapping(self, remote_id: str, dataset_scope: str | None = None):
        finder = getattr(self.registry, "mappings_for_remote_id", None)
        if finder is not None:
            mappings = finder(remote_id)
            if dataset_scope:
                mappings = [
                    item for item in mappings if item.dataset_scope == dataset_scope
                ]
            if len(mappings) == 1:
                return mappings[0]
            return None
        mapping = self.registry.get_mapping_by_remote_id(remote_id)
        if (
            mapping is not None
            and dataset_scope
            and mapping.dataset_scope != dataset_scope
        ):
            return None
        return mapping

    @staticmethod
    def _position_match(
        passages: Sequence[Passage], chunk: dict[str, Any]
    ) -> Passage | None:
        """Resolve v0.27.2 PDF positions, with a four-value legacy fallback.

        RAGFlow's canonical PDF position is ``[page, left, right, top,
        bottom]``.  The local registry stores PyMuPDF's ``[x0, y0, x1,
        y1]``; coordinate matching is intentionally tolerant because the
        server and local extractor can round coordinates differently.
        """
        reference_page = chunk.get("page_number")
        try:
            reference_page = int(reference_page) if reference_page is not None else None
        except (TypeError, ValueError):
            reference_page = None
        if reference_page is not None and reference_page <= 0:
            reference_page += 1
        for position in _position_values(chunk):
            if len(position) < 2:
                continue
            page_value = position[0]
            if isinstance(page_value, (list, tuple)) and page_value:
                page_value = page_value[-1]
            try:
                page = int(cast(Any, page_value))
            except (TypeError, ValueError):
                continue
            if reference_page is not None and page == reference_page - 1:
                page = reference_page
            elif page <= 0:
                page += 1
            if len(position) >= 5:
                try:
                    left, right, top, bottom = (float(position[i]) for i in range(1, 5))
                except (TypeError, ValueError):
                    continue
                left, right = sorted((left, right))
                top, bottom = sorted((top, bottom))
                if right <= left or bottom <= top:
                    continue
                scored = []
                for passage in passages:
                    if passage.page_number != page:
                        continue
                    x0, y0, x1, y1 = passage.bbox
                    overlap_x = max(0.0, min(x1, right) - max(x0, left))
                    overlap_y = max(0.0, min(y1, bottom) - max(y0, top))
                    overlap = overlap_x * overlap_y
                    if overlap <= 0:
                        continue
                    passage_area = max(0.001, (x1 - x0) * (y1 - y0))
                    position_area = max(0.001, (right - left) * (bottom - top))
                    score = overlap / min(passage_area, position_area)
                    scored.append((score, passage))
                if scored:
                    scored.sort(
                        key=lambda item: (
                            -item[0],
                            item[1].paragraph_number,
                            item[1].passage_uid,
                        )
                    )
                    if len(scored) == 1 or scored[0][0] > scored[1][0] + 1e-6:
                        return scored[0][1]
                continue

            # Older adapters/tests used [page, paragraph, start, end].
            try:
                paragraph = int(position[1])
            except (TypeError, ValueError):
                continue
            candidates = [
                p
                for p in passages
                if p.page_number == page and p.paragraph_number == paragraph
            ]
            if len(position) >= 4:
                try:
                    start, end = int(position[2]), int(position[3])
                except (TypeError, ValueError):
                    start = end = -1
                if start >= 0 and end >= start:
                    candidates = [
                        p
                        for p in candidates
                        if p.char_start <= start and end <= p.char_end
                    ]
            if len(candidates) == 1:
                return candidates[0]
        return None

    def resolve(
        self, chunk: dict[str, Any], dataset_scope: str | None = None
    ) -> Citation:
        if not isinstance(chunk, dict):
            return self._unresolved(None, "retrieval reference is not an object")
        remote_id = _remote_id(chunk)
        if not remote_id:
            return self._unresolved(None, "reference has no remote document id")
        mapping = self._mapping(remote_id, dataset_scope)
        if mapping is None:
            return self._unresolved(
                remote_id, "unknown or ambiguous remote document id"
            )
        extraction = self.registry.get_extraction(mapping.version_uid)
        if extraction is None:
            return self._unresolved(remote_id, "local provenance version is missing")
        passages = extraction.passages
        if not passages:
            return self._unresolved(remote_id, "local version has no passages")

        position_match = self._position_match(passages, chunk)
        if position_match is not None:
            return self._from_passage(
                position_match,
                "position",
                1.0,
                remote_id,
                str(
                    chunk.get(
                        "content",
                        chunk.get("text", chunk.get("content_with_weight", "")),
                    )
                    or ""
                ),
            )

        content = str(
            chunk.get(
                "content", chunk.get("text", chunk.get("content_with_weight", ""))
            )
            or ""
        )
        normalized = normalize_text(content)
        if normalized:
            exact = [p for p in passages if normalize_text(p.text) == normalized]
            if len(exact) == 1:
                return self._from_passage(exact[0], "exact", 0.99, remote_id, content)
            scored = sorted(
                ((token_dice(content, passage.text), passage) for passage in passages),
                key=lambda item: (
                    -item[0],
                    item[1].page_number,
                    item[1].paragraph_number,
                    item[1].char_start,
                    item[1].passage_uid,
                ),
            )
            if scored and scored[0][0] >= self.threshold:
                score, passage = scored[0]
                return self._from_passage(
                    passage, "token_dice", score, remote_id, content
                )
        return self._unresolved(
            remote_id, "no unambiguous position or match above threshold", content
        )

    def resolve_many(
        self, chunks: Iterable[dict[str, Any]], dataset_scope: str | None = None
    ) -> list[Citation]:
        return [self.resolve(chunk, dataset_scope=dataset_scope) for chunk in chunks]


def resolve_citation(
    chunk: dict[str, Any],
    registry: Any,
    threshold: float = 0.60,
    dataset_scope: str | None = None,
) -> Citation:
    return CitationResolver(registry, threshold).resolve(
        chunk, dataset_scope=dataset_scope
    )
