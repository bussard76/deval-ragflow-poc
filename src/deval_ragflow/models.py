"""Small serialisable records shared by extraction, storage, and the adapter."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

PROVENANCE_SCHEMA = "deval.pdf-provenance.v1"
EXTRACTOR_VERSION = "pymupdf-1.26.5"
TERMINAL_STATES = frozenset(("DONE", "FAIL", "CANCEL", "CANCELED", "CANCELLED"))
KNOWN_STATES = frozenset(("UNSTART", "RUNNING", "SCHEDULE", "DONE", "FAIL", "CANCEL"))


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Passage:
    passage_uid: str
    document_uid: str
    version_uid: str
    page_number: int
    paragraph_number: int
    section_path: Optional[Tuple[str, ...]]
    bbox: Tuple[float, float, float, float]
    char_start: int
    char_end: int
    text: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passage_uid": self.passage_uid,
            "document_uid": self.document_uid,
            "version_uid": self.version_uid,
            "page_number": self.page_number,
            "paragraph_number": self.paragraph_number,
            "section_path": list(self.section_path)
            if self.section_path is not None
            else None,
            "bbox": list(self.bbox),
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text": self.text,
        }


@dataclass
class DocumentExtraction:
    document_uid: str
    version_uid: str
    sha256: str
    source_basename: str
    classification: str
    page_count: int
    byte_size: int
    canonical_text: str
    passages: List[Passage] = field(default_factory=list)
    parser_config: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    page_classifications: List[str] = field(default_factory=list)
    # Transient input bytes used to avoid uploading a different file if it
    # changes between extraction and the remote step; never serialized/stored.
    raw_bytes: Optional[bytes] = field(default=None, repr=False, compare=False)

    @property
    def uploadable(self) -> bool:
        return self.classification in ("extracted", "mixed")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "document_uid": self.document_uid,
            "version_uid": self.version_uid,
            "sha256": self.sha256,
            "source_basename": self.source_basename,
            "classification": self.classification,
            "page_count": self.page_count,
            "byte_size": self.byte_size,
            "canonical_text": self.canonical_text,
            "passages": [p.as_dict() for p in self.passages],
            "parser_config": self.parser_config,
            "errors": self.errors,
            "warnings": self.warnings,
            "page_classifications": self.page_classifications,
            "uploadable": self.uploadable,
        }


@dataclass(frozen=True)
class DatasetRecord:
    dataset_scope: str
    remote_dataset_id: str
    name: str
    embedding_model: str
    llm_model: str
    config: Dict[str, Any]
    owned: bool = True


@dataclass(frozen=True)
class MappingRecord:
    mapping_uid: int
    dataset_scope: str
    version_uid: str
    remote_document_id: Optional[str]
    remote_name: str
    state: str
    failure_message: Optional[str] = None


@dataclass(frozen=True)
class RemoteStatus:
    state: str
    progress: Optional[float] = None
    progress_msg: str = ""
    progress_messages: Tuple[str, ...] = ()
    chunk_count: Optional[int] = None
    token_count: Optional[int] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def failure_message(self) -> Optional[str]:
        return self.progress_msg if self.state == "FAIL" else None


@dataclass(frozen=True)
class GraphStatus:
    task_id: Optional[str] = None
    state: str = "UNKNOWN"
    progress: Optional[float] = None
    progress_msg: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


@dataclass
class RetrievalResult:
    chunks: List[Dict[str, Any]]
    references: List[Dict[str, Any]]
    raw: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"chunks": self.chunks, "references": self.references, "raw": self.raw}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.chunks[key]
        if key == "raw":
            return self.raw
        if key == "data":
            return self.raw.get("data")
        if key == "chunks":
            return self.chunks
        if key == "references":
            return self.references
        return getattr(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except (AttributeError, KeyError, TypeError):
            return default

    def __len__(self):
        return len(self.chunks)

    # pi-lens-ignore: iter-return-iterator
    def __iter__(self):
        return iter(self.chunks)


@dataclass(frozen=True)
class Citation:
    passage_uid: Optional[str]
    document_uid: Optional[str]
    version_uid: Optional[str]
    page_number: Optional[int]
    paragraph_number: Optional[int]
    char_start: Optional[int]
    char_end: Optional[int]
    method: str
    confidence: float
    matched_text: str = ""
    remote_document_id: Optional[str] = None
    reason: Optional[str] = None
    section_path: Optional[Tuple[str, ...]] = None
    bbox: Optional[Tuple[float, float, float, float]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passage_uid": self.passage_uid,
            "document_uid": self.document_uid,
            "version_uid": self.version_uid,
            "page_number": self.page_number,
            "paragraph_number": self.paragraph_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "method": self.method,
            "confidence": self.confidence,
            "matched_text": self.matched_text,
            "remote_document_id": self.remote_document_id,
            "reason": self.reason,
            "section_path": list(self.section_path)
            if self.section_path is not None
            else None,
            "bbox": list(self.bbox) if self.bbox is not None else None,
        }


@dataclass(frozen=True)
class IngestionResult:
    document_uid: str
    version_uid: str
    classification: str
    state: str
    dataset_scope: str
    remote_dataset_id: Optional[str] = None
    remote_document_id: Optional[str] = None
    progress: Optional[float] = None
    message: str = ""
    extraction: Optional[DocumentExtraction] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "document_uid": self.document_uid,
            "version_uid": self.version_uid,
            "classification": self.classification,
            "state": self.state,
            "dataset_scope": self.dataset_scope,
            "remote_dataset_id": self.remote_dataset_id,
            "remote_document_id": self.remote_document_id,
            "progress": self.progress,
            "message": self.message,
        }


# Friendly aliases; the records remain plain dataclasses.
ProvenancePassage = Passage
PDFExtraction = DocumentExtraction
