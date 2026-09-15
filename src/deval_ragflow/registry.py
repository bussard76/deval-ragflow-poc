"""SQLite provenance registry; it is local state, not a queue or a remote cache."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .errors import RegistryError, UnsafeOperation
from .models import DatasetRecord, DocumentExtraction, MappingRecord, json_dumps


def _json_load(value: Any, default: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _int_load(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS documents (
    document_uid TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL UNIQUE,
    source_basename TEXT NOT NULL,
    classification TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    page_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS document_versions (
    version_uid TEXT PRIMARY KEY,
    document_uid TEXT NOT NULL REFERENCES documents(document_uid) ON DELETE CASCADE,
    provenance_schema TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    parser_config_json TEXT NOT NULL,
    classification TEXT NOT NULL,
    canonical_text TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(document_uid, provenance_schema, extractor_version, parser_config_json)
);
CREATE TABLE IF NOT EXISTS passages (
    passage_uid TEXT PRIMARY KEY,
    version_uid TEXT NOT NULL REFERENCES document_versions(version_uid) ON DELETE CASCADE,
    document_uid TEXT NOT NULL REFERENCES documents(document_uid) ON DELETE CASCADE,
    page_number INTEGER NOT NULL,
    paragraph_number INTEGER NOT NULL,
    section_path_json TEXT,
    bbox_json TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(version_uid, page_number, paragraph_number),
    CHECK(char_start >= 0 AND char_end >= char_start)
);
CREATE TABLE IF NOT EXISTS datasets (
    dataset_scope TEXT PRIMARY KEY,
    remote_dataset_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    embedding_model TEXT NOT NULL DEFAULT '',
    llm_model TEXT NOT NULL DEFAULT '',
    config_json TEXT NOT NULL,
    owned INTEGER NOT NULL CHECK(owned IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS ragflow_documents (
    mapping_uid INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_scope TEXT NOT NULL REFERENCES datasets(dataset_scope) ON DELETE CASCADE,
    version_uid TEXT NOT NULL REFERENCES document_versions(version_uid) ON DELETE CASCADE,
    remote_document_id TEXT,
    remote_name TEXT NOT NULL,
    state TEXT NOT NULL,
    failure_message TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_scope, version_uid),
    UNIQUE(dataset_scope, remote_document_id)
);
CREATE TABLE IF NOT EXISTS indexes (
    index_key TEXT PRIMARY KEY,
    dataset_scope TEXT NOT NULL REFERENCES datasets(dataset_scope) ON DELETE CASCADE,
    version_uid TEXT REFERENCES document_versions(version_uid) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    remote_task_id TEXT,
    state TEXT NOT NULL,
    progress REAL,
    progress_msg TEXT NOT NULL DEFAULT '',
    raw_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS passages_version_page ON passages(version_uid, page_number, paragraph_number);
CREATE INDEX IF NOT EXISTS ragflow_documents_remote ON ragflow_documents(remote_document_id);
CREATE INDEX IF NOT EXISTS indexes_dataset_kind ON indexes(dataset_scope, kind);
"""


class Registry:
    def __init__(self, path: str | Path = Path(".data/registry.sqlite3")):
        self.path = (
            ":memory:" if str(path) == ":memory:" else str(Path(path).expanduser())
        )
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path, timeout=30, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._connection.commit()
        self.initialize()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @property
    def conn(self) -> sqlite3.Connection:
        return self._connection

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def initialize(self) -> None:
        with self._lock:
            self._connection.executescript(SCHEMA)
            self._connection.commit()

    def _transaction(self):
        self._connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _mapping(row: sqlite3.Row) -> MappingRecord:
        return MappingRecord(
            mapping_uid=_int_load(row["mapping_uid"], 0),
            dataset_scope=row["dataset_scope"],
            version_uid=row["version_uid"],
            remote_document_id=row["remote_document_id"],
            remote_name=row["remote_name"],
            state=row["state"],
            failure_message=row["failure_message"],
        )

    @staticmethod
    def _dataset(row: sqlite3.Row) -> DatasetRecord:
        return DatasetRecord(
            dataset_scope=row["dataset_scope"],
            remote_dataset_id=row["remote_dataset_id"],
            name=row["name"],
            embedding_model=row["embedding_model"],
            llm_model=row["llm_model"],
            config=_json_load(row["config_json"], {}),
            owned=bool(row["owned"]),
        )

    def save_extraction(self, extraction: DocumentExtraction) -> None:
        """Atomically save identity, version, and all passages before remote work."""
        from .models import EXTRACTOR_VERSION, PROVENANCE_SCHEMA

        with self._lock:
            try:
                self._transaction()
                self._connection.execute(
                    """INSERT INTO documents(document_uid, sha256, source_basename, classification, byte_size, page_count)
                       VALUES(?, ?, ?, ?, ?, ?)
                       ON CONFLICT(document_uid) DO UPDATE SET
                         source_basename=excluded.source_basename,
                         classification=excluded.classification,
                         byte_size=excluded.byte_size,
                         page_count=excluded.page_count,
                         updated_at=CURRENT_TIMESTAMP""",
                    (
                        extraction.document_uid,
                        extraction.sha256,
                        extraction.source_basename,
                        extraction.classification,
                        extraction.byte_size,
                        extraction.page_count,
                    ),
                )
                self._connection.execute(
                    """INSERT INTO document_versions(
                         version_uid, document_uid, provenance_schema, extractor_version,
                         parser_config_json, classification, canonical_text
                       ) VALUES(?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(version_uid) DO UPDATE SET
                         classification=excluded.classification,
                         canonical_text=excluded.canonical_text""",
                    (
                        extraction.version_uid,
                        extraction.document_uid,
                        PROVENANCE_SCHEMA,
                        EXTRACTOR_VERSION,
                        json_dumps(extraction.parser_config),
                        extraction.classification,
                        extraction.canonical_text,
                    ),
                )
                self._connection.execute(
                    "DELETE FROM passages WHERE version_uid = ?",
                    (extraction.version_uid,),
                )
                self._connection.executemany(
                    """INSERT INTO passages(
                         passage_uid, version_uid, document_uid, page_number, paragraph_number,
                         section_path_json, bbox_json, char_start, char_end, text
                       ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            passage.passage_uid,
                            passage.version_uid,
                            passage.document_uid,
                            passage.page_number,
                            passage.paragraph_number,
                            json_dumps(list(passage.section_path))
                            if passage.section_path is not None
                            else None,
                            json_dumps(list(passage.bbox)),
                            passage.char_start,
                            passage.char_end,
                            passage.text,
                        )
                        for passage in extraction.passages
                    ],
                )
                self._connection.commit()
            except Exception as exc:  # noqa: BLE001 - wrap all transaction failures
                self._connection.rollback()
                raise RegistryError(f"could not save extraction: {exc}")

    # Aliases used by orchestration callers.
    upsert_extraction = save_extraction
    register_extraction = save_extraction

    def get_extraction(self, version_uid: str) -> DocumentExtraction | None:
        from .models import DocumentExtraction, Passage

        with self._lock:
            version = self._connection.execute(
                "SELECT v.*, d.sha256, d.source_basename, d.byte_size, d.page_count FROM document_versions v JOIN documents d ON d.document_uid=v.document_uid WHERE v.version_uid=?",
                (version_uid,),
            ).fetchone()
            if version is None:
                return None
            rows = self._connection.execute(
                "SELECT * FROM passages WHERE version_uid=? ORDER BY page_number, paragraph_number",
                (version_uid,),
            ).fetchall()
        passages = []
        for row in rows:
            section = (
                _json_load(row["section_path_json"], None)
                if row["section_path_json"]
                else None
            )
            passages.append(
                Passage(
                    passage_uid=row["passage_uid"],
                    document_uid=row["document_uid"],
                    version_uid=row["version_uid"],
                    page_number=row["page_number"],
                    paragraph_number=row["paragraph_number"],
                    section_path=tuple(section) if section is not None else None,
                    bbox=tuple(_json_load(row["bbox_json"], [])),
                    char_start=row["char_start"],
                    char_end=row["char_end"],
                    text=row["text"],
                )
            )
        return DocumentExtraction(
            document_uid=version["document_uid"],
            version_uid=version["version_uid"],
            sha256=version["sha256"],
            source_basename=version["source_basename"],
            classification=version["classification"],
            page_count=version["page_count"],
            byte_size=version["byte_size"],
            canonical_text=version["canonical_text"],
            passages=passages,
            parser_config=_json_load(version["parser_config_json"], {}),
        )

    def upsert_dataset(
        self,
        dataset_scope: str,
        remote_dataset_id: str,
        name: str,
        embedding_model: str = "",
        llm_model: str = "",
        config: dict[str, Any] | None = None,
        owned: bool = True,
    ) -> DatasetRecord:
        with self._lock:
            try:
                self._transaction()
                existing = self._connection.execute(
                    "SELECT * FROM datasets WHERE dataset_scope=?", (dataset_scope,)
                ).fetchone()
                if (
                    existing is not None
                    and existing["remote_dataset_id"] != remote_dataset_id
                ):
                    raise RegistryError(
                        f"dataset scope {dataset_scope} is already mapped to another remote dataset"
                    )
                # Ownership is monotonic: an existing unowned dataset can
                # never become deletable merely because an upsert used its
                # default ``owned=True``.
                effective_owned = (
                    bool(existing["owned"]) if existing is not None else bool(owned)
                )
                self._connection.execute(
                    """INSERT INTO datasets(dataset_scope, remote_dataset_id, name, embedding_model, llm_model, config_json, owned)
                       VALUES(?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(dataset_scope) DO UPDATE SET
                         name=excluded.name, embedding_model=excluded.embedding_model,
                         llm_model=excluded.llm_model, config_json=excluded.config_json,
                         owned=excluded.owned, updated_at=CURRENT_TIMESTAMP""",
                    (
                        dataset_scope,
                        remote_dataset_id,
                        name,
                        embedding_model,
                        llm_model,
                        json_dumps(config or {}),
                        int(effective_owned),
                    ),
                )
                row = self._connection.execute(
                    "SELECT * FROM datasets WHERE dataset_scope=?", (dataset_scope,)
                ).fetchone()
                self._connection.commit()
                return self._dataset(row)
            except Exception as exc:
                self._connection.rollback()
                if isinstance(exc, RegistryError):
                    raise
                raise RegistryError(f"could not save dataset: {exc}")

    def get_dataset(self, dataset_scope: str) -> DatasetRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM datasets WHERE dataset_scope=?", (dataset_scope,)
            ).fetchone()
        return self._dataset(row) if row else None

    def list_datasets(self) -> list[DatasetRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM datasets ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [self._dataset(row) for row in rows]

    def get_dataset_by_remote_id(self, remote_dataset_id: str) -> DatasetRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM datasets WHERE remote_dataset_id=?", (remote_dataset_id,)
            ).fetchone()
        return self._dataset(row) if row else None

    def reserve_mapping(
        self, dataset_scope: str, version_uid: str, remote_name: str
    ) -> MappingRecord:
        with self._lock:
            try:
                self._transaction()
                self._connection.execute(
                    """INSERT OR IGNORE INTO ragflow_documents(dataset_scope, version_uid, remote_name, state)
                       VALUES(?, ?, ?, 'RESERVED')""",
                    (dataset_scope, version_uid, remote_name),
                )
                row = self._connection.execute(
                    "SELECT * FROM ragflow_documents WHERE dataset_scope=? AND version_uid=?",
                    (dataset_scope, version_uid),
                ).fetchone()
                self._connection.commit()
                return self._mapping(row)
            except Exception as exc:  # noqa: BLE001 - wrap all transaction failures
                self._connection.rollback()
                raise RegistryError(f"could not reserve remote document: {exc}")

    reserve_ragflow_document = reserve_mapping

    def get_mapping(self, dataset_scope: str, version_uid: str) -> MappingRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM ragflow_documents WHERE dataset_scope=? AND version_uid=?",
                (dataset_scope, version_uid),
            ).fetchone()
        return self._mapping(row) if row else None

    def get_mapping_by_remote_id(self, remote_document_id: str) -> MappingRecord | None:
        mappings = self.mappings_for_remote_id(remote_document_id)
        return mappings[0] if len(mappings) == 1 else None

    def mappings_for_remote_id(self, remote_document_id: str) -> list[MappingRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM ragflow_documents WHERE remote_document_id=? ORDER BY mapping_uid",
                (remote_document_id,),
            ).fetchall()
        return [self._mapping(row) for row in rows]

    def set_mapping_remote(
        self,
        dataset_scope: str,
        version_uid: str,
        remote_document_id: str,
        state: str = "UPLOADED",
        metadata: dict[str, Any] | None = None,
    ) -> MappingRecord:
        with self._lock:
            try:
                self._transaction()
                self._connection.execute(
                    """UPDATE ragflow_documents
                       SET remote_document_id=?, state=?, metadata_json=?, updated_at=CURRENT_TIMESTAMP
                       WHERE dataset_scope=? AND version_uid=?""",
                    (
                        remote_document_id,
                        state,
                        json_dumps(metadata or {}),
                        dataset_scope,
                        version_uid,
                    ),
                )
                row = self._connection.execute(
                    "SELECT * FROM ragflow_documents WHERE dataset_scope=? AND version_uid=?",
                    (dataset_scope, version_uid),
                ).fetchone()
                if row is None:
                    raise RegistryError("mapping reservation does not exist")
                self._connection.commit()
                return self._mapping(row)
            except Exception as exc:
                self._connection.rollback()
                if isinstance(exc, RegistryError):
                    raise
                raise RegistryError(f"could not update remote mapping: {exc}")

    def set_mapping_state(
        self, dataset_scope: str, version_uid: str, state: str, message: str = ""
    ) -> None:
        with self._lock:
            self._connection.execute(
                "UPDATE ragflow_documents SET state=?, failure_message=?, updated_at=CURRENT_TIMESTAMP WHERE dataset_scope=? AND version_uid=?",
                (state, message or None, dataset_scope, version_uid),
            )
            self._connection.commit()

    def delete_mapping(
        self, dataset_scope: str, version_uid: str
    ) -> MappingRecord | None:
        """Delete one collection mapping while retaining shared local provenance."""
        with self._lock:
            try:
                self._transaction()
                row = self._connection.execute(
                    "SELECT * FROM ragflow_documents WHERE dataset_scope=? AND version_uid=?",
                    (dataset_scope, version_uid),
                ).fetchone()
                if row is None:
                    self._connection.commit()
                    return None
                mapping = self._mapping(row)
                version = self._connection.execute(
                    "SELECT document_uid FROM document_versions WHERE version_uid=?",
                    (version_uid,),
                ).fetchone()
                self._connection.execute(
                    "DELETE FROM ragflow_documents WHERE dataset_scope=? AND version_uid=?",
                    (dataset_scope, version_uid),
                )
                still_mapped = self._connection.execute(
                    "SELECT 1 FROM ragflow_documents WHERE version_uid=? LIMIT 1",
                    (version_uid,),
                ).fetchone()
                if still_mapped is None:
                    self._connection.execute(
                        "DELETE FROM document_versions WHERE version_uid=?",
                        (version_uid,),
                    )
                    if version is not None:
                        still_versioned = self._connection.execute(
                            "SELECT 1 FROM document_versions WHERE document_uid=? LIMIT 1",
                            (version["document_uid"],),
                        ).fetchone()
                        if still_versioned is None:
                            self._connection.execute(
                                "DELETE FROM documents WHERE document_uid=?",
                                (version["document_uid"],),
                            )
                self._connection.commit()
                return mapping
            except Exception as exc:
                self._connection.rollback()
                if isinstance(exc, RegistryError):
                    raise
                raise RegistryError(f"could not delete remote mapping: {exc}")

    def mappings_for_dataset(self, dataset_scope: str) -> list[MappingRecord]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM ragflow_documents WHERE dataset_scope=? ORDER BY mapping_uid",
                (dataset_scope,),
            ).fetchall()
        return [self._mapping(row) for row in rows]

    @staticmethod
    def index_key(dataset_scope: str, kind: str, version_uid: str | None = None) -> str:
        return "{}:{}:{}".format(kind, dataset_scope, version_uid or "-")

    def upsert_index(
        self,
        dataset_scope: str,
        kind: str,
        version_uid: str | None = None,
        remote_task_id: str | None = None,
        state: str = "UNKNOWN",
        progress: float | None = None,
        progress_msg: str = "",
        raw: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = self.index_key(dataset_scope, kind, version_uid)
        with self._lock:
            if raw is None:
                prior = self._connection.execute(
                    "SELECT raw_json FROM indexes WHERE index_key=?", (key,)
                ).fetchone()
                raw_json = prior[0] if prior is not None else "{}"
            else:
                raw_json = json_dumps(raw)
            self._connection.execute(
                """INSERT INTO indexes(index_key, dataset_scope, version_uid, kind, remote_task_id, state, progress, progress_msg, raw_json)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(index_key) DO UPDATE SET
                     remote_task_id=excluded.remote_task_id, state=excluded.state,
                     progress=excluded.progress, progress_msg=excluded.progress_msg,
                     raw_json=excluded.raw_json, updated_at=CURRENT_TIMESTAMP""",
                (
                    key,
                    dataset_scope,
                    version_uid,
                    kind,
                    remote_task_id,
                    state,
                    progress,
                    progress_msg,
                    raw_json,
                ),
            )
            self._connection.commit()
            result = self.get_index(dataset_scope, kind, version_uid)
            if result is None:
                raise RegistryError("index write did not persist")
            return result

    def get_index(
        self, dataset_scope: str, kind: str, version_uid: str | None = None
    ) -> dict[str, Any] | None:
        key = self.index_key(dataset_scope, kind, version_uid)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM indexes WHERE index_key=?", (key,)
            ).fetchone()
        if row is None:
            return None
        return {
            "index_key": row["index_key"],
            "dataset_scope": row["dataset_scope"],
            "version_uid": row["version_uid"],
            "kind": row["kind"],
            "remote_task_id": row["remote_task_id"],
            "state": row["state"],
            "progress": row["progress"],
            "progress_msg": row["progress_msg"],
            "raw": _json_load(row["raw_json"], {}),
        }

    def list_passages(self, version_uid: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM passages WHERE version_uid=? ORDER BY page_number, paragraph_number",
                (version_uid,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["section_path"] = (
                _json_load(item.pop("section_path_json"), None)
                if item.get("section_path_json")
                else None
            )
            item["bbox"] = _json_load(item.pop("bbox_json"), [])
            result.append(item)
        return result

    get_passages = list_passages

    def check_writable(self) -> bool:
        with self._lock:
            try:
                self._transaction()
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS __deval_write_probe (id INTEGER PRIMARY KEY)"
                )
                self._connection.execute("DROP TABLE __deval_write_probe")
                self._connection.commit()
                return True
            except Exception:  # noqa: BLE001 - a write probe only reports failure
                self._connection.rollback()
                return False

    def delete_dataset_rows(self, dataset_scope: str, remote_dataset_id: str) -> None:
        """Delete local scope only after the caller has completed remote deletion."""
        with self._lock:
            try:
                self._transaction()
                row = self._connection.execute(
                    "SELECT * FROM datasets WHERE dataset_scope=?", (dataset_scope,)
                ).fetchone()
                if (
                    row is None
                    or not bool(row["owned"])
                    or row["remote_dataset_id"] != remote_dataset_id
                ):
                    raise UnsafeOperation("dataset is not an owned local mapping")
                versions = [
                    r[0]
                    for r in self._connection.execute(
                        "SELECT version_uid FROM ragflow_documents WHERE dataset_scope=?",
                        (dataset_scope,),
                    ).fetchall()
                ]
                version_documents = (
                    {
                        version["version_uid"]: version["document_uid"]
                        for version in self._connection.execute(
                            "SELECT version_uid, document_uid FROM document_versions"
                        ).fetchall()
                        if version["version_uid"] in versions
                    }
                    if versions
                    else {}
                )
                self._connection.execute(
                    "DELETE FROM datasets WHERE dataset_scope=?", (dataset_scope,)
                )
                for version_uid in versions:
                    document_uid = version_documents.get(version_uid)
                    if document_uid is None:
                        continue
                    in_use = self._connection.execute(
                        "SELECT 1 FROM ragflow_documents WHERE version_uid=? LIMIT 1",
                        (version_uid,),
                    ).fetchone()
                    if in_use is None:
                        self._connection.execute(
                            "DELETE FROM document_versions WHERE version_uid=?",
                            (version_uid,),
                        )
                        still_has_version = self._connection.execute(
                            "SELECT 1 FROM document_versions WHERE document_uid=? LIMIT 1",
                            (document_uid,),
                        ).fetchone()
                        if still_has_version is None:
                            self._connection.execute(
                                "DELETE FROM documents WHERE document_uid=?",
                                (document_uid,),
                            )
                self._connection.commit()
            except Exception as exc:
                self._connection.rollback()
                if isinstance(exc, (UnsafeOperation, RegistryError)):
                    raise
                raise RegistryError(f"could not delete local dataset rows: {exc}")

    reset_dataset = delete_dataset_rows
    reset_test_data = delete_dataset_rows

    def counts(self) -> dict[str, int]:
        with self._lock:
            result = {}
            result["documents"] = _int_load(
                self._connection.execute("SELECT COUNT(*) FROM documents").fetchone()[
                    0
                ],
                0,
            )
            result["document_versions"] = _int_load(
                self._connection.execute(
                    "SELECT COUNT(*) FROM document_versions"
                ).fetchone()[0],
                0,
            )
            result["passages"] = _int_load(
                self._connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0],
                0,
            )
            result["datasets"] = _int_load(
                self._connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0],
                0,
            )
            result["ragflow_documents"] = _int_load(
                self._connection.execute(
                    "SELECT COUNT(*) FROM ragflow_documents"
                ).fetchone()[0],
                0,
            )
            result["indexes"] = _int_load(
                self._connection.execute("SELECT COUNT(*) FROM indexes").fetchone()[0],
                0,
            )
            return result


SQLiteRegistry = Registry
