"""Small stdlib HTTP API that connects the first web UI to RAGFlow."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from .adapter import RAGFlowAdapter
from .citations import CitationResolver
from .config import Config
from .errors import DevalError
from .ingestion import IngestionService
from .model_catalog import (  # type: ignore[import-not-found]
    ModelOption,
    remote_model_options,
)
from .registry import Registry

MAX_NAME_LENGTH = 160
MAX_QUESTION_LENGTH = 10_000
RETRIEVAL_CHUNK_MIN = 5
RETRIEVAL_CHUNK_MAX = 20
DEFAULT_RETRIEVAL_CHUNKS = 5
_ALLOWED_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}
_FILENAME_RE = re.compile(r'filename="([^"]*)"', re.IGNORECASE)
CROSS_LANGUAGE_OPTIONS = ("German", "English")
_CROSS_LANGUAGE_SET = frozenset(CROSS_LANGUAGE_OPTIONS)


def _bounded_retrieval_chunks(value: Any) -> int:
    if isinstance(value, bool):
        return DEFAULT_RETRIEVAL_CHUNKS
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return DEFAULT_RETRIEVAL_CHUNKS
    if not RETRIEVAL_CHUNK_MIN <= parsed <= RETRIEVAL_CHUNK_MAX:
        return DEFAULT_RETRIEVAL_CHUNKS
    return parsed


class WebError(Exception):
    """An expected API error with an HTTP status code."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class UploadJob:
    job_id: str
    collection_scope: str
    filenames: list[str]
    operation: str = "upload"
    state: str = "processing"
    completed: int = 0
    error: str = ""


@dataclass(frozen=True)
class ChatHandle:
    collection_scope: str
    model_id: str
    chat_id: str
    session_id: str


def _safe_filename(value: str) -> str:
    name = Path(value).name.strip()
    if not name or name in {".", ".."}:
        raise WebError(400, "each uploaded file needs a filename")
    return name


def _parse_multipart(content_type: str, body: bytes) -> list[tuple[str, bytes]]:
    marker = "boundary="
    if marker not in content_type:
        raise WebError(400, "multipart upload has no boundary")
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    if not boundary:
        raise WebError(400, "multipart upload has an empty boundary")
    delimiter = b"--" + boundary.encode("utf-8")
    files: list[tuple[str, bytes]] = []
    for part in body.split(delimiter)[1:]:
        if part.startswith(b"--"):
            break
        part = part.lstrip(b"\r\n")
        header_bytes, separator, payload = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        headers = header_bytes.decode("latin-1", errors="replace")
        disposition = next(
            (
                line
                for line in headers.split("\r\n")
                if line.lower().startswith("content-disposition:")
            ),
            "",
        )
        match = _FILENAME_RE.search(disposition)
        if not match:
            continue
        filename = _safe_filename(match.group(1))
        if payload.endswith(b"\r\n"):
            payload = payload[:-2]
        files.append((filename, payload))
    if not files:
        raise WebError(400, "upload must contain at least one file")
    return files


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _normalize_cross_languages(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise WebError(400, "cross_languages must be an array")
    if any(not isinstance(language, str) for language in value):
        raise WebError(400, "cross_languages must contain language names")
    if len(set(value)) != len(value):
        raise WebError(400, "cross_languages must not contain duplicates")
    unsupported = [
        language for language in value if language not in _CROSS_LANGUAGE_SET
    ]
    if unsupported:
        raise WebError(
            400,
            "unsupported cross-language; choose German or English",
        )
    # Keep cache keys and RAGFlow prompt settings deterministic regardless of
    # the order in which a client sends the selected checkboxes.
    return [language for language in CROSS_LANGUAGE_OPTIONS if language in value]


def _remove_temp_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def _completion_reference_chunks(completion: dict[str, Any]) -> list[dict[str, Any]]:
    data = completion.get("data")
    if not isinstance(data, dict):
        return []
    reference = data.get("reference")
    if not isinstance(reference, dict):
        return []
    chunks = reference.get("chunks")
    if isinstance(chunks, dict):
        chunks = list(chunks.values())
    if not isinstance(chunks, list):
        return []
    return [chunk for chunk in chunks if isinstance(chunk, dict)]


def _normalize_chat_messages(value: Any, question: str) -> list[dict[str, str]]:
    if value in (None, []):
        return [{"role": "user", "content": question}]
    if not isinstance(value, list):
        raise WebError(400, "messages must be an array")
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise WebError(400, "messages must contain objects")
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise WebError(400, "messages must contain user/assistant text messages")
        messages.append({"role": role, "content": content})
    if not messages or messages[-1]["role"] != "user":
        messages.append({"role": "user", "content": question})
    else:
        messages[-1]["content"] = question
    return messages


_FOLLOW_UP_RE = re.compile(
    r"\b(?:antworte|beantworte|erkläre|erlaeutere|erläutere|vertiefe|"
    r"präzisiere|praezisiere|ausführlich|ausführlicher|detailliert|"
    r"genauer|mehr dazu|more detail|elaborate|explain further|tell me more)\b",
    re.IGNORECASE,
)


def _contextualize_question(messages: list[dict[str, str]], question: str) -> str:
    previous_questions = [
        message["content"] for message in messages if message["role"] == "user"
    ]
    if len(previous_questions) < 2 or not _FOLLOW_UP_RE.search(question):
        return question
    return f"{previous_questions[-2]}\n\n{question}"


class WebApplication:
    """Application services used by the tiny HTTP adapter."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.from_env()
        self.registry = Registry(self.config.registry_path)
        self._lock = threading.RLock()
        self._jobs: dict[str, UploadJob] = {}
        self._chats: dict[str, ChatHandle] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="deval-web"
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self.registry.close()

    def _collection_config(self, name: str) -> Config:
        return replace(self.config, dataset_name=name)

    @staticmethod
    def _retrieval_chunks(dataset: Any) -> int:
        config = getattr(dataset, "config", {})
        value = config.get("retrieval_chunk_count") if isinstance(config, dict) else None
        return _bounded_retrieval_chunks(value)

    def _invalidate_chat_handles(self, collection_scope: str) -> None:
        with self._lock:
            self._chats = {
                key: handle
                for key, handle in self._chats.items()
                if handle.collection_scope != collection_scope
            }

    def update_collection_settings(
        self, scope: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        dataset = self.registry.get_dataset(scope)
        if dataset is None:
            raise WebError(404, "collection not found")
        value = payload.get("retrieval_chunk_count")
        if isinstance(value, bool) or not isinstance(value, int):
            raise WebError(400, "retrieval_chunk_count must be an integer")
        if not RETRIEVAL_CHUNK_MIN <= value <= RETRIEVAL_CHUNK_MAX:
            raise WebError(
                400,
                f"retrieval_chunk_count must be between {RETRIEVAL_CHUNK_MIN} and {RETRIEVAL_CHUNK_MAX}",
            )
        config = dict(dataset.config)
        config["retrieval_chunk_count"] = value
        self.registry.upsert_dataset(
            dataset.dataset_scope,
            dataset.remote_dataset_id,
            dataset.name,
            dataset.embedding_model,
            dataset.llm_model,
            config,
            owned=dataset.owned,
        )
        self._invalidate_chat_handles(scope)
        return self._collection_view(scope)

    def _remote_model_options(self) -> list[ModelOption]:
        async def load() -> list[dict[str, Any]]:
            adapter = RAGFlowAdapter(
                self.config.base_url,
                self.config.api_key,
                timeout=self.config.request_timeout,
            )
            try:
                return await adapter.list_chat_models()
            finally:
                await adapter.aclose()

        try:
            remote_models = asyncio.run(load())
            return remote_model_options(remote_models)
        except WebError:
            raise
        except Exception as exc:
            raise WebError(502, f"RAGFlow model list failed: {exc}") from exc

    def _configured_model_options(self) -> list[ModelOption]:
        return self._remote_model_options()

    def _available_models(self) -> list[tuple[ModelOption, bool]]:
        return [(option, True) for option in self._remote_model_options()]

    def models(self) -> list[dict[str, Any]]:
        try:
            options = self._available_models()
        except (OSError, TypeError, ValueError) as exc:
            raise WebError(500, f"model configuration is invalid: {exc}") from exc
        return [
            {**option.public_dict(), "configured": is_configured}
            for option, is_configured in options
        ]

    async def _ensure_dataset(self, config: Config, adapter: RAGFlowAdapter):
        existing = self.registry.get_dataset(config.dataset_scope)
        remote = await adapter.ensure_dataset(
            config.dataset_name,
            embedding_model=config.embedding_model,
            llm_model=config.llm_model,
            chunk_method=config.chunk_method,
            parser_config=config.parser_config,
        )
        remote_id = str(remote.get("id", ""))
        if not remote_id:
            raise DevalError("RAGFlow dataset response has no id")
        dataset_config = dict(existing.config) if existing is not None else {}
        dataset_config.update(
            {
                "chunk_method": config.chunk_method,
                "parser_config": config.parser_config,
                "embedding_model": config.embedding_model,
                "llm_model": config.llm_model,
            }
        )
        dataset_config.setdefault("retrieval_chunk_count", DEFAULT_RETRIEVAL_CHUNKS)
        return self.registry.upsert_dataset(
            config.dataset_scope,
            remote_id,
            config.dataset_name,
            config.embedding_model,
            config.llm_model,
            dataset_config,
            owned=(
                existing.owned
                if existing is not None
                else bool(adapter.last_dataset_created)
            ),
        )

    def _graph_view(
        self, scope: str, mappings: list[Any] | None = None
    ) -> dict[str, Any]:
        if mappings is None:
            mappings = self.registry.mappings_for_dataset(scope)
        graph = self.registry.get_index(scope, "graph") or {}
        remote_state = str(graph.get("state", "UNKNOWN")).upper()
        with self._lock:
            job = self._jobs.get(scope)
            job_state = job.state if job is not None else ""
            job_error = job.error if job is not None else ""

        remote_update = remote_state in {"RUNNING", "SCHEDULE"} or (
            remote_state == "UNSTART" and bool(graph.get("remote_task_id"))
        )
        unknown_task = remote_state == "UNKNOWN" and bool(graph.get("remote_task_id"))
        if job_state in {"processing", "building"} or remote_update or unknown_task:
            state = "updating"
        elif job_state == "error" or remote_state in {
            "FAIL",
            "CANCEL",
            "CANCELED",
            "NOT_CONFIGURED",
        }:
            state = "error"
        elif not mappings:
            state = "empty"
        elif remote_state == "EMPTY":
            state = "error"
        elif remote_state == "DONE" and all(
            mapping.state == "DONE" for mapping in mappings
        ):
            state = "current"
        else:
            state = "outdated"

        return {
            "state": state,
            "remote_state": remote_state,
            "progress": graph.get("progress"),
            "message": str(job_error or graph.get("progress_msg") or ""),
            "document_count": len(mappings),
        }

    def _collection_status(
        self, scope: str, graph_view: dict[str, Any] | None = None
    ) -> str:
        with self._lock:
            job = self._jobs.get(scope)
            job_state = job.state if job is not None else ""
        graph_state = (graph_view or self._graph_view(scope))["state"]
        if graph_state == "current":
            return "ready"
        if graph_state == "empty":
            return "idle"
        if graph_state == "error":
            return "error"
        if graph_state == "updating":
            return "processing" if job_state == "processing" else "building"
        if graph_state == "outdated":
            return "outdated"
        return "processing"

    def _collection_view(self, scope: str) -> dict[str, Any]:
        dataset = self.registry.get_dataset(scope)
        if dataset is None:
            raise WebError(404, "collection not found")
        mappings = self.registry.mappings_for_dataset(scope)
        documents: list[str] = []
        document_records: list[dict[str, Any]] = []
        for mapping in mappings:
            extraction = self.registry.get_extraction(mapping.version_uid)
            name = (
                extraction.source_basename
                if extraction is not None
                else mapping.remote_name
            )
            documents.append(name)
            document_records.append(
                {
                    "id": mapping.version_uid,
                    "name": name,
                    "state": mapping.state,
                }
            )
        with self._lock:
            job = self._jobs.get(scope)
            job_view = (
                {
                    "id": job.job_id,
                    "files": job.filenames,
                    "operation": job.operation,
                    "completed": job.completed,
                    "total": len(job.filenames),
                    "error": job.error,
                }
                if job is not None
                else None
            )
        graph_view = self._graph_view(scope, mappings)
        return {
            "id": dataset.dataset_scope,
            "name": dataset.name,
            "status": self._collection_status(scope, graph_view),
            "retrieval_chunk_count": self._retrieval_chunks(dataset),
            "documents": documents,
            "document_records": document_records,
            "graph": graph_view,
            "job": job_view,
        }

    def list_collections(self) -> list[dict[str, Any]]:
        return [
            self._collection_view(item.dataset_scope)
            for item in self.registry.list_datasets()
        ]

    def create_collection(self, name: str) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise WebError(400, "collection name must not be empty")
        if len(name) > MAX_NAME_LENGTH:
            raise WebError(400, "collection name is too long")
        if any(
            item.name.casefold() == name.casefold()
            for item in self.registry.list_datasets()
        ):
            raise WebError(409, "a collection with this name already exists")
        config = self._collection_config(name)

        async def run():
            adapter = RAGFlowAdapter(
                config.base_url, config.api_key, timeout=config.request_timeout
            )
            try:
                return await self._ensure_dataset(config, adapter)
            finally:
                await adapter.aclose()

        try:
            dataset = asyncio.run(run())
        except WebError:
            raise
        except Exception as exc:
            raise WebError(
                502, f"could not create collection in RAGFlow: {exc}"
            ) from exc
        return self._collection_view(dataset.dataset_scope)

    def delete_collection(self, scope: str) -> dict[str, Any]:
        dataset = self.registry.get_dataset(scope)
        if dataset is None:
            raise WebError(404, "collection not found")
        if not dataset.owned:
            raise WebError(409, "collection is not owned by this DEval instance")
        with self._lock:
            current = self._jobs.get(scope)
            if current is not None and current.state in {"processing", "building"}:
                raise WebError(409, "collection is still being updated")
        if self._graph_view(scope)["state"] == "updating":
            raise WebError(409, "collection is still being updated")
        config = self._collection_config(dataset.name)

        async def run() -> str:
            adapter = RAGFlowAdapter(
                config.base_url, config.api_key, timeout=config.request_timeout
            )
            try:
                remote_id = dataset.remote_dataset_id
                matches = [
                    item
                    for item in await adapter.list_datasets()
                    if str(item.get("name", "")).casefold() == dataset.name.casefold()
                ]
                if len(matches) > 1:
                    raise DevalError(
                        f"more than one RAGFlow dataset has the name {dataset.name!r}"
                    )
                if matches and matches[0].get("id"):
                    # Resolve a stale local id by the collection name before
                    # deleting; this also repairs the common tenant-reset case.
                    remote_id = str(matches[0]["id"])
                await adapter.delete_owned_dataset(remote_id, remote_id)
                return remote_id
            finally:
                await adapter.aclose()

        try:
            remote_id = asyncio.run(run())
        except WebError:
            raise
        except Exception as exc:
            raise WebError(
                502, f"could not delete collection from RAGFlow: {exc}"
            ) from exc
        self.registry.delete_dataset_rows(scope, dataset.remote_dataset_id)
        with self._lock:
            self._chats = {
                key: value
                for key, value in self._chats.items()
                if value.collection_scope != scope
            }
        return {"deleted": scope, "remote_dataset_id": remote_id}

    def start_upload(
        self, scope: str, files: list[tuple[str, bytes]]
    ) -> dict[str, Any]:
        if self.registry.get_dataset(scope) is None:
            raise WebError(404, "collection not found")
        if not files:
            raise WebError(400, "upload must contain at least one file")
        with self._lock:
            current = self._jobs.get(scope)
            if current is not None and current.state in {"processing", "building"}:
                raise WebError(409, "collection is already processing an upload")
        temp_dir = Path(tempfile.mkdtemp(prefix="deval-web-"))
        paths: list[Path] = []
        for index, (filename, content) in enumerate(files):
            if len(content) > self.config.max_document_bytes:
                _remove_temp_dir(temp_dir)
                raise WebError(
                    413, f"file is larger than the configured PDF limit: {filename}"
                )
            path = temp_dir / str(index) / _safe_filename(filename)
            path.parent.mkdir()
            path.write_bytes(content)
            paths.append(path)
        job = UploadJob(
            job_id=uuid.uuid4().hex,
            collection_scope=scope,
            filenames=[name for name, _ in files],
        )
        with self._lock:
            self._jobs[scope] = job
        self._executor.submit(self._process_upload, job, temp_dir, paths)
        return self._collection_view(scope)

    def start_delete(self, scope: str, version_uid: str) -> dict[str, Any]:
        if self.registry.get_dataset(scope) is None:
            raise WebError(404, "collection not found")
        with self._lock:
            current = self._jobs.get(scope)
            if current is not None and current.state in {"processing", "building"}:
                raise WebError(409, "collection is already being updated")
        if self._graph_view(scope)["state"] == "updating":
            raise WebError(409, "collection is already being updated")
        mapping = self.registry.get_mapping(scope, version_uid)
        if mapping is None:
            raise WebError(404, "document not found in collection")
        if mapping.state != "DONE":
            raise WebError(409, "document is still being processed")
        extraction = self.registry.get_extraction(mapping.version_uid)
        filename = (
            extraction.source_basename
            if extraction is not None
            else mapping.remote_name
        )
        job = UploadJob(
            job_id=uuid.uuid4().hex,
            collection_scope=scope,
            filenames=[filename],
            operation="delete",
            state="building",
        )
        with self._lock:
            self._jobs[scope] = job
        self._executor.submit(self._process_delete, job, version_uid)
        return self._collection_view(scope)

    def _process_delete(self, job: UploadJob, version_uid: str) -> None:
        config = self._collection_config(job.collection_scope)

        async def run():
            adapter = RAGFlowAdapter(
                config.base_url, config.api_key, timeout=config.request_timeout
            )
            try:
                dataset = self.registry.get_dataset(job.collection_scope)
                mapping = self.registry.get_mapping(job.collection_scope, version_uid)
                if dataset is None or mapping is None:
                    raise DevalError("document mapping disappeared before deletion")
                if mapping.remote_document_id:
                    await adapter.delete_document(
                        dataset.remote_dataset_id, mapping.remote_document_id
                    )
                if (
                    self.registry.delete_mapping(job.collection_scope, version_uid)
                    is None
                ):
                    raise DevalError("document mapping disappeared during deletion")
                remaining = self.registry.mappings_for_dataset(job.collection_scope)
                if not remaining:
                    self.registry.upsert_index(
                        job.collection_scope,
                        "graph",
                        state="EMPTY",
                        progress=1.0,
                        progress_msg="no documents in collection",
                    )
                else:
                    self.registry.upsert_index(
                        job.collection_scope,
                        "graph",
                        state="UNSTART",
                        progress=0.0,
                        progress_msg="document removed; graph rebuild accepted",
                    )
                    graph = await IngestionService(
                        config, self.registry, adapter
                    ).build_graph(timeout=config.graph_timeout)
                    if graph.get("state") == "EMPTY":
                        # A completed run may legitimately contain no KG entities;
                        # vector retrieval and chat still work with the parsed docs.
                        self.registry.upsert_index(
                            job.collection_scope,
                            "graph",
                            state="DONE",
                            progress=1.0,
                            progress_msg=graph.get(
                                "message", "graph has no knowledge-graph entities"
                            ),
                        )
                    elif graph.get("state") != "DONE":
                        raise DevalError(
                            "GraphRAG rebuild did not finish successfully: "
                            f"{graph.get('message', graph.get('state', 'unknown'))}"
                        )
                with self._lock:
                    job.completed = 1
                    job.state = "ready"
            finally:
                await adapter.aclose()

        try:
            asyncio.run(run())
        except Exception as exc:  # noqa: BLE001 - surfaced by the status endpoint
            with self._lock:
                job.state = "error"
                job.error = str(exc)

    def _process_upload(
        self, job: UploadJob, temp_dir: Path, paths: list[Path]
    ) -> None:
        config = self._collection_config(job.collection_scope)

        async def run():
            adapter = RAGFlowAdapter(
                config.base_url, config.api_key, timeout=config.request_timeout
            )
            try:
                service = IngestionService(config, self.registry, adapter)
                self.registry.upsert_index(
                    job.collection_scope,
                    "graph",
                    state="UNSTART",
                    progress=0.0,
                    progress_msg="upload accepted",
                )
                semaphore = asyncio.Semaphore(config.upload_concurrency)

                async def ingest_one(path: Path):
                    async with semaphore:
                        result = await service.ingest(
                            path,
                            allow_mixed=True,
                            timeout=config.parse_timeout,
                        )
                    if result.state != "DONE":
                        raise DevalError(
                            f"upload could not be parsed: {path.name} ({result.state})"
                        )
                    with self._lock:
                        job.completed += 1
                    return result

                results = await asyncio.gather(
                    *(ingest_one(path) for path in paths),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
                with self._lock:
                    job.state = "building"
                graph = await service.build_graph(timeout=config.graph_timeout)
                if graph.get("state") == "EMPTY":
                    # RAGFlow can finish successfully without extracted KG
                    # entities; vector retrieval still works for the documents.
                    self.registry.upsert_index(
                        job.collection_scope,
                        "graph",
                        state="DONE",
                        progress=1.0,
                        progress_msg=graph.get(
                            "message", "graph has no knowledge-graph entities"
                        ),
                    )
                elif graph.get("state") != "DONE":
                    raise DevalError(
                        f"GraphRAG build did not finish successfully: {graph.get('message', graph.get('state', 'unknown'))}"
                    )
                with self._lock:
                    job.state = "ready"
            finally:
                await adapter.aclose()

        try:
            asyncio.run(run())
        except Exception as exc:  # noqa: BLE001 - surfaced by the status endpoint
            with self._lock:
                job.state = "error"
                job.error = str(exc)
        finally:
            _remove_temp_dir(temp_dir)

    def _model(self, model_id: str) -> ModelOption:
        try:
            model = next(
                (
                    option
                    for option in self._configured_model_options()
                    if option.id == model_id
                ),
                None,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise WebError(500, f"model configuration is invalid: {exc}") from exc
        if model is None:
            raise WebError(400, "selected model is not configured in RAGFlow")
        return model

    @staticmethod
    def _conversation_id(value: str) -> str:
        conversation_id = value.strip() or uuid.uuid4().hex
        if len(conversation_id) > 120:
            raise WebError(400, "conversation id is too long")
        return conversation_id

    def job_status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = next(
                (item for item in self._jobs.values() if item.job_id == job_id), None
            )
            if job is None:
                raise WebError(404, "job not found")
            return {
                "id": job.job_id,
                "collection_id": job.collection_scope,
                "state": job.state,
                "files": job.filenames,
                "operation": job.operation,
                "completed": job.completed,
                "total": len(job.filenames),
                "error": job.error,
            }

    def graph_status(self, scope: str) -> dict[str, Any]:
        if self.registry.get_dataset(scope) is None:
            raise WebError(404, "collection not found")
        view = self._graph_view(scope)
        return {
            "collection_id": scope,
            "ready": view["state"] == "current",
            "state": view["remote_state"],
            "freshness": view["state"],
            "progress": view["progress"],
            "message": view["message"],
            "document_count": view["document_count"],
        }

    def source_image(self, scope: str, image_id: str) -> tuple[bytes, str]:
        dataset = self.registry.get_dataset(scope)
        if dataset is None:
            raise WebError(404, "collection not found")
        image_id = image_id.strip()
        if not re.fullmatch(
            rf"{re.escape(dataset.remote_dataset_id)}-[A-Za-z0-9_-]+", image_id
        ):
            raise WebError(404, "source image not found")
        config = self._collection_config(dataset.name)

        async def run() -> tuple[bytes, str]:
            adapter = RAGFlowAdapter(
                config.base_url, config.api_key, timeout=config.request_timeout
            )
            try:
                return await adapter.get_document_image(image_id)
            finally:
                await adapter.aclose()

        try:
            return asyncio.run(run())
        except WebError:
            raise
        except Exception as exc:
            raise WebError(502, f"RAGFlow source image failed: {exc}") from exc

    def source_document(self, scope: str, version_uid: str) -> tuple[bytes, str, str]:
        dataset = self.registry.get_dataset(scope)
        if dataset is None:
            raise WebError(404, "collection not found")
        version_uid = version_uid.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", version_uid):
            raise WebError(404, "document not found")
        mapping = self.registry.get_mapping(scope, version_uid)
        remote_document_id = mapping.remote_document_id if mapping else None
        if mapping is None or not remote_document_id:
            raise WebError(404, "document not found")
        extraction = self.registry.get_extraction(version_uid)
        filename = (
            extraction.source_basename
            if extraction is not None
            else mapping.remote_name
        ) or "document.pdf"
        config = self._collection_config(dataset.name)

        async def run() -> tuple[bytes, str]:
            adapter = RAGFlowAdapter(
                config.base_url, config.api_key, timeout=config.request_timeout
            )
            try:
                return await adapter.download_document(
                    dataset.remote_dataset_id, remote_document_id
                )
            finally:
                await adapter.aclose()

        try:
            content, content_type = asyncio.run(run())
            return content, content_type, filename
        except WebError:
            raise
        except Exception as exc:
            raise WebError(502, f"RAGFlow document download failed: {exc}") from exc

    async def _ensure_chat_session(
        self,
        adapter: RAGFlowAdapter,
        dataset: Any,
        model: ModelOption,
        conversation_id: str,
        cross_languages: list[str] | None = None,
        retrieval_chunk_count: int = DEFAULT_RETRIEVAL_CHUNKS,
    ) -> ChatHandle:
        cache_key = conversation_id
        chat_name = f"deval-web-{conversation_id}"
        if self.config.stateless_chat:
            language_key = ",".join(cross_languages or ()) or "none"
            cache_key = f"stateless:{dataset.dataset_scope}:{model.id}:{language_key}"
            chat_name = (
                f"deval-web-stateless-{uuid.uuid5(uuid.NAMESPACE_URL, cache_key)}"
            )
        with self._lock:
            handle = self._chats.get(cache_key)
            if handle is not None:
                if (
                    handle.collection_scope != dataset.dataset_scope
                    or handle.model_id != model.id
                ):
                    raise WebError(
                        409, "chat context and model cannot change during the MVP chat"
                    )
                return handle
        chat = await adapter.ensure_chat(
            chat_name,
            [dataset.remote_dataset_id],
            llm_model=model.ragflow_model,
            cross_languages=cross_languages,
            retrieval_chunk_count=retrieval_chunk_count,
        )
        chat_id = str(chat.get("id", ""))
        if not chat_id:
            raise DevalError("RAGFlow chat has no id")
        session_id = ""
        if not self.config.stateless_chat:
            session = await adapter.create_chat_session(chat_id, name="DEval Webchat")
            session_id = str(session.get("id", ""))
            if not session_id:
                raise DevalError("RAGFlow chat session has no id")
        handle = ChatHandle(dataset.dataset_scope, model.id, chat_id, session_id)
        with self._lock:
            self._chats[cache_key] = handle
        return handle

    async def _retrieve_citations(
        self,
        adapter: RAGFlowAdapter,
        config: Config,
        dataset: Any,
        scope: str,
        question: str,
        cross_languages: list[str] | None = None,
    ) -> tuple[Any, list[Any]]:
        chunk_count = self._retrieval_chunks(dataset)
        retrieved = await adapter.retrieve(
            question,
            [dataset.remote_dataset_id],
            page_size=chunk_count,
            similarity_threshold=0.0,
            knn_top_k=max(20, chunk_count),
            knn_num_candidates=max(40, chunk_count),
            rerank_candidates_count=max(20, chunk_count),
            highlight=False,
            use_kg=False,
            cross_languages=cross_languages,
        )
        citations = CitationResolver(
            self.registry, config.citation_threshold
        ).resolve_many(
            retrieved.references or retrieved.chunks,
            dataset_scope=scope,
        )
        return retrieved, citations

    def retrieve(
        self, scope: str, question: str, cross_languages: Any = None
    ) -> dict[str, Any]:
        languages = _normalize_cross_languages(cross_languages)
        dataset = self.registry.get_dataset(scope)
        if dataset is None:
            raise WebError(404, "collection not found")
        graph = self.registry.get_index(scope, "graph")
        if graph is None or graph.get("state") != "DONE":
            raise WebError(409, "GraphRAG is not ready for this collection")
        question = question.strip()
        if not question:
            raise WebError(400, "question must not be empty")
        config = self._collection_config(dataset.name)

        async def run():
            adapter = RAGFlowAdapter(
                config.base_url, config.api_key, timeout=config.request_timeout
            )
            try:
                retrieved, citations = await self._retrieve_citations(
                    adapter, config, dataset, scope, question, languages
                )
                references = retrieved.references or retrieved.chunks
                return {
                    "question": question,
                    "cross_languages": list(languages),
                    "results": references,
                    "citations": self._citation_views(
                        citations, references, scope, dataset.remote_dataset_id
                    ),
                }
            finally:
                await adapter.aclose()

        try:
            return asyncio.run(run())
        except Exception as exc:
            raise WebError(502, f"RAGFlow retrieval failed: {exc}") from exc

    def create_chat_session(
        self,
        scope: str,
        model_id: str,
        conversation_id: str = "",
        cross_languages: Any = None,
    ) -> dict[str, Any]:
        languages = _normalize_cross_languages(cross_languages)
        dataset = self.registry.get_dataset(scope)
        if dataset is None:
            raise WebError(404, "collection not found")
        graph = self.registry.get_index(scope, "graph")
        if graph is None or graph.get("state") != "DONE":
            raise WebError(409, "GraphRAG is not ready for this collection")
        model = self._model(model_id)
        conversation_id = self._conversation_id(conversation_id)
        if self.config.stateless_chat:
            return {
                "conversation_id": conversation_id,
                "collection_id": scope,
                "model": model.public_dict(),
                "cross_languages": list(languages),
            }
        config = self._collection_config(dataset.name)

        async def run():
            adapter = RAGFlowAdapter(
                config.base_url, config.api_key, timeout=config.request_timeout
            )
            try:
                await self._ensure_chat_session(
                    adapter,
                    dataset,
                    model,
                    conversation_id,
                    languages,
                    self._retrieval_chunks(dataset),
                )
            finally:
                await adapter.aclose()

        try:
            asyncio.run(run())
            return {
                "conversation_id": conversation_id,
                "collection_id": scope,
                "model": model.public_dict(),
                "cross_languages": list(languages),
            }
        except WebError:
            raise
        except Exception as exc:
            raise WebError(502, f"RAGFlow chat setup failed: {exc}") from exc

    @staticmethod
    def _reference_image_id(reference: Any, remote_dataset_id: str) -> str | None:
        if not isinstance(reference, dict) or not remote_dataset_id.strip():
            return None
        value = reference.get("image_id", reference.get("img_id"))
        if not isinstance(value, str):
            return None
        image_id = value.strip()
        if not image_id.startswith(remote_dataset_id.strip() + "-"):
            return None
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,512}", image_id):
            return None
        return image_id

    def _citation_view(
        self,
        citation: Any,
        *,
        collection_scope: str = "",
        remote_dataset_id: str = "",
        reference: Any = None,
    ) -> dict[str, Any]:
        extraction = (
            self.registry.get_extraction(citation.version_uid)
            if citation.version_uid
            else None
        )
        excerpt = citation.matched_text
        if citation.version_uid and citation.passage_uid:
            local_excerpt = next(
                (
                    str(item.get("text", ""))
                    for item in self.registry.list_passages(citation.version_uid)
                    if item.get("passage_uid") == citation.passage_uid
                ),
                "",
            )
            if local_excerpt:
                excerpt = local_excerpt
        source = citation.as_dict()
        image_id = self._reference_image_id(reference, remote_dataset_id)
        if image_id:
            source["image_id"] = image_id
            if isinstance(reference, dict) and reference.get("positions") is not None:
                source["positions"] = reference["positions"]
        view: dict[str, Any] = {
            "document": extraction.source_basename
            if extraction
            else "Unaufgelöste Quelle",
            "page": citation.page_number,
            "excerpt": excerpt,
            "resolved": citation.passage_uid is not None,
            "source": source,
        }
        if collection_scope and citation.version_uid:
            view["document_url"] = (
                f"/api/collections/{quote(collection_scope, safe='')}/documents/"
                f"{quote(citation.version_uid, safe='')}/file"
            )
        if image_id and collection_scope:
            view["image_id"] = image_id
            view["image_url"] = (
                f"/api/collections/{quote(collection_scope, safe='')}/source-images/"
                f"{quote(image_id, safe='')}"
            )
        return view

    def _citation_views(
        self,
        citations: list[Any],
        references: list[Any],
        collection_scope: str,
        remote_dataset_id: str,
    ) -> list[dict[str, Any]]:
        return [
            self._citation_view(
                citation,
                collection_scope=collection_scope,
                remote_dataset_id=remote_dataset_id,
                reference=references[index] if index < len(references) else None,
            )
            for index, citation in enumerate(citations)
        ]

    def answer(
        self,
        collection_scope: str,
        model_id: str,
        question: str,
        conversation_id: str,
        messages: Any = None,
        cross_languages: Any = None,
    ) -> dict[str, Any]:
        languages = _normalize_cross_languages(cross_languages)
        dataset = self.registry.get_dataset(collection_scope)
        if dataset is None:
            raise WebError(404, "collection not found")
        mappings = self.registry.mappings_for_dataset(collection_scope)
        if not mappings or any(mapping.state != "DONE" for mapping in mappings):
            raise WebError(409, "documents are not ready for this collection")
        question = question.strip()
        if not question:
            raise WebError(400, "question must not be empty")
        if len(question) > MAX_QUESTION_LENGTH:
            raise WebError(400, "question is too long")
        chat_messages = _normalize_chat_messages(messages, question)
        retrieval_question = _contextualize_question(chat_messages, question)
        completion_messages = chat_messages
        if retrieval_question != question:
            completion_messages = [
                *chat_messages[:-1],
                {**chat_messages[-1], "content": retrieval_question},
            ]
        conversation_id = self._conversation_id(conversation_id)
        model = self._model(model_id)
        config = self._collection_config(dataset.name)

        async def run() -> dict[str, Any]:
            adapter = RAGFlowAdapter(
                config.base_url, config.api_key, timeout=config.request_timeout
            )
            try:
                handle = await self._ensure_chat_session(
                    adapter,
                    dataset,
                    model,
                    conversation_id,
                    languages,
                    self._retrieval_chunks(dataset),
                )
                completion = await adapter.chat_completion(
                    handle.chat_id,
                    retrieval_question,
                    handle.session_id,
                    messages=completion_messages,
                    stateless=self.config.stateless_chat,
                )
                data = completion.get("data")
                answer = data.get("answer") if isinstance(data, dict) else None
                if not isinstance(answer, str) or not answer.strip():
                    raise DevalError("RAGFlow chat completion returned no answer")
                fallback_retrieved, fallback_citations = await self._retrieve_citations(
                    adapter,
                    config,
                    dataset,
                    collection_scope,
                    retrieval_question,
                    languages,
                )
                reference_chunks = _completion_reference_chunks(completion)
                citations = (
                    CitationResolver(
                        self.registry, config.citation_threshold
                    ).resolve_many(reference_chunks, dataset_scope=collection_scope)
                    if reference_chunks
                    else fallback_citations
                )
                references = reference_chunks or (
                    fallback_retrieved.references or fallback_retrieved.chunks
                )
                return {
                    "answer": answer.strip(),
                    "conversation_id": conversation_id,
                    "collection_id": collection_scope,
                    "model": model.public_dict(),
                    "cross_languages": list(languages),
                    "citations": self._citation_views(
                        citations,
                        references,
                        collection_scope,
                        dataset.remote_dataset_id,
                    ),
                }
            finally:
                await adapter.aclose()

        try:
            return asyncio.run(run())
        except WebError:
            raise
        except Exception as exc:
            raise WebError(502, f"RAGFlow chat request failed: {exc}") from exc


class WebHandler(BaseHTTPRequestHandler):
    application: WebApplication

    def _send(self, status: int, payload: Any) -> None:
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        origin = self.headers.get("Origin")
        if origin in _ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(
        self,
        status: int,
        body: bytes,
        content_type: str,
        content_disposition: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "private, max-age=300")
        self.send_header("X-Content-Type-Options", "nosniff")
        if content_disposition:
            self.send_header("Content-Disposition", content_disposition)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, error: Exception) -> None:
        if isinstance(error, WebError):
            self._send(error.status, {"error": error.message})
            return
        self.log_error("web request failed: %s", error)
        self._send(500, {"error": "internal web service error"})

    def _body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise WebError(400, "invalid Content-Length") from None
        if length < 0 or length > 200 * 1024 * 1024:
            raise WebError(413, "request body is too large")
        return self.rfile.read(length)

    def do_OPTIONS(self) -> None:
        self._send(204, {})

    def do_GET(self) -> None:
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/api/health":
                self._send(200, {"ok": True})
            elif path == "/api/models":
                self._send(200, {"models": self.application.models()})
            elif path == "/api/collections":
                self._send(200, {"collections": self.application.list_collections()})
            elif path.startswith("/api/jobs/"):
                job_id = unquote(path.removeprefix("/api/jobs/")).strip()
                self._send(200, self.application.job_status(job_id))
            elif path.startswith("/api/collections/") and "/source-images/" in path:
                prefix = "/api/collections/"
                scope, image_id = path[len(prefix) :].split("/source-images/", 1)
                body, content_type = self.application.source_image(
                    unquote(scope).strip("/"), unquote(image_id).strip("/")
                )
                self._send_binary(200, body, content_type)
            elif (
                path.startswith("/api/collections/")
                and "/documents/" in path
                and path.endswith("/file")
            ):
                prefix = "/api/collections/"
                scope, document_path = path[len(prefix) :].rsplit("/documents/", 1)
                version_uid = document_path[: -len("/file")]
                body, content_type, filename = self.application.source_document(
                    unquote(scope).strip("/"), unquote(version_uid).strip("/")
                )
                self._send_binary(
                    200,
                    body,
                    content_type,
                    f"inline; filename*=UTF-8''{quote(filename, safe='')}",
                )
            elif path.startswith("/api/collections/") and path.endswith("/graph"):
                prefix = "/api/collections/"
                scope = unquote(path[len(prefix) : -len("/graph")]).strip("/")
                self._send(200, self.application.graph_status(scope))
            elif path.startswith("/api/collections/") and path.endswith("/status"):
                prefix = "/api/collections/"
                scope = unquote(path[len(prefix) : -len("/status")]).strip("/")
                self._send(200, self.application._collection_view(scope))
            elif path.startswith("/api/collections/"):
                scope = unquote(path.removeprefix("/api/collections/")).strip()
                self._send(200, self.application._collection_view(scope))
            else:
                raise WebError(404, "endpoint not found")
        except Exception as exc:  # noqa: BLE001 - convert expected and unexpected errors
            self._error(exc)

    def do_DELETE(self) -> None:
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"
            marker = "/documents/"
            prefix = "/api/collections/"
            if path.startswith(prefix) and marker not in path[len(prefix) :]:
                scope = unquote(path[len(prefix) :]).strip("/")
                self._send(200, self.application.delete_collection(scope))
                return
            if path.startswith(prefix) and marker in path[len(prefix) :]:
                scope, version_uid = path[len(prefix) :].rsplit(marker, 1)
                self._send(
                    202,
                    self.application.start_delete(
                        unquote(scope).strip("/"), unquote(version_uid).strip()
                    ),
                )
                return
            raise WebError(404, "endpoint not found")
        except Exception as exc:  # noqa: BLE001 - convert expected and unexpected errors
            self._error(exc)

    def do_PATCH(self) -> None:
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"
            prefix = "/api/collections/"
            if not path.startswith(prefix):
                raise WebError(404, "endpoint not found")
            payload = json.loads(self._body().decode("utf-8"))
            if not isinstance(payload, dict):
                raise WebError(400, "request body must be an object")
            scope = unquote(path.removeprefix(prefix)).strip("/")
            self._send(200, self.application.update_collection_settings(scope, payload))
        except json.JSONDecodeError:
            self._error(WebError(400, "request body is not valid JSON"))
        except UnicodeDecodeError:
            self._error(WebError(400, "request body is not valid UTF-8"))
        except Exception as exc:  # noqa: BLE001 - convert expected and unexpected errors
            self._error(exc)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path.rstrip("/") or "/"
            body = self._body()
            content_type = self.headers.get("Content-Type", "")
            if path == "/api/collections":
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise WebError(400, "request body must be an object")
                self._send(
                    201,
                    self.application.create_collection(str(payload.get("name", ""))),
                )
                return
            if path.startswith("/api/collections/") and path.endswith("/upload"):
                prefix = "/api/collections/"
                scope = unquote(path[len(prefix) : -len("/upload")]).strip("/")
                files = _parse_multipart(content_type, body)
                self._send(202, self.application.start_upload(scope, files))
                return
            if path == "/api/chat/sessions":
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise WebError(400, "request body must be an object")
                self._send(
                    201,
                    self.application.create_chat_session(
                        str(payload.get("collection_id", "")),
                        str(payload.get("model_id", "")),
                        str(payload.get("conversation_id", "")),
                        payload.get("cross_languages"),
                    ),
                )
                return
            if path == "/api/query":
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise WebError(400, "request body must be an object")
                self._send(
                    200,
                    self.application.retrieve(
                        str(payload.get("collection_id", "")),
                        str(payload.get("question", "")),
                        payload.get("cross_languages"),
                    ),
                )
                return
            if path == "/api/chat":
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise WebError(400, "request body must be an object")
                self._send(
                    200,
                    self.application.answer(
                        str(payload.get("collection_id", "")),
                        str(payload.get("model_id", "")),
                        str(payload.get("question", "")),
                        str(payload.get("conversation_id", "")),
                        payload.get("messages"),
                        payload.get("cross_languages"),
                    ),
                )
                return
            raise WebError(404, "endpoint not found")
        except json.JSONDecodeError:
            self._error(WebError(400, "request body is not valid JSON"))
        except UnicodeDecodeError:
            self._error(WebError(400, "request body is not valid UTF-8"))
        except Exception as exc:  # noqa: BLE001 - convert expected and unexpected errors
            self._error(exc)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the API quiet during the local UI workflow.
        return


def build_server(
    application: WebApplication, host: str, port: int
) -> ThreadingHTTPServer:
    handler = type("DevalWebHandler", (WebHandler,), {"application": application})
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deval-webchat")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args(argv)
    application = WebApplication()
    server = build_server(application, args.host, args.port)
    try:
        print(f"DEval web API listening on http://{args.host}:{args.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        application.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
