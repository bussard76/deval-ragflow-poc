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
from urllib.parse import unquote, urlparse

from .adapter import RAGFlowAdapter
from .citations import CitationResolver
from .config import Config
from .errors import DevalError
from .ingestion import IngestionService
from .model_catalog import ModelCatalog, ModelOption  # type: ignore[import-not-found]
from .registry import Registry

MAX_NAME_LENGTH = 160
MAX_QUESTION_LENGTH = 10_000
_ALLOWED_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}
_FILENAME_RE = re.compile(r'filename="([^"]*)"', re.IGNORECASE)


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


def _remove_temp_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def _remote_model_keys(remote: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for field in ("model_id", "id"):
        value = remote.get(field)
        if isinstance(value, str) and value.strip():
            keys.add(value.strip())
    name = ""
    for field in ("name", "model_name", "model"):
        value = remote.get(field)
        if isinstance(value, str) and value.strip():
            name = value.strip()
            break
    if not name:
        return keys
    keys.add(name)
    instance = remote.get("instance_name")
    provider = remote.get("provider_name")
    if isinstance(instance, str) and isinstance(provider, str):
        instance = instance.strip()
        provider = provider.strip()
        if instance and provider:
            keys.add(f"{name}@{instance}@{provider}")
    return keys


def _completion_reference_chunks(completion: dict[str, Any]) -> list[dict[str, Any]]:
    data = completion.get("data")
    if not isinstance(data, dict):
        return []
    reference = data.get("reference")
    if not isinstance(reference, dict):
        return []
    chunks = reference.get("chunks")
    if not isinstance(chunks, list):
        return []
    return [chunk for chunk in chunks if isinstance(chunk, dict)]


class WebApplication:
    """Application services used by the tiny HTTP adapter."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.from_env()
        self.registry = Registry(self.config.registry_path)
        self.catalog = ModelCatalog(
            self.config.model_config_path, fallback_model=self.config.llm_model
        )
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

    def _configured_model_options(self) -> list[ModelOption]:
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
        except Exception as exc:
            raise WebError(502, f"RAGFlow model list failed: {exc}") from exc
        configured = (
            set().union(*(_remote_model_keys(item) for item in remote_models))
            if remote_models
            else set()
        )
        return [
            option
            for option in self.catalog.options()
            if option.ragflow_model in configured
        ]

    def _available_models(self) -> list[tuple[ModelOption, bool]]:
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
        except Exception as exc:
            raise WebError(502, f"RAGFlow model list failed: {exc}") from exc
        configured = (
            set().union(*(_remote_model_keys(item) for item in remote_models))
            if remote_models
            else set()
        )
        return [
            (option, option.ragflow_model in configured)
            for option in self.catalog.options()
        ]

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
        return self.registry.upsert_dataset(
            config.dataset_scope,
            remote_id,
            config.dataset_name,
            config.embedding_model,
            config.llm_model,
            {
                "chunk_method": config.chunk_method,
                "parser_config": config.parser_config,
                "embedding_model": config.embedding_model,
                "llm_model": config.llm_model,
            },
            owned=(
                existing.owned
                if existing is not None
                else bool(adapter.last_dataset_created)
            ),
        )

    def _collection_status(self, scope: str) -> str:
        with self._lock:
            job = self._jobs.get(scope)
            if job is not None:
                if job.state in {"processing", "building"}:
                    return "processing" if job.state == "processing" else "building"
                if job.state == "error":
                    return "error"
        mappings = self.registry.mappings_for_dataset(scope)
        graph = self.registry.get_index(scope, "graph")
        if graph is not None and graph.get("state") == "DONE":
            return "ready"
        if not mappings:
            return "idle"
        if graph is not None and graph.get("state") in {
            "UNSTART",
            "RUNNING",
            "SCHEDULE",
        }:
            return "building"
        return "processing"

    def _collection_view(self, scope: str) -> dict[str, Any]:
        dataset = self.registry.get_dataset(scope)
        if dataset is None:
            raise WebError(404, "collection not found")
        documents: list[str] = []
        for mapping in self.registry.mappings_for_dataset(scope):
            extraction = self.registry.get_extraction(mapping.version_uid)
            documents.append(
                extraction.source_basename
                if extraction is not None
                else mapping.remote_name
            )
        with self._lock:
            job = self._jobs.get(scope)
            job_view = (
                {
                    "id": job.job_id,
                    "files": job.filenames,
                    "completed": job.completed,
                    "total": len(job.filenames),
                    "error": job.error,
                }
                if job is not None
                else None
            )
        return {
            "id": dataset.dataset_scope,
            "name": dataset.name,
            "status": self._collection_status(scope),
            "documents": documents,
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
                for path in paths:
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
                with self._lock:
                    job.state = "building"
                graph = await service.build_graph(timeout=config.graph_timeout)
                if graph.get("state") != "DONE":
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
                "completed": job.completed,
                "total": len(job.filenames),
                "error": job.error,
            }

    def graph_status(self, scope: str) -> dict[str, Any]:
        if self.registry.get_dataset(scope) is None:
            raise WebError(404, "collection not found")
        index = self.registry.get_index(scope, "graph") or {}
        state = str(index.get("state", "UNKNOWN"))
        return {
            "collection_id": scope,
            "ready": state == "DONE",
            "state": state,
            "progress": index.get("progress"),
            "message": index.get("progress_msg", ""),
        }

    async def _ensure_chat_session(
        self,
        adapter: RAGFlowAdapter,
        dataset: Any,
        model: ModelOption,
        conversation_id: str,
    ) -> ChatHandle:
        with self._lock:
            handle = self._chats.get(conversation_id)
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
            f"deval-web-{conversation_id}",
            [dataset.remote_dataset_id],
            llm_model=model.ragflow_model,
        )
        chat_id = str(chat.get("id", ""))
        if not chat_id:
            raise DevalError("RAGFlow chat has no id")
        session = await adapter.create_chat_session(chat_id, name="DEval Webchat")
        session_id = str(session.get("id", ""))
        if not session_id:
            raise DevalError("RAGFlow chat session has no id")
        handle = ChatHandle(dataset.dataset_scope, model.id, chat_id, session_id)
        with self._lock:
            self._chats[conversation_id] = handle
        return handle

    async def _retrieve_citations(
        self,
        adapter: RAGFlowAdapter,
        config: Config,
        dataset: Any,
        scope: str,
        question: str,
    ) -> tuple[Any, list[Any]]:
        retrieved = await adapter.retrieve(
            question,
            [dataset.remote_dataset_id],
            page_size=5,
            similarity_threshold=0.0,
            knn_top_k=20,
            knn_num_candidates=40,
            rerank_candidates_count=20,
            highlight=False,
            use_kg=False,
        )
        citations = CitationResolver(
            self.registry, config.citation_threshold
        ).resolve_many(
            retrieved.references or retrieved.chunks,
            dataset_scope=scope,
        )
        return retrieved, citations

    def retrieve(self, scope: str, question: str) -> dict[str, Any]:
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
                    adapter, config, dataset, scope, question
                )
                return {
                    "question": question,
                    "results": retrieved.references or retrieved.chunks,
                    "citations": [self._citation_view(item) for item in citations],
                }
            finally:
                await adapter.aclose()

        try:
            return asyncio.run(run())
        except Exception as exc:
            raise WebError(502, f"RAGFlow retrieval failed: {exc}") from exc

    def create_chat_session(
        self, scope: str, model_id: str, conversation_id: str = ""
    ) -> dict[str, Any]:
        dataset = self.registry.get_dataset(scope)
        if dataset is None:
            raise WebError(404, "collection not found")
        graph = self.registry.get_index(scope, "graph")
        if graph is None or graph.get("state") != "DONE":
            raise WebError(409, "GraphRAG is not ready for this collection")
        model = self._model(model_id)
        conversation_id = self._conversation_id(conversation_id)
        config = self._collection_config(dataset.name)

        async def run():
            adapter = RAGFlowAdapter(
                config.base_url, config.api_key, timeout=config.request_timeout
            )
            try:
                await self._ensure_chat_session(
                    adapter, dataset, model, conversation_id
                )
            finally:
                await adapter.aclose()

        try:
            asyncio.run(run())
            return {
                "conversation_id": conversation_id,
                "collection_id": scope,
                "model": model.public_dict(),
            }
        except WebError:
            raise
        except Exception as exc:
            raise WebError(502, f"RAGFlow chat setup failed: {exc}") from exc

    def _citation_view(self, citation: Any) -> dict[str, Any]:
        extraction = (
            self.registry.get_extraction(citation.version_uid)
            if citation.version_uid
            else None
        )
        excerpt = citation.matched_text
        if not excerpt and citation.version_uid and citation.passage_uid:
            excerpt = next(
                (
                    str(item.get("text", ""))
                    for item in self.registry.list_passages(citation.version_uid)
                    if item.get("passage_uid") == citation.passage_uid
                ),
                "",
            )
        return {
            "document": extraction.source_basename
            if extraction
            else "Unaufgelöste Quelle",
            "page": citation.page_number,
            "excerpt": excerpt,
            "resolved": citation.passage_uid is not None,
            "source": citation.as_dict(),
        }

    def answer(
        self,
        collection_scope: str,
        model_id: str,
        question: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        dataset = self.registry.get_dataset(collection_scope)
        if dataset is None:
            raise WebError(404, "collection not found")
        graph = self.registry.get_index(collection_scope, "graph")
        if graph is None or graph.get("state") != "DONE":
            raise WebError(409, "GraphRAG is not ready for this collection")
        question = question.strip()
        if not question:
            raise WebError(400, "question must not be empty")
        if len(question) > MAX_QUESTION_LENGTH:
            raise WebError(400, "question is too long")
        conversation_id = self._conversation_id(conversation_id)
        model = self._model(model_id)
        config = self._collection_config(dataset.name)

        async def run() -> dict[str, Any]:
            adapter = RAGFlowAdapter(
                config.base_url, config.api_key, timeout=config.request_timeout
            )
            try:
                handle = await self._ensure_chat_session(
                    adapter, dataset, model, conversation_id
                )
                completion = await adapter.chat_completion(
                    handle.chat_id, question, handle.session_id
                )
                data = completion.get("data")
                answer = data.get("answer") if isinstance(data, dict) else None
                if not isinstance(answer, str) or not answer.strip():
                    raise DevalError("RAGFlow chat completion returned no answer")
                _, fallback_citations = await self._retrieve_citations(
                    adapter, config, dataset, collection_scope, question
                )
                reference_chunks = _completion_reference_chunks(completion)
                citations = (
                    CitationResolver(
                        self.registry, config.citation_threshold
                    ).resolve_many(reference_chunks, dataset_scope=collection_scope)
                    if reference_chunks
                    else fallback_citations
                )
                return {
                    "answer": answer.strip(),
                    "conversation_id": conversation_id,
                    "collection_id": collection_scope,
                    "model": model.public_dict(),
                    "citations": [self._citation_view(item) for item in citations],
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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
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
