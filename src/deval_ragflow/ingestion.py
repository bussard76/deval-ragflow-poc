"""Local-first ingestion orchestration with bounded polling and cancellation."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from .adapter import RAGFlowAdapter
from .config import Config
from .errors import AdapterError, ReconciliationError
from .models import PROVENANCE_SCHEMA, DocumentExtraction, IngestionResult, RemoteStatus
from .pdf import extract_pdf
from .registry import Registry


def _is_missing_remote_document(exc: Exception) -> bool:
    return (
        isinstance(exc, ReconciliationError)
        or getattr(exc, "status_code", None) == 404
        or getattr(exc, "code", None) == 102
    )


class IngestionService:
    def __init__(self, config: Config, registry: Registry, adapter: RAGFlowAdapter):
        self.config = config
        self.registry = registry
        self.adapter = adapter

    @staticmethod
    def remote_name(extraction: DocumentExtraction) -> str:
        # The hash is the identity; the source basename is metadata, not an id.
        return f"deval-{extraction.version_uid}.pdf"

    @staticmethod
    def remote_metadata(extraction: DocumentExtraction) -> dict[str, Any]:
        return {
            "document_uid": extraction.document_uid,
            "version_uid": extraction.version_uid,
            "sha256": extraction.sha256,
            "provenance_schema": PROVENANCE_SCHEMA,
            "source_basename": extraction.source_basename,
        }

    @staticmethod
    def _remote_matches(
        remote: dict[str, Any], remote_name: str, version_uid: str
    ) -> bool:
        metadata = (
            remote.get("meta_fields")
            or remote.get("metadata")
            or remote.get("document_metadata")
            or {}
        )
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (TypeError, ValueError):
                metadata = {}
        if isinstance(metadata, dict) and metadata.get("version_uid"):
            return metadata.get("version_uid") == version_uid
        if remote.get("name") == remote_name or remote.get("location") == remote_name:
            return True
        # A direct lookup by the locally persisted RAGFlow id is authoritative
        # when this release omits both name and metadata from the response.
        return not remote.get("name") and not remote.get("location")

    async def _dataset(self):
        existing = self.registry.get_dataset(self.config.dataset_scope)
        remote = await self.adapter.ensure_dataset(
            self.config.dataset_name,
            embedding_model=self.config.embedding_model,
            llm_model=self.config.llm_model,
            chunk_method=self.config.chunk_method,
            parser_config=self.config.parser_config,
        )
        remote_id = str(remote.get("id", ""))
        if not remote_id:
            raise AdapterError("RAGFlow dataset response has no id")
        dataset = self.registry.upsert_dataset(
            self.config.dataset_scope,
            remote_id,
            self.config.dataset_name,
            self.config.embedding_model,
            self.config.llm_model,
            {
                "chunk_method": self.config.chunk_method,
                "parser_config": self.config.parser_config,
                "embedding_model": self.config.embedding_model,
                "llm_model": self.config.llm_model,
            },
            owned=existing.owned
            if existing is not None
            else bool(getattr(self.adapter, "last_dataset_created", False)),
        )
        return dataset

    def _result(
        self,
        extraction: DocumentExtraction,
        state: str,
        dataset_id: str | None = None,
        document_id: str | None = None,
        progress: float | None = None,
        message: str = "",
    ) -> IngestionResult:
        return IngestionResult(
            document_uid=extraction.document_uid,
            version_uid=extraction.version_uid,
            classification=extraction.classification,
            state=state,
            dataset_scope=self.config.dataset_scope,
            remote_dataset_id=dataset_id,
            remote_document_id=document_id,
            progress=progress,
            message=message,
            extraction=extraction,
        )

    async def ingest(
        self,
        path: Any,
        *,
        allow_mixed: bool = False,
        timeout: float | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> IngestionResult:
        extraction = extract_pdf(
            path,
            parser_config=self.config.provenance_parser_config,
            max_pages=self.config.max_document_pages,
            max_bytes=self.config.max_document_bytes,
            max_text_chars=self.config.max_text_chars,
        )
        self.registry.save_extraction(extraction)
        if not extraction.uploadable or (
            extraction.classification == "mixed" and not allow_mixed
        ):
            return self._result(
                extraction,
                "NOT_UPLOADED",
                message="classification is not approved for remote upload",
            )

        dataset = await self._dataset()
        dataset_id = dataset.remote_dataset_id
        remote_name = self.remote_name(extraction)
        mapping = self.registry.reserve_mapping(
            self.config.dataset_scope, extraction.version_uid, remote_name
        )
        remote_doc = None
        if mapping.remote_document_id:
            try:
                remote_doc = await self.adapter.get_document(
                    dataset_id, mapping.remote_document_id
                )
            except AdapterError as exc:
                # A crash can leave a stale local remote id. Reconcile a
                # documented not-found response; propagate other failures.
                if not _is_missing_remote_document(exc):
                    raise
            if remote_doc is not None and not self._remote_matches(
                remote_doc, remote_name, extraction.version_uid
            ):
                remote_doc = None
        if remote_doc is None:
            remote_doc = await self.adapter.find_document(
                dataset_id,
                remote_name,
                extraction.version_uid,
                max_pages=self.config.reconcile_max_pages,
            )
        if remote_doc is None:
            if extraction.raw_bytes is None:
                raise AdapterError(
                    "uploadable extraction has no transient source bytes"
                )
            remote_doc = await self.adapter.upload_document(
                dataset_id, remote_name, extraction.raw_bytes
            )
        remote_document_id = str(remote_doc.get("id", ""))
        if not remote_document_id:
            raise AdapterError("RAGFlow document response has no id")
        self.registry.set_mapping_remote(
            self.config.dataset_scope,
            extraction.version_uid,
            remote_document_id,
            state="UPLOADED",
            metadata=self.remote_metadata(extraction),
        )
        await self.adapter.patch_document(
            dataset_id,
            remote_document_id,
            name=remote_name,
            metadata=self.remote_metadata(extraction),
            chunk_method=self.config.chunk_method,
            parser_config=self.config.parser_config,
        )
        self.registry.set_mapping_state(
            self.config.dataset_scope, extraction.version_uid, "METADATA_UPDATED"
        )
        status = await self.adapter.document_status(dataset_id, remote_document_id)
        if status.state == "DONE":
            self._persist_parse(status, extraction, remote_document_id, "DONE")
            return self._result(
                extraction,
                "DONE",
                dataset_id,
                remote_document_id,
                status.progress,
                status.progress_msg,
            )
        if status.state == "FAIL":
            self._persist_parse(status, extraction, remote_document_id, "FAIL")
            return self._result(
                extraction,
                "FAIL",
                dataset_id,
                remote_document_id,
                status.progress,
                status.progress_msg,
            )
        if status.state in ("CANCEL", "CANCELED"):
            self._persist_parse(status, extraction, remote_document_id, "CANCELED")
            return self._result(
                extraction,
                "CANCELED",
                dataset_id,
                remote_document_id,
                status.progress,
                status.progress_msg,
            )
        if status.state == "UNKNOWN":
            return self._unknown(
                extraction,
                dataset_id,
                remote_document_id,
                status.progress_msg,
                status.progress,
                status.raw,
            )
        if status.state not in ("RUNNING", "SCHEDULE"):
            await self.adapter.start_parse(dataset_id, [remote_document_id])
        self._persist_parse(status, extraction, remote_document_id, "RUNNING")
        return await self._poll_parse(
            extraction,
            dataset_id,
            remote_document_id,
            timeout if timeout is not None else self.config.parse_timeout,
            cancel_event,
        )

    def _persist_parse(
        self,
        status: RemoteStatus,
        extraction: DocumentExtraction,
        remote_document_id: str,
        state: str,
    ) -> None:
        self.registry.upsert_index(
            self.config.dataset_scope,
            "parse",
            extraction.version_uid,
            state=state,
            progress=status.progress,
            progress_msg=status.progress_msg,
            raw=status.raw,
        )
        self.registry.set_mapping_state(
            self.config.dataset_scope,
            extraction.version_uid,
            state,
            status.progress_msg,
        )

    def _unknown(
        self, extraction, dataset_id, document_id, message, progress=None, raw=None
    ):
        self.registry.upsert_index(
            self.config.dataset_scope,
            "parse",
            extraction.version_uid,
            state="UNKNOWN",
            progress=progress,
            progress_msg=message,
            raw=raw,
        )
        self.registry.set_mapping_state(
            self.config.dataset_scope, extraction.version_uid, "UNKNOWN", message
        )
        return self._result(
            extraction, "UNKNOWN", dataset_id, document_id, progress, message
        )

    async def _poll_parse(
        self,
        extraction: DocumentExtraction,
        dataset_id: str,
        document_id: str,
        timeout: float,
        cancel_event: asyncio.Event | None,
    ) -> IngestionResult:
        deadline = time.monotonic() + max(0.0, timeout)
        delay = max(0.0, self.config.poll_interval)
        cancel_sent = False
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    cancel_sent = True
                    return await self._cancel_and_grace(
                        extraction, dataset_id, document_id, "cancel requested"
                    )
                if time.monotonic() >= deadline:
                    cancel_sent = True
                    return await self._cancel_and_grace(
                        extraction, dataset_id, document_id, "parse timeout"
                    )
                status = await self.adapter.document_status(dataset_id, document_id)
                if status.state == "DONE":
                    self._persist_parse(status, extraction, document_id, "DONE")
                    return self._result(
                        extraction,
                        "DONE",
                        dataset_id,
                        document_id,
                        status.progress,
                        status.progress_msg,
                    )
                if status.state == "FAIL":
                    self._persist_parse(status, extraction, document_id, "FAIL")
                    return self._result(
                        extraction,
                        "FAIL",
                        dataset_id,
                        document_id,
                        status.progress,
                        status.progress_msg,
                    )
                if status.state in ("CANCEL", "CANCELED"):
                    self._persist_parse(status, extraction, document_id, "CANCELED")
                    return self._result(
                        extraction,
                        "CANCELED",
                        dataset_id,
                        document_id,
                        status.progress,
                        status.progress_msg,
                    )
                if status.state == "UNKNOWN":
                    return self._unknown(
                        extraction,
                        dataset_id,
                        document_id,
                        status.progress_msg,
                        status.progress,
                        status.raw,
                    )
                self._persist_parse(status, extraction, document_id, status.state)
                remaining = max(0.0, deadline - time.monotonic())
                if remaining <= 0:
                    cancel_sent = True
                    return await self._cancel_and_grace(
                        extraction, dataset_id, document_id, "parse timeout"
                    )
                await asyncio.sleep(min(delay, remaining))
                delay = min(5.0, max(0.05, delay * 2 if delay else 0.05))
        except asyncio.CancelledError:
            # Cancellation is best effort, but never silently reports success.
            if not cancel_sent:
                await self._send_cancel(
                    extraction, dataset_id, document_id, "task cancelled"
                )
            raise

    async def _send_cancel(
        self, extraction, dataset_id: str, document_id: str, message: str
    ) -> bool:
        try:
            await self.adapter.cancel_parse(dataset_id, [document_id])
        except AdapterError:
            self.registry.set_mapping_state(
                self.config.dataset_scope, extraction.version_uid, "UNKNOWN", message
            )
            self.registry.upsert_index(
                self.config.dataset_scope,
                "parse",
                extraction.version_uid,
                state="UNKNOWN",
                progress_msg=message,
            )
            return False
        self.registry.set_mapping_state(
            self.config.dataset_scope,
            extraction.version_uid,
            "CANCEL_REQUESTED",
            message,
        )
        self.registry.upsert_index(
            self.config.dataset_scope,
            "parse",
            extraction.version_uid,
            state="CANCEL_REQUESTED",
            progress_msg=message,
        )
        return True

    async def _cancel_and_grace(
        self,
        extraction,
        dataset_id: str,
        document_id: str,
        reason: str,
        cancel_sent: bool = False,
    ) -> IngestionResult:
        if not cancel_sent:
            await self._send_cancel(extraction, dataset_id, document_id, reason)
        deadline = time.monotonic() + max(0.0, self.config.cancel_timeout)
        while time.monotonic() < deadline:
            try:
                status = await self.adapter.document_status(dataset_id, document_id)
            except AdapterError:
                self.registry.set_mapping_state(
                    self.config.dataset_scope,
                    extraction.version_uid,
                    "UNKNOWN",
                    "remote cancellation status could not be confirmed",
                )
                self.registry.upsert_index(
                    self.config.dataset_scope,
                    "parse",
                    extraction.version_uid,
                    state="UNKNOWN",
                    progress_msg="remote cancellation status could not be confirmed",
                )
                return self._result(
                    extraction,
                    "UNKNOWN",
                    dataset_id,
                    document_id,
                    message="remote cancellation status could not be confirmed",
                )
            if status.state == "DONE":
                self._persist_parse(status, extraction, document_id, "DONE")
                return self._result(
                    extraction,
                    "DONE",
                    dataset_id,
                    document_id,
                    status.progress,
                    status.progress_msg,
                )
            if status.state == "FAIL":
                self._persist_parse(status, extraction, document_id, "FAIL")
                return self._result(
                    extraction,
                    "FAIL",
                    dataset_id,
                    document_id,
                    status.progress,
                    status.progress_msg,
                )
            if status.state in ("CANCEL", "CANCELED"):
                self._persist_parse(status, extraction, document_id, "CANCELED")
                return self._result(
                    extraction,
                    "CANCELED",
                    dataset_id,
                    document_id,
                    status.progress,
                    status.progress_msg,
                )
            await asyncio.sleep(
                min(
                    max(0.05, self.config.poll_interval),
                    max(0.0, deadline - time.monotonic()),
                )
            )
        message = "remote cancellation status could not be confirmed"
        self.registry.set_mapping_state(
            self.config.dataset_scope, extraction.version_uid, "UNKNOWN", message
        )
        self.registry.upsert_index(
            self.config.dataset_scope,
            "parse",
            extraction.version_uid,
            state="UNKNOWN",
            progress_msg=message,
        )
        return self._result(
            extraction, "UNKNOWN", dataset_id, document_id, message=message
        )

    async def cancel(self, version_uid: str) -> dict[str, Any]:
        mapping = self.registry.get_mapping(self.config.dataset_scope, version_uid)
        if mapping is None or not mapping.remote_document_id:
            raise ReconciliationError(
                f"no remote document mapping for version {version_uid}"
            )
        dataset = self.registry.get_dataset(self.config.dataset_scope)
        if dataset is None:
            raise ReconciliationError("no local dataset mapping")
        raw = await self.adapter.cancel_parse(
            dataset.remote_dataset_id, [mapping.remote_document_id]
        )
        self.registry.set_mapping_state(
            self.config.dataset_scope,
            version_uid,
            "CANCEL_REQUESTED",
            "cancel requested",
        )
        self.registry.upsert_index(
            self.config.dataset_scope,
            "parse",
            version_uid,
            state="CANCEL_REQUESTED",
            progress_msg="cancel requested",
        )
        return raw

    async def build_graph(self, *, timeout: float | None = None) -> dict[str, Any]:
        dataset = self.registry.get_dataset(self.config.dataset_scope)
        if dataset is None:
            raise ReconciliationError(
                "ingest at least one document before building a graph"
            )
        if not self.config.llm_model:
            self.registry.upsert_index(
                self.config.dataset_scope,
                "graph",
                state="NOT_CONFIGURED",
                progress_msg="RAGFLOW_LLM_MODEL is not configured",
            )
            return {
                "state": "NOT_CONFIGURED",
                "message": "RAGFLOW_LLM_MODEL is not configured",
            }
        mappings = self.registry.mappings_for_dataset(self.config.dataset_scope)
        if not mappings or any(mapping.state != "DONE" for mapping in mappings):
            return {
                "state": "NOT_READY",
                "message": "all mapped documents must have DONE parsing state",
            }
        existing = self.registry.get_index(self.config.dataset_scope, "graph")
        if existing and existing.get("state") == "DONE":
            graph = await self.adapter.get_graph(dataset.remote_dataset_id)
            graph_obj = graph.get("graph", graph) if isinstance(graph, dict) else None
            if not isinstance(graph_obj, dict) or (
                not graph_obj.get("nodes") and not graph_obj.get("edges")
            ):
                return {
                    "state": "EMPTY",
                    "task_id": existing.get("remote_task_id"),
                    "graph": graph,
                    "message": "graph has no nodes or edges",
                }
            return {
                "state": "DONE",
                "task_id": existing.get("remote_task_id"),
                "graph": graph,
            }

        # A persisted active/unknown task is resumed rather than queued again.
        # The trace endpoint is dataset-scoped and exposes the current task.
        task_id = (
            existing.get("remote_task_id")
            if existing
            and existing.get("state") in ("UNSTART", "RUNNING", "SCHEDULE", "UNKNOWN")
            else None
        )
        started = existing.get("raw", {}) if task_id and existing else {}
        if not task_id:
            task_id, started = await self.adapter.start_graph(dataset.remote_dataset_id)
            self.registry.upsert_index(
                self.config.dataset_scope,
                "graph",
                remote_task_id=task_id,
                state="RUNNING",
                raw=started,
            )
        deadline = time.monotonic() + (
            timeout if timeout is not None else self.config.graph_timeout
        )
        while time.monotonic() < deadline:
            status = await self.adapter.graph_status(dataset.remote_dataset_id)
            self.registry.upsert_index(
                self.config.dataset_scope,
                "graph",
                remote_task_id=task_id,
                state=status.state,
                progress=status.progress,
                progress_msg=status.progress_msg,
                raw=status.raw,
            )
            if status.state == "FAIL":
                return {
                    "state": "FAIL",
                    "task_id": task_id,
                    "message": status.progress_msg,
                    "raw": status.raw,
                }
            if status.state in ("CANCEL", "CANCELED"):
                return {
                    "state": "CANCELED",
                    "task_id": task_id,
                    "message": status.progress_msg,
                    "raw": status.raw,
                }
            if status.state == "DONE":
                graph = await self.adapter.get_graph(dataset.remote_dataset_id)
                graph_obj = (
                    graph.get("graph", graph) if isinstance(graph, dict) else None
                )
                if not isinstance(graph_obj, dict) or (
                    not graph_obj.get("nodes") and not graph_obj.get("edges")
                ):
                    return {
                        "state": "EMPTY",
                        "task_id": task_id,
                        "graph": graph,
                        "message": "graph has no nodes or edges",
                    }
                return {
                    "state": "DONE",
                    "task_id": task_id,
                    "graph": graph,
                    "raw": status.raw,
                }
            await asyncio.sleep(
                min(
                    max(0.05, self.config.poll_interval),
                    max(0.0, deadline - time.monotonic()),
                )
            )
        self.registry.upsert_index(
            self.config.dataset_scope,
            "graph",
            remote_task_id=task_id,
            state="UNKNOWN",
            progress_msg="graph timeout",
        )
        return {"state": "UNKNOWN", "task_id": task_id, "message": "graph timeout"}


async def ingest_pdf(
    path: Any,
    config: Config,
    registry: Registry,
    adapter: RAGFlowAdapter,
    *,
    allow_mixed: bool = False,
    timeout: float | None = None,
    cancel_event: asyncio.Event | None = None,
) -> IngestionResult:
    """Convenience entry point; all remote work still goes through the adapter."""
    return await IngestionService(config, registry, adapter).ingest(
        path, allow_mixed=allow_mixed, timeout=timeout, cancel_event=cancel_event
    )
