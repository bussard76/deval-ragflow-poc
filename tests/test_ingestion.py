from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from deval_ragflow.ingestion import IngestionService
from deval_ragflow.models import GraphStatus, RemoteStatus

PDF = Path(__file__).parent / "fixtures" / "golden.pdf"


class FakeAdapter:
    def __init__(self, statuses=None):
        self.uploads = 0
        self.uploaded_names = []
        self.finds = 0
        self.starts = 0
        self.cancels = 0
        self.statuses = list(statuses or [])

    async def ensure_dataset(self, name, **kwargs):
        return {"id": "dataset", "name": name}

    async def find_document(
        self, dataset_id, name, version_uid, max_pages=20
    ) -> dict[str, Any] | None:
        self.finds += 1

    async def upload_document(self, dataset_id, filename, content):
        self.uploads += 1
        self.uploaded_names.append(filename)
        return {"id": "document", "name": filename}

    async def get_document(self, dataset_id, document_id):
        return {"id": document_id, "run": "UNSTART"}

    async def patch_document(self, *args, **kwargs):
        return {"id": "document"}

    async def start_parse(self, dataset_id, document_ids):
        self.starts += 1
        return {"code": 0}

    async def document_status(self, dataset_id, document_id):
        if self.statuses:
            return self.statuses.pop(0)
        return RemoteStatus("DONE", 1.0, "done", ("done",), 1, 1, {"run": "DONE"})

    async def cancel_parse(self, dataset_id, document_ids):
        self.cancels += 1
        return {"code": 0}

    async def start_graph(self, dataset_id):
        return "task", {"code": 0}

    async def graph_status(self, dataset_id):
        return GraphStatus("task", "DONE", 1, "done", {"progress": 1})

    async def get_graph(self, dataset_id):
        return {"graph": {"nodes": [{"id": "n"}], "edges": []}}


def run(coro):
    return asyncio.run(coro)


def test_ingest_is_idempotent_and_commits_local_first(config, registry):
    adapter = FakeAdapter(
        [
            RemoteStatus("UNSTART", 0, "", (), 0, 0, {"run": "UNSTART"}),
            RemoteStatus("DONE", 1, "done", ("done",), 1, 1, {"run": "DONE"}),
        ]
    )
    service = IngestionService(config, registry, adapter)
    first = run(service.ingest(PDF))
    second = run(service.ingest(PDF))
    assert first.state == "DONE" and second.state == "DONE"
    assert adapter.uploads == 1 and adapter.starts == 1
    assert adapter.uploaded_names == [PDF.name]
    assert registry.counts()["documents"] == 1
    assert registry.counts()["ragflow_documents"] == 1


def test_scanned_or_mixed_without_approval_never_calls_remote(
    config, registry, tmp_path
):
    import base64

    import fitz

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(fitz.Rect(0, 0, 20, 20), stream=png)
    scanned = tmp_path / "scanned.pdf"
    scanned.write_bytes(doc.tobytes())
    doc.close()
    adapter = FakeAdapter()
    result = run(IngestionService(config, registry, adapter).ingest(scanned))
    assert result.state == "NOT_UPLOADED"
    assert adapter.uploads == 0


def test_timeout_sends_one_cancel_and_records_terminal_cancel(config, registry):
    adapter = FakeAdapter(
        [
            RemoteStatus(
                "RUNNING", 0.2, "running", ("running",), 0, 0, {"run": "RUNNING"}
            ),
            RemoteStatus(
                "CANCEL", 0.2, "cancelled", ("cancelled",), 0, 0, {"run": "CANCEL"}
            ),
        ]
    )
    service = IngestionService(config, registry, adapter)
    result = run(service.ingest(PDF, timeout=0))
    assert result.state == "CANCELED"
    assert adapter.cancels == 1
    assert (
        registry.get_index(config.dataset_scope, "parse", result.version_uid)["state"]
        == "CANCELED"
    )


def test_unknown_remote_state_never_counts_as_success(config, registry):
    adapter = FakeAdapter(
        [
            RemoteStatus(
                "UNKNOWN",
                None,
                "future state",
                ("future state",),
                None,
                None,
                {"run": "FUTURE"},
            )
        ]
    )
    result = run(IngestionService(config, registry, adapter).ingest(PDF))
    assert result.state == "UNKNOWN"
    assert (
        registry.get_mapping(config.dataset_scope, result.version_uid).state
        == "UNKNOWN"
    )


def test_graph_requires_llm_and_persists_task(config, registry):
    adapter = FakeAdapter()
    registry.upsert_dataset(config.dataset_scope, "dataset", config.dataset_name)
    service = IngestionService(config, registry, adapter)
    result = run(service.build_graph())
    assert result["state"] == "NOT_CONFIGURED"
