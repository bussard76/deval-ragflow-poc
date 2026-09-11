import asyncio
import json

import httpx  # type: ignore[import-not-found]
import pytest

from deval_ragflow.adapter import (
    RAGFlowAdapter,
    normalize_progress_message,
    normalize_run_state,
)
from deval_ragflow.errors import (
    AdapterError,
    DatasetConfigurationError,
    RAGFlowBusinessError,
    ReconciliationError,
    UnsafeOperation,
)


def run(coro):
    return asyncio.run(coro)


def test_state_and_message_normalization():
    assert normalize_run_state("3") == "DONE"
    assert normalize_run_state(4) == "FAIL"
    assert normalize_run_state("SCHEDULE") == "SCHEDULE"
    assert normalize_run_state(5) == "SCHEDULE"
    assert normalize_run_state("future-state") == "UNKNOWN"
    assert normalize_progress_message(["one", 2]) == ("one", "2")


def test_adapter_routes_multipart_patch_status_retrieval_graph_and_delete():
    requests = []
    status_runs = iter(["RUNNING", "DONE"])

    def handler(request):
        requests.append(request)
        path = request.url.path
        if path == "/api/v1/models" and request.method == "GET":
            assert request.url.params["type"] == "chat"
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": [
                        {
                            "name": "model",
                            "instance_name": "local",
                            "provider_name": "Test",
                            "model_type": ["chat"],
                        }
                    ],
                },
            )
        if path == "/api/v1/datasets" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": [
                        {
                            "id": "dataset",
                            "name": "kb",
                            "embedding_model": "e",
                            "chunk_method": "naive",
                            "parser_config": {"x": 1},
                        }
                    ],
                },
            )
        if path == "/api/v1/chats" and request.method == "GET":
            return httpx.Response(
                200, json={"code": 0, "data": {"chats": [], "total": 0}}
            )
        if path == "/api/v1/chats" and request.method == "POST":
            body = json.loads(request.content)
            assert body["dataset_ids"] == ["dataset"] and body["llm_id"] == "model"
            assert body["llm_setting"]["max_completion_tokens"] == 512
            assert body["prompt_config"]["quote"] is True
            return httpx.Response(
                200, json={"code": 0, "data": {"id": "chat", "name": body["name"]}}
            )
        if path == "/api/v1/chat/completions":
            body = json.loads(request.content)
            assert body == {
                "chat_id": "chat",
                "session_id": "session",
                "question": "fact?",
                "stream": False,
                "quote": True,
            }
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "answer": "The answer.",
                        "reference": {
                            "chunks": [
                                {
                                    "document_id": "remote-doc",
                                    "positions": [[1, 1, 0, 1]],
                                }
                            ]
                        },
                    },
                },
            )
        if path.endswith("/sessions") and request.method == "POST":
            return httpx.Response(200, json={"code": 0, "data": {"id": "session"}})
        if path.endswith("/documents") and request.method == "POST":
            body = request.content
            assert b"golden.pdf" in body and b"pdf-bytes" in body
            return httpx.Response(
                200,
                json={"code": 0, "data": [{"id": "remote-doc", "name": "golden.pdf"}]},
            )
        if path.endswith("/documents/remote-doc") and request.method == "PATCH":
            body = json.loads(request.content)
            assert body["meta_fields"]["version_uid"] == "version"
            assert body["enabled"] == 1
            return httpx.Response(200, json={"code": 0, "data": {"id": "remote-doc"}})
        if path.endswith("/documents") and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "docs": [
                            {
                                "id": "remote-doc",
                                "run": next(status_runs),
                                "progress": 1 if False else 0.5,
                                "progress_msg": ["step", "failure detail"],
                                "chunk_count": 1,
                            }
                        ]
                    },
                },
            )
        if path.endswith("/chunks") and request.method == "POST":
            assert json.loads(request.content) == {"document_ids": ["remote-doc"]}
            return httpx.Response(200, json={"code": 0})
        if path.endswith("/chunks") and request.method == "DELETE":
            assert json.loads(request.content) == {"document_ids": ["remote-doc"]}
            return httpx.Response(200, json={"code": 0})
        if path == "/api/v1/retrieval":
            body = json.loads(request.content)
            assert (
                body["knn_top_k"] == 10
                and body["knn_num_candidates"] == 20
                and body["use_kg"] is True
            )
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "chunks": [
                            {
                                "id": "chunk",
                                "document_id": "remote-doc",
                                "positions": [[1, 1, 0, 1]],
                            }
                        ]
                    },
                },
            )
        if path.endswith("/run_graphrag"):
            return httpx.Response(
                200, json={"code": 0, "data": {"graphrag_task_id": "task"}}
            )
        if path.endswith("/trace_graphrag"):
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {"id": "task", "progress": 1, "progress_msg": "done"},
                },
            )
        if path.endswith("/knowledge_graph"):
            return httpx.Response(
                200, json={"code": 0, "data": {"graph": {"nodes": [], "edges": []}}}
            )
        if path == "/api/v1/datasets" and request.method == "DELETE":
            assert json.loads(request.content) == {
                "ids": ["dataset"],
                "delete_all": False,
            }
            return httpx.Response(200, json={"code": 0})
        raise AssertionError(f"unexpected {request.method} {path}")

    async def exercise():
        adapter = RAGFlowAdapter(
            "http://localhost:9380/api/v1",
            "secret",
            transport=httpx.MockTransport(handler),
        )
        try:
            assert (await adapter.list_chat_models())[0]["name"] == "model"
            dataset = await adapter.ensure_dataset(
                "kb", embedding_model="e", chunk_method="naive", parser_config={"x": 1}
            )
            assert dataset["id"] == "dataset"
            chat = await adapter.ensure_chat(
                "deval-cli-dataset", ["dataset"], llm_model="model"
            )
            assert chat["id"] == "chat"
            session = await adapter.create_chat_session("chat")
            assert session["id"] == "session"
            completion = await adapter.chat_completion("chat", "fact?", "session")
            assert completion["data"]["answer"] == "The answer."
            uploaded = await adapter.upload_document(
                "dataset", "golden.pdf", b"pdf-bytes"
            )
            assert uploaded["id"] == "remote-doc"
            await adapter.patch_document(
                "dataset",
                "remote-doc",
                name="golden.pdf",
                metadata={"version_uid": "version"},
            )
            first = await adapter.document_status("dataset", "remote-doc")
            second = await adapter.document_status("dataset", "remote-doc")
            assert first.state == "RUNNING" and "failure detail" in first.progress_msg
            assert second.state == "DONE"
            await adapter.start_parse("dataset", ["remote-doc"])
            await adapter.cancel_parse("dataset", ["remote-doc"])
            result = await adapter.retrieve(
                "question",
                ["dataset"],
                knn_top_k=10,
                knn_num_candidates=20,
                use_kg=True,
            )
            assert result.references[0]["document_id"] == "remote-doc"
            task, _ = await adapter.start_graph("dataset")
            assert task == "task"
            assert (await adapter.graph_status("dataset")).state == "DONE"
            assert (await adapter.get_graph("dataset"))["graph"] is not None
            await adapter.delete_owned_dataset("dataset", "dataset")
            assert any("authorization" in dict(r.headers) for r in requests)
        finally:
            await adapter.aclose()

    run(exercise())


def test_ensure_chat_reapplies_llm_id_to_existing_chat():
    def handler(request):
        if request.url.path == "/api/v1/chats" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "chats": [{"id": "chat", "name": "deval-cli-dataset"}],
                        "total": 1,
                    },
                },
            )
        if request.method == "PATCH" and request.url.path == "/api/v1/chats/chat":
            body = json.loads(request.content)
            assert body["llm_id"] == "model"
            return httpx.Response(
                200,
                json={"code": 0, "data": {"id": "chat", "name": "deval-cli-dataset"}},
            )
        if request.url.path == "/api/v1/chats" and request.method == "POST":
            raise AssertionError("chat should be patched, not created")
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    async def exercise():
        adapter = RAGFlowAdapter(
            transport=httpx.MockTransport(handler),
        )
        try:
            chat = await adapter.ensure_chat(
                "deval-cli-dataset", ["dataset"], llm_model="model"
            )
            assert chat["id"] == "chat"
        finally:
            await adapter.aclose()

    run(exercise())


def test_graph_progress_is_running_until_explicit_completion():
    async def exercise():
        adapter = RAGFlowAdapter(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, json={"code": 0, "data": {"id": "task", "progress": 0.5}}
                )
            )
        )
        try:
            status = await adapter.graph_status("dataset")
            assert status.state == "RUNNING"
            assert status.raw["data"]["progress"] == 0.5
        finally:
            await adapter.aclose()

    run(exercise())


def test_business_code_and_safe_delete_fail_closed():
    def business(_):
        return httpx.Response(200, json={"code": 101, "message": "bad request"})

    async def exercise():
        adapter = RAGFlowAdapter(transport=httpx.MockTransport(business))
        try:
            with pytest.raises(RAGFlowBusinessError):
                await adapter.list_datasets()
            with pytest.raises(UnsafeOperation):
                await adapter.delete_owned_dataset("remote", "other")
        finally:
            await adapter.aclose()

    run(exercise())


def test_conflicting_deterministic_document_metadata_fails_closed():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "docs": [
                        {
                            "id": "remote",
                            "name": "deval-version.pdf",
                            "meta_fields": {"version_uid": "different"},
                        }
                    ]
                },
            },
        )

    async def exercise():
        adapter = RAGFlowAdapter(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(ReconciliationError):
                await adapter.find_document("dataset", "deval-version.pdf", "version")
        finally:
            await adapter.aclose()

    run(exercise())


def test_existing_dataset_config_mismatch_and_multiple_reconciliation():
    def handler(request):
        if request.url.path == "/api/v1/datasets":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": [{"id": "d", "name": "kb", "embedding_model": "different"}],
                },
            )
        raise AssertionError(request.url)

    async def exercise():
        adapter = RAGFlowAdapter(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(DatasetConfigurationError):
                await adapter.ensure_dataset("kb", embedding_model="expected")
        finally:
            await adapter.aclose()

    run(exercise())


def test_chat_completion_rejects_error_answers():
    async def exercise():
        adapter = RAGFlowAdapter(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"code": 0, "data": {"answer": "**ERROR**: provider failed"}},
                )
            )
        )
        try:
            with pytest.raises(AdapterError, match="provider failed"):
                await adapter.chat_completion("chat", "fact?", "session")
        finally:
            await adapter.aclose()

    run(exercise())
