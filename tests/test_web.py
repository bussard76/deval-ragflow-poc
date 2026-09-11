import json
import time
from dataclasses import replace
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import ClassVar

import pytest

from deval_ragflow.models import GraphStatus, RemoteStatus, RetrievalResult
from deval_ragflow.web import (  # type: ignore[import-not-found]
    WebApplication,
    WebError,
    _parse_multipart,
    build_server,
)

PDF = Path(__file__).parent / "fixtures" / "golden.pdf"


class FakeAdapter:
    instances: ClassVar[list["FakeAdapter"]] = []

    def __init__(self, *args, **kwargs):
        self.last_dataset_created = True
        self.chat_llm_model = ""
        self.retrieve_calls: list[dict[str, object]] = []
        type(self).instances.append(self)

    async def aclose(self):
        return None

    async def list_chat_models(self):
        return [
            {
                "name": "model",
                "instance_name": "local",
                "provider_name": "Test",
                "model_type": ["chat"],
            }
        ]

    async def ensure_dataset(self, name, **kwargs):
        return {"id": "remote-dataset", "name": name}

    async def find_document(self, *args, **kwargs):
        return None

    async def upload_document(self, dataset_id, filename, content):
        return {"id": "remote-doc", "name": filename}

    async def patch_document(self, *args, **kwargs):
        return {"id": "remote-doc"}

    async def document_status(self, *args, **kwargs):
        return RemoteStatus("DONE", 1.0, "done", ("done",), 1, 1, {"run": "DONE"})

    async def ensure_chat(self, name, dataset_ids, *, llm_model=""):
        self.chat_llm_model = llm_model
        return {"id": "remote-chat"}

    async def create_chat_session(self, chat_id, *, name=""):
        return {"id": "remote-session"}

    async def chat_completion(self, chat_id, question, session_id, **kwargs):
        return {
            "code": 0,
            "data": {
                "answer": "Eine echte Antwort [ID:0].",
                "reference": {
                    "chunks": [
                        {
                            "document_id": "remote-doc",
                            "positions": [[1, 1, 0, 1]],
                            "content": "A cited passage.",
                        }
                    ]
                },
            },
        }

    async def retrieve(self, *args, **kwargs):
        self.retrieve_calls.append(kwargs)
        return RetrievalResult(
            chunks=[],
            references=[
                {
                    "document_id": "remote-doc",
                    "positions": [[1, 1, 0, 1]],
                }
            ],
            raw={"code": 0},
        )

    async def start_graph(self, dataset_id):
        return "graph-task", {"code": 0}

    async def graph_status(self, dataset_id):
        return GraphStatus("graph-task", "DONE", 1.0, "done", {"progress": 1})

    async def get_graph(self, dataset_id):
        return {"graph": {"nodes": [{"id": "n"}], "edges": []}}


def test_parse_multipart_extracts_multiple_files():
    body = (
        b"--demo\r\n"
        b'Content-Disposition: form-data; name="files"; filename="one.pdf"\r\n'
        b"Content-Type: application/pdf\r\n\r\n"
        b"one\r\n"
        b"--demo\r\n"
        b'Content-Disposition: form-data; name="files"; filename="two.pdf"\r\n'
        b"Content-Type: application/pdf\r\n\r\n"
        b"two\r\n"
        b"--demo--\r\n"
    )

    assert _parse_multipart("multipart/form-data; boundary=demo", body) == [
        ("one.pdf", b"one"),
        ("two.pdf", b"two"),
    ]


def test_web_application_wires_collection_upload_graph_and_chat(
    monkeypatch, config, tmp_path
):
    model_config = tmp_path / "models.json"
    model_config.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "local",
                        "name": "Local",
                        "provider": "Test",
                        "ragflow_model": "model@local@Test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    app_config = replace(
        config,
        llm_model="model@local@Test",
        model_config_path=model_config,
    )
    monkeypatch.setattr("deval_ragflow.web.RAGFlowAdapter", FakeAdapter)
    app = WebApplication(app_config)
    try:
        collection = app.create_collection("Web collection")
        assert collection["id"] == "Web collection"
        assert app.models()[0]["id"] == "local"

        pending = app.start_upload("Web collection", [("golden.pdf", PDF.read_bytes())])
        assert pending["status"] == "processing"
        current = app._collection_view("Web collection")
        for _ in range(100):
            current = app._collection_view("Web collection")
            if current["status"] == "ready":
                break
            time.sleep(0.02)
        else:
            pytest.fail(f"upload did not become ready: {current}")

        assert current["documents"] == ["golden.pdf"]
        response = app.answer(
            "Web collection", "local", "What is the result?", "conversation"
        )
        assert response["answer"] == "Eine echte Antwort [ID:0]."
        assert response["model"]["id"] == "local"
        assert len(response["citations"]) == 1
        assert response["citations"][0]["excerpt"]
        adapter = FakeAdapter.instances[-1]
        assert adapter.chat_llm_model == "model@local@Test"
        assert adapter.retrieve_calls[-1].get("use_kg") is False
    finally:
        app.close()


def test_http_api_exposes_health_and_models(config, tmp_path, monkeypatch):
    model_config = tmp_path / "models.json"
    model_config.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "local",
                        "name": "Local",
                        "provider": "Test",
                        "ragflow_model": "model",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("deval_ragflow.web.RAGFlowAdapter", FakeAdapter)
    app = WebApplication(replace(config, model_config_path=model_config))
    server = build_server(app, "127.0.0.1", 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=2)
    try:
        connection.request("GET", "/api/health", headers={"Origin": "http://evil.test"})
        health = connection.getresponse()
        assert health.status == 200
        assert json.loads(health.read()) == {"ok": True}
        assert health.headers.get("Access-Control-Allow-Origin") is None

        connection.request("GET", "/api/models")
        models = connection.getresponse()
        assert models.status == 200
        assert json.loads(models.read())["models"][0]["id"] == "local"
    finally:
        connection.close()
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        app.close()


def test_models_filter_catalog_to_ragflow_chat_models(config, tmp_path, monkeypatch):
    model_config = tmp_path / "models.json"
    model_config.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "local",
                        "name": "Local",
                        "provider": "Test",
                        "ragflow_model": "model@local@Test",
                    },
                    {
                        "id": "missing",
                        "name": "Missing",
                        "provider": "Test",
                        "ragflow_model": "missing@codex@Test",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("deval_ragflow.web.RAGFlowAdapter", FakeAdapter)
    app = WebApplication(replace(config, model_config_path=model_config))
    try:
        models = app.models()
        assert {item["id"] for item in models} == {"local", "missing"}
        assert (
            next(item for item in models if item["id"] == "local")["configured"] is True
        )
        assert (
            next(item for item in models if item["id"] == "missing")["configured"]
            is False
        )
        with pytest.raises(WebError, match="not configured in RAGFlow"):
            app._model("missing")
    finally:
        app.close()


def test_chat_requires_ready_graph(config, tmp_path, monkeypatch):
    model_config = tmp_path / "models.json"
    model_config.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "local",
                        "name": "Local",
                        "provider": "Test",
                        "ragflow_model": "model",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    app = WebApplication(replace(config, model_config_path=model_config))
    try:
        app.registry.upsert_dataset("test-dataset", "remote", "test-dataset")
        with pytest.raises(WebError, match="GraphRAG is not ready"):
            app.answer("test-dataset", "local", "question", "conversation")
    finally:
        app.close()
