import json
import time
from dataclasses import replace
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import Any, ClassVar

import pytest

from deval_ragflow.models import (
    DocumentExtraction,
    GraphStatus,
    Passage,
    RemoteStatus,
    RetrievalResult,
)
from deval_ragflow.web import (  # type: ignore[import-not-found]
    WebApplication,
    WebError,
    _contextualize_question,
    _parse_multipart,
    build_server,
)

PDF = Path(__file__).parent / "fixtures" / "golden.pdf"


class FakeAdapter:
    instances: ClassVar[list["FakeAdapter"]] = []
    empty_graph: ClassVar[bool] = False

    def __init__(self, *args, **kwargs):
        self.last_dataset_created = True
        self.chat_llm_model = ""
        self.chat_cross_languages: list[str] = []
        self.chat_session_calls = 0
        self.chat_completion_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[dict[str, object]] = []
        self.retrieve_questions: list[str] = []
        type(self).instances.append(self)

    async def aclose(self):
        return None

    async def list_chat_models(self):
        return [
            {
                "model_id": "remote-model",
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

    async def delete_document(self, *args, **kwargs):
        return {"code": 0}

    async def document_status(self, *args, **kwargs):
        return RemoteStatus("DONE", 1.0, "done", ("done",), 1, 1, {"run": "DONE"})

    async def ensure_chat(self, name, dataset_ids, *, llm_model="", cross_languages=()):
        self.chat_llm_model = llm_model
        self.chat_cross_languages = list(cross_languages)
        return {"id": "remote-chat"}

    async def create_chat_session(self, chat_id, *, name=""):
        self.chat_session_calls += 1
        return {"id": "remote-session"}

    async def chat_completion(self, chat_id, question, session_id, **kwargs):
        self.chat_completion_calls.append(
            {
                "chat_id": chat_id,
                "question": question,
                "session_id": session_id,
                **kwargs,
            }
        )
        return {
            "code": 0,
            "data": {
                "answer": "Eine echte Antwort [ID:0].",
                "reference": {
                    "chunks": [
                        {
                            "document_id": "remote-doc",
                            "image_id": "remote-dataset-chunk",
                            "positions": [[1, 1, 0, 1]],
                            "content": "A cited passage.",
                        }
                    ]
                },
            },
        }

    async def get_document_image(self, image_id, **kwargs):
        assert image_id == "remote-dataset-chunk"
        return b"jpeg-bytes", "image/jpeg"

    async def download_document(self, dataset_id, document_id, **kwargs):
        assert dataset_id == "remote-dataset" and document_id == "remote-doc"
        return b"%PDF-1.7", "application/pdf"

    async def retrieve(self, *args, **kwargs):
        self.retrieve_calls.append(kwargs)
        self.retrieve_questions.append(str(args[0]) if args else "")
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
        if type(self).empty_graph:
            return {"graph": {"nodes": [], "edges": []}}
        return {"graph": {"nodes": [{"id": "n"}], "edges": []}}


def test_follow_up_question_keeps_previous_topic_for_retrieval():
    messages = [
        {"role": "user", "content": "Was sind food shortage Indikatoren?"},
        {"role": "assistant", "content": "Eine Liste von Indikatoren."},
        {"role": "user", "content": "Antworte ausführlich"},
    ]
    assert _contextualize_question(messages, "Antworte ausführlich") == (
        "Was sind food shortage Indikatoren?\n\nAntworte ausführlich"
    )
    assert _contextualize_question(messages, "Welche Quellen gibt es?") == (
        "Welche Quellen gibt es?"
    )


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


def test_web_application_wires_collection_upload_graph_and_chat(monkeypatch, config):
    app_config = replace(config, llm_model="model@local@Test")
    monkeypatch.setattr("deval_ragflow.web.RAGFlowAdapter", FakeAdapter)
    app = WebApplication(app_config)
    try:
        collection = app.create_collection("Web collection")
        assert collection["id"] == "Web collection"
        assert app.models()[0]["id"] == "model"

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
        assert current["graph"]["state"] == "current"
        assert current["graph"]["document_count"] == 1

        app.registry.upsert_index(
            "Web collection",
            "graph",
            state="RUNNING",
            progress=0.4,
            progress_msg="building",
        )
        updating = app._collection_view("Web collection")
        assert updating["status"] == "building"
        assert updating["graph"]["state"] == "updating"
        assert updating["graph"]["progress"] == 0.4
        app.registry.upsert_index("Web collection", "graph", state="DONE", progress=1.0)

        response = app.answer(
            "Web collection",
            "model",
            "What is the result?",
            "conversation",
            cross_languages=["English", "German"],
        )
        assert response["answer"] == "Eine echte Antwort [ID:0]."
        assert response["model"]["id"] == "model"
        assert response["cross_languages"] == ["German", "English"]
        assert len(response["citations"]) == 1
        citation = response["citations"][0]
        passage_uid = citation["source"]["passage_uid"]
        expected_excerpt = next(
            item["text"]
            for item in app.registry.list_passages(current["document_records"][0]["id"])
            if item["passage_uid"] == passage_uid
        )
        assert citation["excerpt"] == expected_excerpt
        assert citation["excerpt"] != "A cited passage."
        assert citation["image_id"] == "remote-dataset-chunk"
        assert citation["document_url"] == (
            "/api/collections/Web%20collection/documents/"
            f"{current['document_records'][0]['id']}/file"
        )
        assert response["citations"][0]["image_url"] == (
            "/api/collections/Web%20collection/source-images/remote-dataset-chunk"
        )
        adapter = FakeAdapter.instances[-1]
        assert app.source_image("Web collection", "remote-dataset-chunk") == (
            b"jpeg-bytes",
            "image/jpeg",
        )
        assert app.source_document(
            "Web collection", current["document_records"][0]["id"]
        ) == (b"%PDF-1.7", "application/pdf", "golden.pdf")
        assert adapter.chat_session_calls == 0
        completion_call = adapter.chat_completion_calls[-1]
        assert completion_call["stateless"] is True
        assert completion_call["session_id"] == ""
        assert completion_call["messages"][-1] == {
            "role": "user",
            "content": "What is the result?",
        }
        assert adapter.chat_cross_languages == ["German", "English"]
        assert adapter.retrieve_calls[-1]["cross_languages"] == [
            "German",
            "English",
        ]
        app.answer(
            "Web collection",
            "model",
            "Antworte ausführlich",
            "conversation",
            messages=[
                {"role": "user", "content": "Wie wirkt sich EZ aus?"},
                {"role": "assistant", "content": "Eine Antwort."},
                {"role": "user", "content": "Was sind food shortage Indikatoren?"},
                {"role": "assistant", "content": "Eine Liste."},
            ],
            cross_languages=["German", "English"],
        )
        followup_adapter = FakeAdapter.instances[-1]
        expected_followup = (
            "Was sind food shortage Indikatoren?\n\nAntworte ausführlich"
        )
        assert (
            followup_adapter.chat_completion_calls[-1]["question"] == expected_followup
        )
        assert followup_adapter.retrieve_questions[-1] == expected_followup
        app.answer(
            "Web collection",
            "model",
            "What is the result?",
            "conversation",
            cross_languages=[],
        )
        assert len(app._chats) == 2

        second = DocumentExtraction(
            "second-doc",
            "second-version",
            "b" * 64,
            "second.pdf",
            "extracted",
            1,
            12,
            "second text",
            [
                Passage(
                    "second-p",
                    "second-doc",
                    "second-version",
                    1,
                    1,
                    None,
                    (0.0, 0.0, 10.0, 10.0),
                    0,
                    12,
                    "second text",
                )
            ],
        )
        app.registry.save_extraction(second)
        app.registry.reserve_mapping(
            "Web collection", "second-version", "second-remote.pdf"
        )
        app.registry.set_mapping_remote(
            "Web collection", "second-version", "remote-doc-2"
        )
        app.registry.set_mapping_state("Web collection", "second-version", "DONE")
        monkeypatch.setattr(FakeAdapter, "empty_graph", True)

        first_delete = app.start_delete(
            "Web collection", current["document_records"][0]["id"]
        )
        assert first_delete["job"]["operation"] == "delete"
        after_first_delete = first_delete
        for _ in range(100):
            after_first_delete = app._collection_view("Web collection")
            if after_first_delete["status"] == "ready":
                break
            time.sleep(0.02)
        else:
            pytest.fail(f"delete rebuild did not finish: {after_first_delete}")
        assert after_first_delete["documents"] == ["second.pdf"]
        assert after_first_delete["graph"]["state"] == "current"

        second_delete = app.start_delete(
            "Web collection", after_first_delete["document_records"][0]["id"]
        )
        assert second_delete["job"]["operation"] == "delete"
        after_delete = second_delete
        for _ in range(100):
            after_delete = app._collection_view("Web collection")
            if after_delete["status"] == "idle":
                break
            time.sleep(0.02)
        else:
            pytest.fail(f"delete did not finish: {after_delete}")
        assert after_delete["documents"] == []
        assert after_delete["graph"]["state"] == "empty"
        assert adapter.chat_llm_model == "model@local@Test"
        assert adapter.retrieve_calls[-1].get("use_kg") is False
    finally:
        app.close()


def test_upload_accepts_completed_empty_graph(config, tmp_path, monkeypatch):
    monkeypatch.setattr("deval_ragflow.web.RAGFlowAdapter", FakeAdapter)
    monkeypatch.setattr(FakeAdapter, "empty_graph", True)
    app = WebApplication(replace(config, llm_model="model@local@Test"))
    try:
        app.create_collection("Empty graph collection")
        pending = app.start_upload(
            "Empty graph collection", [("golden.pdf", PDF.read_bytes())]
        )
        job_id = pending["job"]["id"]
        job = app.job_status(job_id)
        for _ in range(100):
            job = app.job_status(job_id)
            if job["state"] in {"ready", "error"}:
                break
            time.sleep(0.02)
        assert job["state"] == "ready", job
        view = app._collection_view("Empty graph collection")
        assert view["status"] == "ready"
        assert view["graph"]["state"] == "current"
    finally:
        app.close()


def test_http_api_exposes_health_and_models(config, monkeypatch):
    monkeypatch.setattr("deval_ragflow.web.RAGFlowAdapter", FakeAdapter)
    app = WebApplication(config)
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
        assert json.loads(models.read())["models"][0]["id"] == "model"
    finally:
        connection.close()
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        app.close()


def test_http_api_proxies_source_image(config, monkeypatch):
    monkeypatch.setattr("deval_ragflow.web.RAGFlowAdapter", FakeAdapter)
    app = WebApplication(config)
    app.registry.upsert_dataset("test23", "remote-dataset", "test23")
    app.registry.save_extraction(
        DocumentExtraction(
            "document", "version", "a" * 64, "golden.pdf", "extracted", 1, 1, "", []
        )
    )
    app.registry.reserve_mapping("test23", "version", "golden.pdf")
    app.registry.set_mapping_remote("test23", "version", "remote-doc")
    app.registry.set_mapping_state("test23", "version", "DONE")
    server = build_server(app, "127.0.0.1", 0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection("127.0.0.1", server.server_address[1], timeout=2)
    try:
        connection.request(
            "GET", "/api/collections/test23/source-images/remote-dataset-chunk"
        )
        image = connection.getresponse()
        assert image.status == 200
        assert image.headers["Content-Type"] == "image/jpeg"
        assert image.read() == b"jpeg-bytes"

        connection.request("GET", "/api/collections/test23/documents/version/file")
        document = connection.getresponse()
        assert document.status == 200
        assert document.headers["Content-Type"] == "application/pdf"
        assert document.headers["Content-Disposition"].startswith("inline;")
        assert document.read() == b"%PDF-1.7"
    finally:
        connection.close()
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        app.close()


def test_models_are_loaded_from_ragflow_without_local_configuration(
    config, monkeypatch
):
    class DynamicFakeAdapter(FakeAdapter):
        async def list_chat_models(self):
            return [
                {
                    "model_id": "remote-model",
                    "name": "model",
                    "instance_name": "local",
                    "provider_name": "Test",
                    "model_type": ["chat"],
                },
                {
                    "model_id": "second-model",
                    "name": "second-model",
                    "instance_name": "cloud",
                    "provider_name": "OpenAI",
                    "model_type": ["chat"],
                },
            ]

    monkeypatch.setattr("deval_ragflow.web.RAGFlowAdapter", DynamicFakeAdapter)
    app = WebApplication(config)
    try:
        models = app.models()
        assert {item["id"] for item in models} == {"model", "second-model"}
        assert all(item["configured"] is True for item in models)
        assert {item["provider"] for item in models} == {
            "Test · local",
            "OpenAI · cloud",
        }
        assert app._model("second-model").ragflow_model == ("second-model@cloud@OpenAI")
    finally:
        app.close()


def test_cross_language_validation_rejects_unsupported_api_values(config):
    app = WebApplication(config)
    try:
        with pytest.raises(WebError, match="cross_languages must be an array"):
            app.retrieve("missing", "question", "English")
        with pytest.raises(WebError, match="unsupported cross-language"):
            app.retrieve("missing", "question", ["French"])
        with pytest.raises(WebError, match="must not contain duplicates"):
            app.retrieve("missing", "question", ["German", "German"])
    finally:
        app.close()


def test_chat_requires_parsed_documents(config):
    app = WebApplication(config)
    try:
        app.registry.upsert_dataset("test-dataset", "remote", "test-dataset")
        with pytest.raises(WebError, match="documents are not ready"):
            app.answer("test-dataset", "missing-model", "question", "conversation")
    finally:
        app.close()


def test_chat_works_before_graph_build_finishes(config, monkeypatch):
    monkeypatch.setattr("deval_ragflow.web.RAGFlowAdapter", FakeAdapter)
    app = WebApplication(replace(config, llm_model="model@local@Test"))
    try:
        app.registry.upsert_dataset("test-dataset", "remote-dataset", "test-dataset")
        app.registry.save_extraction(
            DocumentExtraction(
                "document",
                "version",
                "a" * 64,
                "golden.pdf",
                "extracted",
                1,
                1,
                "",
                [],
            )
        )
        app.registry.reserve_mapping("test-dataset", "version", "golden.pdf")
        app.registry.set_mapping_remote("test-dataset", "version", "remote-doc")
        app.registry.set_mapping_state("test-dataset", "version", "DONE")
        app.registry.upsert_index("test-dataset", "graph", state="UNSTART")

        view = app._collection_view("test-dataset")
        assert view["status"] == "outdated"
        assert view["graph"]["state"] == "outdated"
        response = app.answer("test-dataset", "model", "question", "conversation")
        assert response["answer"] == "Eine echte Antwort [ID:0]."
    finally:
        app.close()
