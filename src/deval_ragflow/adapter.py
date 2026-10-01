"""The only network boundary: a small async HTTP adapter for RAGFlow v0.27.2."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

try:
    import httpx
except ImportError:  # pragma: no cover - package installation supplies this
    httpx = None

from .config import normalize_base_url
from .errors import (
    AdapterError,
    DatasetConfigurationError,
    RAGFlowBusinessError,
    RAGFlowHTTPError,
    ReconciliationError,
    UnsafeOperation,
)
from .models import KNOWN_STATES, GraphStatus, RemoteStatus, RetrievalResult

NUMERIC_STATES = {
    0: "UNSTART",
    1: "RUNNING",
    2: "CANCEL",
    3: "DONE",
    4: "FAIL",
    5: "SCHEDULE",
}


def normalize_run_state(value: Any) -> str:
    """Map only the six numeric states documented by the pinned release."""
    if isinstance(value, bool):
        return "UNKNOWN"
    if isinstance(value, int):
        return NUMERIC_STATES.get(value, "UNKNOWN")
    if isinstance(value, float) and value.is_integer():
        try:
            return NUMERIC_STATES.get(int(value), "UNKNOWN")
        except (OverflowError, ValueError):
            return "UNKNOWN"
    if isinstance(value, str):
        text = value.strip().upper()
        if text.isdigit():
            try:
                return NUMERIC_STATES.get(int(text), "UNKNOWN")
            except ValueError:
                return "UNKNOWN"
        if text in ("CANCELED", "CANCELLED"):
            return "CANCEL"
        return text if text in KNOWN_STATES else "UNKNOWN"
    return "UNKNOWN"


def normalize_progress_message(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if isinstance(item, (list, tuple)):
                result.extend(normalize_progress_message(item))
            elif isinstance(item, dict):
                result.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                result.append(str(item))
        return tuple(result)
    return (str(value),)


def _as_progress(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _success_code(value: Any) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool) and value == 0
    ) or value == "0"


def _data(payload: Mapping[str, Any]) -> Any:
    return payload.get("data")


def _documents(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        docs = data.get("docs", data.get("documents"))
        if isinstance(docs, list):
            return [item for item in docs if isinstance(item, dict)]
        if isinstance(docs, dict):
            return [docs]
        if data.get("id") is not None:
            return [data]
    return []


def _chats(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        chats = data.get("chats", data.get("dialogs", data.get("data", [])))
        if isinstance(chats, list):
            return [item for item in chats if isinstance(item, dict)]
        if isinstance(chats, dict):
            return [chats]
        if data.get("id") is not None:
            return [data]
    return []


class RAGFlowAdapter:
    """Async HTTP adapter. Version-specific paths and response quirks live here."""

    api_prefix = "/api/v1"

    def __init__(
        self,
        base_url: str = "http://localhost:9380",
        api_key: str = "",
        timeout: float = 30.0,
        transport: Any = None,
        client: Any = None,
    ):
        if httpx is None:
            raise AdapterError("httpx is required; install the project dependencies")
        self.base_url = normalize_base_url(base_url)
        self.timeout = timeout
        self.api_key = api_key
        self.last_dataset_created = False
        self._owns_client = client is None
        if client is not None:
            self._client = client
            if api_key:
                self._client.headers.update({"Authorization": "Bearer " + api_key})
        else:
            headers = {"Accept": "application/json"}
            if api_key:
                headers.update({"Authorization": "Bearer " + api_key})
            self._client = httpx.AsyncClient(
                timeout=timeout, headers=headers, transport=transport
            )

    @staticmethod
    def _segment(value: Any) -> str:
        return quote(str(value), safe="")

    def _url(self, path: str) -> str:
        return self.base_url + self.api_prefix + "/" + path.lstrip("/")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def close(self) -> None:
        await self.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        files: Any = None,
        timeout: float | None = None,
        require_code: bool = True,
    ) -> dict[str, Any]:
        if httpx is None:
            raise AdapterError("httpx is required; install the project dependencies")
        attempts = 3 if method.upper() in {"GET", "HEAD"} else 1
        retryable = (
            httpx.RemoteProtocolError,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.ReadTimeout,
        )
        response: Any = None
        for attempt in range(attempts):
            try:
                response = await self._client.request(
                    method,
                    self._url(path),
                    params=params,
                    json=json_body,
                    files=files,
                    timeout=self.timeout if timeout is None else timeout,
                )
                break
            except Exception as exc:
                if isinstance(exc, AdapterError):
                    raise
                if attempt + 1 < attempts and isinstance(exc, retryable):
                    await asyncio.sleep(0.25 * (attempt + 1))
                    continue
                raise AdapterError(
                    f"RAGFlow request failed ({method.upper()} {path}): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        if response is None:
            raise AdapterError(f"RAGFlow request failed ({method.upper()} {path})")
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError):
            payload = {}
        if response.status_code >= 400:
            message = (
                payload.get("message", response.text[:500])
                if isinstance(payload, dict)
                else response.text[:500]
            )
            raise RAGFlowHTTPError(response.status_code, str(message), payload)
        if not isinstance(payload, dict):
            raise AdapterError("RAGFlow returned a non-object JSON response")
        code = payload.get("code")
        if require_code and not _success_code(code):
            raise RAGFlowBusinessError(
                code, str(payload.get("message", "unknown business error")), payload
            )
        if not require_code and "code" in payload and not _success_code(code):
            raise RAGFlowBusinessError(
                code, str(payload.get("message", "unknown business error")), payload
            )
        return payload

    async def _binary_request(
        self,
        path: str,
        *,
        label: str,
        content_type_prefix: str,
        timeout: float | None = None,
    ) -> tuple[bytes, str]:
        try:
            response = await self._client.request(
                "GET",
                self._url(path),
                timeout=self.timeout if timeout is None else timeout,
            )
        except Exception as exc:
            if isinstance(exc, AdapterError):
                raise
            raise AdapterError(
                f"RAGFlow {label} request failed: {type(exc).__name__}: {exc}"
            ) from exc
        if response.status_code >= 400:
            try:
                payload = response.json()
            except (ValueError, json.JSONDecodeError):
                payload = {}
            message = (
                payload.get("message", f"{label} unavailable")
                if isinstance(payload, dict)
                else f"{label} unavailable"
            )
            raise RAGFlowHTTPError(response.status_code, str(message), payload)
        content_type = (
            response.headers.get("content-type", "application/octet-stream")
            .split(";", 1)[0]
            .strip()
        )
        if not content_type.startswith(content_type_prefix):
            raise AdapterError(f"RAGFlow {label} response is not a file")
        return response.content, content_type

    async def get_document_image(
        self, image_id: str, *, timeout: float | None = None
    ) -> tuple[bytes, str]:
        """Fetch a RAGFlow source screenshot/blob without parsing it as JSON."""
        image_id = str(image_id).strip()
        if not image_id:
            raise AdapterError("RAGFlow image id must not be empty")
        content, content_type = await self._binary_request(
            f"/documents/images/{self._segment(image_id)}",
            label="source image",
            content_type_prefix="image/",
            timeout=timeout,
        )
        return content, content_type

    async def download_document(
        self, dataset_id: str, document_id: str, *, timeout: float | None = None
    ) -> tuple[bytes, str]:
        """Download the original file stored for a RAGFlow document."""
        if not str(dataset_id).strip() or not str(document_id).strip():
            raise AdapterError("RAGFlow dataset and document ids must not be empty")
        return await self._binary_request(
            f"/datasets/{self._segment(dataset_id)}/documents/{self._segment(document_id)}",
            label="document",
            content_type_prefix="application/",
            timeout=timeout,
        )

    async def healthz(self) -> dict[str, Any]:
        # healthz is documented as an HTTP readiness endpoint; accept a successful
        # body without code, but still reject an explicit non-zero business code.
        return await self._request("GET", "/system/healthz", require_code=False)

    async def list_datasets(
        self,
        *,
        page: int = 1,
        page_size: int = 30,
        name: str | None = None,
        dataset_id: str | None = None,
        max_pages: int = 100,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        current = max(1, page)
        for _ in range(max(1, max_pages)):
            params: dict[str, Any] = {
                "page": current,
                "page_size": page_size,
                "orderby": "create_time",
                "desc": "true",
            }
            if name is not None:
                params["name"] = name
            if dataset_id is not None:
                params["id"] = dataset_id
            payload = await self._request("GET", "/datasets", params=params)
            data = _data(payload)
            if isinstance(data, list):
                items = [item for item in data if isinstance(item, dict)]
                total = payload.get("total_datasets", payload.get("total"))
            elif isinstance(data, dict):
                candidate = data.get("datasets", data.get("data", []))
                items = (
                    [item for item in candidate if isinstance(item, dict)]
                    if isinstance(candidate, list)
                    else []
                )
                total = data.get("total_datasets", data.get("total"))
            else:
                items, total = [], None
            result.extend(items)
            try:
                total_count = int(total) if total is not None else None
            except (TypeError, ValueError):
                total_count = None
            if dataset_id or name:
                # The server applies exact filters in this API; don't hide a
                # second page when a test server omits totals.
                if len(items) < page_size:
                    break
            elif (
                total_count is not None
                and len(result) >= total_count
                or len(items) < page_size
            ):
                break
            current += 1
        return result

    async def list_chats(
        self,
        *,
        page: int = 1,
        page_size: int = 30,
        max_pages: int = 100,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        current = max(1, page)
        for _ in range(max(1, max_pages)):
            payload = await self._request(
                "GET",
                "/chats",
                params={
                    "page": current,
                    "page_size": page_size,
                    "orderby": "create_time",
                    "desc": "true",
                },
            )
            data = _data(payload)
            items = _chats(data)
            result.extend(items)
            total = data.get("total") if isinstance(data, dict) else None
            try:
                total_count = int(total) if total is not None else None
            except (TypeError, ValueError):
                total_count = None
            if (
                total_count is not None
                and len(result) >= total_count
                or len(items) < page_size
            ):
                break
            current += 1
        return result

    async def list_chat_models(self) -> list[dict[str, Any]]:
        """Return the active tenant chat models known by RAGFlow."""
        payload = await self._request("GET", "/models", params={"type": "chat"})
        data = _data(payload)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("models", data.get("data", []))
        else:
            items = []
        if not isinstance(items, list):
            raise AdapterError("RAGFlow model response has no models list")
        return [item for item in items if isinstance(item, dict)]

    async def ensure_chat(
        self,
        name: str,
        dataset_ids: Sequence[str],
        *,
        llm_model: str = "",
        cross_languages: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise AdapterError("RAGFlow chat name must not be empty")
        ids = [str(dataset_id) for dataset_id in dataset_ids if dataset_id]
        if not ids:
            raise AdapterError("RAGFlow chat needs at least one dataset")
        name = name.strip()
        matches = [
            item
            for item in await self.list_chats()
            if isinstance((chat_name := item.get("name")), str)
            and chat_name.casefold() == name.casefold()
        ]
        if len(matches) > 1:
            raise ReconciliationError(
                f"more than one RAGFlow chat has the deterministic name {name!r}"
            )
        chat_settings = {
            "dataset_ids": ids,
            # Give the model several retrieved chunks so broad summary questions
            # summarize the source material instead of one arbitrary chunk.
            "top_n": 5,
            "top_k": 20,
            "rerank_candidates_count": 20,
            "similarity_threshold": 0.0,
            "llm_setting": {"temperature": 0.1, "max_completion_tokens": 512},
            # Keep the summary broad and let RAGFlow place source markers next
            # to the facts it used. Source resolution remains local below.
            "prompt_config": {
                "quote": True,
                "cross_languages": list(cross_languages or ()),
                "system": (
                    "Antworte sachlich auf Deutsch und richte die Ausführlichkeit nach der Anfrage. "
                    "Wenn ausdrücklich eine ausführliche Antwort verlangt wird, erläutere die relevanten "
                    "Inhalte der Wissensbasis vollständig statt sie nur aufzuzählen. "
                    "Nutze ausschließlich die Wissensbasis. "
                    "Berücksichtige bei kurzen Rückfragen den bisherigen Gesprächskontext. "
                    "Fasse bei allgemeinen Fragen die relevanten Ausschnitte zusammen. "
                    "Belege jede sachliche Aussage direkt mit dem Quellenmarker von RAGFlow. "
                    "Wenn die Antwort nicht darin steht, sage: Nicht in der Wissensbasis gefunden.\n"
                    "Wissensbasis:\n{knowledge}"
                ),
            },
        }
        if llm_model:
            chat_settings["llm_id"] = llm_model
        if matches:
            chat = matches[0]
            if not chat.get("id"):
                raise AdapterError("RAGFlow chat response has no id")
            await self._request(
                "PATCH",
                f"/chats/{self._segment(chat['id'])}",
                json_body=chat_settings,
            )
            return chat
        body: dict[str, Any] = {"name": name, **chat_settings}
        if llm_model:
            body["llm_id"] = llm_model
        payload = await self._request("POST", "/chats", json_body=body)
        data = _data(payload)
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, dict) or not data.get("id"):
            raise AdapterError("RAGFlow chat create response has no id")
        return data

    async def create_chat_session(
        self, chat_id: str, *, name: str = "Deval CLI"
    ) -> dict[str, Any]:
        payload = await self._request(
            "POST",
            f"/chats/{self._segment(chat_id)}/sessions",
            json_body={"name": name},
        )
        data = _data(payload)
        if not isinstance(data, dict) or not data.get("id"):
            raise AdapterError("RAGFlow chat session response has no id")
        return data

    async def chat_completion(
        self,
        chat_id: str,
        question: str,
        session_id: str,
        *,
        messages: Sequence[Mapping[str, str]] | None = None,
        stateless: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if not question.strip():
            raise AdapterError("question must not be empty")
        request_timeout = max(360.0, self.timeout) if timeout is None else timeout
        if stateless:
            request_messages = [dict(message) for message in (messages or ())]
            if not request_messages:
                request_messages = [{"role": "user", "content": question}]
            openai_payload = await self._request(
                "POST",
                f"/openai/{self._segment(chat_id)}/chat/completions",
                json_body={
                    "model": "model",
                    "stream": False,
                    "messages": request_messages,
                    "extra_body": {"reference": True},
                },
                timeout=request_timeout,
                require_code=False,
            )
            choices = openai_payload.get("choices")
            first_choice = choices[0] if isinstance(choices, list) and choices else None
            message = (
                first_choice.get("message") if isinstance(first_choice, dict) else None
            )
            answer = message.get("content") if isinstance(message, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                raise AdapterError("RAGFlow stateless completion returned no answer")
            if answer.startswith("**ERROR**"):
                raise AdapterError(answer)
            return {
                "code": 0,
                "data": {
                    "answer": answer,
                    "reference": message.get("reference", {})
                    if isinstance(message, dict)
                    else {},
                },
                "raw": openai_payload,
            }

        payload = await self._request(
            "POST",
            "/chat/completions",
            json_body={
                "chat_id": chat_id,
                "session_id": session_id,
                "question": question,
                "stream": False,
                "quote": True,
            },
            timeout=request_timeout,
        )
        data = _data(payload)
        if not isinstance(data, dict):
            raise AdapterError("RAGFlow chat completion response has no data")
        answer = data.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise AdapterError("RAGFlow chat completion returned an empty answer")
        if answer.startswith("**ERROR**"):
            raise AdapterError(answer)
        return payload

    @staticmethod
    def _check_dataset_config(
        remote: Mapping[str, Any],
        embedding_model: str,
        chunk_method: str,
        parser_config: Mapping[str, Any],
    ) -> None:
        remote_embedding = remote.get("embedding_model") or remote.get("embd_id")
        if embedding_model and remote_embedding not in (None, "", embedding_model):
            raise DatasetConfigurationError(
                "existing dataset embedding model does not match configuration"
            )
        remote_chunk_method = remote.get("chunk_method") or remote.get("parser_id")
        if chunk_method and remote_chunk_method not in (None, "", chunk_method):
            raise DatasetConfigurationError(
                "existing dataset chunk method does not match configuration"
            )
        remote_parser = remote.get("parser_config")
        if isinstance(remote_parser, str):
            try:
                remote_parser = json.loads(remote_parser)
            except ValueError:
                remote_parser = None
        if parser_config and isinstance(remote_parser, dict):
            for key, expected in parser_config.items():
                if key in remote_parser and remote_parser[key] != expected:
                    raise DatasetConfigurationError(
                        "existing dataset parser config does not match configuration"
                    )

    async def ensure_dataset(
        self,
        name: str,
        *,
        embedding_model: str = "",
        llm_model: str = "",
        chunk_method: str = "naive",
        parser_config: dict[str, Any] | None = None,
        description: str = "deval PDF provenance PoC",
    ) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise AdapterError("RAGFlow dataset name must not be empty")
        name = name.strip()
        # The pinned server treats a missing name-filter result as a
        # business error, so list pages and filter locally before deciding to
        # create a dataset.
        wanted_name = name.casefold()
        matches = [
            item
            for item in await self.list_datasets()
            if isinstance((remote_name := item.get("name")), str)
            and remote_name.casefold() == wanted_name
        ]
        if len(matches) > 1:
            raise ReconciliationError(
                f"more than one RAGFlow dataset has the deterministic name {name!r}"
            )
        if matches:
            remote = matches[0]
            self._check_dataset_config(
                remote, embedding_model, chunk_method, parser_config or {}
            )
            if not remote.get("id"):
                raise AdapterError("RAGFlow dataset response has no id")
            self.last_dataset_created = False
            return remote
        body: dict[str, Any] = {
            "name": name,
            "description": description,
            "permission": "me",
            "chunk_method": chunk_method,
            "parser_config": parser_config or {},
        }
        if embedding_model:
            body["embedding_model"] = embedding_model
        # v0.27.2 has no documented create-dataset llm field. Keep it out of
        # the request; the model is checked by doctor/GraphRAG setup.
        payload = await self._request("POST", "/datasets", json_body=body)
        data = _data(payload)
        if isinstance(data, list):
            data = data[0] if data else None
        if not isinstance(data, dict) or not data.get("id"):
            raise AdapterError("RAGFlow dataset create response has no id")
        self.last_dataset_created = True
        return data

    async def list_documents(
        self,
        dataset_id: str,
        *,
        page: int = 1,
        page_size: int = 30,
        name: str | None = None,
        document_id: str | None = None,
        max_pages: int = 20,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        current = max(1, page)
        for _ in range(max(1, max_pages)):
            params: dict[str, Any] = {
                "page": current,
                "page_size": page_size,
                "orderby": "create_time",
                "desc": "true",
            }
            if name is not None:
                params["name"] = name
            if document_id is not None:
                params["id"] = document_id
            payload = await self._request(
                "GET", f"/datasets/{self._segment(dataset_id)}/documents", params=params
            )
            data = _data(payload)
            items = _documents(data)
            result.extend(items)
            total = (
                data.get(
                    "total", data.get("total_documents", data.get("total_datasets"))
                )
                if isinstance(data, dict)
                else None
            )
            try:
                total_count = int(total) if total is not None else None
            except (TypeError, ValueError):
                total_count = None
            if document_id and items:
                break
            if total_count is not None and len(result) >= total_count:
                break
            if len(items) < page_size:
                break
            current += 1
        return result

    async def _fetch_document(
        self, dataset_id: str, document_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = await self._request(
            "GET",
            f"/datasets/{self._segment(dataset_id)}/documents",
            params={
                "id": document_id,
                "page": 1,
                "page_size": 1,
                "orderby": "create_time",
                "desc": "true",
            },
        )
        docs = _documents(_data(payload))
        if not docs or not docs[0].get("id"):
            raise ReconciliationError(f"RAGFlow document {document_id} was not found")
        if str(docs[0]["id"]) != str(document_id):
            raise ReconciliationError(
                f"RAGFlow returned a different document id for {document_id}"
            )
        return docs[0], payload

    async def get_document(self, dataset_id: str, document_id: str) -> dict[str, Any]:
        document, _ = await self._fetch_document(dataset_id, document_id)
        return document

    async def find_document(
        self, dataset_id: str, remote_name: str, version_uid: str, max_pages: int = 20
    ) -> dict[str, Any] | None:
        # v0.27.2 returns a business error when a name filter matches no
        # document. List bounded pages and match locally so a first ingestion
        # can correctly proceed to upload.
        docs = await self.list_documents(dataset_id, max_pages=max_pages)
        matches = []
        for doc in docs:
            metadata = (
                doc.get("meta_fields")
                or doc.get("metadata")
                or doc.get("document_metadata")
                or {}
            )
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except ValueError:
                    metadata = {}
            remote_doc_name = doc.get("name") or doc.get("location")
            metadata_version = (
                metadata.get("version_uid") if isinstance(metadata, dict) else None
            )
            if metadata_version:
                if metadata_version == version_uid:
                    matches.append(doc)
                # The visible filename is not the identity; another version
                # may legitimately use the same uploaded name.
            elif remote_doc_name == remote_name:
                # A document uploaded before its metadata PATCH is still
                # recoverable by deterministic name; the caller patches it.
                matches.append(doc)
        if len(matches) > 1:
            raise ReconciliationError(
                f"multiple remote documents match version {version_uid}"
            )
        return matches[0] if matches else None

    async def upload_document(
        self, dataset_id: str, filename: str, content: bytes
    ) -> dict[str, Any]:
        files = {"file": (filename, content, "application/pdf")}
        payload = await self._request(
            "POST", f"/datasets/{self._segment(dataset_id)}/documents", files=files
        )
        docs = _documents(_data(payload))
        if not docs:
            data = _data(payload)
            if isinstance(data, dict) and data.get("id"):
                docs = [data]
        if not docs or not docs[0].get("id"):
            raise AdapterError("RAGFlow upload response has no document id")
        return docs[0]

    async def patch_document(
        self,
        dataset_id: str,
        document_id: str,
        *,
        name: str,
        metadata: dict[str, Any],
        chunk_method: str = "naive",
        parser_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = {
            "name": name,
            "meta_fields": metadata,
            "chunk_method": chunk_method,
            "parser_config": parser_config or {},
            "enabled": 1,
        }
        payload = await self._request(
            "PATCH",
            f"/datasets/{self._segment(dataset_id)}/documents/{self._segment(document_id)}",
            json_body=body,
        )
        data = _data(payload)
        return data if isinstance(data, dict) else {"raw": payload}

    async def delete_document(
        self, dataset_id: str, document_id: str
    ) -> dict[str, Any]:
        """Remove one document from a dataset (RAGFlow v0.27.2)."""
        return await self._request(
            "DELETE",
            f"/datasets/{self._segment(dataset_id)}/documents",
            json_body={"ids": [document_id]},
        )

    async def start_parse(
        self, dataset_id: str, document_ids: Sequence[str]
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/datasets/{self._segment(dataset_id)}/chunks",
            json_body={"document_ids": list(document_ids)},
        )

    async def document_status(self, dataset_id: str, document_id: str) -> RemoteStatus:
        document, response_payload = await self._fetch_document(dataset_id, document_id)
        raw_run = document.get("run", document.get("status"))
        state = normalize_run_state(raw_run)
        messages = normalize_progress_message(document.get("progress_msg"))
        return RemoteStatus(
            state=state,
            progress=_as_progress(document.get("progress")),
            progress_msg="\n".join(messages),
            progress_messages=messages,
            chunk_count=document.get("chunk_count"),
            token_count=document.get("token_count"),
            raw=response_payload,
        )

    async def cancel_parse(
        self, dataset_id: str, document_ids: Sequence[str]
    ) -> dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/datasets/{self._segment(dataset_id)}/chunks",
            json_body={"document_ids": list(document_ids)},
        )

    async def retrieve(
        self,
        question: str,
        dataset_ids: Sequence[str],
        *,
        document_ids: Sequence[str] | None = None,
        page: int = 1,
        page_size: int = 30,
        similarity_threshold: float = 0.2,
        vector_similarity_weight: float = 0.3,
        knn_top_k: int = 1024,
        knn_num_candidates: int | None = None,
        rerank_candidates_count: int = 64,
        keyword: bool = False,
        highlight: bool = True,
        use_kg: bool = False,
        cross_languages: Sequence[str] | None = None,
        reference_metadata: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        body: dict[str, Any] = {
            "question": question,
            "dataset_ids": list(dataset_ids),
            "document_ids": list(document_ids or []),
            "page": page,
            "page_size": page_size,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "knn_top_k": knn_top_k,
            "knn_num_candidates": max(2048, knn_top_k)
            if knn_num_candidates is None
            else knn_num_candidates,
            "rerank_candidates_count": rerank_candidates_count,
            "keyword": keyword,
            "highlight": highlight,
            "use_kg": use_kg,
        }
        if cross_languages is not None:
            body["cross_languages"] = list(cross_languages)
        if reference_metadata is not None:
            body["reference_metadata"] = reference_metadata
        payload = await self._request("POST", "/retrieval", json_body=body)
        data = _data(payload)
        if not isinstance(data, dict):
            data = {}
        chunks_value = data.get("chunks")
        chunks = (
            [item for item in chunks_value if isinstance(item, dict)]
            if isinstance(chunks_value, list)
            else []
        )
        references_value = data.get("references", data.get("reference"))
        if isinstance(references_value, dict):
            references_value = references_value.get("chunks", [])
        references = (
            [item for item in references_value if isinstance(item, dict)]
            if isinstance(references_value, list)
            else chunks
        )
        return RetrievalResult(chunks=chunks, references=references, raw=payload)

    async def start_graph(self, dataset_id: str) -> tuple[str, dict[str, Any]]:
        payload = await self._request(
            "POST",
            f"/datasets/{self._segment(dataset_id)}/index",
            params={"type": "graph"},
        )
        data = _data(payload)
        task_id = data.get("graphrag_task_id") if isinstance(data, dict) else None
        if not task_id and isinstance(data, dict):
            task_id = data.get("task_id") or data.get("id")
        if not task_id:
            raise AdapterError("RAGFlow GraphRAG response has no task id")
        return str(task_id), payload

    async def graph_status(self, dataset_id: str) -> GraphStatus:
        payload = await self._request(
            "GET",
            f"/datasets/{self._segment(dataset_id)}/index",
            params={"type": "graph"},
        )
        data = _data(payload)
        raw = data if isinstance(data, dict) else {}
        raw_state = raw.get("run", raw.get("state", raw.get("status")))
        state = normalize_run_state(raw_state) if raw_state is not None else "UNKNOWN"
        if state == "UNKNOWN":
            text = str(raw_state).upper() if raw_state is not None else ""
            if text in ("SUCCESS", "SUCCEEDED", "COMPLETED", "COMPLETE"):
                state = "DONE"
            elif text in ("FAILED", "FAILURE"):
                state = "FAIL"
            elif text in ("CANCELED", "CANCELLED"):
                state = "CANCEL"
        progress = _as_progress(raw.get("progress"))
        messages = normalize_progress_message(raw.get("progress_msg"))
        message_text = "\n".join(messages).lower()
        if (
            raw_state is None
            and state == "UNKNOWN"
            and progress is not None
            and progress < 0
        ):
            state = "FAIL"
        if (
            raw_state is None
            and state == "UNKNOWN"
            and any(word in message_text for word in ("failed", "failure", "error"))
        ):
            state = "FAIL"
        # Graph task progress is documented on a 0..1 scale: sub-terminal
        # progress is running, while 1.0 is terminal. Missing fields never
        # imply success.
        if (
            raw_state is None
            and state == "UNKNOWN"
            and progress is not None
            and 0 <= progress < 1.0
        ):
            state = "RUNNING"
        if (
            raw_state is None
            and state == "UNKNOWN"
            and progress is not None
            and progress >= 1.0
        ):
            state = "DONE"
        return GraphStatus(
            task_id=str(raw.get("id")) if raw.get("id") is not None else None,
            state=state,
            progress=progress,
            progress_msg="\n".join(messages),
            raw=payload,
        )

    async def get_graph(self, dataset_id: str) -> dict[str, Any]:
        payload = await self._request(
            "GET", f"/datasets/{self._segment(dataset_id)}/graph"
        )
        data = _data(payload)
        return data if isinstance(data, dict) else {"raw": payload}

    async def delete_owned_dataset(
        self, dataset_id: str, owned_dataset_id: str | None
    ) -> dict[str, Any]:
        if not dataset_id or owned_dataset_id != dataset_id:
            raise UnsafeOperation(
                "dataset deletion requires an exact owned local dataset id"
            )
        # Never add delete_all here: the pinned API treats omitted/null ids as
        # a possible delete-all operation.
        return await self._request(
            "DELETE", "/datasets", json_body={"ids": [dataset_id], "delete_all": False}
        )

    async def delete_dataset(
        self, dataset_id: str, *, owned_dataset_id: str | None = None
    ) -> dict[str, Any]:
        return await self.delete_owned_dataset(dataset_id, owned_dataset_id)

    # Small semantic aliases keep callers from reaching around this boundary.
    upload = upload_document
    parse = start_parse
    status = document_status
    cancel = cancel_parse
    query = retrieve
    start_graphrag = start_graph
    trace_graphrag = graph_status
    knowledge_graph = get_graph
    delete_owned = delete_owned_dataset
    get_status = document_status
    cancel_documents = cancel_parse
    retrieve_chunks = retrieve


AsyncRAGFlowAdapter = RAGFlowAdapter
RAGFlowHTTPAdapter = RAGFlowAdapter
