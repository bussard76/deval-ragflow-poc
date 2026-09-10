# Decisions and unresolved assumptions

## Pinned upstream

This PoC pins official RAGFlow tag `v0.27.1` at commit `b9df87c4c75a5b0d35c90d15329fc0f6f91cb73e`, based on the official quickstart, HTTP reference, SDK, and source review. `scripts/ragflow-compose.sh` verifies and stages the exact upstream `docker/` directory at runtime. A release upgrade must update this commit, the compose wrapper, the route/response contract, fixtures, and tests together; `main` must not be mixed with this release.

The exact wrapper/runtime choices are intentional: active profiles are `elasticsearch,cpu,metadata-mysql,tei-cpu`, with the official TEI service using `BAAI/bge-small-en-v1.5` for a practical CPU embedding default. Dependency passwords are random and ignored, published ports are loopback-bound, official named volumes persist data, and sandbox/privileged/Docker-socket services are inactive. The staged directory also has a bookkeeping marker so unmanaged directories are never replaced. Docker Compose >=2.26.1, x86 hardware, 4 CPUs, 16 GB RAM, 50 GB disk, and `vm.max_map_count >= 262144` are upstream operational assumptions.

## Adapter route table

The single async `httpx` adapter normalizes `RAGFLOW_BASE_URL` (default `http://localhost:9380`) without `/api/v1`, then owns these v0.27.1 paths:

| Operation | Route |
| --- | --- |
| readiness | `GET /system/healthz` |
| dataset list/create | `GET/POST /datasets` |
| document list/metadata | `GET /datasets/{dataset_id}/documents?id={document_id}` |
| document upload | `POST /datasets/{dataset_id}/documents` (multipart) |
| document metadata | `PATCH /datasets/{dataset_id}/documents/{document_id}` |
| start built-in parsing | `POST /datasets/{dataset_id}/chunks` |
| cancel built-in parsing | `DELETE /datasets/{dataset_id}/chunks` |
| retrieval | `POST /retrieval` |
| chat assistant/session/completion | `GET/POST/PATCH /chats`, `POST /chats/{chat_id}/sessions`, `POST /chat/completions` |
| GraphRAG start/status/get | `POST run_graphrag`, `GET trace_graphrag`, `GET knowledge_graph` |
| owned dataset deletion | `DELETE /datasets` |

The dataset creation body uses only fields accepted by the pinned strict request model (`name`, description, permission, chunk method, parser config, and optional embedding model); it deliberately omits the older illustrative `language` field. The dataset deletion payload is exactly `{"ids":[owned_id],"delete_all":false}`. The document-list API uses `page`, `page_size`, `name`, and `id`; however, the pinned handler returns a business error when `name`/`id` matches nothing, so the adapter lists bounded pages and matches locally before upload/reconciliation. Reconciliation fails on multiple deterministic matches. GraphRAG's v0.27.1 aliases return `graphrag_task_id` and a trace object whose progress/message fields differ from document parsing; completion is accepted only from an explicit terminal state or documented `progress >= 1.0`, never from a missing field. The current `main` `/index?type=graph` and `/graph` routes are deliberately not substituted.

## Identity and provenance

`document_uid` is lowercase SHA-256 of raw PDF bytes. `version_uid` is SHA-256 of canonical sorted compact JSON containing `document_uid`, `provenance_schema`, `extractor_version`, and `parser_config`. Remote dataset/document IDs are opaque. A source revision key beyond this content/parser identity is out of scope.

The SQLite schema has `datasets`, `documents`, `document_versions`, `passages`, `ragflow_documents`, and `indexes`. Foreign keys, WAL, busy timeout, unique constraints, and short `BEGIN IMMEDIATE` transactions provide local idempotence. `ragflow_documents` is unique by `(dataset_scope, version_uid)`; the local dataset row includes ownership and configuration. Ownership is true only for a dataset created by this adapter run or an already-owned local mapping; reusing a pre-existing same-name dataset cannot make it resettable. A crash after remote upload and before mapping commit can still duplicate a remote upload because RAGFlow has no idempotency key; deterministic name/metadata reconciliation handles one match and surfaces multiple matches.

PyMuPDF uses sorted 1-based text blocks as paragraphs. Canonical text joins normalized non-empty block text with newline separators and page form-feed separators. Character ranges are Unicode-code-point half-open offsets. Coordinates are PDF points rounded to two decimals. `section_path` is the active PDF outline/TOC path, or null; it is not semantic heading detection. Malformed, encrypted/password-protected, empty, and fully scanned files are registered but not uploaded. Mixed files need explicit approval. Byte/page/text limits run before remote work.

## Failure and citation behavior

Only documented document states `UNSTART`, `RUNNING`, `SCHEDULE`, `DONE`, `FAIL`, and `CANCEL` and numeric mappings `0..5` are recognized. Unknown states are `UNKNOWN`; HTTP success with a non-zero JSON business code is failure. Progress messages can be strings or lists and are preserved/normalized without discarding failure text.

Ingestion commits local extraction first, ensures a matching dataset, reserves/reconciles/upload metadata, starts `/chunks`, and polls against a monotonic deadline with bounded backoff. Timeout or cancellation sends one `DELETE /chunks` and polls a bounded grace period. Remote cancellation is eventual; partial chunks are possible and an unconfirmed outcome is `UNKNOWN`.

Citation normalization is Unicode NFKC, casefold, and whitespace collapse. v0.27.1 PDF positions are `[page, left, right, top, bottom]` and are matched against local PyMuPDF bboxes; a position is used only when it maps unambiguously to the local remote-document mapping. Otherwise exact normalized text is tried, then multiset token Dice overlap with a default threshold of `0.60`. Ties sort by score, page, paragraph, character offset, and passage UID. Unknown IDs, ambiguous mappings, and low scores are unresolved; filename/page guessing and LLM citation matching are prohibited.

GraphRAG and `ask` require parsed chunks and a configured RAGFlow LLM. The PoC documents a local Ollama `qwen2.5:0.5b` provider and keeps its model data in a named Docker volume; Ollama is a separate local sidecar because the official RAGFlow compose files do not include an LLM. `ask` uses a deterministic chat name and concise CPU-oriented settings, then resolves displayed citations from a separate retrieval response. Empty graph, provider/model failure, and timeout are truthful outcomes. Query sends `use_kg=true` only when explicitly requested.

## Safe reset and open risks

`reset-test-data` requires an exact owned dataset ID and `--confirm 'DELETE DEVAL TEST DATA'`. It never deletes arbitrary IDs, enables `delete_all`, or removes Docker volumes. Local rows are removed only after remote deletion succeeds; remote failure retains them.

Open risks: upstream may change list filters, business codes, graph shapes, progress fields, or reference positions; PyMuPDF block/TOC/float behavior can vary outside the pin; the official stack is resource-heavy and x86-oriented; provider calls may leave the machine; remote cancellation and crash recovery are eventually consistent; the 0.5B CPU chat model may produce short or low-quality answers, so citations remain the source of truth.
