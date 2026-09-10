# Architecture

```text
PDF bytes
   │ SHA-256 + PyMuPDF (local)
   ▼
DocumentExtraction ── SQLite registry (.data/registry.sqlite3)
   │ passages, offsets, bboxes, version identity
   ▼
IngestionService ── RAGFlowAdapter (async httpx, /api/v1 only)
   │ dataset ensure → deterministic upload/reconcile → metadata → chunks parse
   ▼
RAGFlow v0.27.1
   ├─ retrieval / GraphRAG aliases ── raw chunks + CitationResolver ── local passages
   └─ chat/session completion ── concise answer
                                      │
                                      └─ separate retrieval provides displayed citations
```

## Boundaries

- `pdf.py` is the only PDF parser. It hashes bytes before opening them and never stores an absolute source path.
- `registry.py` is one SQLite database with foreign keys, WAL, busy timeout, unique identities, and short `BEGIN IMMEDIATE` writes. It stores source basenames and provenance, not files or secrets.
- `adapter.py` is the only network layer. It owns the base URL normalization, `/api/v1` prefix, bearer header, HTTP/business-code checks, multipart upload, status normalization, raw responses, retrieval/chat controls, GraphRAG aliases, and safe dataset deletion.
- `ingestion.py` is local-first and has no queue. It commits extraction, reserves `(dataset_scope, version_uid)`, reconciles deterministic remote names/metadata, polls with a monotonic deadline, sends one cancellation request, and records uncertain cancellation as `UNKNOWN`.
- `citations.py` never guesses from filename/page and never calls an LLM. A remote position must map unambiguously; text fallbacks are normalized and scored deterministically. `ask` deliberately resolves citations from a separate retrieval response rather than trusting generated text.
- `cli.py` is an argparse shell over these components. Docker orchestration is deliberately outside Python in `scripts/ragflow-compose.sh`.

## State

The local mapping states include `RESERVED`, `UPLOADED`, `METADATA_UPDATED`, parse states (`RUNNING`, `DONE`, `FAIL`, `CANCEL_REQUESTED`, `CANCELED`, `UNKNOWN`), and graph task states. RAGFlow's documented document run values are `UNSTART`, `RUNNING`, `SCHEDULE`, `DONE`, `FAIL`, and `CANCEL`; unknown values never count as success.

A `version_uid` is a parser-policy version, not a claim that a source has a globally meaningful revision number. The local registry may map one version to one deterministic remote document per dataset scope. Remote IDs remain opaque.

## Deployment provenance

The wrapper downloads and verifies the official RAGFlow v0.27.1 commit, then stages its full `docker/` directory at runtime. It generates an ignored runtime `.env` and a bookkeeping marker; no service replacement is committed. Active profiles are Elasticsearch, CPU, MySQL metadata, and the official TEI CPU embedding service (`BAAI/bge-small-en-v1.5`). Host ports are loopback-only, named data volumes persist, and the sandbox profile/Docker socket are disabled. The documented Ollama chat sidecar is separate from the official stack. See `docs/DECISIONS.md` for the route table and upgrade caveats.
