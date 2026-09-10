# Deval RAGFlow PDF Provenance PoC

A small, local-first proof of concept: PyMuPDF extracts deterministic PDF provenance, SQLite records it, and one async `httpx` adapter talks to the official RAGFlow HTTP API. `ask` uses the configured local RAGFlow chat model for a concise answer; citation matching remains deterministic and never uses an LLM.

## Requirements

- Python 3.9–3.12 (the tested host is Python 3.9.6)
- Docker Compose >= 2.26.1 for RAGFlow, x86-oriented hardware, at least 4 CPUs/16 GB RAM/50 GB disk, and Elasticsearch's `vm.max_map_count >= 262144`
- The official stack is resource-heavy; ARM hosts may need a locally built RAGFlow image.

Install in an isolated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
cp .env.example .env
```

`.env` is ignored. `RAGFLOW_BASE_URL` defaults to `http://localhost:9380`; omit `/api/v1` because the adapter owns that route prefix. Set `RAGFLOW_API_KEY` from RAGFlow's Avatar → API screen. The reproducible local defaults are `BAAI/bge-small-en-v1.5@Builtin` for the official TEI CPU service and `qwen2.5:0.5b@local@Ollama` for a local chat model; configure the Ollama provider as described below. In v0.27.1 the LLM value is a local readiness label: GraphRAG uses the model provider configured for the RAGFlow tenant because dataset creation has no documented per-dataset LLM field.

## Official RAGFlow deployment

The wrapper stages the official `docker/` directory from commit `b9df87c4c75a5b0d35c90d15329fc0f6f91cb73e` (tag `v0.27.1`) into ignored `.data/ragflow-v0.27.1/`; the upstream compose files are not patched. It does not vendor a partial service graph or add an application container. A generated runtime `.env` randomizes dependency passwords, uses the official Elasticsearch/CPU/MySQL profiles plus the official `tei-cpu` embedding service, binds published ports to loopback, and leaves data in official named volumes (`esdata01`, `mysql_data`, `minio_data`, `redis_data`). The CPU profile uses `BAAI/bge-small-en-v1.5` to keep the local model practical on ARM/CPU hosts. The sandbox profile and Docker socket are not enabled.

```bash
./scripts/ragflow-compose.sh verify
./scripts/ragflow-compose.sh config
./scripts/ragflow-compose.sh up
./scripts/ragflow-compose.sh wait
```

`config` redacts generated credentials even when Compose has interpolated them into healthcheck or command arguments.

### Local chat model

Retrieval only needs the TEI embedding model. GraphRAG and the live-test configuration also require a usable chat model. Start the pinned Ollama image and pull the small local model:

```bash
docker run -d --name deval-ollama \
  --restart unless-stopped \
  -p 127.0.0.1:11434:11434 \
  -v deval_ollama_data:/root/.ollama \
  ollama/ollama@sha256:684d8674b4315fa18f4f0e973a118ec2652ed96f67563277839985175858e0ba
# If the digest is unavailable on the selected platform, use ollama/ollama:latest.
docker exec deval-ollama ollama pull qwen2.5:0.5b
```

In RAGFlow **Settings → Model providers → Ollama**, add instance `local` with base URL `http://host.docker.internal:11434` and model `qwen2.5:0.5b` as a chat model. Keep the model labels in `.env.example`; do not put provider credentials in tracked files. CPU generation is intentionally small and can take a few minutes; use `ask --timeout 360` when needed.

`wait` checks the official readiness endpoint, not process startup order:
`http://127.0.0.1:9380/api/v1/system/healthz`. Use `ps`, `logs`, and `down` for ordinary lifecycle operations. The wrapper never removes persistent volumes. If the upstream release changes, update the commit, routes, compose provenance, fixtures, and tests together; see [DECISIONS.md](DECISIONS.md).

## CLI

```bash
python -m deval_ragflow doctor
python -m deval_ragflow extract ./paper.pdf
python -m deval_ragflow ingest ./paper.pdf
python -m deval_ragflow ingest ./mixed.pdf --allow-mixed
python -m deval_ragflow status VERSION_UID
python -m deval_ragflow query "What does the paper conclude?"
python -m deval_ragflow ask "Welche zentralen Ergebnisse nennt die Studie?" --timeout 360
python -m deval_ragflow query "Find related entities" --use-kg
python -m deval_ragflow graph
python -m deval_ragflow cancel VERSION_UID
python -m deval_ragflow reset-test-data --dataset-id REMOTE_ID --confirm 'DELETE DEVAL TEST DATA'
```

`query` returns retrieved evidence and deterministic local citations. `ask` creates/reuses a deterministic chat, asks RAGFlow for a concise answer, then performs a separate retrieval call for the displayed citations; an explicit chat/provider failure is an error, never a fake success. `doctor` reports health, authenticated dataset access when a key is present, SQLite writability, and a non-secret list of missing model/key configuration without printing the key. `reset-test-data` accepts only an explicitly owned local dataset ID and the exact confirmation string; remote deletion succeeds before local rows are removed.

## Provenance and limits

`document_uid` is the lowercase SHA-256 of raw PDF bytes. `version_uid` hashes the document identity, provenance schema, extractor version, and canonical parser JSON. Text blocks are deterministic 1-based page paragraphs; coordinates are PDF points rounded to two decimals; character ranges are Unicode-code-point half-open offsets into canonical text (newline between blocks, form-feed between pages). PDF outlines provide `section_path` when available; no semantic heading detection is claimed.

Malformed, encrypted, empty, and image-only PDFs are saved by ingestion in the local registry but not uploaded. Mixed text/image PDFs require `--allow-mixed`. Byte, page, and text limits run before any remote request. A crash after a remote upload and before SQLite mapping commit cannot be made exactly-once; deterministic names and metadata reconcile one match and surface multiple matches.

Retrieval returns raw RAGFlow chunks and deterministic citations. `ask` adds concise RAGFlow chat generation but resolves citations through the independent retrieval path. Resolution uses a verified v0.27.1 PDF bbox position, exact normalized text, then multiset token Dice overlap (default threshold 0.60); citations include page/paragraph, section, bbox, character range, method, and confidence. Unknown IDs and low-confidence matches remain unresolved. Graph construction requires parsed chunks and a configured RAGFlow LLM.

Die vollständige Schritt-für-Schritt-Anleitung steht auf Deutsch in [docs/REPRODUKTION.md](docs/REPRODUKTION.md). See [ARCHITECTURE.md](ARCHITECTURE.md), [TESTPLAN.md](TESTPLAN.md), and [docs/DECISIONS.md](docs/DECISIONS.md).
