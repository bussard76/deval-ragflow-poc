# Test plan

## Unit suite

Run:

```bash
python -m compileall -q src tests
python -m pytest -q
```

The tests cover:

- identical PDF bytes at different paths, SHA-256/document/version identities, canonical block offsets, TOC paths, rounded bboxes, limits, malformed/encrypted/empty/scanned/mixed classification;
- SQLite foreign keys, unique `(dataset_scope, version_uid)` mappings, idempotent extraction, concurrent reservations, persisted parse/graph indexes, and owned-reset behavior;
- adapter HTTP errors, non-zero business codes, multipart upload, metadata PATCH, pagination, numeric/text/unknown states, list progress messages, retrieval request fields and raw references, GraphRAG aliases, and deletion body safety;
- ingestion local-first/no-upload classifications, reconciliation, repeat idempotence, parse polling, monotonic timeout, one cancellation request, failure, and unknown state handling;
- citation verification for RAGFlow PDF bboxes (plus legacy page/paragraph positions), Unicode/whitespace normalization, exact matching, token Dice threshold, deterministic ties, unknown IDs, and unresolved output;
- CLI command/help, `ask` chat failure handling and independent citation retrieval, and reset confirmation/ownership gates.

## Live test policy

`tests/test_live_ragflow.py` uses both real PDFs in `data/input/` and the actual async adapter. Credentials/models load from the same `.env`/process configuration as the CLI, but the live flag must be explicitly set in the process environment; it skips with a precise reason unless `RUN_LIVE_RAGFLOW_TESTS=true`, `RAGFLOW_API_KEY` is set, and `RAGFLOW_EMBEDDING_MODEL` and `RAGFLOW_LLM_MODEL` are set. Once those conditions are met, connection, upload, parsing, retrieval, citation, or model errors fail the test; no fake transport is used and no error is converted to a skip.

```bash
RUN_LIVE_RAGFLOW_TESTS=false python -m pytest -q -rs tests/test_live_ragflow.py
RUN_LIVE_RAGFLOW_TESTS=true python -m pytest -q -rs tests/test_live_ragflow.py
python -m deval_ragflow ask "Welche zentralen Ergebnisse nennt die Studie?" --timeout 360
```

Live runs create real datasets/documents and consume local/provider resources. Use a disposable API account/dataset and reset it only with the CLI's exact confirmation. Live success does not prove exactly-once recovery after a process crash or GraphRAG provider availability across upgrades. The `ask` smoke command also depends on the configured local chat provider and may be slow on CPU.

## Optional Docker validation

On a resource-capable host, run `verify`, `config`, `up`, `wait`, `curl` against the official healthz URL, then `doctor`. ARM compatibility, kernel sysctl, provider credentials, upstream response changes, and asynchronous remote cancellation remain deployment risks rather than unit-test claims.
