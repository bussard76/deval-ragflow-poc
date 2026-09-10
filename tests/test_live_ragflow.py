import asyncio
import os
from pathlib import Path

import pytest

from deval_ragflow.adapter import RAGFlowAdapter
from deval_ragflow.citations import CitationResolver
from deval_ragflow.config import Config
from deval_ragflow.ingestion import IngestionService
from deval_ragflow.registry import Registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_PDFS = sorted((PROJECT_ROOT / "data" / "input").glob("*.pdf"))


@pytest.mark.integration
def test_live_ragflow_uses_the_two_real_deval_pdfs():
    # Require an explicit process-level opt-in so an existing .env cannot make
    # the ordinary unit-suite command contact a live service.
    if os.environ.get("RUN_LIVE_RAGFLOW_TESTS", "").strip().lower() != "true":
        pytest.skip("RUN_LIVE_RAGFLOW_TESTS is not true")
    config = Config.from_env()
    if not config.run_live_tests:
        pytest.skip("RUN_LIVE_RAGFLOW_TESTS is not true")
    if not config.api_key:
        pytest.skip("RAGFLOW_API_KEY is not configured")
    if not config.embedding_model:
        pytest.skip("RAGFLOW_EMBEDDING_MODEL is not configured")
    if not config.llm_model:
        pytest.skip("RAGFLOW_LLM_MODEL is not configured")
    assert [path.name for path in LIVE_PDFS] == [
        "2024_DEval_Minderungsstudie_Web__Zusammenfassung.pdf",
        "2025_DEval_Dezentralisierung_Afrika_Zusammenfassung.pdf",
    ]

    registry = Registry(config.registry_path)

    async def run():
        adapter = RAGFlowAdapter(
            config.base_url, config.api_key, timeout=config.request_timeout
        )
        try:
            health = await adapter.healthz()
            assert health is not None
            service = IngestionService(config, registry, adapter)
            ingested = []
            for pdf_path in LIVE_PDFS:
                # Both supplied summaries contain extractable text and images;
                # explicitly approve that classification for this live PoC.
                first = await service.ingest(pdf_path, allow_mixed=True)
                assert first.state == "DONE", f"{pdf_path.name}: {first.message}"
                assert first.remote_dataset_id and first.remote_document_id
                assert first.extraction is not None
                assert first.extraction.source_basename == pdf_path.name
                assert first.extraction.passages
                assert registry.get_extraction(first.version_uid) is not None

                # A second pass must reconcile the deterministic remote mapping
                # instead of uploading a duplicate document.
                second = await service.ingest(pdf_path, allow_mixed=True)
                assert second.state == "DONE", f"{pdf_path.name}: {second.message}"
                assert second.remote_document_id == first.remote_document_id
                ingested.append(first)

            dataset = registry.get_dataset(config.dataset_scope)
            assert dataset is not None
            assert len(registry.mappings_for_dataset(config.dataset_scope)) == len(
                LIVE_PDFS
            )
            resolver = CitationResolver(registry, config.citation_threshold)

            for result in ingested:
                extraction = result.extraction
                assert extraction is not None
                query = " ".join(extraction.passages[0].text.split())
                retrieved = await adapter.retrieve(
                    query,
                    [dataset.remote_dataset_id],
                    document_ids=[result.remote_document_id],
                    page_size=10,
                    similarity_threshold=0.0,
                    knn_top_k=20,
                    knn_num_candidates=20,
                    rerank_candidates_count=20,
                    highlight=False,
                )
                assert retrieved.chunks, (
                    f"no RAGFlow chunks retrieved for {extraction.source_basename}"
                )
                references = retrieved.references or retrieved.chunks
                citations = resolver.resolve_many(
                    references, dataset_scope=config.dataset_scope
                )
                assert any(
                    citation.method != "unresolved"
                    and citation.version_uid == result.version_uid
                    for citation in citations
                ), (
                    f"no deterministic citation resolved for {extraction.source_basename}"
                )
        finally:
            await adapter.aclose()

    try:
        # No exception is converted to a skip after configuration says this
        # should run; a missing/unhealthy stack is a genuine live-test failure.
        asyncio.run(run())
    finally:
        registry.close()
