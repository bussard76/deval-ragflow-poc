"""Argparse CLI for the local provenance PoC."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .adapter import RAGFlowAdapter
from .citations import CitationResolver
from .config import Config
from .errors import DevalError
from .ingestion import IngestionService
from .pdf import extract_pdf
from .registry import Registry

CONFIRMATION = "DELETE DEVAL TEST DATA"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deval-ragflow", description="Local PDF provenance and RAGFlow PoC"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser(
        "doctor", help="check RAGFlow readiness, auth, models, and SQLite"
    )
    doctor.set_defaults(command_handler="doctor")

    extract = sub.add_parser(
        "extract", help="validate and extract local PDF provenance"
    )
    extract.add_argument("pdf", type=Path)
    extract.set_defaults(command_handler="extract")

    ingest = sub.add_parser(
        "ingest", help="extract, register, upload, parse, and poll one PDF"
    )
    ingest.add_argument("pdf", type=Path)
    ingest.add_argument(
        "--allow-mixed",
        action="store_true",
        help="permit mixed text/image PDFs to upload",
    )
    ingest.add_argument("--timeout", type=float, default=None)
    ingest.set_defaults(command_handler="ingest")

    status = sub.add_parser(
        "status", help="show remote parsing status for a local version"
    )
    status.add_argument("version_uid")
    status.set_defaults(command_handler="status")

    query = sub.add_parser(
        "query", help="retrieve raw chunks and resolve local citations"
    )
    query.add_argument("question")
    query.add_argument("--use-kg", action="store_true")
    query.set_defaults(command_handler="query")

    ask = sub.add_parser("ask", help="answer a factual question with RAGFlow citations")
    ask.add_argument("question")
    ask.add_argument("--timeout", type=float, default=None)
    ask.add_argument(
        "--json",
        action="store_true",
        help="print the complete answer and citations as JSON",
    )
    ask.set_defaults(command_handler="ask")

    graph = sub.add_parser(
        "graph", help="construct and inspect the dataset knowledge graph"
    )
    graph.add_argument("--timeout", type=float, default=None)
    graph.set_defaults(command_handler="graph")

    cancel = sub.add_parser("cancel", help="request cancellation of a parsing task")
    cancel.add_argument("version_uid")
    cancel.set_defaults(command_handler="cancel")

    reset = sub.add_parser(
        "reset-test-data", help="delete one explicitly owned test dataset"
    )
    reset.add_argument("--dataset-id", required=True)
    reset.add_argument("--confirm", default="")
    reset.set_defaults(command_handler="reset-test-data")
    return parser


def _json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str))


def _print_answer(answer: str, citations: list, registry: Registry) -> None:
    print(answer.strip())
    print("\nQuellen:")
    if not citations:
        print("- RAGFlow hat keine Referenzen zurückgegeben.")
        return
    for number, citation in enumerate(citations, 1):
        if citation.passage_uid is None:
            print(f"- [{number}] nicht deterministisch aufgelöst: {citation.reason}")
            continue
        source = "unbekannte Quelle"
        if citation.version_uid:
            extraction = registry.get_extraction(citation.version_uid)
            if extraction is not None:
                source = extraction.source_basename
        location = []
        if citation.page_number is not None:
            location.append(f"S. {citation.page_number}")
        if citation.paragraph_number is not None:
            location.append(f"Absatz {citation.paragraph_number}")
        if citation.section_path:
            location.append("Abschnitt: " + " > ".join(citation.section_path))
        print(
            f"- [{number}] {source}; {', '.join(location) or 'Position unbekannt'}; "
            f"{citation.method}, Konfidenz {citation.confidence:.2f}"
        )


def _runtime(config: Config):
    registry = Registry(config.registry_path)
    try:
        adapter = RAGFlowAdapter(
            config.base_url, config.api_key, timeout=config.request_timeout
        )
    except Exception:
        registry.close()
        raise
    return registry, adapter


async def _doctor(config: Config) -> int:
    registry = None
    missing = []
    if not config.api_key:
        missing.append("RAGFLOW_API_KEY")
    if not config.embedding_model:
        missing.append("RAGFLOW_EMBEDDING_MODEL")
    if not config.llm_model:
        missing.append("RAGFLOW_LLM_MODEL")
    result: dict[str, Any] = {
        "base_url": config.base_url,
        "sqlite_writable": False,
        "embedding_model_configured": bool(config.embedding_model),
        "llm_model_configured": bool(config.llm_model),
        "models_ready": bool(config.embedding_model and config.llm_model),
        "api_key_configured": bool(config.api_key),
        "missing_configuration": missing,
    }
    try:
        try:
            registry = Registry(config.registry_path)
            result["sqlite_writable"] = registry.check_writable()
            if not result["sqlite_writable"]:
                result["sqlite_error"] = "SQLite write probe failed"
        except Exception as exc:  # noqa: BLE001 - doctor reports any probe failure
            result["sqlite_error"] = str(exc)

        adapter = RAGFlowAdapter(
            config.base_url, config.api_key, timeout=config.request_timeout
        )
        try:
            try:
                health = await adapter.healthz()
                result["health"] = {"ok": True, "raw": health}
            except Exception as exc:  # noqa: BLE001 - doctor reports any service failure
                result["health"] = {"ok": False, "error": str(exc)}
            if config.api_key:
                try:
                    datasets = await adapter.list_datasets(page_size=1, max_pages=1)
                    result["authenticated_dataset_access"] = {
                        "ok": True,
                        "dataset_count_sample": len(datasets),
                    }
                except Exception as exc:  # noqa: BLE001 - doctor reports any auth failure
                    result["authenticated_dataset_access"] = {
                        "ok": False,
                        "error": str(exc),
                    }
            else:
                result["authenticated_dataset_access"] = {
                    "ok": False,
                    "skipped": "RAGFLOW_API_KEY is not configured",
                }
            _json(result)
            return 0 if result["sqlite_writable"] and result["health"]["ok"] else 1
        finally:
            await adapter.aclose()
    finally:
        if registry is not None:
            registry.close()


async def _reset_command(config: Config, args: argparse.Namespace) -> int:
    if args.confirm != CONFIRMATION:
        print(f"refusing reset: pass --confirm {CONFIRMATION!r}", file=sys.stderr)
        return 2
    registry = Registry(config.registry_path)
    try:
        dataset = registry.get_dataset_by_remote_id(args.dataset_id)
        if dataset is None or not dataset.owned:
            print(
                "refusing reset: dataset id is not an owned local mapping",
                file=sys.stderr,
            )
            return 2
        adapter = RAGFlowAdapter(
            config.base_url, config.api_key, timeout=config.request_timeout
        )
        try:
            result = await adapter.delete_owned_dataset(
                args.dataset_id, dataset.remote_dataset_id
            )
        finally:
            await adapter.aclose()
        registry.delete_dataset_rows(dataset.dataset_scope, dataset.remote_dataset_id)
        _json({"deleted_remote_dataset_id": args.dataset_id, "remote": result})
        return 0
    finally:
        registry.close()


async def _network_command(config: Config, args: argparse.Namespace) -> int:
    if args.command_handler == "reset-test-data":
        return await _reset_command(config, args)
    registry, adapter = _runtime(config)
    service = IngestionService(config, registry, adapter)
    try:
        if args.command_handler == "ingest":
            result = await service.ingest(
                args.pdf, allow_mixed=args.allow_mixed, timeout=args.timeout
            )
            _json(result.as_dict())
            return 0 if result.state in ("DONE", "NOT_UPLOADED") else 1
        if args.command_handler == "status":
            mapping = registry.get_mapping(config.dataset_scope, args.version_uid)
            dataset = registry.get_dataset(config.dataset_scope)
            if mapping is None or not mapping.remote_document_id or dataset is None:
                raise DevalError(
                    f"no local remote mapping for version {args.version_uid}"
                )
            status = await adapter.document_status(
                dataset.remote_dataset_id, mapping.remote_document_id
            )
            _json(
                {
                    "version_uid": args.version_uid,
                    "remote_document_id": mapping.remote_document_id,
                    "local_state": mapping.state,
                    "local_failure": mapping.failure_message,
                    "state": status.state,
                    "progress": status.progress,
                    "progress_msg": status.progress_msg,
                    "failure": status.failure_message,
                    "raw": status.raw,
                }
            )
            return 0
        if args.command_handler == "ask":
            dataset = registry.get_dataset(config.dataset_scope)
            if dataset is None:
                raise DevalError("ingest a document before asking a question")
            chat = await adapter.ensure_chat(
                "deval-cli-" + dataset.remote_dataset_id,
                [dataset.remote_dataset_id],
                llm_model=config.llm_model,
            )
            chat_id = chat.get("id")
            if not chat_id:
                raise DevalError("RAGFlow chat has no id")
            session = await adapter.create_chat_session(str(chat_id))
            session_id = session.get("id")
            if not session_id:
                raise DevalError("RAGFlow chat session has no id")
            completion = await adapter.chat_completion(
                str(chat_id), args.question, str(session_id), timeout=args.timeout
            )
            data = completion.get("data")
            if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
                raise DevalError("RAGFlow chat completion returned no answer")
            retrieved = await adapter.retrieve(
                args.question,
                [dataset.remote_dataset_id],
                page_size=3,
                similarity_threshold=0.0,
                knn_top_k=10,
                knn_num_candidates=20,
                rerank_candidates_count=20,
                highlight=False,
            )
            citations = CitationResolver(
                registry, config.citation_threshold
            ).resolve_many(
                retrieved.references or retrieved.chunks,
                dataset_scope=config.dataset_scope,
            )
            if args.json:
                _json(
                    {
                        "answer": data["answer"],
                        "citations": [citation.as_dict() for citation in citations],
                        "references": retrieved.references,
                        "raw": {"completion": completion, "retrieval": retrieved.raw},
                    }
                )
            else:
                _print_answer(data["answer"], citations, registry)
            return 0
        if args.command_handler == "query":
            dataset = registry.get_dataset(config.dataset_scope)
            if dataset is None:
                raise DevalError("ingest a document before querying")
            retrieved = await adapter.retrieve(
                args.question, [dataset.remote_dataset_id], use_kg=args.use_kg
            )
            citations = CitationResolver(
                registry, config.citation_threshold
            ).resolve_many(retrieved.references, dataset_scope=config.dataset_scope)
            _json(
                {
                    "chunks": retrieved.chunks,
                    "references": retrieved.references,
                    "citations": [citation.as_dict() for citation in citations],
                    "raw": retrieved.raw,
                }
            )
            return 0
        if args.command_handler == "graph":
            result = await service.build_graph(timeout=args.timeout)
            _json(result)
            return 0 if result.get("state") in ("DONE", "EMPTY", "NOT_READY") else 1
        if args.command_handler == "cancel":
            result = await service.cancel(args.version_uid)
            _json(result)
            return 0
        raise DevalError("unknown command")
    finally:
        await adapter.aclose()
        registry.close()


def _extract_command(args: argparse.Namespace, config: Config) -> int:
    extraction = extract_pdf(
        args.pdf,
        parser_config=config.provenance_parser_config,
        max_pages=config.max_document_pages,
        max_bytes=config.max_document_bytes,
        max_text_chars=config.max_text_chars,
    )
    _json(extraction.as_dict())
    return 0


def main(argv: Any | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = Config.from_env()
        if args.command_handler == "extract":
            return _extract_command(args, config)
        if args.command_handler == "doctor":
            return asyncio.run(_doctor(config))
        return asyncio.run(_network_command(config, args))
    except (DevalError, OSError, ValueError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
