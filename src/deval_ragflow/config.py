"""Configuration with a deliberately dependency-free .env reader."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .errors import ConfigurationError


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def normalize_base_url(value: str) -> str:
    value = value.strip().rstrip("/")
    if not value:
        raise ConfigurationError("RAGFLOW_BASE_URL must not be empty")
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc or parts.scheme not in ("http", "https"):
        raise ConfigurationError("RAGFLOW_BASE_URL must be an http(s) URL")
    if parts.query or parts.fragment or parts.username or parts.password:
        raise ConfigurationError(
            "RAGFLOW_BASE_URL must not contain query, fragment, or user credentials"
        )
    path = parts.path.rstrip("/")
    for suffix in ("/api/v1", "/api"):
        if path.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            break
    return urlunsplit(
        (parts.scheme, parts.netloc, path, parts.query, parts.fragment)
    ).rstrip("/")


@dataclass(frozen=True)
class Config:
    base_url: str = "http://localhost:9380"
    api_key: str = field(default="", repr=False)
    dataset_name: str = "deval-poc"
    embedding_model: str = ""
    llm_model: str = ""
    chunk_method: str = "naive"
    chunk_token_num: int = 512
    delimiter: str = "\n"
    registry_path: Path = field(default_factory=lambda: Path(".data/registry.sqlite3"))
    request_timeout: float = 30.0
    parse_timeout: float = 1800.0
    cancel_timeout: float = 60.0
    poll_interval: float = 2.0
    max_document_pages: int = 2000
    max_document_bytes: int = 50 * 1024 * 1024
    max_text_chars: int = 20 * 1000 * 1000
    citation_threshold: float = 0.60
    reconcile_max_pages: int = 20
    graph_timeout: float = 1800.0
    run_live_tests: bool = False
    model_config_path: Path = field(default_factory=lambda: Path("config/models.json"))
    parser_config: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "base_url", normalize_base_url(self.base_url))
        if self.chunk_token_num <= 0:
            raise ConfigurationError("chunk_token_num must be positive")
        for name in (
            "request_timeout",
            "parse_timeout",
            "cancel_timeout",
            "poll_interval",
            "graph_timeout",
            "max_document_pages",
            "max_document_bytes",
            "max_text_chars",
            "reconcile_max_pages",
        ):
            value = getattr(self, name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ConfigurationError(f"{name} must be finite")
            if value < 0:
                raise ConfigurationError(f"{name} must not be negative")
        if not self.delimiter:
            raise ConfigurationError("RAGFLOW_DELIMITER must not be empty")
        if not 0 <= self.citation_threshold <= 1:
            raise ConfigurationError("citation_threshold must be between 0 and 1")
        dataset_name = self.dataset_name.strip()
        if not dataset_name:
            raise ConfigurationError("RAGFLOW_DATASET_NAME must not be empty")
        object.__setattr__(self, "dataset_name", dataset_name)
        object.__setattr__(self, "api_key", self.api_key.strip())
        chunk_method = self.chunk_method.strip()
        if not chunk_method:
            raise ConfigurationError("RAGFLOW_CHUNK_METHOD must not be empty")
        object.__setattr__(self, "chunk_method", chunk_method)
        object.__setattr__(self, "embedding_model", self.embedding_model.strip())
        object.__setattr__(self, "llm_model", self.llm_model.strip())
        parser = dict(self.parser_config)
        parser.setdefault("chunk_token_num", self.chunk_token_num)
        parser.setdefault("delimiter", self.delimiter)
        object.__setattr__(self, "parser_config", parser)

    @property
    def provenance_parser_config(self) -> dict[str, object]:
        policy = dict(self.parser_config)
        policy["chunk_method"] = self.chunk_method
        return policy

    @property
    def dataset_scope(self) -> str:
        return self.dataset_name

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        dotenv_path: Path | None = None,
    ):
        # Process variables override .env; an injected mapping has the highest
        # priority, which keeps configuration tests deterministic.
        source = dict(_read_dotenv(Path(dotenv_path or ".env")))
        source.update(os.environ)
        if environ is not None:
            source.update(environ)

        def value(name: str, default: str) -> str:
            result = source.get(name)
            return default if result is None else result

        def integer(name: str, default: int) -> int:
            raw = value(name, str(default))
            try:
                result = int(raw)
            except ValueError:
                raise ConfigurationError(f"{name} must be an integer") from None
            if result < 0:
                raise ConfigurationError(f"{name} must not be negative")
            return result

        def number(name: str, default: float) -> float:
            raw = value(name, str(default))
            try:
                result = float(raw)
            except ValueError:
                raise ConfigurationError(f"{name} must be a number") from None
            if not math.isfinite(result):
                raise ConfigurationError(f"{name} must be finite")
            if result < 0:
                raise ConfigurationError(f"{name} must not be negative")
            return result

        def boolean(name: str, default: bool) -> bool:
            raw = value(name, "true" if default else "false").strip().lower()
            if raw in ("1", "true", "yes", "on"):
                return True
            if raw in ("0", "false", "no", "off", ""):
                return False
            raise ConfigurationError(f"{name} must be true or false")

        return cls(
            base_url=value("RAGFLOW_BASE_URL", "http://localhost:9380"),
            api_key=value("RAGFLOW_API_KEY", ""),
            dataset_name=value("RAGFLOW_DATASET_NAME", "deval-poc"),
            embedding_model=value("RAGFLOW_EMBEDDING_MODEL", ""),
            llm_model=value("RAGFLOW_LLM_MODEL", ""),
            chunk_method=value("RAGFLOW_CHUNK_METHOD", "naive"),
            chunk_token_num=integer("RAGFLOW_CHUNK_TOKEN_NUM", 512),
            delimiter=value("RAGFLOW_DELIMITER", "\n").replace("\\n", "\n"),
            registry_path=Path(
                value("RAGFLOW_REGISTRY_PATH", ".data/registry.sqlite3")
            ),
            request_timeout=number("RAGFLOW_REQUEST_TIMEOUT", 30),
            parse_timeout=number("RAGFLOW_PARSE_TIMEOUT", 1800),
            cancel_timeout=number("RAGFLOW_CANCEL_TIMEOUT", 60),
            poll_interval=number("RAGFLOW_POLL_INTERVAL", 2),
            max_document_pages=integer("RAGFLOW_MAX_DOCUMENT_PAGES", 2000),
            max_document_bytes=integer("RAGFLOW_MAX_DOCUMENT_BYTES", 50 * 1024 * 1024),
            max_text_chars=integer("RAGFLOW_MAX_TEXT_CHARS", 20 * 1000 * 1000),
            citation_threshold=number("RAGFLOW_CITATION_THRESHOLD", 0.60),
            reconcile_max_pages=integer("RAGFLOW_RECONCILE_MAX_PAGES", 20),
            graph_timeout=number("RAGFLOW_GRAPH_TIMEOUT", 1800),
            run_live_tests=boolean("RUN_LIVE_RAGFLOW_TESTS", False),
            model_config_path=Path(value("DEVAL_MODEL_CONFIG", "config/models.json")),
        )


load_config = Config.from_env
