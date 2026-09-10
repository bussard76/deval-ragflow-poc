import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from deval_ragflow.config import Config
from deval_ragflow.registry import Registry


@pytest.fixture
def registry(tmp_path):
    value = Registry(tmp_path / "registry.sqlite3")
    try:
        yield value
    finally:
        value.close()


@pytest.fixture
def config(tmp_path):
    return Config(
        base_url="http://localhost:9380",
        dataset_name="test-dataset",
        registry_path=tmp_path / "registry.sqlite3",
        poll_interval=0,
        parse_timeout=2,
        cancel_timeout=0.2,
        graph_timeout=1,
        reconcile_max_pages=2,
    )
