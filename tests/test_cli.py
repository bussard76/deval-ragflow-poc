from pathlib import Path

from deval_ragflow.cli import CONFIRMATION, build_parser, main
from deval_ragflow.config import Config


def test_config_repr_does_not_expose_api_key(tmp_path):
    config = Config(api_key="top-secret", registry_path=tmp_path / "registry.sqlite3")
    assert "top-secret" not in repr(config)


def test_help_lists_all_commands():
    text = build_parser().format_help()
    for name in (
        "doctor",
        "extract",
        "ingest",
        "status",
        "query",
        "ask",
        "graph",
        "cancel",
        "reset-test-data",
    ):
        assert name in text


def test_reset_requires_exact_confirmation_and_owned_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RAGFLOW_REGISTRY_PATH", str(tmp_path / "registry.sqlite3"))
    assert main(["reset-test-data", "--dataset-id", "x"]) == 2
    assert "confirm" in capsys.readouterr().err
    assert (
        main(["reset-test-data", "--dataset-id", "x", "--confirm", CONFIRMATION]) == 2
    )
    assert "owned" in capsys.readouterr().err


def test_extract_command_does_not_need_ragflow(tmp_path, capsys):
    fixture = Path(__file__).parent / "fixtures" / "golden.pdf"
    assert main(["extract", str(fixture)]) == 0
    assert '"classification": "extracted"' in capsys.readouterr().out
