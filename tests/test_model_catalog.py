import json
from pathlib import Path

from deval_ragflow.model_catalog import ModelCatalog  # type: ignore[import-not-found]


def test_model_catalog_exposes_enabled_entries_without_provider_secrets(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "local",
                        "name": "Local",
                        "provider": "Ollama",
                        "ragflow_model": "qwen@local@Ollama",
                        "api_key": "must-not-be-returned",
                        "enabled": True,
                    },
                    {
                        "id": "disabled",
                        "name": "Disabled",
                        "provider": "Other",
                        "ragflow_model": "other",
                        "enabled": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    models = ModelCatalog(path).options()

    assert len(models) == 1
    assert models[0].ragflow_model == "qwen@local@Ollama"
    assert "api_key" not in models[0].public_dict()


def test_model_catalog_falls_back_to_configured_model_when_file_is_missing(tmp_path):
    models = ModelCatalog(tmp_path / "missing.json", fallback_model="model").options()

    assert [model.id for model in models] == ["default"]
    assert models[0].ragflow_model == "model"


def test_default_catalog_includes_pi_codex_gpt_56_models():
    path = Path(__file__).parents[1] / "config" / "models.json"
    models = {model.id: model for model in ModelCatalog(path).options()}

    model_ids = {"gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"}
    assert model_ids <= models.keys()
    for model_id in model_ids:
        assert models[model_id].provider == "OpenAI Codex · Pi.dev"
        assert models[model_id].ragflow_model == f"{model_id}@codex@OpenAI"
