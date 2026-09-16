from deval_ragflow.model_catalog import (
    remote_model_options,  # type: ignore[import-not-found]
)


def test_remote_model_catalog_exposes_every_chat_model_without_secrets():
    models = remote_model_options(
        [
            {
                "model_id": "model-a-id",
                "name": "model-a",
                "instance_name": "local",
                "provider_name": "Ollama",
                "api_key": "must-not-be-returned",
            },
            {
                "model_id": "model-b-id",
                "name": "model-b",
                "instance_name": "cloud",
                "provider_name": "OpenAI",
            },
        ]
    )

    assert [model.id for model in models] == ["model-a", "model-b"]
    assert [model.ragflow_model for model in models] == [
        "model-a@local@Ollama",
        "model-b@cloud@OpenAI",
    ]
    assert models[0].provider == "Ollama · local"
    assert "api_key" not in models[0].public_dict()


def test_remote_model_catalog_skips_duplicate_model_references():
    rows = [
        {
            "model_id": "first-id",
            "name": "model",
            "instance_name": "cloud",
            "provider_name": "OpenAI",
        },
        {
            "model_id": "second-id",
            "name": "model",
            "instance_name": "cloud",
            "provider_name": "OpenAI",
        },
    ]

    models = remote_model_options(rows)

    assert len(models) == 1
    assert models[0].ragflow_model == "model@cloud@OpenAI"
