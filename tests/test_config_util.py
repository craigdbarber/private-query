"""Tests config_util."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from chroma_util import ChromaClientConfig
from config_util import load_yaml_config
from ollama_util import OllamaClientConfig


def test_load_yaml_config_loads_yaml(tmp_path: Path):
    """Test successfully loads yaml config."""
    tmp_config = tmp_path / "test_config.yaml"
    tmp_config.write_text("""
key: "value"
nested:
    nested_key: "nested_value"
                """)
    config = load_yaml_config(tmp_config)
    assert config is not None
    value = config["key"]
    assert value is not None
    assert value == "value"
    nested = config["nested"]
    assert nested is not None
    nested_value = nested["nested_key"]
    assert nested_value is not None
    assert nested_value == "nested_value"


def test_load_yaml_config_returns_raises_fnf_error():
    """Test raises FileNotFound error when no file exists."""
    with pytest.raises(FileNotFoundError, match=r"File does not exist:.*"):
        load_yaml_config("none existant file")


def test_chroma_client_config_validation_error_with_malformed_input(tmp_path: Path):
    """Test raises ValidationError for bad text input."""
    with pytest.raises(ValidationError):
        tmp_config = (tmp_path / "config.yaml").resolve()
        tmp_config.write_text("""
embedding_model: "all-MiniLM-L6-v2"
embedding_model_revision: "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
model_cache_directory: "some_dir"
database:
    type: "local"
    # missing persist_directory
""")
        config = load_yaml_config(tmp_config)
        ChromaClientConfig.model_validate(config)


def test_chroma_config_util_validation_error_with_invalid_dict():
    """Test raises ValidationError for bad dict."""
    with pytest.raises(ValidationError):
        config_dict = {
            # missing embedding_model
            "embedding_model_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
            "model_cache_directory": "some_dir",
            "database": {"type": "local", "persist_directory": "some_dir"},
        }
        ChromaClientConfig.model_validate(config_dict)


def test_ollama_client_config_validation_error_with_malformed_input(tmp_path: Path):
    """Test raises Validation error for bad text input."""
    with pytest.raises(ValidationError):
        tmp_config = (tmp_path / "config.yaml").resolve()
        tmp_config.write_text("""
host: localhost:8080
# model missing
""")
        config = load_yaml_config(tmp_config)
        OllamaClientConfig.model_validate(config)


def test_ollama_client_config_validation_error_with_invalid_dict():
    """Test raises ValidationERror for bad dict."""
    with pytest.raises(ValidationError):
        config_dict = {
            "model": "llama3.2:latest",
            # missing host
        }
        OllamaClientConfig.model_validate(config_dict)
