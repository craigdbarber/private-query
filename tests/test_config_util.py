"""Tests config_util."""

import tempfile
from typing import assert_type, cast

import pytest

from config_util import NestedDict, get_config_dict, get_config_str, load_yaml_config


def test_get_config_dict():
    """Test successfully retrieves nested dict."""
    config: NestedDict = {"nested_dict": {"value": "bob"}}
    nested_dict = get_config_dict(config, "nested_dict")
    assert nested_dict is not None
    assert_type(nested_dict, NestedDict)
    assert (
        cast(
            dict[str, str],
            nested_dict,
        )["value"]
        == "bob"
    )


def test_get_config_dict_returns_none():
    """Test properly returns None."""
    value = get_config_dict({}, "some_value", False)
    assert value is None


def test_get_config_dict_throws_error():
    """Test properly throws error."""
    with pytest.raises(ValueError):
        get_config_dict({}, "value")


def test_get_config_str():
    """Test successfully retrieves str value."""
    config: NestedDict = {"value": "bob"}
    value = get_config_str(config, "value")
    assert value is not None
    assert value == "bob"


def test_get_config_returns_none():
    """Test properly returns None."""
    config: NestedDict = {}
    value = get_config_str(config, "value", False)
    assert value is None


def test_get_config_str_throws_error():
    """Test properly throws error."""
    config: NestedDict = {}
    with pytest.raises(ValueError):
        get_config_str(config, "value")


def test_load_yaml_config():
    """Test successfully loads yaml config."""
    with tempfile.NamedTemporaryFile(mode="w+t") as tmp:
        tmp.write("""
test_config: "test_value"
                  """)
        tmp.flush()
        config: NestedDict = load_yaml_config(tmp.name)
        assert config is not None
        data_dir = get_config_str(config, "test_config")
        assert data_dir is not None
        assert data_dir == "test_value"


def test_load_yaml_config_returns_none():
    """Test properly returns None when no file exists."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("none existant file")


def test_integration_loading_project_config():
    with tempfile.NamedTemporaryFile(mode="w+t") as tmp:
        tmp.write("""
settings:
  chroma:
    persist_directory: "./vector_store"

    embedding_model: "multi-qa-MiniLM-L6-cos-v1"
  ollama:
    host: "localhost"
    model: "llama3.2:latest"

  data_directory: "./data"
                  """)
        tmp.flush()
        config = load_yaml_config(tmp.name)
        assert config is not None
        settings = get_config_dict(config, "settings")
        assert settings is not None

        data_dir = get_config_str(settings, "data_directory")
        assert data_dir == "./data"

        chroma_config = get_config_dict(settings, "chroma")
        assert chroma_config is not None
        persist_dir = get_config_str(chroma_config, "persist_directory")
        assert persist_dir == "./vector_store"
        embedding_model = get_config_str(chroma_config, "embedding_model")
        assert embedding_model == "multi-qa-MiniLM-L6-cos-v1"

        ollama_config = get_config_dict(settings, "ollama")
        assert ollama_config is not None
        host = get_config_str(ollama_config, "host")
        assert host == "localhost"
        model = get_config_str(ollama_config, "model")
        assert model == "llama3.2:latest"
