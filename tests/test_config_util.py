"""Tests config_util."""

import tempfile
from typing import Any

import pytest

from config_util import (
    get_config_value,
    get_config_value_or_none,
    load_yaml_config,
)


def test_get_config_value_retrieves_str():
    """Test successfully retrieves str."""
    value = "sue"
    config = {"key": value}
    retrieved_value = get_config_value(config, "key", type(value))
    assert retrieved_value is not None
    assert isinstance(retrieved_value, type(value))
    assert retrieved_value == value


def test_get_config_value_retrieves_dict():
    """Test successfully retrieves dict."""
    nested_dict = {"value": "bob"}
    config = {"nested_dict": nested_dict}
    retrieved_nested_dict = get_config_value(config, "nested_dict", type(nested_dict))
    assert retrieved_nested_dict is not None
    assert isinstance(retrieved_nested_dict, type(nested_dict))
    assert retrieved_nested_dict == nested_dict


def test_get_config_value_raises_key_error():
    """Test raises ValueError for missing value."""
    config = {}
    with pytest.raises(KeyError):
        get_config_value(config, "key", str)


def test_get_config_value_raises_type_error():
    """Test raises TypeError for wrong types."""
    config: dict[str, Any] = {"key": 1}
    with pytest.raises(TypeError):
        get_config_value(config, "key", str)


def test_get_config_value_or_none_returns_value():
    """Test correctly returns value."""
    value = "foo"
    config = {"key": value}
    retrieved_value = get_config_value_or_none(config, "key", type(value))
    assert retrieved_value is not None
    assert retrieved_value == value


def test_get_config_value_or_none_returns_none():
    """Test correctly returns None."""
    config = {}
    retrieved_value = get_config_value_or_none(config, "key", str)
    assert retrieved_value is None


def test_load_yaml_config_loads_str():
    """Test successfully loads yaml config."""
    with tempfile.NamedTemporaryFile(mode="w+t") as tmp:
        tmp.write("""
test_config: "test_value"
                  """)
        tmp.flush()
        config = load_yaml_config(tmp.name)
        assert config is not None
        data_dir = get_config_value(config, "test_config", str)
        assert data_dir is not None
        assert data_dir == "test_value"


def test_load_yaml_config_returns_raises_fnf_error():
    """Test raises FileNotFound error when no file exists."""
    with pytest.raises(FileNotFoundError):
        load_yaml_config("none existant file")


def test_load_yaml_config_loads_full_config():
    """Tests loading a project config."""
    with tempfile.NamedTemporaryFile(mode="w+t") as tmp:
        tmp.write("""
settings:
  collection_name: "private-query-collection"
  chroma:
    persist_directory: "./vector_store"
    embedding_model: "all-MiniLM-L6-v2"
    embedding_model_revision: "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    model_cache_directory: "./model-cache"
  ollama:
    host: "localhost:8080"
    model: "llama3.2:latest"
                  """)
        tmp.flush()
        config = load_yaml_config(tmp.name)
        assert config is not None
        settings = get_config_value(config, "settings", dict[str, Any])
        assert settings is not None
        assert len(settings) > 0

        collection_name = get_config_value(settings, "collection_name", str)
        assert collection_name == "private-query-collection"

        chroma_config = get_config_value(settings, "chroma", dict[str, str])
        assert chroma_config is not None
        assert len(chroma_config) > 0
        persist_dir = get_config_value(chroma_config, "persist_directory", str)
        assert persist_dir == "./vector_store"
        embedding_model = get_config_value(chroma_config, "embedding_model", str)
        assert embedding_model == "all-MiniLM-L6-v2"
        model_cache_directory = get_config_value(
            chroma_config, "model_cache_directory", str
        )
        assert model_cache_directory == "./model-cache"

        ollama_config = get_config_value(settings, "ollama", dict[str, str])
        assert ollama_config is not None
        assert len(ollama_config) > 0
        host = get_config_value(ollama_config, "host", str)
        assert host == "localhost:8080"
        model = get_config_value(ollama_config, "model", str)
        assert model == "llama3.2:latest"
