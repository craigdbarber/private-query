"""A test suite for the private-query module."""

import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_util import start_local_ollama, stop_process

from chroma_util import ChromaClient
from config_util import get_config_dict, load_yaml_config
from ollama_util import OllamaClient


@dataclass
class _SessionData:
    chroma: ChromaClient
    ollama: OllamaClient
    model: str


@pytest.fixture(scope="session", name="session_data")
def suite_setup_teardown() -> Generator[_SessionData, None, None]:
    """Provide setup and teardown functionality for test suite."""
    chroma: ChromaClient
    ollama: OllamaClient

    # setup config
    ollama_host = "localhost:8080"
    ollama_model = "llama3.2:latest"
    tmp_persist_dir = tempfile.TemporaryDirectory()
    tmp_persist_dir_path = Path(tmp_persist_dir.name)
    tmp_cache_dir = tempfile.TemporaryDirectory()
    tmp_cache_dir_path = Path(tmp_cache_dir.name)
    tmp_config = tempfile.NamedTemporaryFile(mode="w+t")
    tmp_config.write(f"""
chroma:
    type: "local"
    persist_directory: "{tmp_persist_dir_path}"
    embedding_model: "all-MiniLM-L6-v2"
    model_cache_directory: "{tmp_cache_dir_path}"
ollama:
    host: "{ollama_host}"
    model: "{ollama_model}"
                         """)
    tmp_config.flush()

    # load chroma client
    config = load_yaml_config(tmp_config.name)
    chroma_config = get_config_dict(config, "chroma")
    assert chroma_config is not None
    chroma = ChromaClient(chroma_config)

    # load ollama
    ollama_home = tempfile.TemporaryDirectory()
    start_local_ollama(ollama_host, ollama_home)
    ollama_config = get_config_dict(config, "ollama")
    assert ollama_config is not None
    ollama = OllamaClient(ollama_config)

    yield _SessionData(chroma=chroma, ollama=ollama, model=ollama_model)

    stop_process("ollama")
    ollama_home.cleanup()
    tmp_persist_dir.cleanup()
    tmp_cache_dir.cleanup()
    tmp_config.close()


def test_chroma_client(session_data: _SessionData):
    """Tests the chroma client successfully loaded."""
    chroma = session_data.chroma
    assert chroma is not None and chroma.is_alive()


def test_ollama_client(session_data: _SessionData):
    """Tests ollama client successfully loaded."""
    ollama = session_data.ollama
    assert ollama is not None
    assert ollama.contains_model(session_data.model)
