"""A test suite for the private-query module."""

import random
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_util import start_local_ollama, stop_process

from chroma_util import ChromaClient
from config_util import get_config_dict, load_yaml_config
from ollama_util import OllamaClient
from private_query import PrivateQuery


@dataclass
class _SessionData:
    private_query: PrivateQuery
    chroma: ChromaClient


@pytest.fixture(scope="session", name="session_data")
def suite_setup_teardown(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[_SessionData, None, None]:
    """Provide setup and teardown functionality for test suite."""
    chroma: ChromaClient
    ollama: OllamaClient

    # setup config
    ollama_host = "localhost:8080"
    ollama_model = "llama3.2:latest"
    tmp_persist_dir = tmp_path_factory.mktemp("persist_dir")
    tmp_cache_dir = tmp_path_factory.mktemp("cache_dir")
    tmp_config_dir = tmp_path_factory.mktemp("config")
    tmp_config = tmp_config_dir / "config.yaml"
    tmp_config.write_text(f"""
chroma:
    type: "local"
    persist_directory: "{tmp_persist_dir}"
    embedding_model: "all-MiniLM-L6-v2"
    model_cache_directory: "{tmp_cache_dir}"
ollama:
    host: "{ollama_host}"
    model: "{ollama_model}"
                         """)

    # load chroma client
    config = load_yaml_config(str(tmp_config.absolute()))
    chroma_config = get_config_dict(config, "chroma")
    assert chroma_config is not None
    chroma = ChromaClient(chroma_config)

    # load ollama
    ollama_home = tmp_path_factory.mktemp("ollama_home")
    start_local_ollama(ollama_host, ollama_home)
    ollama_config = get_config_dict(config, "ollama")
    assert ollama_config is not None
    ollama = OllamaClient(ollama_config)

    yield _SessionData(PrivateQuery(chroma=chroma, ollama=ollama), chroma=chroma)

    stop_process("ollama")


def test_embed_documents(session_data: _SessionData, tmp_path: Path):
    """Test embed_vector_data."""
    word_pool = [
        "apple",
        "sunny",
        "mountain",
        "shadow",
        "galaxy",
        "breeze",
        "ocean",
        "river",
    ]
    collection_name = "test_collection123"
    doc_paths: list[Path] = []
    file_idx = 0
    file_count = 100
    while file_idx < file_count:
        tmp_file = tmp_path / f"tmp_file_{file_idx}.txt"
        tmp_file.write_text(" ".join(random.choices(word_pool, k=500)))  # noqa: S311
        doc_paths.append(tmp_file.absolute())
        file_idx += 1

    ids = session_data.private_query.embed_documents(
        collection_name=collection_name, document_paths=doc_paths
    )

    collection = session_data.chroma.get_or_create_collection(collection_name)
    result = collection.get(ids)
    result_docs = result["documents"]
    assert result_docs
    assert len(result_docs) >= file_count
    for doc in result_docs:
        assert str(doc.encode(encoding="UTF-8"))
    result_ids = result["ids"]
    assert result_ids
    assert ids.sort() == result_ids.sort()
