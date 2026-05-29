"""A test suite for the private-query module."""

from collections.abc import Generator
from dataclasses import dataclass

import pytest
from test_util import start_local_ollama

from chroma_util import ChromaClient, ChromaClientConfig
from config_util import load_yaml_config
from ollama_util import OllamaClient, OllamaClientConfig
from private_query import PrivateQuery
from resource_util import resolve_directory


@dataclass
class _PrivateQuerySessionData:
    private_query: PrivateQuery
    chroma: ChromaClient


@pytest.fixture(scope="session", name="pq_session_data")
def suite_setup_teardown(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[_PrivateQuerySessionData, None, None]:
    """Provide setup and teardown functionality for test suite."""
    chroma: ChromaClient
    ollama: OllamaClient

    # setup config
    ollama_host = "localhost:11434"
    collection_name = "private_query_test_collection"
    tmp_persist_dir = tmp_path_factory.mktemp("persist_dir")
    tmp_cache_dir = tmp_path_factory.mktemp("cache_dir")
    tmp_config_dir = tmp_path_factory.mktemp("config")
    tmp_config = tmp_config_dir / "config.yaml"
    tmp_config.write_text(f"""
chroma:
    embedding_model: "all-MiniLM-L6-v2"
    embedding_model_revision: "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
    model_cache_directory: "{tmp_cache_dir}"
    database:
        type: "local"
        persist_directory: "{tmp_persist_dir}"
ollama:
    host: "{ollama_host}"
    model: "llama3.2:latest"
                         """)

    # load chroma client
    config = load_yaml_config(str(tmp_config.absolute()))
    chroma_config = ChromaClientConfig.model_validate(config["chroma"])
    chroma = ChromaClient(chroma_config)

    # load ollama
    ollama_home = tmp_path_factory.mktemp("ollama_home")
    start_local_ollama(ollama_host, ollama_home)
    ollama_config = OllamaClientConfig.model_validate(config["ollama"])
    ollama = OllamaClient(ollama_config)

    yield _PrivateQuerySessionData(
        PrivateQuery(chroma=chroma, ollama=ollama, collection_name=collection_name),
        chroma=chroma,
    )


def test_embed_documents(pq_session_data: _PrivateQuerySessionData):
    """Test embed_documents."""
    test_data_dir = resolve_directory("tests/data")
    test_data_files = list(test_data_dir.glob("*"))
    assert test_data_files and len(test_data_files) > 0

    ids = pq_session_data.private_query.embed_documents(document_paths=test_data_files)
    collection = pq_session_data.private_query._semantic_collection  # noqa # pylint: disable=protected-access
    result = collection.get(ids)
    result_docs = result["documents"]

    assert result_docs
    assert len(result_docs) >= len(test_data_files)
    for doc in result_docs:
        assert str(doc.encode(encoding="UTF-8"))
    result_ids = result["ids"]
    assert result_ids
    assert sorted(ids) == sorted(result_ids)


def test_process_prompt(pq_session_data: _PrivateQuerySessionData):
    """Test process prompt."""
    test_data_dir = resolve_directory("tests/data")
    test_txt_files = list(test_data_dir.glob("*.txt"))
    assert test_txt_files and len(test_txt_files) > 0

    pq_session_data.private_query.embed_documents(document_paths=test_txt_files)

    response = pq_session_data.private_query.process_prompt(
        prompt="Who was sherlock holmes?"
    )
    assert response


def test_system_prompt(pq_session_data: _PrivateQuerySessionData):
    """Test the system prompt works correctly."""
    test_data_dir = resolve_directory("tests/data")
    test_txt_files = list(test_data_dir.glob("*.txt"))
    assert test_txt_files and len(test_txt_files) > 0

    pq_session_data.private_query.embed_documents(document_paths=test_txt_files)

    response = pq_session_data.private_query.process_prompt("who was watson?")
    assert "References" in response


def test_merge_and_slice_document():
    """Test _merge_and_slice_document."""
    from private_query import _merge_and_slice_document

    text = "01234569abcdefghij"
    path = "test.txt"
    # Overlapping: (0,5) and (3,8) -> (0,8)
    # Disjoint: (10, 15)
    # Out of bounds: (-5,2) -> (0,2)
    bounds = [(0, 5), (3, 8), (10, 15), (-5, 2)]
    slices = _merge_and_slice_document(path=path, char_bounds=bounds, text=text)
    assert len(slices) == 2
    assert "character_boundaries: (0-8)" in slices[0]
    assert "character_boundaries: (10-15)" in slices[1]
