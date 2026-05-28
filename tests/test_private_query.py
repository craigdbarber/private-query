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
        PrivateQuery(chroma=chroma, ollama=ollama), chroma=chroma
    )


def test_embed_documents(pq_session_data: _PrivateQuerySessionData):
    """Test embed_documents."""
    test_data_dir = resolve_directory("tests/data")
    test_data_files = list(test_data_dir.glob("*"))
    assert test_data_files and len(test_data_files) > 0
    collection_name = "test_embed_documents"

    ids = pq_session_data.private_query.embed_documents(
        collection_name=collection_name, document_paths=test_data_files
    )
    collection = pq_session_data.chroma.get_or_create_collection(collection_name)
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
    collection_name = "test_process_prompt"

    pq_session_data.private_query.embed_documents(
        collection_name=collection_name, document_paths=test_txt_files
    )

    response = pq_session_data.private_query.process_prompt(
        prompt="Who was sherlock holmes?",
        collection_name=collection_name,
    )
    assert response
