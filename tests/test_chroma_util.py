"""Tests chroma_util."""

import tempfile
from pathlib import Path

import pytest

from chroma_util import _CHROMA_MAX_BATCH_SIZE, ChromaClient, batched_upsert
from config_util import NestedDict, load_yaml_config

# The test chroma client for the session
_chroma_client: ChromaClient


@pytest.fixture(scope="session", autouse=True)
def suite_setup_teardown():
    """Set up chroma client for test session."""
    tmp_persist_dir = tempfile.TemporaryDirectory()
    tmp_persist_dir_path = Path(tmp_persist_dir.name)
    tmp_cache_dir = tempfile.TemporaryDirectory()
    tmp_cache_dir_path = Path(tmp_cache_dir.name)
    client: ChromaClient
    with tempfile.NamedTemporaryFile(mode="w+t") as tmp_config:
        tmp_config.write(f"""
type: "local"
persist_directory: "{tmp_persist_dir_path}"
embedding_model: "all-MiniLM-L6-v2"
model_cache_directory: "{tmp_cache_dir_path}"
                    """)
        tmp_config.flush()
        chroma_config: NestedDict = load_yaml_config(tmp_config.name)
        assert chroma_config is not None
        client = ChromaClient(chroma_config)
        assert client is not None
        global _chroma_client
        _chroma_client = client

    yield

    tmp_persist_dir.cleanup()
    tmp_cache_dir.cleanup()


def test_client_setup():
    """Test client is successfully setup."""
    assert _chroma_client is not None


def test_get_or_create_collection():
    """Test get_or_create_collection."""
    collection = _chroma_client.get_or_create_collection("test_collection")
    assert collection is not None
    assert collection.name == "test_collection"
    other_collection = _chroma_client.get_or_create_collection("other_collection")
    assert other_collection is not None
    assert other_collection.name == "other_collection"
    same_collection = _chroma_client.get_or_create_collection("test_collection")
    assert same_collection is not None
    assert same_collection.name == "test_collection"


def test_batched_upsert():
    """Test batched_upsert."""
    collection = _chroma_client.get_or_create_collection("test_collection1")
    docs: list[str] = []
    metadatas: list[dict[str, str]] = []
    ids: list[str] = []
    index = 0
    while index < 2 * _CHROMA_MAX_BATCH_SIZE:
        docs.append(f"document_{index}")
        metadatas.append({"id": f"{index}"})
        ids.append(str(index))
        index += 1
    batched_upsert(collection, docs, metadatas, ids)
    results = collection.get(ids=ids)
    assert results is not None
    results_docs = results["documents"]
    assert results_docs is not None
    assert results_docs.sort() == docs.sort()
    results_ids = results["ids"]
    assert results_ids is not None
    assert results_ids.sort() == ids.sort()
    results_metadatas = results["metadatas"]
    assert results_metadatas is not None
    assert results_metadatas.sort(key=lambda d: d["id"]) == metadatas.sort(  # type: ignore # ty: ignore
        key=lambda d: d["id"]
    )
