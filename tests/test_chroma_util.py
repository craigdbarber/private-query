"""Tests chroma_util."""

import random
from collections.abc import Generator
from dataclasses import dataclass

import pytest

from chroma_util import ChromaClient, ChromaClientConfig
from config_util import load_yaml_config


# The test chroma client for the session
@dataclass
class _ChromaSessionData:
    client: ChromaClient


@pytest.fixture(scope="session", name="chroma_session_data")
def suite_setup_teardown(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[_ChromaSessionData, None, None]:
    """Set up chroma client for test session."""
    # suite setup
    tmp_persist_dir = tmp_path_factory.mktemp("persist_dir")
    tmp_cache_dir = tmp_path_factory.mktemp("cache_dir")
    tmp_config_dir = tmp_path_factory.mktemp("config")
    tmp_config = tmp_config_dir / "config.yaml"
    tmp_config.write_text(f"""
embedding_model: "all-MiniLM-L6-v2"
embedding_model_revision: "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
model_cache_directory: "{tmp_cache_dir}"
database:
  type: "local"
  persist_directory: "{tmp_persist_dir}"
                """)
    yaml_config = load_yaml_config(str(tmp_config.absolute()))
    chroma_config = ChromaClientConfig.model_validate(yaml_config)
    client = ChromaClient(chroma_config)
    assert client

    yield _ChromaSessionData(client=client)


def test_client_setup(chroma_session_data: _ChromaSessionData):
    """Test client is successfully setup."""
    client = chroma_session_data.client
    assert client
    assert client.is_alive()


def test_get_or_create_collection(chroma_session_data: _ChromaSessionData):
    """Test get_or_create_collection."""
    client = chroma_session_data.client
    collection = client.get_or_create_collection("test_collection")
    assert collection
    assert collection.name == "test_collection"
    other_collection = client.get_or_create_collection("other_collection")
    assert other_collection
    assert other_collection.name == "other_collection"
    same_collection = client.get_or_create_collection("test_collection")
    assert same_collection
    assert same_collection.name == "test_collection"


def test_batched_upsert(chroma_session_data: _ChromaSessionData):
    """Test batched_upsert."""
    client = chroma_session_data.client
    docs: list[str] = []
    metadatas: list[dict[str, str]] = []
    ids: list[str] = []
    index = 0
    while index < 1000:
        docs.append(f"document contents for id:{index}")
        metadatas.append({"id": f"{index}"})
        ids.append(str(index))
        index += 1
    client.batched_upsert("test_collection1", docs, metadatas, ids)
    collection = client.get_or_create_collection("test_collection1")
    results = collection.get(ids=ids)
    assert results
    results_docs = results["documents"]
    assert results_docs
    assert [
        str(doc.encode(encoding="UTF-8")) for doc in results_docs
    ].sort() == docs.sort()
    results_ids = results["ids"]
    assert results_ids
    assert results_ids.sort() == ids.sort()
    results_metadatas = results["metadatas"]
    assert results_metadatas
    assert results_metadatas.sort(key=lambda d: d["id"]) == metadatas.sort(
        key=lambda d: d["id"]
    )


def test_chunk_text_by_tokens(chroma_session_data: _ChromaSessionData):
    """Test chunk_text_by_tokens."""
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
    document = " ".join(random.choices(word_pool, k=1000))  # noqa: S311
    chunks = chroma_session_data.client.chunk_text_by_tokens(document)
    assert chunks
    assert len(chunks) != 0
    assert all(chunk for chunk in chunks)
