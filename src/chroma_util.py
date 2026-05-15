"""A module providing utility functionality for chromadb."""

from enum import StrEnum
from typing import Any

import torch
from chromadb import Collection, PersistentClient
from chromadb.api import ClientAPI
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from more_itertools import batched

from config_util import NestedDict, get_config_str

_CHROMA_MAX_BATCH_SIZE = 166


class ChromaType(StrEnum):
    """An enum defining supported types in the configuration."""

    LOCAL = "local"
    REMOTE = "remote"


def _chroma_type_from_str(value: str) -> ChromaType:
    """Parse a configuration chroma type from the specified string.
    Raises a ValueError if undefined.

    Args:
        value: The value type to be parsed.
    Returns: The parsed configuration chroma type.

    """
    if value == ChromaType.LOCAL:
        return ChromaType.LOCAL
    if value == ChromaType.REMOTE:
        return ChromaType.REMOTE
    raise ValueError(f"Failed to parse chroma type: ${value}")


class ChromaClient:
    """A utility class for chromadb which encapsulates a client and embedding
    function.
    """

    def __init__(self, config: NestedDict):
        """Intialize new client.

        Args:
            config: A dictionary containing the configuration values to parse.

        """
        # instance variables
        self._client: ClientAPI
        self._embedding_model: SentenceTransformerEmbeddingFunction
        self._model_cache_dir: str

        # load the config
        chroma_type = get_config_str(config, "type")
        assert chroma_type is not None
        db_type = _chroma_type_from_str(chroma_type)
        if db_type == ChromaType.LOCAL:
            persist_dir = get_config_str(config, "persist_directory")
            assert persist_dir is not None
            self._client = PersistentClient(path=persist_dir)
        else:
            raise NotImplementedError(f"Chroma type not supported: {chroma_type}")
        # TODO: impemented support for remote type

        embedding_model = get_config_str(config, "embedding_model")
        assert embedding_model is not None
        model_cache_dir = get_config_str(config, "model_cache_directory")
        assert model_cache_dir is not None
        self._embedding_func = _get_embedding_function(embedding_model, model_cache_dir)

    def get_or_create_collection(
        self, name: str, distance_func: str = ""
    ) -> Collection:
        """Get or create a collection.

        Args:
            name: The name of the collection.
            distance_func: The name of the distance function to utilize.
        Returns: The exising or created collection.

        """
        metadata = {}
        if distance_func != "":
            metadata["hnsw:space"] = distance_func
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_func,  # type: ignore # ty: ignore
            metadata=metadata if metadata else None,  # type: ignore # ty: ignore
        )


def batched_upsert(
    collection: Collection,
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    """Perform batched upserts of the specified documents into the specified collection.

    Args:
        collection: The collection to be modified.
        documents: The documents to be upserted.
        metadatas: The metadata associated with the documents.
        ids: The ids associated with the documents.

    """
    document_indices = list(range(len(documents)))
    for batch in batched(document_indices, _CHROMA_MAX_BATCH_SIZE):
        start_idx = batch[0]
        end_idx = batch[-1]
        collection.upsert(
            ids=ids[start_idx:end_idx],
            documents=documents[start_idx:end_idx],
            metadatas=metadatas[start_idx:end_idx],  # type:ignore # ty: ignore # noqa: E501
        )


def _get_embedding_function(
    model: str, model_cache_dir: str
) -> SentenceTransformerEmbeddingFunction:
    """Create an embedding function with the given model name.

    Args:
        model: The name of the embedding model to be used.
        model_cache_dir: The directory to be used for model caching.

    Returns: The embedding function for the specified model name.

    """
    kwargs: dict[str, Any] = {
        "cache_folder": model_cache_dir,
        "token": False,
    }
    system_device = torch.accelerator.current_accelerator(check_available=True)
    if system_device is not None:
        kwargs["device"] = str(system_device)
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model, **kwargs
    )
