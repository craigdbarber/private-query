"""
A module providing utility functionality for chromadb.
"""

from enum import StrEnum

import chromadb
from chromadb.api import ClientAPI, Collection
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from more_itertools import batched

from config_util import get_config_value
from device_util import retrieve_current_accelerator_type

_CHROMA_MAX_BATCH_SIZE = 166


class ChromaType(StrEnum):
    """An enum defining supported types in the configuration."""

    LOCAL = "local"
    REMOTE = "remote"

    @staticmethod
    def from_str(value: str) -> ChromaType:
        """Parses a configuration chroma type from the specified string.
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
    """A utility class for chromadb which encapsulates a client and embedding function.

    Args:
        config: A dictionary containing the configuration values to parse.
    """

    def __init__(self, config: dict[str, str]):
        # instance variables
        self._client: ClientAPI
        self._embedding_model: SentenceTransformerEmbeddingFunction

        # load the config
        db_type = ChromaType.from_str(get_config_value(config, "type"))
        if db_type == ChromaType.LOCAL:
            self._client = chromadb.PersistentClient(
                get_config_value(config, "persist_directory")
            )
        else:
            raise NotImplementedError(f"Chroma type not supported: ${config['type']}")
        # TODO: impemented support for remote type

        embedding_model = get_config_value(config, "embedding_model")
        self._embedding_func = get_embedding_function(embedding_model)

    def get_or_create_collection(
        self, name: str, embedding_model: str, distance_func: str = ""
    ) -> Collection:
        """Gets or creates a collection.
        Args:
            name: The name of the collection.
            embedding_model: The name of the embedding model to utilize.
            distance_func: The name of the distance function to utilize.
        Returns: The exising or created collection.
        """
        metadata = {}
        embedding_func = get_embedding_function(embedding_model)
        if distance_func != "":
            metadata["hnsw:space"] = distance_func
        return self._client.get_or_create_collection(
            name=name, embedding_function=embedding_func, metadata=metadata
        )


def batched_upsert(
    collection: chromadb.Collection,
    documents: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    """Performs batched upserts of the specified documents into the specified collection.

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
            metadatas=metadatas[start_idx:end_idx],
        )


def get_embedding_function(model: str) -> SentenceTransformerEmbeddingFunction:
    """Creates an embedding function with the given model name.
    Args:
        model: The name of the embedding model to be used.

    Returns: The embedding function for the specified model name.
    """
    device = retrieve_current_accelerator_type()
    if device != "":
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model, device=device
        )
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)
