"""A module providing utility functionality for chromadb."""

from enum import StrEnum
from typing import Any

import torch
from chromadb import Collection, PersistentClient
from chromadb.api import ClientAPI
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from more_itertools import batched
from transformers import AutoTokenizer

from config_util import NestedDict, get_config_str

_CHROMA_MAX_BATCH_SIZE = 166


class _ChromaType(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


def _chroma_type_from_str(value: str) -> _ChromaType:
    """Parse a configuration chroma type from the specified string.
    Raises a ValueError if undefined.

    Args:
        value: The value type to be parsed.
    Returns: The parsed configuration chroma type.

    """
    if value == _ChromaType.LOCAL:
        return _ChromaType.LOCAL
    if value == _ChromaType.REMOTE:
        return _ChromaType.REMOTE
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
        self._embedding_func: SentenceTransformerEmbeddingFunction
        self._model_cache_dir: str

        # load the config
        chroma_type = get_config_str(config, "type")
        assert chroma_type
        db_type = _chroma_type_from_str(chroma_type)
        if db_type == _ChromaType.LOCAL:
            persist_dir = get_config_str(config, "persist_directory")
            assert persist_dir
            self._client = PersistentClient(path=persist_dir)
        else:
            raise NotImplementedError(f"Chroma type not supported: {chroma_type}")
            # TODO: impemented support for remote type

        embedding_model = get_config_str(config, "embedding_model")
        assert embedding_model
        model_cache_dir = get_config_str(config, "model_cache_directory")
        assert model_cache_dir
        self._model_cache_dir = model_cache_dir
        self._embedding_func = _get_embedding_function(
            embedding_model, self._model_cache_dir
        )

    def get_or_create_collection(
        self, name: str, distance_func: str | None = None
    ) -> Collection:
        """Get or create a collection.

        Args:
            name: The name of the collection.
            distance_func: The name of the distance function to utilize.
        Returns: The exising or created collection.

        """
        metadata: dict[str, str] = {}
        if distance_func:
            metadata["hnsw:space"] = distance_func
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_func,  # type: ignore # ty: ignore
            metadata=metadata if metadata else None,  # type: ignore # ty: ignore
        )

    def is_alive(self) -> bool:
        """Return whether the client connection is alive."""
        time = self._client.heartbeat()
        return time is not None

    def batched_upsert(
        self,
        collection_name: str,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """Perform batched upserts of the specified documents into the specified
        collection.

        Args:
            collection_name: The name of the collection to be modified.
            documents: The documents to be upserted.
            metadatas: The metadata associated with the documents.
            ids: The ids associated with the documents.

        """
        collection = self.get_or_create_collection(collection_name)
        document_indices = list(range(len(documents)))
        for batch in batched(document_indices, self._client.get_max_batch_size()):
            start_idx = batch[0]
            end_idx = batch[-1]
            collection.upsert(
                ids=ids[start_idx:end_idx],
                documents=documents[start_idx:end_idx],
                metadatas=metadatas[start_idx:end_idx],  # type:ignore # ty: ignore # noqa: E501
            )

    def chunk_text_by_tokens(
        self, text: str, chunk_size: int = 256, overlap: int = 32
    ) -> list[str]:
        """Split the specified text into overlapping chunks of tokens for better LLM
        performance. Tailors the tokenization method to the specified embedding model.

        Args:
            text: The text to be chunked.
            chunk_size: The number of tokens in each chunk size.
            overlap: The number of tokens to overlap between chunks.
            cache_dir: The cache director for the embedding model.
        Returns: A list of the generated text chunks.

        """
        kwargs: dict[str, Any] = {"token": False, "cache_dir": self._model_cache_dir}
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=f"sentence-transformers/{self._embedding_func.model_name}",
            **kwargs,
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks: list[str] = []
        start_index = 0
        while start_index < len(tokens):
            end_index = start_index + chunk_size
            chunk_tokens = tokens[start_index:end_index]
            tokens_text = tokenizer.decode(
                chunk_tokens, clean_up_tokenization_spaces=True
            )
            chunks.append(tokens_text)
            start_index += chunk_size - overlap
            if end_index >= len(tokens):
                break

        # drop empty chunks
        return [chunk for chunk in chunks if chunk]  # say that ten times fast ;)


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
    if system_device:
        kwargs["device"] = str(system_device)
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model, **kwargs
    )
