"""A module providing utility functionality for chromadb."""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import logging
from typing import Any, Literal

import torch
from chromadb import Collection, HttpClient, PersistentClient
from chromadb.api import ClientAPI
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from loguru import logger
from more_itertools import batched
from pydantic import BaseModel, Field, HttpUrl
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from retry_util import random_exponential_retry

_MAX_TOKENIZATION_LIMIT = 1_000_000  # A sensible token limit for the tokenizer


class LocalDbConfig(BaseModel):
    """Configuration model for a local database."""

    type: Literal["local"] = "local"
    persist_directory: str


class RemoteDbConfig(BaseModel):
    """Configuration model for a remote database."""

    type: Literal["remote"] = "remote"
    host: HttpUrl
    database_name: str
    auth_token: str | None


class ChromaClientConfig(BaseModel):
    """Configuration model for ChromaClient."""

    embedding_model: str
    embedding_model_revision: str
    model_cache_directory: str
    database: LocalDbConfig | RemoteDbConfig = Field(discriminator="type")


class ChromaClient:
    """A utility class for chromadb which encapsulates a client and embedding
    function.
    """

    def __init__(self, config: ChromaClientConfig):
        """Intialize new client.

        Args:
            config: The configuration this client will use.

        Raises:
            ConnectionError: If a client connection could not be established.

        """
        # instance variables
        self._client: ClientAPI
        self._embedding_func: SentenceTransformerEmbeddingFunction
        self._model_cache_dir: str
        self._tokenizer: PreTrainedTokenizerBase

        # load the database client connection
        match config.database:
            case LocalDbConfig(persist_directory=persist_dir):
                logger.info(
                    f"Setting up local chroma client: persist_directory: {persist_dir}"
                )
                self._client = PersistentClient(persist_dir)
            case RemoteDbConfig(
                host=hostname, database_name=database, auth_token=token
            ):
                logger.info(
                    f"Setting up remote chroma client: host: {hostname} database_name: \
                        {database}"
                )
                try:
                    kwargs: dict[str, Any] = {}
                    kwargs["ssl"] = hostname.scheme == "https"
                    if token is not None:
                        settings = Settings(
                            chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                            chroma_client_auth_credentials=token,
                        )
                        kwargs["settings"] = settings
                    self._client = random_exponential_retry(
                        lambda: HttpClient(
                            host=str(hostname), database=database, **kwargs
                        )
                    )
                except Exception as e:
                    err_msg = (
                        f"Failed to create remote chroma client: host: {hostname} \
                        database: {database}"
                    )
                    logger.error(err_msg)
                    raise ConnectionError(err_msg) from e

        # reduce the logging noise from huggingface_hub
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        # load the embedding model
        self._model_cache_dir = config.model_cache_directory
        embedding_model = config.embedding_model
        embedding_model_revision = config.embedding_model_revision
        self._embedding_func = _get_embedding_function(
            embedding_model, self._model_cache_dir, embedding_model_revision
        )

        # load the tokenizer
        model_path = f"sentence-transformers/{self._embedding_func.model_name}"
        try:
            tokenizer = random_exponential_retry(
                lambda: AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=model_path,
                    revision=embedding_model_revision,
                    token=False,
                    cache_dir=self._model_cache_dir,
                )
            )
            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                raise ValueError(f"Unexpected tokenizer type: {type(tokenizer)}")
            self._tokenizer = tokenizer

        except Exception as e:
            err_msg = f"Failed to retrieve toknizer for model: {model_path}"
            logger.error(err_msg)
            raise ConnectionError(err_msg) from e

    def get_or_create_collection(
        self, name: str, distance_func: str | None = None
    ) -> Collection:
        """Get or create a collection.

        Args:
            name: The name of the collection.
            distance_func: The name of the distance function to utilize.
        Returns: The exising or created collection.

        Raises:
            ConnectionError: If the chroma connection failed.

        """
        metadata: dict[str, str] = {}
        if distance_func:
            metadata["hnsw:space"] = distance_func
        try:
            return random_exponential_retry(
                lambda: self._client.get_or_create_collection(
                    name=name,
                    embedding_function=self._embedding_func,  # ty: ignore # type: ignore
                    metadata=metadata or None,
                )
            )
        except Exception as e:
            err_msg = f"Failed to get collection: {name}"
            logger.error(err_msg)
            raise ConnectionError(err_msg) from e

    def is_alive(self) -> bool:
        """Return whether the client connection is alive."""
        try:
            self._client.heartbeat()
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            logger.info("Client connection is not alive.")
            return False

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

        Raises:
            ConnectionError: If the ChromaDB connection failed.

        """
        collection = self.get_or_create_collection(collection_name)
        document_indices = list(range(len(documents)))
        for batch in batched(document_indices, self._client.get_max_batch_size()):
            start_idx = batch[0]
            end_idx = batch[-1] + 1
            try:
                random_exponential_retry(
                    lambda start_idx=start_idx, end_idx=end_idx: collection.upsert(
                        ids=ids[start_idx:end_idx],
                        documents=documents[start_idx:end_idx],
                        metadatas=metadatas[start_idx:end_idx],  # type: ignore # noqa: E501
                    )
                )
            except Exception as e:
                err_msg = f"Failed to upsert for collection: {collection_name}"
                logger.error(err_msg)
                raise ConnectionError(err_msg) from e

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

        Raises: RuntimeError If the tokenizer for the embedding model could not be
        retrieved.

        """
        self._tokenizer.model_max_length = _MAX_TOKENIZATION_LIMIT
        # tokenization safety rail assuming ~4 chars per token average
        if len(text) > _MAX_TOKENIZATION_LIMIT * 10:
            err_msg = f"Input text is too large for tokenization safety rails: len: \
                {len(text)}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        tokens = self._tokenizer.encode(
            text, add_special_tokens=False, truncation=False
        )
        chunks: list[str] = []
        start_index = 0
        while start_index < len(tokens):
            end_index = start_index + chunk_size
            chunk_tokens = tokens[start_index:end_index]
            tokens_text = self._tokenizer.decode(
                chunk_tokens, clean_up_tokenization_spaces=True
            )
            if tokens_text is not None and tokens_text:
                chunks.append(str(tokens_text))
            start_index += chunk_size - overlap
            if end_index >= len(tokens):
                break

        # drop empty chunks
        return [chunk for chunk in chunks if chunk]  # say that ten times fast ;)


def _get_embedding_function(
    model: str, model_cache_dir: str, model_revision: str
) -> SentenceTransformerEmbeddingFunction:
    """Create an embedding function with the given model name.

    Args:
        model: The name of the embedding model to be used.
        model_cache_dir: The directory to be used for model caching.
        model_revision: The revision of the model.

    Returns: The embedding function for the specified model name.

    Raises:
        ConnectionError: If the connection failed to the model repository.

    """
    kwargs: dict[str, Any] = {}
    system_device = torch.accelerator.current_accelerator(check_available=True)
    if system_device:
        kwargs["device"] = str(system_device)
    try:
        logger.info(f"Retrieving embedding model: {model}")
        return random_exponential_retry(
            lambda: embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model,
                token=False,
                cache_folder=model_cache_dir,
                revision=model_revision,
                **kwargs,
            )
        )
    except Exception as e:
        err_msg = f"Failed to retrieve embedding model: {model}"
        logger.error(err_msg)
        raise ConnectionError(err_msg) from e
