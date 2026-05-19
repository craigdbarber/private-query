"""Contains the core business logic of the private query system."""

from pathlib import Path
from typing import Final

from chroma_util import ChromaClient, batched_upsert
from ollama_util import OllamaClient

TEXT_CHUNK_SIZE: Final[int] = 1000
TEXT_CHUNK_OVERLAP: Final[int] = 100


class PrivateQuery:
    """A library class which encapsulates the logic of loading the
    configuration, intializing system components, and executing prompt queries.
    """

    def __init__(self, chroma: ChromaClient, ollama: OllamaClient):
        """Initialize the PrivateQuery class.

        Args:
            chroma: The chroma client to be utilized.
            ollama: The ollama client to be utilized.

        """
        # Instance variables
        self._chroma = chroma
        self._ollama = ollama

    def embed_vector_data(
        self, collection_name: str, document_paths: list[Path]
    ) -> list[str]:
        """Embed data into the vector db.

        Args:
            collection_name: The name of the collection the data is to be embedded into.
            document_paths: The paths of the documents to be embedded.

        Returns: A list of ids created for the embedded documents.

        """
        docs: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []

        for path in document_paths:
            with open(path, encoding="UTF-8") as file:
                text = file.read()
                file_path = str(Path(file.name))
                if len(text) > TEXT_CHUNK_SIZE:
                    chunks = chunk_text(text)
                    chunk_count = 0
                    for chunk in chunks:
                        docs.append(chunk)
                        ids.append(f"id_{file_path}_{chunk_count}")
                        metadatas.append({"path:": file_path})
                        chunk_count += 1
                else:
                    docs.append(file.read())
                    ids.append(f"id_{file_path}")
                    metadatas.append({"path": file_path})

        # batch upsert the docs
        collection = self._chroma.get_or_create_collection(collection_name)
        batched_upsert(
            collection=collection, documents=docs, ids=ids, metadatas=metadatas
        )

        return ids

    def process_prompt_query(self) -> str:
        """Process the query."""
        # query for context from the vector db
        # build context string
        # execute the query against the LLM
        # TODO: implement me
        raise NotImplementedError()


def chunk_text(text: str) -> list[str]:
    """Split the specified text into overlapping chunks for better LLM performance.

    Args:
        text: The text to be chunked.
    Returns: A list of the generated text chunks.

    """
    chunks: list[str] = []
    start_index = 0
    while start_index < len(text):
        end_index = (
            start_index + TEXT_CHUNK_SIZE
            if start_index + TEXT_CHUNK_SIZE < len(text)
            else len(text)
        )
        chunk = text[start_index:end_index]
        # prefer clean sentence breaks
        if end_index < len(text):
            last_period_index = chunk.rfind(".")
            if last_period_index > start_index + TEXT_CHUNK_SIZE / 2:
                end_index = start_index + last_period_index + 1
                chunk = text[start_index:end_index]
        chunks.append(chunk.strip())
        start_index = (
            end_index - TEXT_CHUNK_OVERLAP if end_index < len(text) else end_index
        )

    # drop empty chunks
    return [chunk for chunk in chunks if chunk]  # say that ten times fast ;)
