"""Contains the core business logic of the private query system."""

from pathlib import Path

from chroma_util import ChromaClient
from ollama_util import OllamaClient


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

    def embed_documents(
        self, collection_name: str, document_paths: list[Path]
    ) -> list[str]:
        """Embed the specified documents into the chroma db.

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
                chunks = self._chroma.chunk_text_by_tokens(text)
                for idx, chunk in enumerate(chunks):
                    docs.append(chunk)
                    ids.append(f"id_{file_path}_{idx}")
                    metadatas.append({"path:": file_path})

        # batch upsert the docs
        self._chroma.batched_upsert(
            collection_name=collection_name,
            documents=docs,
            ids=ids,
            metadatas=metadatas,
        )

        return ids

    def process_prompt_query(self) -> str:
        """Process the query."""
        # query for context from the vector db
        # build context string
        # execute the query against the LLM
        # TODO: implement me
        raise NotImplementedError()
