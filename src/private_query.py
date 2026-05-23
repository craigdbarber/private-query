"""Contains the core business logic of the private query system."""

from pathlib import Path

from loguru import logger

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

        logger.info("Processing documents:")

        for path in document_paths:
            logger.info(f"Processing doc: {path}")
            with open(path, encoding="UTF-8") as file:
                text = file.read()
                file_path = str(Path(file.name))
                chunks = self._chroma.chunk_text_by_tokens(text)
                for idx, chunk in enumerate(chunks):
                    docs.append(chunk)
                    ids.append(f"id_{file_path}_{idx}")
                    metadatas.append({"path": file_path})

        logger.info(f"Batch upserting documents into collection: {collection_name}")
        self._chroma.batched_upsert(
            collection_name=collection_name,
            documents=docs,
            ids=ids,
            metadatas=metadatas,
        )

        return ids

    def process_prompt(self, prompt: str, collection_name: str) -> str:
        """Process a prompt using the specified collection as context.

        Args:
            prompt: The prompt to be processed.
            collection_name: The name of the collection to be used as context.
        Returns: The result of the prompt.

        """
        # query for context from the vector db
        logger.info(f"Querying vector collection: {collection_name}")
        collection = self._chroma.get_or_create_collection(collection_name)
        query_results = collection.query(
            query_texts=[prompt], include=["documents", "metadatas"]
        )
        # build context string
        result_docs = (
            query_results["documents"][0]
            if query_results["documents"] is not None
            else []
        )
        result_metadatas = (
            query_results["metadatas"][0]
            if query_results["metadatas"] is not None
            else []
        )
        results_zip = list(zip(result_docs, result_metadatas))
        logger.info(f"Query results len: {len(results_zip)}")
        context = ""
        for doc, metadata in results_zip:
            logger.info(f"Query result path: {metadata['path']}")
            context += f"\n\npath: {metadata['path']}"
            context += f"\n\ndocument: {doc}"

        context_prompt = f"""
        Your role is a knowledge base provider, answering questions about information
        contained in a local set of documents. Answer the question below only using the
        provided context. Provide a descriptive and complete answer, including a final
        summary. Do not include the exact contents of context documents in your answer,
        just include your synopsis and explanation. Do not include mention of source
        documents in-line with your answer. If the context does not contain
        enough information to accurately answer the question, say so clearly. After the
        summary, include a well formatted references section which provides a list of
        the unique file paths from the context used to generate your answer, making sure
        to just include the file paths not the contents of the documents, also
        making sure to format each as follows: "* [file path]". If you could not
        provide an answer due to lack of context information, omit the references
        section.

        Context:
        {context}

        Question: {prompt}
        """
        # execute the query against the LLM
        response = self._ollama.execute_prompt(context_prompt)
        return response
