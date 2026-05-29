"""Contains the core business logic of the private query system."""

from inspect import cleandoc
from pathlib import Path
from typing import Any

from chromadb.base_types import Metadata
from loguru import logger

from chroma_util import ChromaClient
from ollama_util import OllamaClient
from transformer_util import extract_text_from_file


class PrivateQuery:
    """A library class which encapsulates the logic of loading the
    configuration, intializing system components, and executing prompt queries.
    """

    def __init__(
        self,
        chroma: ChromaClient,
        ollama: OllamaClient,
        collection_name: str,
        n_results: int = 10,
        char_radius: int = 1500,
    ):
        """Initialize the PrivateQuery class.

        Args:
            chroma: The chroma client to be utilized.
            ollama: The ollama client to be utilized.
            collection_name: The name of the collection to be used.
            n_results: The number of results to retrieve with vector queries.
            char_radius: The character radius to be used for constructing contexts for
            prompts.

        """
        # Instance variables
        self._chroma = chroma
        self._ollama = ollama
        self._n_results = n_results
        self._char_radius = char_radius
        self._semantic_collection = self._chroma.get_or_create_collection(
            f"private_query_semantic_{collection_name}"
        )
        self._raw_collection = self._chroma.get_or_create_collection_no_embedding(
            f"private_query_raw_{collection_name}"
        )

    def embed_documents(self, document_paths: list[Path]) -> list[str]:
        """Embed the specified documents into the chroma db.

        Args:
            collection_name: The name of the collection the data is to be embedded into.
            document_paths: The paths of the documents to be embedded.

        Returns: A list of ids created for the embedded documents.

        """
        docs: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []
        raw_doc_ids: list[str] = []
        raw_docs: list[str] = []

        logger.info("Processing documents:")

        for path in document_paths:
            resolved_path = path.resolve()
            logger.info(f"Processing doc: {resolved_path}")

            text = extract_text_from_file(resolved_path)
            str_path = str(resolved_path)
            raw_doc_ids.append(str_path)
            raw_docs.append(text)

            chunks: list[ChromaClient.TextChunk] = self._chroma.chunk_text_by_tokens(
                text
            )
            for idx, chunk in enumerate(chunks):
                docs.append(chunk.text)
                ids.append(f"id_{str(resolved_path)}_{idx}")
                metadata: dict[str, Any] = {}
                metadata["path"] = str_path
                metadata["start_token"] = chunk.start_token
                metadata["char_start"] = chunk.char_start
                metadata["char_end"] = chunk.char_end
                metadatas.append(metadata)

        logger.info("Batch upserting documents into chromadb.")
        self._chroma.batched_upsert(
            self._semantic_collection,
            documents=docs,
            ids=ids,
            metadatas=metadatas,
        )
        self._chroma.batched_upsert(
            self._raw_collection, documents=raw_docs, ids=raw_doc_ids
        )

        return ids

    def process_prompt(self, prompt: str) -> str:
        """Process a prompt using the specified collection as context.

        Args:
            prompt: The prompt to be processed.

        Returns: The result of the prompt.

        """
        context = self._retrieve_overlap_safe_context(prompt)

        system_prompt = cleandoc("""
### ROLE
Your role is a precise and helpful Knowledge Base Assistant. You answer
questions based exclusively on the provided context source blocks.

### INSTRUCTIONS
1. **Grounding:** Answer the user's question ONLY using the provided context.
2. **Synthesis:** Provide a descriptive, complete answer. Do not quote large
   blocks of text; instead, provide a clear synopsis and explanation in your
   own words.
3. **Structure:**
   - [Analysis]: A detailed answer to the question.
   - [Summary]: A concise 1-2 sentence conclusion.
   - [References]: A list of unique source paths used, formatted as:
     `* [file path]`.
4. **Constraints:**
   - Do not mention source documents or paths in-line within your answer.
   - Do not repeat paths in the References section.
   - Do not include character boundaries in the References section.
   - If the context is insufficient, state: "I do not have enough information
     to answer this question." and omit the References section.

### OUTPUT FORMAT
Follow the structure above strictly. Ensure the tone is professional and
unbiased.
""")
        # execute the query against the LLM
        response = self._ollama.execute_prompt(
            system_prompt=system_prompt, context=context, prompt=prompt
        )
        return response

    def _retrieve_overlap_safe_context(self, prompt: str) -> str:
        # query the semantic collection for document chunks related to the prompt
        query_results = self._semantic_collection.query(
            query_texts=[prompt],
            include=["metadatas"],
            n_results=self._n_results,
        )
        result_metadatas = (
            query_results["metadatas"][0]
            if query_results["metadatas"] is not None
            else []
        )

        # Calculate a list of expanded character bounds for each of the chunks
        # using char_radius, mapping them to the file path associated chunks.
        # This is to ensure surrounding context isn't lost,
        doc_char_bounds_map: dict[str, list[tuple[int, int]]] = {}
        for metadata in result_metadatas:
            path = _get_metadata_with_type(metadata, "path", str)
            char_start = _get_metadata_with_type(metadata, "char_start", int)
            char_end = _get_metadata_with_type(metadata, "char_end", int)
            expanded_bounds = (
                char_start - self._char_radius,
                char_end + self._char_radius,
            )
            doc_char_bounds_map.setdefault(path, []).append(expanded_bounds)

        # retrieve the set of unique raw document texts corresponding to the chunks
        target_doc_ids = list(doc_char_bounds_map.keys())
        raw_documents = self._raw_collection.get(
            ids=target_doc_ids, include=["documents"]
        )
        raw_documents = raw_documents if raw_documents is not None else {}
        # create a mapping of document path to raw text
        path_to_text_map: dict[str, str] = {}

        for path, text in zip(  # ty: ignore[not-iterable]
            raw_documents["ids"],
            raw_documents["documents"],  # ty: ignore[invalid-argument-type]
            strict=True,
        ):
            path_to_text_map[path] = text

        # Merge the character bounds for each document, and retrieve corresponding
        # slices of the raw text.
        merged_and_sliced_texts: list[str] = []
        for path, char_bounds in doc_char_bounds_map.items():
            if not path_to_text_map.get(path):
                continue
            text = path_to_text_map[path]
            merged_and_sliced_texts.extend(
                _merge_and_slice_document(path=path, char_bounds=char_bounds, text=text)
            )
        return "\n".join(merged_and_sliced_texts)


def _merge_and_slice_document(
    path: str, char_bounds: list[tuple[int, int]], text: str
) -> list[str]:
    # sort the bounds by the starting characters
    char_bounds.sort(key=lambda c: c[0])

    # merge overlapping character boundaries
    merged_bounds: list[tuple[int, int]] = []
    text_len = len(text)
    for char_start, char_end in char_bounds:
        # clamp the bounds by the document start and length
        char_start = max(0, char_start)
        char_end = min(text_len, char_end)
        if not merged_bounds:
            merged_bounds.append((char_start, char_end))
            continue
        prev_start, prev_end = merged_bounds[-1]
        if char_start <= prev_end:
            merged_bounds[-1] = (prev_start, max(prev_end, char_end))
        else:
            merged_bounds.append((char_start, char_end))

    # create labeled slices for each of the merged boundaries
    slices: list[str] = []
    for char_start, char_end in merged_bounds:
        slice_text = text[char_start:char_end].strip()
        labeled_text = (
            f"---SOURCE: path: {path} character_boundaries: ({char_start}"
            f"-{char_end}) ---\n{slice_text}"
        )
        slices.append(labeled_text)
    return slices


def _get_metadata_with_type[T](metadata: Metadata, key: str, cls: type[T]) -> T:
    if key not in metadata:
        err_msg = f"Metadata['{key}']: key not found."
        logger.error(err_msg)
        raise KeyError(err_msg)
    value = metadata[key]
    if value is None:
        err_msg = f"Metadata['{key}'] value is None."
        logger.error(err_msg)
        raise ValueError(err_msg)
    if not isinstance(value, cls):
        err_msg = str(
            f"Metadata['{key}']: expected type: {cls.__name__}"
            f" got: {type(value).__name__}."
        )
        logger.error(err_msg)
        raise ValueError(err_msg)
    return value
