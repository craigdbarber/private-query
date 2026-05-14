from chroma_util import ChromaClient
from ollama_util import OllamaClient


class PrivateQuery:
    def __init__(self, chroma: ChromaClient, ollama: OllamaClient):
        """A library class which encapsulates the logic of loading the
        configuration, intializing system components, and executing prompt queries.
        """
        # Instance variables
        self._chroma = chroma
        self._ollama = ollama

    def load_vector_data(self):
        """Load data into the vector db."""
        # TODO: implement me
        raise NotImplementedError()

    def process_prompt_query(self) -> str:
        """Process the query."""
        # query for context from the vector db
        # build context string
        # execute the query against the LLM
        # TODO: implement me
        raise NotImplementedError()


def chunk_text(text: str) -> list[str]:
    """Chunks the text into parts."""
    # TODO: implement me
    raise NotImplementedError()
