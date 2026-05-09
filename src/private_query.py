import config_util
from chroma_util import ChromaClient
from ollama_util import OllamaClient


class PrivateQuery:
    _CONFIG_FILE_PATH = "config.yaml"

    def __init__(self):
        """A library class which encapsulates the logic of loading the
        configuration, intializing system components, and executing prompt queries.
        """
        # Instance variables
        self._chroma_util: ChromaClient
        self._ollama_client: OllamaClient
        self.myVar = ""

        config = config_util.load_yaml_config(PrivateQuery._CONFIG_FILE_PATH)
        chroma_config = config_util.get_config_value(config, "chroma")
        assert chroma_config is not None
        self._chroma_util = ChromaClient(chroma_config)
        ollama_config = config_util.get_config_value(config, "ollama")
        assert ollama_config is not None
        self._ollama_client = OllamaClient(ollama_config)

    def load_vector_data(self):
        """Load data into the vector db"""
        # TODO: implement me
        raise NotImplementedError()

    def process_prompt_query(self) -> str:
        """Process the query"""
        # query for context from the vector db
        # build context string
        # execute the query against the LLM
        # TODO: implement me
        raise NotImplementedError()


def chunk_text(text: str) -> list[str]:
    """This will chunk text."""
    # TODO: implement me
    raise NotImplementedError()
