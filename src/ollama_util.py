"""A module providing utility functionality for ollama."""

from ollama import Client

from config_util import NestedDict, get_config_str


class OllamaClient:
    """A library class which encapsulates loading the ollama configuration and
    maintaining the client.
    """

    def __init__(self, config: NestedDict):
        """Initialize the ollama client.

        Args:
        config: The ollama configuration to be used.

        """
        # Instance variables
        self._client: Client
        self._model: str

        model = get_config_str(config, "model")
        assert model is not None
        self._model = model
        host = get_config_str(config, "host")
        assert host is not None
        api_key = get_config_str(config, "api_key", raise_error=False)
        if api_key is not None:
            self._client = Client(
                host=host, headers={"Authorization": f"Bearer ${api_key}"}
            )
        else:
            self._client = Client(host)
        self._client.pull(self._model)

    def contains_model(self, model: str) -> bool:
        """Return whether the specified model is loaded.

        Args:
            model: The model name.

        Returns: Whether the specified model is loaded.

        """
        client_model_found = False
        models = self._client.list().models
        for m in models:
            if m.model is not None and m.model == model:
                client_model_found = True
        return client_model_found

    def execute_prompt(self, prompt: str) -> str:
        """Execute the specified prompt against the configured model.

        Args:
            prompt: The prompt to be executed.
        Returns: The model's response.

        """
        response = self._client.chat(
            model=self._model, messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
