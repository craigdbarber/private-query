"""
A module providing utility functionality for ollama.
"""

from ollama import Client
import config_util


class OllamaClient:
    def __init__(self, config: dict[str, str]):
        """A library class which encapsulates loading the ollama configuration
        and maintaining the client.
        Args:
            config: The ollama configuration to be used.
        """
        # Instance variables
        self._client: Client = None
        self._model: str = None

        self._model = config_util.get_config_value(config, "model")
        host = config_util.get_config_value(config, "host")
        api_key = config_util.get_config_value(config, "api_key")
        if api_key is not None:
            self._client = Client(
                host=host, headers={"Authorization": f"Bearer ${api_key}"}
            )
        else:
            self._client = Client(host)

    def execute_prompt(self, prompt: str) -> str:
        """Executes the specified prompt against the configured model.
        Args:
            prompt: The prompt to be executed.
        Returns: The model's response.
        """
        response = self._client.generate(model=self._model, prompt=prompt, stream=False)
        return response["response"]
