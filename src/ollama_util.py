"""A module providing utility functionality for ollama."""

import logging
import time
from typing import Any

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

        Raises:
            ConnectionError: If a connection could not be established with Ollama.

        """
        # Instance variables
        self._client: Client
        self._model: str

        model = get_config_str(config, "model")
        assert model
        self._model = model
        host = get_config_str(config, "host")
        assert host
        kwargs: dict[str, Any] = {}
        api_key = get_config_str(config, "api_key", raise_error=False)
        if api_key is not None:
            kwargs["headers"] = {"Authorization": f"Bearer ${api_key}"}
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        max_retries = 3
        retry_delay = 2
        attempts = 0
        success = False
        while not success and attempts < max_retries:
            try:
                self._client = Client(host=host, **kwargs)
                self._client.pull(self._model)
                success = True
            except Exception:  # pylint: disable=broad-exception-caught
                attempts += 1
                time.sleep(retry_delay)
        if not success:
            raise ConnectionError(
                f"Failed to establish connection with Ollama at host: {host}"
            )

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
