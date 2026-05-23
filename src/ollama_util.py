"""A module providing utility functionality for ollama."""

import logging
from collections.abc import Callable
from typing import Any

from loguru import logger
from ollama import Client
from tenacity import retry, stop_after_attempt, wait_random_exponential

from config_util import get_config_value, get_config_value_or_none


class OllamaClient:
    """A library class which encapsulates loading the ollama configuration and
    maintaining the client.
    """

    def __init__(self, config: dict[str, str]):
        """Initialize the ollama client.

        Args:
        config: The ollama configuration to be used.

        Raises:
            ConnectionError: If a connection could not be established with Ollama.

        """
        # Instance variables
        self._client: Client
        self._model: str

        model = get_config_value(config, "model", str)
        self._model = model
        host = get_config_value(config, "host", str)
        kwargs: dict[str, Any] = {}
        api_key = get_config_value_or_none(config, "api_key", str)
        if api_key is not None:
            kwargs["headers"] = {"Authorization": f"Bearer ${api_key}"}
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        @retry(
            wait=wait_random_exponential(max=10),
            stop=stop_after_attempt(3),
        )
        def attempt_client_connect(connect_func: Callable[..., Client]) -> Client:
            return connect_func()

        try:
            logger.info(f"Attempting to connect to ollama, host: {host}")
            self._client = attempt_client_connect(lambda: Client(host=host, **kwargs))
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(
                f"Failed to establish ollama client connection with host: {host}"
            )
            raise ConnectionError("Failed to establish ollama client connection") from e

        logger.info(f"Pulling model: {self._model}")
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
        logger.debug(f"Executing ollama prompt: {prompt}")
        response = self._client.chat(
            model=self._model, messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
