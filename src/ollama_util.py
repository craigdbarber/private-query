"""A module providing utility functionality for ollama."""

import logging
from typing import Any

from loguru import logger
from ollama import Client
from pydantic import BaseModel, HttpUrl, field_validator

from retry_util import random_exponential_retry


class OllamaClientConfig(BaseModel):
    """A configuration model for the OllamaClient."""

    model: str
    host: HttpUrl
    api_key: str | None = None

    @field_validator("host", mode="before")
    @classmethod
    def _allows_schemeless_host(cls, value: str) -> str:
        if value and not value.startswith(("http://", "https://")):
            return f"http://{value}"
        return value


class OllamaClient:
    """A library class which encapsulates loading the ollama configuration and
    maintaining the client.
    """

    def __init__(self, config: OllamaClientConfig):
        """Initialize the ollama client.

        Args:
        config: The ollama configuration to be used.

        Raises:
            ConnectionError: If a connection could not be established with Ollama.

        """
        # Instance variables
        self._client: Client
        self._model = config.model
        host = config.host
        kwargs: dict[str, Any] = {}
        api_key = config.api_key
        if api_key is not None:
            kwargs["headers"] = {"Authorization": f"Bearer ${api_key}"}
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        try:
            logger.info(f"Attempting to connect to ollama, host: {host}")
            self._client = random_exponential_retry(
                lambda: Client(host=str(host), **kwargs)
            )
        except Exception as e:
            logger.error(
                f"Failed to establish ollama client connection with host: {host}"
            )
            raise ConnectionError("Failed to establish ollama client connection") from e

        try:
            logger.info(f"Pulling model: {self._model}")
            random_exponential_retry(lambda: self._client.pull(self._model))
        except Exception as e:
            err_msg = f"Failed to pull model: {self._model} "
            logger.error(err_msg)
            raise ConnectionError(err_msg) from e

    def contains_model(self, model: str) -> bool:
        """Return whether the specified model is loaded.

        Args:
            model: The model name.

        Returns: Whether the specified model is loaded.

        """
        models = self._client.list().models
        return any(m.model == model for m in models if m is not None)

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
