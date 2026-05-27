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
            err_msg = f"Failed to establish Ollama client connection with host: {host}"
            logger.error(err_msg)
            raise ConnectionError(err_msg) from e

        try:
            logger.info(f"Pulling model: {self._model}")
            random_exponential_retry(lambda: self._client.pull(self._model))
        except Exception as e:
            err_msg = f"Failed to pull model: {self._model}"
            logger.error(err_msg)
            raise ConnectionError(err_msg) from e

    def contains_model(self, model: str) -> bool:
        """Return whether the specified model is loaded.

        Args:
            model: The model name.

        Returns: Whether the specified model is loaded.

        Raises:
            ConnectionError: If the connection to Ollama failed.

        """
        try:
            models = random_exponential_retry(lambda: self._client.list().models)
        except Exception as e:
            err_msg = "Failed to list models"
            logger.error(err_msg)
            raise ConnectionError(err_msg) from e
        return any(m.model == model for m in models if m is not None)

    def execute_prompt(self, prompt: str) -> str:
        """Execute the specified prompt against the configured model.

        Args:
            prompt: The prompt to be executed.

        Returns: The model's response.

        Raises:
            ConnectionError: If the connection to Ollama failed.

        """
        logger.debug(f"Executing ollama prompt: {prompt}")
        try:
            response = random_exponential_retry(
                lambda: self._client.chat(
                    model=self._model, messages=[{"role": "user", "content": prompt}]
                )
            )
        except Exception as e:
            err_msg = "Failed executing prompt."
            logger.error(err_msg)
            raise ConnectionError(err_msg) from e
        return response["message"]["content"]
