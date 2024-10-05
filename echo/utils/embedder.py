from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import openai
import os
import tiktoken
import logging


OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-small": {
        "input_token_limit": 8191,
    },
    "text-embedding-3-large": {
        "input_token_limit": 8191,
    },
    "text-embedding-ada-002": {
        "input_token_limit": 8191,
    },
}
DEFAULT_DIMENSIONS = 1024


class Embedder(ABC):
    """Generic class for embedding text."""

    def __init__(self, dimensions: int, **kwargs: Any):
        self.dimensions = dimensions

    @abstractmethod
    def embed(self, text: str, timeout: float) -> List[float]:
        """
        Generate an embedding for the given text.

        :param text: The text to embed.
        :param timeout: The maximum time to wait for the embedding to be generated.
        """
        pass


class OpenAIEmbedder(Embedder):
    """
    Embedder implementation that uses OpenAI's embedding models.

    :param embedding_model: The OpenAI embedding model to use.
    :param dimensions: The number of dimensions for the embedding.
    :param api_key: The OpenAI API key to use.
    """

    def __init__(
        self,
        embedding_model: str,
        dimensions: int = DEFAULT_DIMENSIONS,
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the OpenAIEmbedder with a model and API key."""
        if embedding_model not in OPENAI_EMBEDDING_MODELS.keys():
            raise ValueError(f"Invalid OpenAI embedding model: {embedding_model}")
        super().__init__(dimensions=dimensions)
        self._logger = logging.getLogger(__name__)
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self._client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._encoding = tiktoken.encoding_for_model(self.embedding_model)

    def embed(self, text: str, timeout: float = 120) -> List[float]:
        """
        Generate an embedding for the given text using the OpenAI API.

        :param text: The text to embed.
        :param timeout: The maximum time to wait for the embedding to be generated.
        """

        if not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        if len(self._encoding.encode(text)) > OPENAI_EMBEDDING_MODELS[self.embedding_model]["input_token_limit"]:
            raise ValueError("Input text is too long for embedding model.")
        try:
            response = self._client.embeddings.create(
                model=self.embedding_model,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float",
                timeout=timeout,
            )
            if not hasattr(response, "data") or not response.data:
                raise RuntimeError("Invalid response from embedding API.")
            return response.data[0].embedding
        except (TimeoutError, RuntimeError) as e:
            self._logger.error(f"Error occurred while generating embedding: {e}")
            return None
