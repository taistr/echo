from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import openai
import os
import tiktoken
import logging

class Embedder(ABC):
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

OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-small": {
        "max_input_length": 8191,
    },
    "text-embedding-3-large": {
        "max_input_length": 8191,
    },
    "text-embedding-ada-002": {
        "max_input_length": 8191,
    },
}

class OpenAIEmbedder(Embedder):
    def __init__(self, embedding_model: str, dimensions: int = 1024, api_key: str | None = None, **kwargs: Any):
        if embedding_model not in OPENAI_EMBEDDING_MODELS.keys():
            raise ValueError(f"Invalid OpenAI embedding model: {embedding_model}")
        
        super().__init__(dimensions=dimensions)

        self._logger = logging.getLogger(__name__)
        self._embedding_model = embedding_model
        self._dimensions = dimensions
        self._client = openai.OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
        self._encoding = tiktoken.encoding_for_model(self._embedding_model)

    def embed(self, text: str, timeout: float) -> List[float] | None:
        if not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        if len(self._encoding.encode(text)) > 8191:
            raise ValueError("Input text is too long for embedding model.")

        try:
            response = self._client.embeddings.create(
                model=self._embedding_model,
                input=text,
                dimensions=self._dimensions,
                encoding_format="float",
                timeout=timeout,
            )
            if not hasattr(response, "data") or not response.data:
                raise RuntimeError("Invalid response from embedding API.")

            return response.data[0].embedding
        except (TimeoutError, RuntimeError) as e:
            if isinstance(e, TimeoutError):
                self._logger.error("Timeout error occurred while generating embedding.")
            elif isinstance(e, RuntimeError):
                self._logger.error("Runtime error occurred while generating embedding.")
            return None
        