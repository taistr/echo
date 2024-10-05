from abc import ABC, abstractmethod
from pathlib import Path
import google.generativeai as google_ai
import os
import logging
import typing_extensions
import json

# Models with all required features for PDF summary
GOOGLE_SUMMARY_MODELS = {
    "gemini-1.5-flash": {
        "input_token_limit": 1048576,
    },
    "gemini-1.5-pro": {
        "input_token_limit": 2097152,
    },
}


class Summariser(ABC):
    """Generic class for summarising PDF Documents."""

    @abstractmethod
    def summarise(self, pdf_file_path: Path, timeout: float) -> str:
        """
        Generate a summary for the given text.

        :param text: The text to summarise.
        :param timeout: The maximum time to wait for the summary to be generated.
        """
        pass


class GoogleAISummariser(Summariser):
    """Summarises PDF documents using the Google AI API."""

    class Summary(typing_extensions.TypedDict):
        summary: str

    def __init__(self, model: str, api_key: str | None = None) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        if model not in GOOGLE_SUMMARY_MODELS.keys():
            raise ValueError(f"Invalid Google AI summarisation model: {model}")

        try:
            google_ai.configure(api_key=api_key or os.environ["GOOGLE_API_KEY"])
        except KeyError:
            raise ValueError("No API key was provided or available as an environment variable.")

        self.model_name = model
        self._summary_model = google_ai.GenerativeModel(model_name=model)

    def summarise(self, pdf_file_path: Path, timeout: float = 120) -> list[str]:
        """
        Generate a summary for the given PDF file.

        :param pdf_file_path: The path to the PDF file to summarise.
        :param timeout: The maximum time to wait for the summary to be generated.
        :param max_attempts: The maximum number of attempts to generate the summary.
        """
        if not pdf_file_path.exists():
            raise FileNotFoundError(f"PDF file '{pdf_file_path}' not found.")

        if timeout <= 0:
            raise ValueError("Timeout must be a positive number.")

        self._logger.debug(f"Uploading PDF file '{pdf_file_path}' ...")
        pdf_reference = google_ai.upload_file(str(pdf_file_path))

        self._logger.debug(f"Generating summary for PDF file '{pdf_file_path}' ...")
        response = self._summary_model.generate_content(
            [self._summary_prompt, pdf_reference],
            generation_config=google_ai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[self.Summary],
            ),
            request_options={"timeout": timeout},
        )

        summaries = json.loads(response.text)

        google_ai.delete_file(pdf_reference)
        return [summary["summary"] for summary in summaries]

    @property
    def _summary_prompt(self) -> str:
        prompt = (
            "Summarize the following document into a series of one-sentence summaries. Each summary should:\n"
            "1. Capture a distinct key point or finding.\n"
            "2. Use proper nouns and include a clear subject.\n"
            "3. Incorporate specific details such as actions, outcomes, or entities.\n"
            "4. Avoid vague or abstract language.\n"
            "5. Use active voice and be no longer than 20 words.\n\n"
            "For example:\n\n"
            "BAD: This work investigates the extent to which LLMs used in such embodied contexts can reason over sources of feedback provided\n"
            "through natural language, without any additional training.\n\n"
            "GOOD: Large Language Models (LLMs) can be applied to domains beyond natural language processing, such as planning and interaction\n"
            "for robots.\n\n"
            "Now, please summarize the document accordingly."
        )
        return prompt
