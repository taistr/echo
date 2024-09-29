import os
import argparse
import json
import logging
import hashlib
import uuid
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple, Dict
import openai
import tiktoken
import google.generativeai as google_ai
from tqdm import tqdm
import typing_extensions as typing
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)


# Type hint for summary dict structure
@dataclass
class Metadata:
    """Represents metadata for the dataset."""

    date: str
    dimensions: int
    embedding_model: str

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "dimensions": self.dimensions,
            "embedding_model": self.embedding_model,
        }


@dataclass
class Record:
    """Represents a single record in the dataset."""

    id: int
    vector: List[float]
    text: str
    category: str
    document: str

    def to_dict(self) -> dict:
        """
        Convert the record to a dictionary.

        :return: A dictionary representation of the record - note that id is a 64-bit integer.
        """
        return {
            "id": self.id,
            "vector": self.vector,
            "text": self.text,
            "category": self.category,
            "document": self.document,
        }


class Summary(typing.TypedDict):
    summary: str


class DatasetGenerator:
    """Generates a dataset for the MSM vector database."""

    def __init__(
        self,
        google_summary_model: str = "gemini-1.5-flash-latest",
        openai_embedding_model: str = "text-embedding-3-small",
        dimensions: int = 1024,
    ) -> None:
        self._logger = logging.getLogger(__name__)

        # Configure APIs
        google_ai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self._summary_model = google_ai.GenerativeModel(model_name=google_summary_model)

        self._embedding_model = openai_embedding_model
        self._embedding_dimensions = dimensions
        self._encoding = tiktoken.encoding_for_model(self._embedding_model)
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, directory_path: Path | str) -> None:
        """
        Generate the dataset from a directory of documents.

        :param directory_path: The path to the directory containing subdirectories and documents.
        """
        dataset_directory_path = Path(directory_path)
        if not dataset_directory_path.exists():
            raise FileNotFoundError(f"Dataset path '{directory_path}' not found.")

        # Load the existing dataset
        dataset_file_path = dataset_directory_path / "dataset.json"
        old_metadata, old_dataset = self._load_dataset(dataset_file_path)

        old_docs = {record.document for record in old_dataset}
        current_docs_map = self._map_documents(dataset_directory_path)
        current_docs = set(current_docs_map.keys())

        new_dataset: List[Record] = []
        if (
            old_metadata
            and old_metadata["dimensions"] == self._embedding_dimensions
            and old_metadata["embedding_model"] == self._embedding_model
        ):
            common_docs = current_docs.intersection(old_docs)
            new_dataset.extend(record for record in old_dataset if record.document in common_docs)
            new_docs = current_docs - old_docs
        else:
            new_docs = current_docs

        # Generate embeddings for new documents
        for doc in tqdm(new_docs):
            self._logger.info(f"Generating summaries for document: %", doc)
            summaries = self._generate_summaries(current_docs_map[doc])

            for summary in tqdm(summaries, leave=False):
                record = Record(
                    id=uuid.uuid4().int >> 64, # 64-bit integer
                    vector=self._generate_embedding(summary, self._embedding_dimensions),
                    text=summary,
                    category=current_docs_map[doc].parent.name,
                    document=doc,
                )
                new_dataset.append(record)

        # Add metadata and save dataset
        metadata = Metadata(
            date=datetime.now(timezone.utc).isoformat(),
            dimensions=self._embedding_dimensions,
            embedding_model=self._embedding_model,
        )
        dataset = {
            "metadata": metadata.to_dict(),
            "data": [record.to_dict() for record in new_dataset],
        }

        with open(dataset_file_path, "w") as fh:
            json.dump(dataset, fh, indent=4)

    def _generate_summaries(self, pdf_path: Path) -> List[str]:
        """
        Generate summaries from a PDF document.

        :param pdf_path: The path to the PDF document.
        :return: A list of summaries as strings.
        """
        pdf_file_path = Path(pdf_path)
        if not pdf_file_path.exists():
            raise FileNotFoundError(f"PDF file '{pdf_path}' not found.")

        self._logger.debug(f"Uploading PDF file: {pdf_file_path}")
        summary_pdf = google_ai.upload_file(str(pdf_file_path))

        self._logger.debug("Generating summaries for the PDF file...")
        response = self._summary_model.generate_content(
            [self._summary_prompt, summary_pdf],
            generation_config=google_ai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[Summary],
            ),
            request_options={"timeout": 120},
        )

        google_ai.delete_file(summary_pdf)

        summaries = json.loads(response.text)
        return [summary["summary"] for summary in summaries]

    def _generate_embedding(self, text: str, dimensions: int) -> List[float]:
        """
        Generate an embedding vector for a text string.

        :param text: The text to generate the embedding for.
        :param dimensions: The embedding dimension size.
        :return: A list of floats representing the embedding vector.
        """
        if not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        if len(self._encoding.encode(text)) > 8191:
            raise ValueError("Input text is too long for embedding model.")

        response = self._client.embeddings.create(
            model=self._embedding_model,
            input=text,
            dimensions=dimensions,
            encoding_format="float",
        )
        if not hasattr(response, "data") or not response.data:
            raise RuntimeError("Invalid response from embedding API.")

        return response.data[0].embedding

    def _map_documents(self, dataset_dir_path: Path) -> Dict[str, Path]:
        """
        Map document file paths to their SHA256 hashes.

        :param dataset_dir_path: The directory path containing the documents.
        :return: A dictionary mapping file hashes to file paths.
        """
        document_map = {}
        for file_path in dataset_dir_path.rglob("*.pdf"):
            document_hash = self._file_hash(file_path)
            document_map[document_hash] = file_path

        return document_map

    @property
    def _summary_prompt(self) -> str:
        return (
            "Summarize the following document into a series of one-sentence summaries. "
            "Extract the key points. Always use proper nouns and include a subject in each summary."
        )

    @staticmethod
    def _load_dataset(file_path: Path) -> Tuple[dict, List[Record]]:
        """
        Load a dataset from a JSON file.

        :param file_path: The path to the dataset JSON file.
        :return: A tuple containing the metadata and dataset records.
        """
        if not file_path.exists():
            return {}, []

        with open(file_path) as fh:
            json_data = json.load(fh)

        metadata = json_data.get("metadata", {})
        data = json_data.get("data", [])
        records = [Record(**record) for record in data]

        return metadata, records

    @staticmethod
    def _file_hash(file_path: Path, algorithm: str = "sha256") -> str:
        """
        Generate a hash for a file using the specified algorithm.

        :param file_path: The path to the file.
        :param algorithm: The hashing algorithm to use.
        :return: The file's hash as a string.
        """
        hash_function = hashlib.new(algorithm)
        with open(file_path, "rb") as fh:
            while chunk := fh.read(8192):
                hash_function.update(chunk)

        return hash_function.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset for the MSM project.")
    parser.add_argument("directory", help="The directory containing the dataset subdirectories and documents")
    args = parser.parse_args()

    generator = DatasetGenerator()
    generator.generate(args.directory)
