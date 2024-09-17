import os
import argparse
from pathlib import Path
import openai
import google.generativeai as google_ai
import json
from dataclasses import dataclass
import hashlib
import logging
import typing_extensions as typing
import uuid
from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG)

class Summary(typing.TypedDict):
    summary: str

@dataclass
class Record:
    id: int
    vector: list[float]
    text: str
    category: str
    document: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "vector": self.vector,
            "text": self.text,
            "category": self.category,
            "document": self.document,
        }


class DatasetGenerator:
    """Generates a dataset for the MSM vector database."""

    def __init__(self, google_summary_model: str = "gemini-1.5-flash-latest") -> None:
        self._logger = logging.getLogger(__name__)

        self._summary_model = google_ai.GenerativeModel(
            model_name=google_summary_model,
        )
        self._client = openai.OpenAI()


    def generate(self, directory_path: Path | str) -> None:
        """
        Generate the dataset

        :param directory_path: The path to the directory containing the dataset subdirectories and documents
        """
        # Check if the path exists
        dataset_directory_path = Path(directory_path)
        if not dataset_directory_path.exists():
            raise FileNotFoundError("Could not find the dataset path.")

        # Load the current dataset
        dataset_file_path = dataset_directory_path / "dataset.json"
        try:
            old_dataset = self._load_dataset(dataset_file_path)
        except (FileNotFoundError, ValueError):
            old_dataset = []

        # Copy over records from documents that are already in the dataset
        # TODO: fix up the names here
        old_docs = {record.document for record in old_dataset}
        current_docs_map = self._map_documents(dataset_directory_path)
        current_docs = set(current_docs_map.keys())

        new_dataset = []
        common_docs = current_docs.intersection(old_docs)  # probably a more efficient way to do this
        for doc in common_docs:
            for record in old_dataset:
                if record.document == doc:
                    new_dataset.append(record)

        # Generate embeddings for new documents
        new_docs = current_docs - old_docs
        for doc in tqdm(new_docs):
            self._logger.info(f"Generating summaries for document: {doc}")
            summaries = self._generate_summaries(current_docs_map[doc])

            for summary in tqdm(summaries, leave=False):
                record = Record(
                    id=uuid.uuid4().int,
                    vector=self._generate_embedding(summary),
                    text=summary,
                    category=current_docs_map[doc].parent.name,
                    document=doc,
                )
                new_dataset.append(record)

        # Save the new dataset
        # TODO: make this safer
        with open(dataset_file_path, "w") as fh:
            json.dump([record.to_dict() for record in new_dataset], fh)

    def _generate_summaries(self, pdf_path: Path | str) -> list[str]:
        """
        Generate records for a PDF document.

        :param pdf_path: The path to the PDF document
        """
        pdf_file_path = Path(pdf_path)
        if not pdf_file_path.exists():
            raise FileNotFoundError(f"Could not find the PDF file: {pdf_path}")

        summary_pdf = google_ai.upload_file(str(pdf_file_path))
        response = self._summary_model.generate_content( # TODO: handle the timeout
            self._summary_prompt,
            generation_config=google_ai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=list[Summary],
            ),
            request_options={"timeout": 120}
        )
        self._logger.debug("Generated summaries")

        summaries = json.loads(response.text) #TODO: make this safer - entirely possible to fail atm
        
        return [summary["summary"] for summary in summaries]
    
    def _generate_embedding(self, text: str) -> list[float]:
        """
        Generate a vector for a text.

        :param text: The text to generate a vector for
        """
        # TODO: check the input

        # TODO: make this safer
        response = self._client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float",
        )

        return response.data[0].embedding

    def _map_documents(self, dataset_dir_path: Path | str) -> dict[str, Path]:
        """
        Map document file paths to their hashes.

        :param file_path: The path to the directory containing the documents
        """
        dataset_dir_path = Path(dataset_dir_path)

        document_map = {}

        for file_path in dataset_dir_path.iterdir():
            if file_path.is_dir():
                for document_path in file_path.iterdir():
                    if document_path.is_file() and document_path.suffix == ".pdf":
                        document_hash = self._file_hash(document_path)
                        document_map[document_hash] = document_path
                    else:
                        self._logger.warning(f"Encountered non-PDF file: {document_path}")

        return document_map

    @property
    def _summary_prompt(self) -> str:
        return (
            "Summarize the following document into a series of one-sentence summaries. "
            "Extract the most important information that someone may need to know—i.e., only the key points. "
            "Always use proper nouns when referring to the subject. "
            "Always include a subject in each summary. "
        )

    @staticmethod
    def _load_dataset(file_path: Path | str) -> list[Record]:
        """
        Load a dataset from a JSON file.

        :param file_path: The path to the JSON file
        """
        dataset_file_path = Path(file_path)

        if not dataset_file_path.exists():
            raise FileNotFoundError(f"Could not find the dataset file: {file_path}")

        try:
            with open(dataset_file_path) as fh:
                data = json.load(fh)
        except json.JSONDecodeError:
            raise ValueError(f"Could not decode the dataset file: {file_path}")

        return [Record(**record) for record in data]

    @staticmethod
    def _file_hash(file_path: Path | str, algorithm: str = "sha256") -> str:
        """
        Generate a hash for a PDF file.

        :param file_path: The path to the file
        :param algorithm: The hashing algorithm to use (default is 'md5')
        """

        hash_function = hashlib.new(algorithm)
        with open(file_path, "rb") as fh:
            while chunk := fh.read(8192):
                hash_function.update(chunk)

        return hash_function.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset for the Echo project")
    parser.add_argument("directory", help="The directory containing the dataset subdirectories and documents")
    args = parser.parse_args()

    generator = DatasetGenerator()
    generator.generate(args.directory)
