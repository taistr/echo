from enum import Enum
from typing import Any
from .utils.summariser import GoogleAISummariser
from .utils.embedder import OpenAIEmbedder
from pathlib import Path
import logging
import hashlib
from typing import Callable
from dataclasses import dataclass
import numpy as np
import uuid
from datetime import datetime, timezone
import argparse
import json
import logging

@dataclass
class Summary:
    id: np.int64
    text: str
    vector: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "text": self.text,
            "vector": self.vector,
        }

@dataclass
class Document:
    name: str
    tags: list[str]
    hash: str
    summaries: list[Summary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tags": self.tags,
            "hash": self.hash,
            "summaries": [summary.to_dict() for summary in self.summaries],
        }

@dataclass
class Metadata:
    date: str
    dimensions: int
    embedding_model: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "dimensions": self.dimensions,
            "embedding_model": self.embedding_model,
        }

@dataclass
class Dataset:
    metadata: Metadata
    documents: list[Document]

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataset to a dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "documents": [doc.to_dict() for doc in self.documents],
        }

class DatasetGenerator:
    """Generates a dataset from a directory of documentation for the MSM's agents to access during operation."""

    def __init__(self):
        self._summariser = GoogleAISummariser(
            model="gemini-1.5-flash",
        )

        self._embedder = OpenAIEmbedder(
            embedding_model="text-embedding-3-small",
        )

        self._logger = logging.getLogger(self.__class__.__name__)

    def generate(self, directory_path: Path) -> dict:
        """
        Generate a dataset from the given directory.

        :param directory_path: The path to the directory containing the documentation.
        :return: A dataset of documentation records.
        """
        if not isinstance(directory_path, Path):
            self._logger.warning(f"Converting directory path to Path: {directory_path}")
            directory_path = Path(directory_path)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory path '{directory_path}' not found.")
        
        # Get a list of all PDF files in the directory
        pdf_files = list(directory_path.rglob("*.pdf"))
        
        documents = []
        for pdf_path in pdf_files:
            documents.append(
                Document(
                    name=pdf_path.name,
                    tags=self.get_directories(pdf_path, directory_path),
                    hash=self.hash_file(pdf_path, hashlib.sha256),
                    summaries=self.generate_summaries(pdf_path),
                )
            )

        # Generate metadata for the dataset
        metadata = Metadata(
            date=datetime.now(timezone.utc).isoformat(),
            dimensions=self._embedder.dimensions,
            embedding_model=self._embedder.embedding_model,
        )

        dataset = Dataset( 
            metadata=metadata,
            documents=documents,
        )

        return dataset.to_dict()

    def generate_summaries(self, pdf_path: Path) -> list[Summary]:
        """
        Generate summaries for the given PDF file.
        
        :param pdf_path: The path to the PDF file.
        :return: A list of summary objects.
        """
        if not isinstance(pdf_path, Path):
            raise TypeError(f"Invalid PDF path: {pdf_path}")

        summaries = self._summariser.summarise(pdf_path)

        summary_objects = []

        for summary in summaries:
            summary = Summary(
                id=self.generate_id(),
                text=summary,
                vector=self._embedder.embed(summary),
            )
            summary_objects.append(summary)

        return summary_objects

    @staticmethod
    def generate_id() -> np.int64:
        """Generate a unique ID for a document."""
        uuid_int = (uuid.uuid4().int >> 64) & 0xFFFFFFFFFFFFFFFF

        if uuid_int >= 0x8000000000000000:
            uuid_int -= 0x10000000000000000

        return np.int64(uuid_int)
        

    @staticmethod
    def hash_file(file_path: Path, hash_fn: Callable) -> str:
        """Hashes the contents of a file."""
        if not isinstance(file_path, Path):
            raise TypeError(f"Invalid file path: {file_path}")

        with open(file_path, "rb") as file:
            file_hash = hashlib.file_digest(file, hash_fn).hexdigest()

        return file_hash
    
    @staticmethod
    def get_directories(file_path: Path, base_path: Path) -> list[str]:
        """Get a list of all directories in the given directory."""
        if not isinstance(file_path, Path):
            raise TypeError(f"Invalid file path: {file_path}")

        if not isinstance(base_path, Path):
            raise TypeError(f"Invalid base path: {base_path}")

        relative_path = file_path.relative_to(base_path)
        
        return list(relative_path.parent.parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset from a directory of documentation.")
    parser.add_argument("directory", help="The directory containing the documentation.")
    parser.add_argument("--output", help="The output file for the dataset.", required=True)
    args = parser.parse_args()

    dataset = DatasetGenerator().generate(Path(args.directory))

    if args.output:
        with open(args.output, "w") as file:
            json.dump(dataset, file, indent=4)

    print(f"Dataset generated and saved to {args.output}")


    



