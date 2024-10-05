from echo.utils.mqtt_node import MQTTNode, MQTTSubscription
from echo.utils.embedder import OpenAIEmbedder
from paho.mqtt.client import MQTTMessage, Client
from paho.mqtt.subscribeoptions import SubscribeOptions
from datetime import datetime
import lancedb
from lancedb.pydantic import LanceModel, Vector
from pathlib import Path
from xdg_base_dirs import xdg_data_home
from enum import Enum
import json
from dataclasses import dataclass
from typing import Any
import yaml
import signal
import logging

from echo.dataset import Dataset
from echo.utils.mqtt_messages import Message

DEFAULT_DATABASE_DIRECTORY = xdg_data_home() / "echo" / "database"
DEFAULT_DATABASE_NAME = "lancedb"
DEFAULT_DATASET_NAME = "dataset.json"
DEFAULT_METADATA_NAME = "metadata.yaml"


class QueryType(Enum):
    SEMANTIC = "semantic"  # vector search
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class Query(Message):
    search_text: str
    tags: list[str] | None
    limit: int
    type: QueryType

    def __init__(
        self,
        search_text: str,
        tags: list[str] | None = None,
        limit: int = 10,
        query_type: QueryType = QueryType.SEMANTIC,
        **kwargs,
    ) -> None:
        """
        Create a new query message.

        :param search_text: The text to search for.
        :param tags: The tags to search for.
        :param limit: The maximum number of results to return.
        :param query_type: The type of query to perform.
        """
        self.search_text = search_text
        self.tags = tags
        self.limit = limit
        self.query_type = query_type

    @classmethod
    def from_json(cls, json: dict) -> "Query":
        return cls(**json)

    def to_json(self) -> dict:
        return {
            "search_text": self.search_text,
            "tags": self.tags,
            "limit": self.limit,
            "type": self.query_type.value,
        }


class DatabaseTables(Enum):
    DOCUMENTATION = "documentation"
    MEMORIES = "memories"


@dataclass
class MetadataFile:
    date: str
    dimensions: int
    embedding_model: str
    documents: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "dimensions": self.dimensions,
            "embedding_model": self.embedding_model,
            "documents": self.documents,
        }


class DatabaseService(MQTTNode):
    """
    Service for managing the Echo database. Exposes a set of MQTT topics for interfacing with the database.
    """

    def __init__(
        self,
        database_directory: Path = DEFAULT_DATABASE_DIRECTORY,  # will expect a dataset.json file at the location as well
    ) -> None:
        super().__init__(client_id="database-service")

        self._db_connection = self._initialise_database(database_directory)

        self._logger.info("Database Service is online!")

    def _initialise_database(self, database_directory: Path) -> lancedb.DBConnection:
        """
        Initialise the database connection.
        """

        if (
            not (database_directory / DEFAULT_DATABASE_NAME).exists()
            or not (database_directory / DEFAULT_METADATA_NAME).exists()
        ):
            dataset_path = database_directory / DEFAULT_DATASET_NAME
            if dataset_path.exists():
                # load the dataset
                dataset: Dataset = self.load_dataset(dataset_path)

                # initialise the database
                database_connection = lancedb.connect(database_directory / DEFAULT_DATABASE_NAME)

                # specify the schema and insert data
                database_connection.create_table(
                    name=DatabaseTables.DOCUMENTATION.value,
                    data=self.create_data(dataset),
                    schema=self.documentation_schema(dataset.metadata.dimensions),
                )

                self.create_metadata_file(dataset, database_directory)
            else:
                raise FileNotFoundError(
                    f"Neither a database was found nor a dataset to initialise one at {database_directory}"
                )
        else:
            # Initialise the database
            database_connection = lancedb.connect(database_directory / DEFAULT_DATABASE_NAME)

            # Check if there is a dataset file in the same directory
            dataset_path = database_directory / DEFAULT_DATASET_NAME
            if dataset_path.exists():
                # Load the dataset
                dataset: Dataset = self.load_dataset(dataset_path)

                # Load the metadata file
                metadata_path = database_directory / DEFAULT_METADATA_NAME
                metadata = self.load_metadata(metadata_path)

                # Check if the dataset file is newer than the database file's metadata file
                if datetime.fromisoformat(dataset.metadata.date) > datetime.fromisoformat(metadata.date):
                    # Update the database with the dataset
                    database_connection.create_table(
                        name=DatabaseTables.DOCUMENTATION.value,
                        data=self.create_data(dataset),
                        schema=self.documentation_schema(dataset.metadata.dimensions),
                        mode="overwrite",
                    )

                    # Update the metadata file
                    self.create_metadata_file(dataset, database_directory)

        return database_connection

    def _subscriptions(self) -> list[MQTTSubscription]:
        return [
            MQTTSubscription(
                topic="assistant/database/documentation/query",
                options=SubscribeOptions(qos=1),
                callback=self._query_callback,
            ),
        ]

    def _query_callback(self, client: Client, userdata: Any, message: MQTTMessage) -> None:
        """
        Callback for when a query message is received.
        """
        self._logger.info(f"Received query message: {message.payload}")

    @staticmethod
    def load_dataset(dataset_path: Path) -> Dataset:
        """
        Load a dataset from a JSON file.
        """
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        try:
            with open(dataset_path, "r") as file:
                dataset_dict = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in dataset file: {e}")

        return Dataset.from_dict(dataset_dict)

    @staticmethod
    def load_metadata(metadata_path: Path) -> MetadataFile:
        """
        Load a metadata file from a YAML file.
        """
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        try:
            with open(metadata_path, "r") as file:
                metadata_dict = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in metadata file: {e}")

        return MetadataFile(**metadata_dict)

    def create_data(self, dataset: Dataset) -> list[Any]:
        """
        Create the data for the database from a dataset.

        :param dataset: The dataset to create the data from.
        :return: A list of data to insert into the database.
        """
        schema = self.documentation_schema(dataset.metadata.dimensions)

        data = []
        for document in dataset.documents:
            for summary in document.summaries:
                data.append(
                    schema(
                        id=int(summary.id),  # TODO: check this
                        tags=document.tags,
                        title=document.name,
                        vector=summary.vector,
                        text=summary.text,
                    )
                )

        return data

    def create_metadata_file(self, dataset: Dataset, dataset_directory: Path):
        """
        Create the metadata for the database from a dataset.

        :param dataset: The dataset to create the metadata from.
        :param directory: The directory containing the dataset.
        :return: A dictionary of metadata to write to a YAML file.
        """

        metadata = MetadataFile(
            date=dataset.metadata.date,
            dimensions=dataset.metadata.dimensions,
            embedding_model=dataset.metadata.embedding_model,
            documents=[{"text": doc.name, "hash": doc.hash} for doc in dataset.documents],
        )

        metadata_path = dataset_directory / DEFAULT_METADATA_NAME
        with open(metadata_path, "w") as file:
            yaml.dump(metadata.to_dict(), file)

    @staticmethod
    def documentation_schema(dimensions: int):
        class Excerpt(LanceModel):
            id: int
            tags: list[str]
            title: str
            vector: Vector(dimensions)  # type: ignore
            text: str

        return Excerpt


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    DatabaseService()

    try:
        signal.pause()
    except KeyboardInterrupt:
        pass
