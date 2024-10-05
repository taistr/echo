from echo.utils.mqtt_node import MQTTNode, MQTTSubscription
from echo.utils.embedder import OpenAIEmbedder, Embedder
from paho.mqtt.client import MQTTMessage, Client
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.subscribeoptions import SubscribeOptions
from paho.mqtt.properties import Properties
from datetime import datetime
import lancedb
from lancedb.pydantic import LanceModel, Vector
from pathlib import Path
from xdg_base_dirs import xdg_data_home
from enum import Enum, IntEnum
import json
from dataclasses import dataclass
from typing import Any, Optional
import yaml
import signal
import logging
import http

from echo.dataset import Dataset
from echo.utils.mqtt_messages import Message

DEFAULT_DATABASE_DIRECTORY = xdg_data_home() / "echo" / "database"
DEFAULT_DATABASE_NAME = "lancedb"
DEFAULT_DATASET_NAME = "dataset.json"
DEFAULT_METADATA_NAME = "metadata.yaml"
DEFAULT_TIMEOUT = 1  # seconds


class QueryType(Enum):
    SEMANTIC = "semantic"  # vector search
    KEYWORD = "keyword"
    HYBRID = "hybrid"

class QueryTable(Enum):
    DOCUMENTATION = "documentation"
    MEMORIES = "memories"


class Query(Message):
    table: QueryTable
    search_text: str
    tags: list[str] | None
    limit: int
    type: QueryType

    def __init__(
        self,
        table: QueryTable,
        search_text: str,
        tags: list[str] | None = None,
        limit: int = 10,
        query_type: QueryType = QueryType.SEMANTIC,
        strict: bool = False,
        **kwargs,
    ) -> None:
        """
        Create a new query message.

        :param table: The table to search in (either "documentation" or "memories").
        :param search_text: The text to search for.
        :param tags: The tags to search for.
        :param limit: The maximum number of results to return.
        :param query_type: The type of query to perform.
        :param strict: Whether to search for exact matches only.
        """
        self.table = table
        self.search_text = search_text
        self.tags = tags
        self.limit = limit
        self.query_type = query_type
        self.strict = strict

    @classmethod
    def from_dict(cls, dict_object: dict) -> "Query":
        return cls(
            table=QueryTable(dict_object["table"]),
            search_text=dict_object["search_text"],
            tags=dict_object["tags"],
            limit=dict_object["limit"],
            query_type=QueryType(dict_object["type"]),
        )

    def to_dict(self) -> dict:
        return {
            "table": self.table.value,
            "search_text": self.search_text,
            "tags": self.tags,
            "limit": self.limit,
            "type": self.query_type.value,
        }

@dataclass
class QueryResponse(Message): # TODO: might be good to make response messages their own base class
    status: IntEnum
    query_table: QueryTable
    error_message: Optional[str] = None
    results: Optional[list[dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, dict_object: dict) -> "QueryResponse":
        return cls(
            status=http.HTTPStatus(dict_object["status"]),
            query_table=QueryTable(dict_object["query_table"]),
            error_message=dict_object["error_message"],
            results=dict_object["results"],
        )
    
    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "query_table": self.query_table.value,
            "error_message": self.error_message,
            "results": self.results,
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

        self._db_connection, self._embedder = self._initialise_database(database_directory)

        self._logger.info("Database Service is online!")

    def _initialise_database(self, database_directory: Path) -> tuple[lancedb.DBConnection, Embedder]:
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

                # initialise the embedder
                embedder = OpenAIEmbedder(
                    embedding_model=dataset.metadata.embedding_model,
                    dimensions=dataset.metadata.dimensions,
                )
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
                embedding_model = metadata.embedding_model
                dimensions = metadata.dimensions
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

                    embedding_model = dataset.metadata.embedding_model
                    dimensions = dataset.metadata.dimensions
                
                # Initialise the embedder
                embedder = OpenAIEmbedder(embedding_model=embedding_model, dimensions=dimensions)

        return database_connection, embedder

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
        self._logger.debug(f"Received query message: {message.payload}")

    def _serve_query(self, message: MQTTMessage) -> None:
        """
        Serve a query message.
        
        :param message: The query message to serve.
        """
        # Check for a response topic and correlation data
        try: 
            response_topic = ""
            correlation_data = ""

            response_topic = message.properties.__getattribute__("response_topic")
            correlation_data = message.properties.__getattribute__("correlation_data")
        except AttributeError:
            self._logger.error(
                f"Query message missing response topic {response_topic} or correlation data {correlation_data}"
            )
            return

        try:
            payload = message.payload.decode("utf-8")
        except UnicodeDecodeError as e:
            self._logger.error(f"Failed to decode query message payload: {e}")
            # TODO: respond with error
            return
        
        try:
            query = Query.from_dict(json.loads(payload))
        except json.JSONDecodeError as e:
            # TODO: respond with error
            self._logger.error(f"Failed to decode query message JSON: {e}")
            return
        
        # TODO: implement the SQL filter 
        self._logger.debug(f"Received query: {query}")
        match query.query_type:
            case QueryType.SEMANTIC:
                self._logger.debug(f"Querying the search text for semantic search: {query.search_text}")

                embedding = self._embedder.embed(query.search_text, timeout=DEFAULT_TIMEOUT)
                if embedding is None:
                    self._logger.error(f"Failed to generate embedding for semantic search: {query.search_text}")
                    return
                
                results = self._db_connection[query.table.value]\
                    .search(embedding)\
                    .where(self._generate_filter(query.tags, strict=query.strict))\
                    .limit(query.limit)\
                    .to_list()
            case QueryType.KEYWORD:  # TODO: respond with error
                self._logger.debug(f"Querying the search text for keyword search: {query.search_text}")
                results = self._db_connection[query.table.value]\
                    .search(query.search_text)\
                    .where(self._generate_filter(query.tags, strict=query.strict))\
                    .limit(query.limit)\
                    .to_list()
            case QueryType.HYBRID:  # TODO: respond with error
                self._logger.debug(f"Querying the search text for hybrid search: {query.search_text}")

                embedding = self._embedder.embed(query.search_text, timeout=DEFAULT_TIMEOUT)
                if embedding is None:
                    self._logger.error(f"Failed to generate embedding for hybrid search: {query.search_text}")
                    return  # TODO: respond with error
                
                results = self._db_connection[query.table.value]\
                    .search(query_type="hybrid")\
                    .vector(embedding)\
                    .text(query.search_text)\
                    .where(self._generate_filter(query.tags, strict=query.strict))\
                    .limit(query.limit)\
                    .to_list()

            case _:
                self._logger.error(f"Invalid query type: {query.query_type}")
                return  # TODO: respond with error

        # TODO: respond with results
        response = QueryResponse(
            status=http.HTTPStatus.OK,
            query_table=query.table,
            results=results,
        )

        self._logger.debug(f"Responding with query results: {response}")
        properties = Properties(PacketTypes.PUBLISH)
        setattr(properties, "response_topic", response_topic)
        setattr(properties, "correlation_data", correlation_data)
        self._client.publish(
            topic=response_topic,
            payload=json.dumps(response.to_dict()),
            properties=properties,
            qos=1,
        )


    @staticmethod
    def _generate_filter(tags: list[str] | None, strict: bool = False) -> str:
        """
        Generate a SQL filter string for a lancedb search query.

        :param tags: The tags to filter by.
        :param strict: Whether to search for exact matches only.
        :return: The SQL filter string.
        """

        if tags is None:
            return ""
        
        separator = " AND " if strict else " OR "
        statements = []
        for tag in tags:
            statements.append(f"({tag} IN tags)")
        
        return separator.join(statements)

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
