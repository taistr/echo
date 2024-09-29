from paho.mqtt.client import Client, MQTTMessage, MQTTv5, ConnectFlags, DisconnectFlags
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.reasoncodes import ReasonCode
from paho.mqtt.subscribeoptions import SubscribeOptions
from paho.mqtt.enums import CallbackAPIVersion
from paho.mqtt.properties import Properties
from pymilvus import MilvusClient
import os
from pathlib import Path
import json
import logging
from typing import List, Tuple, Dict, Optional
from dataset import Record
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future, wait
from utils.embedder import OpenAIEmbedder
import threading
from xdg_base_dirs import xdg_state_home
import numpy as np

from typing import Any

class QueryCategory(Enum):
    """An enumeration of query categories."""
    DOCUMENTATION = "documentation"
    MEMORY = "memory"

class Query:
    """A query message class that parses query requests."""

    _query: str | None
    _category: QueryCategory | None
    _limit: int
    _response_topic: str | None
    _correlation_data: str | None
    result: List[str] | None

    DEFAULT_LIMIT = 5

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

        self._query = None
        self._category = None
        self._limit = self.DEFAULT_LIMIT
        self._response_topic = None
        self._correlation_data = None

        self.result = None

    @property
    def text(self) -> str | None:
        return self._query
    
    @property
    def result_limit(self) -> int:
        return self._limit

    def parse(self, message: MQTTMessage) -> None:
        """
        Parse a query message.

        :param message: The MQTT message containing the query.
        """
        try:
            payload = message.payload.decode("utf-8")
        except UnicodeDecodeError as e:
            self._logger.error(f"Failed to decode query message: {e}")
            return

        try:
            query = json.loads(payload)

            self._query = query["query"]
            self._category = QueryCategory(query["category"])
            self._limit = query.get("limit", None)
            self._response_topic = message.properties.__getattribute__("ResponseTopic")
            self._correlation_data = message.properties.__getattribute__("CorrelationData")
        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to parse query message: {e}")
            return
        except (KeyError, AttributeError) as e:
            if isinstance(e, KeyError):
                self._logger.error(f"Missing key in query message: {e}")
            else:
                self._logger.error(f"Failed to extract properties from query message: {e}")

            self._query = None
            self._category = None
            self._limit = self.DEFAULT_LIMIT
            self._response_topic = None
            self._correlation_data = None

            return
        
    def response_args(self) -> dict[str, Any]:
        """
        Create a response message.

        :return: a dictionary of kwargs: topic, payload, properties
        """
        if self.result is None:
            raise ValueError("No result to publish")
        if self._correlation_data is None or self._response_topic is None:
            raise ValueError("Correlation data or response topic missing")

        publish_properties = Properties(PacketTypes.PUBLISH)
        publish_properties.CorrelationData = self._correlation_data
        publish_properties.ResponseTopic = self._response_topic
        return {
            "topic": self._response_topic,
            "payload": self.result,
            "properties": publish_properties,
        }
            
class DatabaseService:
    """Database Service class that exposes a vector database via MQTT"""

    _logger: logging.Logger
    collection_name: str
    _mqtt_client: Client
    _database_client: MilvusClient

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        collection_name: str = "echo",
    ) -> None:

        self._logger = logging.getLogger(__name__)
        self.collection_name = collection_name

        self._executor = ThreadPoolExecutor(max_workers=8)
        self._embedder = OpenAIEmbedder(embedding_model="text-embedding-3-small", dimensions=1024)

        # Initialise database
        self._database_client = self._initialise_database(dataset_path)

        # set up mqtt
        self._mqtt_client = self._setup_client()

        self._logger.info("Database initialised")

    def run(self) -> None:
        try:
            self._mqtt_client.loop_forever()
        except KeyboardInterrupt:
            self._mqtt_client.disconnect()
            self._mqtt_client.loop_stop()

    def _initialise_database(
        self, 
        dataset_path: Path | str | None, 
        database_path: Path = xdg_state_home() / "echo" / "milvus" / "echo.db"
    ) -> MilvusClient:
        """
        Initialise the database

        :param dataset_path: Path to the dataset to populate the database with
        :param database_path: Path to the database
        :return: MilvusClient
        """
        # Initialise database and populate it with the dataset
        if unpopulated := not database_path.exists():
            database_path.parent.mkdir(parents=True, exist_ok=True)

        client = MilvusClient(str(database_path))

        self._populate_database(client, str(dataset_path))

        return client

    def _populate_database(self, client: MilvusClient, dataset_path: Path | str) -> None:
        """
        Populate the database with a dataset

        :param client: MilvusClient
        :param dataset_path: Path to the dataset
        """
        if not Path(str(dataset_path)).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        # Load the dataset
        metadata, dataset = self._load_dataset(Path(dataset_path))

        # Create the collection
        if client.has_collection(self.collection_name):
            client.drop_collection(self.collection_name)
        client.create_collection(self.collection_name, dimension=metadata.get("dimensions"))  # TODO: make safer

        # Insert the records
        data = [record.to_dict() for record in dataset]

        for dict_record in data:
            dict_record["id"] = np.int64(dict_record["id"])
            
        client.insert(collection_name=self.collection_name, data=[record.to_dict() for record in dataset])

    def _setup_client(self) -> Client:
        """
        Set up the client with the necessary callbacks and connect to the broker.
        """
        # Init client
        client = Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="database",
            protocol=MQTTv5,
        )

        # Set up callbacks
        client.on_connect = self._connect_callback
        client.message_callback_add("database/query", self._query_callback)
        # client.message_callback_add("database/insert", self._input_callback)
        # client.message_callback_add("database/delete", self._input_callback)
        # client.message_callback_add("database/update", self._input_callback)
        client.on_disconnect = self._disconnect_callback
        client.on_publish = self._publish_callback
        client.on_subscribe = self._subscribe_callback

        return client

    def _connect_callback(
        self,
        client: Client,
        user_data: Any,
        flags: ConnectFlags,
        reason_code: ReasonCode,
        properties: Properties | None,
    ) -> None:
        """
        Callback for when the client connects to the broker.

        :param client: The client instance for this callback
        :param userdata: The private user data as set in Client() or userdata_set()
        :param flags: Response flags sent by the broker
        :param reason_code: The connection result
        :param properties: The properties returned by the broker
        """
        rc_name = reason_code.getName()
        self._logger.info(
            f"Successfully connected to the broker with reason code {reason_code.getId(rc_name)} ({rc_name})"
        )

        subscriptions = [
            ("database/query", SubscribeOptions(qos=1)),
            ("database/insert", SubscribeOptions(qos=1)),
            ("database/delete", SubscribeOptions(qos=1)),
            ("database/update", SubscribeOptions(qos=1)),
        ]

        # set up subscriptions
        try:
            client.subscribe(subscriptions)
        except ValueError as e:
            raise e  # TODO: deal with this later

    def _disconnect_callback(
        self,
        client: Client,
        user_data: Any,
        disconnect_flags: DisconnectFlags,
        reason_code: ReasonCode,
        properties: Properties | None,
    ) -> None:
        """
        Callback for when the client disconnects from the broker.
        """
        if not disconnect_flags.is_disconnect_packet_from_server:
            self._logger.error("Encountered an unexpected disconnection to the broker!")
        else:
            rc_name = reason_code.getName()
            self._logger.warning(
                f"Disconnected from the broker with reason code {reason_code.getId(rc_name)} ({rc_name})"
            )

    def _publish_callback(
        self,
        client: Client,
        user_data: Any,
        message_id: int,
        reason_code: ReasonCode,
        properties: Properties | None,
    ) -> None:
        """
        Callback for when the client publishes a message.
        """
        raise NotImplementedError

    def _subscribe_callback(
        self, client: Client, user_data: Any, message_id: int, granted_qos: List[ReasonCode], properties: Properties
    ) -> None:
        """
        Callback for when the database client receives a SUBACK response from the broker.

        :param client: The client instance for this callback
        :param userdata: The private user data as set in Client() or userdata_set()
        :param mid: The message ID of the subscribe request
        :param granted_qos: The granted QoS for each subscription
        :param properties: The properties returned by the broker
        """
        self._logger.debug(f"Subscribed to a topic with QoS {granted_qos}")

    def _query_callback(self, client: Client, user_data: Any, message: MQTTMessage) -> None:
        """
        Callback for when the client receives a message on the query topic.

        :param client: The client instance for this callback
        :param userdata: The private user data as set in Client() or userdata_set()
        :param message: The message received from the broker
        """
        query = Query()
        query.parse(message)

        # spawn a new thread to service the query
        future = self._executor.submit(self._service_query, query)

        # spawn a new thread to monitor the thread
        threading.Thread(target=self._monitor_thread, args=[future]).start()

    def _monitor_thread(self, future: Future, query_timeout: int = 10) -> None:
        """
        Monitor the status of a thread servicing a query.

        :param future: The future object representing the thread.
        :param query_timeout: The maximum time to wait for the query to complete.
        """
        done, _ = wait([future], timeout=query_timeout, return_when="FIRST_COMPLETED")

        if not done:
            self._logger.warning(f"A query exceeded overall time limit: {query_timeout}")

    def _service_query(self, query: Query) -> None:
        """
        Service a database query.

        :param query: Query
        """
        # Encode the query
        if query.text is None:
            raise ValueError("Query text is missing")
        if query.result_limit is None:
            raise ValueError("Query limit is missing")
        
        encoding = self._embedder.embed(query.text, timeout=3) # TODO: what happens if this fails?

        # Query the database
        result = self._database_client.search(
            collection_name=self.collection_name,
            data=[encoding],
            limit=query.result_limit,
            output_fields=["text", "category"],
            search_params={"metric_type": "COSINE"},
        )

        # publish the result on the response topic
        result = [record["entity"]["text"] for record in result.pop()]
        query.result = result

        message_info = self._mqtt_client.publish(**query.response_args(), qos=1)

        try:
            message_info.wait_for_publish(timeout=3)
        except (RuntimeError, ValueError) as e:
            self._logger.error(f"Failed to publish query response: {e}")

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

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialise the database service
    database_service = DatabaseService(dataset_path="/home/tyson/echo/resources/example_dataset/dataset.json")

    # Run the database service
    database_service.run()
