from paho.mqtt.client import Client, MQTTMessage, MQTTv5, ConnectFlags, DisconnectFlags
from paho.mqtt.reasoncodes import ReasonCode
from paho.mqtt.subscribeoptions import SubscribeOptions
from paho.mqtt.enums import CallbackAPIVersion
from pymilvus import MilvusClient
import os
from pathlib import Path
import json
import logging
from typing import List, Tuple, Dict, Optional
from dataset import Record

from typing import TYPE_CHECKING, Any

# Import Properties only when type checking
if TYPE_CHECKING:
    from paho.mqtt.properties import Properties


class DatabaseService:
    """Database Service class that exposes a vector database via MQTT"""

    def __init__(
        self,
        dataset_path: Path | str | None = None,
        collection_name: str = "echo",
    ) -> None:

        self._logger = logging.getLogger(__name__)
        self.collection_name = collection_name

        # Initialise database
        self._initialise_database(dataset_path)

        # TODO: Initialise MQTT connections
        self._client = self._setup_client()

        self._logger.info("Database initialised")

    def run(self) -> None:
        try:
            self._client.loop_forever()
        except KeyboardInterrupt:
            self._client.disconnect()
            self._client.loop_stop()

    def _initialise_database(
        self, dataset_path: Path | str | None, database_path: Path = Path("/var/lib/echo/milvus/echo.db")
    ) -> MilvusClient:
        """
        Initialise the database

        :param dataset_path: Path to the dataset to populate the database with
        :param database_path: Path to the database
        :return: MilvusClient
        """
        # Initialise database and populate it with the dataset
        if unpopulated := not database_path.exists():
            database_path.mkdir(parents=True, exist_ok=True)

        if not Path(str(dataset_path)).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

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
        """
        raise NotImplementedError

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
    database_service = DatabaseService(dataset_path="data/dataset.json")

    # Run the database service
    database_service.run()
