from paho.mqtt.client import Client, MQTTMessage, MQTTv5, ConnectFlags, DisconnectFlags
from paho.mqtt.packettypes import PacketTypes
from paho.mqtt.reasoncodes import ReasonCode
from paho.mqtt.subscribeoptions import SubscribeOptions
from paho.mqtt.enums import CallbackAPIVersion, MQTTProtocolVersion
from paho.mqtt.properties import Properties
from abc import ABC, abstractmethod
from typing import Any, List, Callable
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class MQTTSubscription:
    topic: str
    options: SubscribeOptions
    callback : Callable[[Client, Any, MQTTMessage], None]

class MQTTNode:
    """
    Abstract class for an MQTT client node. 

    :param client_id: The client ID for the MQTT node.
    :param broker_host: The host of the MQTT broker.
    :param broker_port: The port of the MQTT broker.
    :param publish_threads: The number of threads to use for servicing received messages.
    """

    _logger: logging.Logger
    _client: Client
    _service_executor: ThreadPoolExecutor

    def __init__(
            self, 
            client_id: str, 
            broker_host: str = "localhost", 
            broker_port: int = 1883, 
            publish_threads: int = 8
        ) -> None:
        """
        Initialize the MQTT node. Sets up an MQTT client, starts a Paho network loop and connects to the broker.
        Additionally, initialises the a thread pool executor for servicing received messages.
        
        :param client_id: The client ID for the MQTT node.
        :param broker_host: The host of the MQTT broker (default: "localhost").
        :param broker_port: The port of the MQTT broker (default: 1883).
        :param publish_threads: The number of threads to use for servicing received messages (default: 8).
        """
        self._logger = logging.getLogger(self.__class__.__name__)

        # Set up client
        self._client = Client(
            client_id=client_id,
            callback_api_version=CallbackAPIVersion.VERSION2,
            protocol=MQTTProtocolVersion.MQTTv5,
        )

        self._client.on_connect = self._on_connect_callback
        self._client.on_disconnect = self._on_disconnect_callback
        self._client.on_publish = self._on_publish_callback
        self._client.on_subscribe = self._on_subscribe_callback
        for subscription in self._subscriptions():
            self._client.message_callback_add(subscription.topic, subscription.callback)

        self._client.loop_start()
        self._client.connect(
            host=broker_host, 
            port=broker_port,
        )

        # Should be used for servicing received messages - no blocking calls in message callbacks!
        self._service_executor = ThreadPoolExecutor(max_workers=publish_threads)

    def __del__(self) -> None:
        self._logger.info("Disconnecting from broker and stopping loop.")
        if self._client.is_connected():
            self._client.disconnect()
        self._client.loop_stop()

    @abstractmethod
    def _subscriptions(self) -> List[MQTTSubscription]:
        """
        Provides a list of MQTTSubscriptions which are used for automatically subscribing to when initialised.

        :return: A list of MQTT subscriptions.
        """
        pass

    def _on_connect_callback(
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
        if reason_code.is_failure:
            self._logger.error(
                f"Failed to connect to broker with reason code {reason_code.getId(rc_name)} ({rc_name})"
            )
            return

        self._logger.info(
            f"Connected to broker with reason code {reason_code.getId(rc_name)} ({rc_name})"
        )
        for subscription in self._subscriptions():
            self._logger.info(f"Subscribing to topic: {subscription.topic}")
            self._client.subscribe(
                topic=subscription.topic, 
                options=subscription.options,
            )         

    @abstractmethod
    def _on_disconnect_callback(
        self,
        client: Client,
        user_data: Any,
        disconnect_flags: DisconnectFlags,
        reason_code: ReasonCode,
        properties: Properties | None,
    ) -> None:
        """
        Callback for when the client disconnects from the broker.

        :param client: The client instance for this callback
        :param userdata: The private user data as set in Client() or userdata_set()
        :param disconnect_flags: Response flags sent by the broker
        :param reason_code: The connection result
        :param properties: The properties returned by the broker
        """
        pass

    @abstractmethod
    def _on_publish_callback(
        self,
        client: Client,
        user_data: Any,
        message_id: int,
        reason_code: ReasonCode,
        properties: Properties | None,
    ) -> None:
        """
        Callback for when the client publishes a message.

        :param client: The client instance for this callback
        :param user_data: The private user data as set in Client() or userdata_set()
        :param message_id: The message ID
        :param reason_code: The connection result
        :param properties: The properties returned by the broker
        """
        pass

    @abstractmethod
    def _on_subscribe_callback(
        self, 
        client: Client, 
        user_data: Any, 
        message_id: int, 
        granted_qos: List[ReasonCode], 
        properties: Properties,
    ) -> None:
        """
        Callback for when the client subscribes to a topic.

        :param client: The client instance for this callback
        :param user_data: The private user data as set in Client() or userdata_set()
        :param message_id: The message ID
        :param reason_code: The connection result
        :param properties: The properties returned by the broker
        """
        pass

    
