import paho.mqtt.client as mqtt
import logging
import json
import threading
from openai import OpenAI

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Echo:
    def __init__(self, mqtt_hostname: str = "localhost", mqtt_port: int = 1883):
        self._logger = logging.getLogger(__name__)

        # set up the client
        self._client = self._setup_client()

        # connect to the broker
        self._client.connect(host=mqtt_hostname, port=mqtt_port)

        # setup model provider
        self.client = OpenAI()

        # for now, set up a simple chat history
        self.chat_history = []

        self._logger.info("Echo initialised")

    def run(self):
        try:
            self._client.loop_forever()
        except KeyboardInterrupt:
            self._client.disconnect()
            self._client.loop_stop()

    def _setup_client(self) -> mqtt.Client:
        """
        Set up the client with the necessary callbacks and connect to the broker.
        """
        # Init client
        client = mqtt.Client(client_id="echo", userdata=None, protocol=mqtt.MQTTv5, transport="tcp")

        # Set up callbacks
        client.on_connect = self._connect_callback
        client.on_message = self._message_callback
        client.message_callback_add("echo/input", self._input_callback)
        # client.on_disconnect = self._disconnect_callback
        # client.on_publish = self._publish_callback
        client.on_subscribe = self._subscribe_callback

        return client

    def _input_callback(self, client: mqtt.Client, userdata: any, message: mqtt.MQTTMessage) -> None:
        """
        Callback for when the client receives a message on the chat topic.

        :param client: The client instance for this callback
        :param userdata: The private user data as set in Client() or userdata_set()
        :param message: An instance of MQTTMessage. This is a class with members topic, payload, qos, retain.
        """
        self._logger.info(f"Received a message on the chat topic")
        try:
            payload = json.loads(message.payload)
        except json.JSONDecodeError as e:
            self._logger.error(f"Failed to decode the payload: {e}")
            return

        # handle message cases
        message = payload.get("message")

        # TODO: lock the chat

        # append the message to the chat history
        self.chat_history.append({"role": "user", "content": message})

        # spawn a thread and generate a response
        threading.Thread(target=self._generate_response, args=(message,)).start()

    def _generate_response(self, message: str) -> None:
        """
        Generate a response to the user's message.

        :param message: The user's message
        """
        # TODO: generalise this to work with various other models
        completion = self.client.chat.completions.create(model="gpt-4o", messages=self.chat_history)

        self.chat_history.append({"role": "system", "content": completion.choices[0].message.content})

        # publish the response to the chat topic
        self._client.publish(
            "echo/output",
            json.dumps({"message": completion.choices[0].message.content}),
        )

    def _message_callback(self, client: mqtt.Client, userdata: any, message: mqtt.MQTTMessage) -> None:
        """
        Callback for when the client receives a message from the server.

        :param client: The client instance for this callback
        :param userdata: The private user data as set in Client() or userdata_set()
        :param message: An instance of MQTTMessage. This is a class with members topic, payload, qos, retain.
        """
        self._logger.info(f"Received a message on unrecognised topic {message.topic} with payload {message.payload}")

    def _subscribe_callback(
        self,
        client: mqtt.Client,
        userdata: any,
        mid: int,
        reasonCodes: list[mqtt.ReasonCodes],
        properties: mqtt.Properties,
    ) -> None:
        """
        Callback for when the client receives a SUBACK response from the server.

        :param client: The client instance for this callback
        :param userdata: The private user data as set in Client() or userdata_set()
        :param mid: The message ID of the subscribe request
        :param granted_qos: The granted QoS for each subscription
        :param properties: The properties returned by the broker
        """
        # TODO: log the topic and QoS
        self._logger.info(f"Subscribed to a topic with QoS")

        # TODO: publish initial message to the chat state topic

    def _connect_callback(
        self,
        client: mqtt.Client,
        userdata: any,
        flags: dict[str, int],
        reason_code: mqtt.ReasonCodes,
        properties: mqtt.Properties,
    ) -> None:
        """ "
        Callback for when the client receives a CONNACK response from the server.

        :param client: The client instance for this callback
        :param userdata: The private user data as set in Client() or userdata_set()
        :param flags: Response flags sent by the broker
        :param reason_code: The connection result
        :param properties: The properties returned by the broker
        """
        self._logger.info(f"Connected to the broker with result code {reason_code}")

        # TODO: handle the case the subscription fails
        self._client.subscribe("echo/input")


if __name__ == "__main__":
    echo = Echo()
    echo.run()
