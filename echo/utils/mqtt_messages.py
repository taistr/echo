from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Message(ABC):
    @abstractmethod
    def from_json(cls, json: dict) -> "Message":
        """
        Create a new message from a JSON object.

        :param json: The JSON object to create the message from.
        :return: The created message.
        """

        pass

    @abstractmethod
    def to_json(self) -> dict:
        """
        Convert the message to a JSON object.

        :return: The JSON object representing the message.
        """
        pass
