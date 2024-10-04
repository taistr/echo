from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class Message(ABC):
    @abstractmethod
    def from_json(cls, json: dict) -> 'Message':
        pass

    @abstractmethod
    def to_json(self) -> dict:
        pass 

class SearchType(Enum):
    ANY = "any"
    ALL = "all"
    EXACT = "exact"

@dataclass
class QueryTag:
    type: SearchType
    tags: list[str]

@dataclass
class QueryFilter:
    tags: list[QueryTag]
    date: str

@dataclass
class DatabaseQuery(Message):
    search_text: str
    filters: QueryFilter
    limit: int

    @classmethod
    def from_json(cls, json: dict) -> 'DatabaseQuery':
        return cls(json['query'])

    def to_json(self) -> dict:
        return {'query': self.query}
    
