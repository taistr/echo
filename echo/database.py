from pymilvus import MilvusClient
import os
from pathlib import Path


class DatabaseService:
    """Database Service class that exposes a vector database via MQTT"""

    def __init__(self) -> None:
        # TODO: Initialise database
        # TODO: Populate the database if it did not previously exist

        # TODO: Initialise MQTT connections
        pass

    def initialise_database(self) -> MilvusClient:
        """Initialise the database"""

        database_path = Path("/var/lib/echo/milvus/echo.db")  # TODO: make this configurable

        # TODO: add a .yaml file with metadata about the database

        # Check if the database path exists, if not initialise it and populate it
        if unpopulated := not database_path.exists():
            database_path.mkdir(parents=True, exist_ok=True)

        client = MilvusClient(str(database_path))

        if unpopulated:
            # Populate the database
            pass

        return client
