from lib.base_ingestor import BaseIngestor
from contextlib import contextmanager


class TestIngestor(BaseIngestor):
    __connected = False

    def __init__(self):
        self.__connection = None

        pass

    @contextmanager
    def scan(self):
        if self.__connection:
            yield
        else:
            self.__connect()

    def __connect(self):
        pass

    def connection(self):
        pass

    @property
    def connection(self):
        return self.__connection

    @connection.setter
    def connection(self, value):
        self.__connection = value

    @connection.deleter
    def connection(self):
        del self.__connection