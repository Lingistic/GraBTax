from lib.base_ingestor import BaseIngestor
from contextlib import contextmanager


class TestIngestor(BaseIngestor):
    __connected = False

    def __init__(self):
        self.__connection = None

        pass

    def scan(self):
        if self.connection:
            yield
        else:
            self.__connect()
            self.scan()

    def __connect(self):
        pass

    def post(self, url):
        if self.connection:
            pass
        else:
            self.__connect()
            self.post(url)


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