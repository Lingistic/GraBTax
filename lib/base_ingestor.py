from abc import ABC, abstractmethod
from contextlib import contextmanager


class BaseIngestor(ABC):

    @abstractmethod
    @contextmanager
    def scan(self):
        pass

    @abstractmethod
    def post(self, url):
        pass

    @abstractmethod
    @property
    def connection(self):
        return  #return connection


    @abstractmethod
    @connection.setter
    def connection(self, value):
        pass

    @abstractmethod
    @connection.deleter
    def connection(self):
        # del connection
        pass

BaseIngestor.register(tuple)

assert issubclass(tuple, BaseIngestor)
assert isinstance((), BaseIngestor)