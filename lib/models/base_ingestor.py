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
    def connection(self):
        pass

BaseIngestor.register(tuple)

assert issubclass(tuple, BaseIngestor)
assert isinstance((), BaseIngestor)