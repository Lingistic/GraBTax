from abc import ABC, abstractmethod


class BaseIngestor(ABC):
    @abstractmethod
    def scan(self):
        pass


BaseIngestor.register(tuple)

assert issubclass(tuple, BaseIngestor)
assert isinstance((), BaseIngestor)