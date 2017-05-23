from lib.models.base_ingestor import BaseIngestor
from newspaper import Article
from collections import namedtuple
from tempfile import NamedTemporaryFile
import csv
from lib.file_lock import FileLock, FileLockException
import os


class UrlIngestor(BaseIngestor):
    __resource_location = "lib/models/default/resources/"
    __article_data = namedtuple("article_data", ["url", "title", "text"])

    def __init__(self):
        self.__connection = None
        pass

    def scan(self):
        raise NotImplementedError
        """
        if self.connection:
            yield
        else:
            self.__connect()
            self.scan()
        """

    def __connect(self):
        if not os.path.exists(self.get_index_file()):
            with open(self.get_index_file(), "a"):
                os.utime(self.get_index_file(), None)

        self.connection = True

    def get_index_file(self):
        return os.path.join(self.__resource_location, "index.tsv")

    def __write_flat_file(self, data):
        with NamedTemporaryFile(mode="w", dir=self.__resource_location, delete=False) as flat_file:
            with FileLock(self.get_index_file()):
                with open(self.get_index_file(), "a") as index:
                    writer = csv.writer(index, delimiter="\t")
                    writer.writerow([flat_file.name, data.url, data.title])
            flat_file.write(data.text)

    def post(self, url):
        if self.connection:
            article = Article(url)
            article.download()
            article.parse()
            data = self.__article_data(url, article.title, article.text)
            self.__write_flat_file(data)
        else:
            try:
                self.__connect()
                self.post(url)
            except Exception as ex:
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