from lib.models.base_ingestor import BaseIngestor
from newspaper import Article
from collections import namedtuple
from tempfile import NamedTemporaryFile
import csv
from lib.file_lock import FileLock
import os
import logging
from lib.models.default.configmap import Config

config = Config()


logging.basicConfig(level=logging.DEBUG)


class RCPIngestor(BaseIngestor):
    __resource_location = config.map("Storage")['storage_dir']
    __article_data = namedtuple("article_data", ["url", "title", "text"])
    __article_data.__new__.__defaults__ = (None,) * len(__article_data._fields)

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
        def check_params():
            if data:
                if data.url:
                    if data.title:
                        if data.text:
                            return True
            return False
        if not check_params():
            raise SyntaxError("url_ingestor.__write_flat_file: check_params failed.")
        with NamedTemporaryFile(mode="w", dir=self.__resource_location, delete=False) as flat_file:
            with FileLock(self.get_index_file()):
                with open(self.get_index_file(), "a") as index:
                    writer = csv.writer(index, delimiter="\t")
                    writer.writerow([flat_file.name, data.url, data.title])
            flat_file.write(data.text)

    def post(self, url):
        data = None
        try:
            if self.connection:
                article = Article(url)
                article.download()
                article.parse()
                data = self.__article_data(url, article.title, article.text)
                self.__write_flat_file(data)
            else:
                self.__connect()
                return self.post(url)
        except Exception as ex:
            logging.error("url_ingestor.post: error occurred during post. The index may or may not have been updated: ex = {0}".format(ex))
            return

        logging.info(("url_ingestor.post: successfully posted: url = {0}, with title len = {1} and text length = {2}".format(
                      url, len(data.title), len(data.text))))

    @property
    def connection(self):
        return self.__connection

    @connection.setter
    def connection(self, value):
        self.__connection = value

    @connection.deleter
    def connection(self):
        del self.__connection
