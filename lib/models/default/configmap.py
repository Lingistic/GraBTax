import configparser
import logging
import os

logging.basicConfig(level=logging.DEBUG)


class Config:
    def __init__(self):
        self.config = configparser.ConfigParser()
        ini_path = os.path.dirname(os.path.realpath(__file__))
        self.config.read(os.path.join(ini_path, "default.ini"))

    # simple pattern from https://wiki.python.org/moin/ConfigParserExamples
    def map(self, section):
        dict1 = {}
        options = Config.options(section)
        for option in options:
            try:
                dict1[option] = Config.get(section, option)
                if dict1[option] == -1:
                    logging.debug("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1