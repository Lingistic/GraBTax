import gensim
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import os
import csv
from gensim.corpora import Dictionary, MmCorpus
import logging


logging.basicConfig(level=logging.DEBUG)

def generate_matrix_market(dictionary, save=False,
                           file='/home/rob/git/GraBTax/lib/models/default/resources/corpus.mm'):
    corpus = iter_docs(dictionary)

    if save:
        MmCorpus.serialize(file, corpus)

    return corpus


def build_dictionary(trim=False, save=False,
                     file='/home/rob/git/GraBTax/lib/models/default/resources/dictionary'):

    def load():
        docs = load_docs()
        for doc in docs:
            yield process_document(doc)

    dictionary = Dictionary(load())
    # ignore words that appear in less than 10% of documents or more than 90% of documents
    if trim:
        dictionary.filter_extremes(no_below=(dictionary.num_docs * .1), no_above=0.9)  # todo find right numbers
    logging.info("dictionary completed: length={0}".format(dictionary.num_docs))
    if save:
        dictionary.save(file)
    return dictionary


def build_corpus():
    dictionary = build_dictionary(save=True)
    corpus = generate_matrix_market(dictionary, save=True)
    return corpus, dictionary


def iter_docs(dictionary):
    docs = load_docs()
    for doc in docs:
        processed = process_document(doc)
        yield dictionary.doc2bow(processed)


def load_docs():
    with open("/home/rob/git/GraBTax/lib/models/default/resources/index.tsv", "r") as index:
        reader = csv.reader(index, delimiter="\t")
        for row in reader:
            with open(row[0], "r", encoding='UTF-8') as parsed_doc:
                yield parsed_doc.readlines()


def train_lda(corpus, dictionary):
    corpus = gensim.corpora.MmCorpus('/home/rob/git/GraBTax/lib/models/default/resources/corpus.mm')
    dictionary = gensim.corpora.Dictionary.load('/home/rob/git/GraBTax/lib/models/default/resources/dictionary')
    lda = LdaModel(corpus = corpus, id2word = dictionary, num_topics = 50, update_every = 1, chunksize = 10000, passes = 1)
    cm = CoherenceModel(model=lda, corpus=corpus, coherence='u_mass')
    cm.get_coherence()
    import pyLDAvis.gensim
    topicmodel = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    html = pyLDAvis.display(topicmodel)
    import webbrowser
    f = open('viz.html', 'w')

    message = html.data

    f.write(message)
    f.close()
    webbrowser.open_new_tab('viz.html')
    print("d")


def process_document(document):
    tokens = []
    for line in document:
        if len(line) > 5:
            tokens.extend(tokenize(line.replace("\n", "").lower())) # todo replace string assignment with mutable type
    return tokens


def tokenize(text):
    """
    wraps gensim's simple_preprocess tokenizer
    :param text:
    :return: list of tokens, with stop words removed
    """
    return [token.lower() for token in simple_preprocess(text) if token not in STOPWORDS]

if __name__ == "__main__":

    #corpus, dictionary = build_corpus()
    train_lda(None, None)