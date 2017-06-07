import gensim
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import csv
from gensim.corpora import Dictionary, MmCorpus
import logging
from lib.models.reuters.configmap import Config
import os
config = Config()

logging.basicConfig(level=logging.DEBUG)

with open(os.path.join(config.map("Storage")['stopwords']), "r") as stopwords:
    reuters_stopwords = stopwords.read()


def generate_matrix_market(dictionary, save=False,
                           file=os.path.join(config.map("Storage")['storage_dir'] + 'corpus.mm')):
    corpus = iter_docs(dictionary)

    if save:
        MmCorpus.serialize(file, corpus)

    return corpus


def build_dictionary(trim=False, save=False,
                     file=os.path.join(config.map("Storage")['storage_dir'] + 'dictionary')):

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
    for dirpath, dnames, fnames in os.walk("/home/rob/nltk_data/corpora/reuters/training/"):
        for f in fnames:
            with open(os.path.join(dirpath, f), "r") as training_file:
                yield training_file.readlines()


def train_lda(corpus, dictionary, save=False, file=os.path.join(config.map("Storage")['storage_dir'] + 'lda.mdl')):
    lda = LdaModel(corpus = corpus, id2word = dictionary, num_topics = 50, update_every = 1, chunksize = 10000, passes = 10)
    if save:
        lda.save(file)

    #cm = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    #print(cm.get_coherence())
    import pyLDAvis.gensim
    topicmodel = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    html = pyLDAvis.display(topicmodel)
    import webbrowser
    with open('viz.html', 'w') as f:

        message = html.data

        f.write(message)

    webbrowser.open_new_tab('viz.html')


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
    return [token.lower() for token in simple_preprocess(text) if token not in STOPWORDS and token not in reuters_stopwords]

if __name__ == "__main__":

    #corp, corp_dict = build_corpus()
    corp = gensim.corpora.MmCorpus('/home/rob/git/Lingistic/GraBTax/lib/models/reuters/resources/corpus.mm')
    corp_dict = gensim.corpora.Dictionary.load('/home/rob/git/Lingistic/GraBTax/lib/models/reuters/resources/dictionary')
    lda = LdaModel.load('/home/rob/git/Lingistic/GraBTax/lib/models/reuters/resources/lda.mdl')
    import pyLDAvis.gensim

    topicmodel = pyLDAvis.gensim.prepare(lda, corp, corp_dict)
    html = pyLDAvis.display(topicmodel)
    import webbrowser

    with open('viz.html', 'w') as f:
        message = html.data

        f.write(message)

    webbrowser.open_new_tab('viz.html')
    train_lda(corp, corp_dict)