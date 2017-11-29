import gensim
from gensim.models import LdaModel, CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import csv
from gensim.corpora import Dictionary, MmCorpus
import logging
from lib.models.default.configmap import Config
import os
import pickle

config = Config()

logging.basicConfig(level=logging.DEBUG)


def generate_matrix_market(dictionary,
                           file=os.path.join(config.map("Storage")['storage_dir'], 'corpus.mm')):

    MmCorpus.serialize(file, bag_docs(dictionary))
    return gensim.corpora.MmCorpus(file)


def build_dictionary(trim=False, save=False,
                     file=os.path.join(config.map("Storage")['storage_dir'], 'dictionary')):

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
    corpus = generate_matrix_market(dictionary)
    return corpus, dictionary


def bag_docs(dictionary):
    docs = load_docs()
    for doc in docs:
        processed = process_document(doc)
        yield dictionary.doc2bow(processed)


def load_docs():
    with open(os.path.join(config.map("Storage")['storage_dir'], "index.tsv"), "r") as index:
        reader = csv.reader(index, delimiter="\t")
        for row in reader:
            with open(row[0], "r", encoding='UTF-8') as parsed_doc:
                yield parsed_doc.readlines()


def write_topic_words(lda):
    with open(os.path.join(config.map("Storage")['storage_dir'], "topic_words.tsv"), "w") as outfile:
        writer = csv.writer(outfile, delimiter="\t")
        for topic in lda.show_topics(num_topics=lda.num_topics, formatted=False):
            writer.writerow([tup[0] for tup in topic[1]])


def write_theta_matrix(lda):
    with open(os.path.join(config.map("Storage")['storage_dir'], "theta.pkl"), "wb") as outfile:
        theta, _ = lda.inference(bag_docs(lda.id2word))
        theta /= theta.sum(axis=1)[:, None]
        pickle.dump(theta, outfile)

def train_lda(corpus, dictionary, save=False, file=os.path.join(config.map("Storage")['storage_dir'], 'lda.mdl')):
    lda = LdaModel(corpus = corpus, id2word = dictionary, num_topics = 5, update_every = 1, chunksize = 10000, passes = 10)
    if save:
        lda.save(file)

    cm = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    print(cm.get_coherence())
    import pyLDAvis.gensim
    topicmodel = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    html = pyLDAvis.display(topicmodel)
    import webbrowser
    with open(os.path.join(config.map("Storage")['storage_dir'], 'viz.html'), 'w') as f:
        message = html.data
        f.write(message)

    webbrowser.open_new_tab('viz.html')
    return lda

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

    corp, corp_dict = build_corpus()
    #corp = gensim.corpora.MmCorpus('/home/rob/git/GraBTax/lib/models/default/resources/corpus.mm')
    #corp_dict = gensim.corpora.Dictionary.load('/home/rob/git/GraBTax/lib/models/default/resources/dictionary')
    train_lda(corp, corp_dict)