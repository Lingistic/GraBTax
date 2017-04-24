import numpy
from networkx import Graph, write_graphml
import logging
logging.basicConfig(level=logging.DEBUG)

"""
processes a document topic matrix and determines the strength of a topic as a function of it's co-occurrences among
the corpus, beyond a threshold
"""


def make_boolean_topic_matrix(doc_topic_matrix, threshold=0.05):
    """
    return a bool matrix for N documents X M topics where topic strength is > threshold
    :param doc_topic_matrix: NxM document topic matrix (topics over documents)
    :param threshold: minimum topic strength
    :return:
    """
    logging.info("preparing boolean topic matrix")
    m = doc_topic_matrix > threshold
    return m


def jaccard_similarity(bool_topic_matrix):
    """
    given a document topic matrix, calculate the jaccard similarity score (intersection over union) between each topic
    :param bool_topic_matrix: a boolean matrix of n documents X m topics with TRUE if the topic is represented
    :return: dictionary of jaccard indexes between i and j values
    """
    logging.info("calculating jaccard indexes for all topics")
    num_topics = bool_topic_matrix.shape[1]
    jaccard_matrix = numpy.zeros((num_topics, num_topics))
    logging.debug(num_topics)
    for i in range(num_topics):
        logging.debug(i)
        topic_i = bool_topic_matrix[:, i]
        jaccard_matrix[i, i] = 1.0
        for j in range(i + 1, num_topics):
                topic_j = bool_topic_matrix[:, j]
                intersection = numpy.logical_and(topic_i, topic_j)
                union = numpy.logical_or(topic_i, topic_j)
                jaccard = intersection.sum() / float(union.sum())
                jaccard_matrix[i, j] = jaccard
                jaccard_matrix[j, i] = jaccard
    return jaccard_matrix


def calculate_cooccurences(bool_topic_matrix):
    """
    given a boolean topic matrix (n observations X m topics where TRUE exists when a topic exists in a doc), count the
    total number of document co-occurrences between topic_i and topic_j
    :return: topic_i X topic_j co-occurrence matrix with co-occurrence counts between topics i and j
    """
    logging.info("calculating co-occurrences")
    num_topics = bool_topic_matrix.shape[1]
    cooccurrence_matrix = numpy.zeros((num_topics, num_topics))
    logging.debug(num_topics)
    for i in range(num_topics):
        logging.debug(i)
        topic_i = bool_topic_matrix[:, i]
        cooccurrence_matrix[i, i] = numpy.nan
        for j in range(i + 1, num_topics):
            topic_j = bool_topic_matrix[:, j]
            count_ij = bool_topic_matrix[numpy.where(topic_i & topic_j)].shape[0]
            cooccurrence_matrix[i, j] = count_ij
            cooccurrence_matrix[j, i] = count_ij
    return cooccurrence_matrix


def add_vertices(cooccurrence_matrix, g):
    """

    :param cooccurrence_matrix:
    :param g:
    :return:
    """
    logging.info("Adding vertices to graph")
    num_topics = cooccurrence_matrix.shape[1]
    logging.debug(num_topics)
    for i in range(num_topics):
        logging.debug(i)
        topic_i = cooccurrence_matrix[:, i]
        sum_i = numpy.nansum(topic_i)
        g.add_node(i, weight=int(sum_i))
    return g


def add_weights(sims_matrix, cooccurrence_matrix, g):
    logging.info("Adding weights to edges in graph")
    num_topics = cooccurrence_matrix.shape[1]
    logging.debug(num_topics)
    for i in range(num_topics):
        logging.debug(i)
        topic_i = cooccurrence_matrix[:, i]
        colocations = numpy.where(topic_i > 0)[0]
        lambda1 = 1
        lambda2 = 1
        for j in colocations:
            rank = 1 if get_rank(i, j, cooccurrence_matrix) == 1 or get_rank(j, i, cooccurrence_matrix) == 1 else 0
            count = cooccurrence_matrix[i, j]
            jac = sims_matrix[i, j]
            weight_ij = (1 + (lambda1 * rank) + (lambda2 * jac)) * count
            g.add_edge(i, j, weight=float(weight_ij))
    return g


def get_rank(i, j, cooccurrence_matrix):
    rank_count = 0
    # first get topics with greater strength than topic j
    topic_j = cooccurrence_matrix[:, j]
    sum_j = numpy.nansum(topic_j)
    candidate_h = []
    num_topics = cooccurrence_matrix.shape[1]
    for h in range(num_topics):
        if j != h and i != h:
            topic_h = cooccurrence_matrix[:, h]
            sum_h = numpy.nansum(topic_h)
            if sum_h > sum_j:
                candidate_h.append(h)

    for h in candidate_h:
        h_given_j = get_conditional_topic_prob(h, j, cooccurrence_matrix)
        i_given_j = get_conditional_topic_prob(i, j, cooccurrence_matrix)
        if h_given_j > i_given_j:
            rank_count += 1
            break

    rank = rank_count + 1
    return rank


def get_conditional_topic_prob(i, j, cooccurrence_matrix):
    topic_j = cooccurrence_matrix[:, j]
    sum_j = numpy.nansum(topic_j)
    count_i_given_j = numpy.nansum(cooccurrence_matrix[i,j])
    if sum_j == 0:
        return 0.0

    return count_i_given_j / sum_j


if __name__ == "__main__":
    import pickle
    with open("theta.pkl", "rb") as f:
        foo = pickle.load(f)
    b = make_boolean_topic_matrix(foo)
    sims = jaccard_similarity(b)
    cooccurrences = calculate_cooccurences(b)
    g = Graph()
    g = add_vertices(cooccurrences, g)
    g = add_weights(sims, cooccurrences, g)
    write_graphml(g, "wiki_topics.gml")


