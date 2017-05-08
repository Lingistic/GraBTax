import numpy
from networkx import Graph, write_graphml, read_graphml
import logging
import pickle
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


def jaccard_similarity(g, bool_topic_matrix):
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
                try:
                    if "count" in g.edge[i][j].keys():
                        g.add_edge(i, j, similarity=int(jaccard*100))
                except KeyError:
                    pass
    return g


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


def add_vertices(cooccurrence_matrix, g, topic_words):
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
        g.add_node(i, weight=int(sum_i), words=topic_words[i])
        colocations = numpy.where(topic_i > 0)[0]
        for j in colocations:
            g.add_edge(i, j, count=int(numpy.nansum(cooccurrence_matrix[i,j])))
    return g


def add_edge_weights(g):
    logging.info("Adding weights to edges in graph")
    num_topics = len(g)
    logging.debug(num_topics)
    for i in range(num_topics):
        logging.debug(i)
        topic_i = g.node[i]["weight"]
        colocations = [key for key in g.edge[i].keys()]
        lambda1 = 1
        lambda2 = 1
        for j in colocations:
            rank_ij = get_rank(i, j, g)
            rank_ji = get_rank(j, i, g)
            rank = 1 if rank_ij == 1 or rank_ji == 1 else 0
            count = g.edge[i][j]["count"]
            jac = g.edge[i][j]["similarity"]
            weight_ij = (1 + (lambda1 * rank) + (lambda2 * jac)) * count
            g.add_edge(i, j, weight=int(weight_ij))
    return g


def get_rank(i, j, g):
    rank_count = 0
    # first get topics with greater strength than topic j
    topic_j_s = g.node[j]["weight"]

    candidate_h = []
    num_topics = len(g)
    for h in range(num_topics):
        topic_h = g.nodes()[h]
        if j != topic_h and i != topic_h:
            topic_h_s = g.node[topic_h]["weight"]

            if topic_h_s > topic_j_s:
                candidate_h.append(topic_h)

    for h in candidate_h:
        h_given_j = get_conditional_topic_prob(h, j, g)
        i_given_j = get_conditional_topic_prob(i, j, g)
        if h_given_j > i_given_j:
            rank_count += 1

    rank = rank_count + 1
    return rank


def get_conditional_topic_prob(i, j, g):
    if i == j:
        return 1.0
    topic_j_s = g.node[j]["weight"]
    try:
        count_i_given_j = g.edge[i][j]["count"]
    except KeyError:
        return 0.0
    if topic_j_s == 0:
        return 0.0

    return count_i_given_j / topic_j_s


def save(name, g):
    write_graphml(g, "graphs//" + name + ".graphml")


def load(name):
    g = read_graphml("graphs//" + name + ".graphml", node_type=int)
    return g


if __name__ == "__main__":
    from lib.subgraph import get_subgraph
    import csv
    import metis
    import networkx

    with open("topic_words.tsv", "r") as infile:
        reader = csv.reader(infile, delimiter="\t")
        topic_words = {int(rows[0]): rows[1] for rows in reader}
    if False:
        with open("theta.pkl", "rb") as f:
            foo = pickle.load(f)
        b = make_boolean_topic_matrix(foo)
        cooccurrences = calculate_cooccurences(b)
        g = Graph()
        g = add_vertices(cooccurrences, g, topic_words)
        g = jaccard_similarity(g, b)
        g = add_edge_weights(g)
        g.graph["edge_weight_attr"] = "weight"
        g.graph["node_weight_attr"] = "weight"
        save("wiki_topics", g)
    else:
        g = load("wiki_topics")
    g.remove_node(179) #outlier
    g.remove_node(245)
    g.remove_node(47)
    g.remove_node(106)
    g.remove_node(13)
    g.remove_node(24)
    g.remove_node(230)
    g.remove_node(59)
    g.remove_node(183)
    taxonomy = Graph()

    def recursive_partition(g, taxonomy, query_topic, k = 4):
        taxonomy.add_node(query_topic, weight=g.node[query_topic]["weight"])
        g_sub = get_subgraph(g, query_topic)
        g_part = None
        if len(g_sub) > 1:
            x = metis.networkx_to_metis(g_sub)
            (edgecuts, parts) = metis.part_graph(x, k)

            for part in range(k):
                max_degree = 0
                max_node = None
                for node in [g_sub.nodes()[i] for i, j in enumerate(parts) if j == part]:
                    degree = g_sub.degree(node)
                    if degree > max_degree:
                        max_node = node
                        max_degree = degree
                if max_node != None:
                    g_part, head = recursive_partition(
                        g_sub.subgraph([g_sub.nodes()[i] for i, j in enumerate(parts) if j == part]), taxonomy, max_node)
                    taxonomy.add_node(max_node, weight=g_sub.node[max_node]["weight"])
                    taxonomy.add_edge(query_topic, max_node)

        #else:
            #g_part = g
            #for node in g_part.nodes():
            #    if node != query_topic:
            #        taxonomy.add_edge(query_topic, node)

        return taxonomy, query_topic

    taxonomy = recursive_partition(g, taxonomy, 43)
    for node in taxonomy[0].nodes():
        taxonomy[0].node[node]["label"] = topic_words[node]

    taxonomy = recursive_partition(g, taxonomy[0], 84)
    for node in taxonomy[0].nodes():
        taxonomy[0].node[node]["label"] = topic_words[node]

    save("84_taxonomy", taxonomy[0])
