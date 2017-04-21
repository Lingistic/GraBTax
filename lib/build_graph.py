import numpy
from networkx import Graph
from collections import Counter

"""
processes a document topic matrix and determines the strength of a topic as a function of it's co-occurrences among
the corpus, beyond a threshold
"""

def jaccard_similarity


def add_vertices(doc_topic_matrix, g, threshold=.05):
    """
    given a topic matrix (n observations X m topics), create an edge between topics which co-occur beyond
    a threshold, and weight each vertex with the sum of its co-occurrence with other topics.
    :return: networkx graph
    """
    cooccurrences = {}
    num_topics = doc_topic_matrix.shape[1]
    for i in range(num_topics):
        sum_i = 0
        for j in range(num_topics):
            if i != j:
                count_ij = doc_topic_matrix[numpy.where((doc_topic_matrix[:, i] > threshold)
                                                          & (doc_topic_matrix[:, j] > threshold))].shape[0]
                if count_ij > 0:
                    g.add_edge(i, j)
                    sum_i += count_ij
        cooccurrences[i] = sum_i
        if i in g.edge.keys():
            g.node[i]["weight"] = sum_i
    return g

if __name__ == "__main__":
    import pickle
    with open("theta.pkl", "rb") as f:
        foo = pickle.load(f)
    graph = add_vertices(foo)
    print(graph)


