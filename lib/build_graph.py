#!/usr/bin/env python
"""
processes a document topic matrix and determines the strength of a topic as a function of it's co-occurrences among
the corpus, beyond a threshold
"""

import numpy
from networkx import Graph, write_graphml, read_graphml
import logging
import pickle
import csv
import metis
import networkx
from networkx.drawing import draw_spectral
__author__ = "Rob McDaniel <robmcdan@gmail.com>"
__copyright__ = """
Copyright 2017 LiveStories

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
__credits__ = ["Rob McDaniel"]
__license__ = "ALv2"
__version__ = "0.0.1"
__maintainer__ = "Rob McDaniel"
__email__ = "robmcdan@gmail.com"
__status__ = "Development"

logging.basicConfig(level=logging.DEBUG)


def make_boolean_topic_matrix(doc_topic_matrix, threshold=0.15):
    """
    return a bool matrix for N documents X M topics where topic strength is > threshold
    :param (matrix) doc_topic_matrix: NxM document topic matrix (topics over documents)
    :param (float) threshold: minimum topic strength
    :return:
    """
    logging.info("preparing boolean topic matrix")
    m = doc_topic_matrix > threshold
    return m


def add_jaccard_weighted_edges(g, bool_topic_matrix):
    """
    given a document topic matrix, calculate the jaccard similarity score (intersection over union) between each topic
    as a weighted edge between topics
    :param (matrix) bool_topic_matrix: a boolean matrix of n documents X m topics with TRUE if the topic is represented
    :param (networkX graph) g: a graph object to populate
    :return: graph object with jaccard-weighted edges between topics
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
    :param (matrix) bool_topic_matrix: document X topic matrix with bool values where a topic exists in a doc.
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


def add_vertices(cooccurrence_matrix, g, topic_labels):
    """
    adds topic vertices and weights (based on co-occurence) -- vertex weighted by total co-occurence, edges weighted
    by co-occurence between v_i and v_j
    :param cooccurrence_matrix: topic X topic co-occurrence matrix
    :param g: graph object to populate
    :param topic_labels: list of labels to associate with each topic (in order)
    :return: graph with weighted vertices, with edges
    """
    logging.info("Adding vertices to graph")
    num_topics = cooccurrence_matrix.shape[1]
    logging.debug(num_topics)
    for i in range(num_topics):
        logging.debug(i)
        topic_i = cooccurrence_matrix[:, i]
        sum_i = numpy.nansum(topic_i)
        g.add_node(i, weight=int(sum_i), label=topic_labels[i])
        colocations = numpy.where(topic_i > 0)[0]
        for j in colocations:
            g.add_edge(i, j, count=int(numpy.nansum(cooccurrence_matrix[i,j])))
    return g


def update_edge_weights(g):
    """
    adds edge-weights to an existing graph which already contains jaccard-weighted edges. Edge weight is based on
    jaccard and rank calculations (see get_rank())
    :param g: target graph
    :return: graph with updated edge-weights
    """
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
    """
    calculates the rank score between topic i and topic j -- selects all nodes that have a higher weight than j, and
    then counts how many of them have a higher conditional probability than i. Score ranges from 1 to (N(vertices) - 2)
    Rank score of 1 means that topic_i is more predictive of topic_j than any other vertex with higher weight than
    topic_j.
    :param i: topic node
    :param j: topic node
    :param g: populated graph
    :return: returns the rank score
    """
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
    """
    gets the conditional probability of topic_i given topic_j
    :param i: topic_i
    :param j: topic_j
    :param g: the populated graph with weighted edges and vertices
    :return: 0.0 < P(i|j) < 1.0
    """
    if i == j:
        return 1.0
    topic_j_s = g.node[j]["weight"]
    try:
        count_i_given_j = g.edge[i][j]["count"]
    except KeyError: # might not be an edge connecting these vertices
        return 0.0
    if topic_j_s == 0:
        return 0.0

    return count_i_given_j / topic_j_s


def save(name, g):
    """
    saves a graph in graphml format
    :param name: friendly name of the graph
    :param g: the graph to save
    :return: None
    """
    write_graphml(g, "graphs//" + name + ".graphml")


def load(name):
    """
    loads a previously-saved graph from graphml format using its friendly name.
    :param name: the friendly name of the graph
    :return: the loaded graph
    """
    g = read_graphml("graphs//" + name + ".graphml", node_type=int)
    return g


def build_graph(theta_matrix, labels, friendly_name=None):
    """
    builds a vertex and edge-weighted graph based on a topic-proportion matrix
    :param theta_matrix: Documents X topic_proportions matrix, values should be between 0.0 and 1.0
    :param labels: list of size = N(Documents) with topic labels
    :param friendly_name: the friendly name to use to save the graph (optional)
    :return: build graph
    """
    b_matrix = make_boolean_topic_matrix(theta_matrix)
    cooccurrences = calculate_cooccurences(b_matrix)
    g = Graph()
    g = add_vertices(cooccurrences, g, labels)
    g = add_jaccard_weighted_edges(g, b_matrix)
    g = update_edge_weights(g)
    g = blacklisted_topics(g)

    # add these for METIS
    g.graph["edge_weight_attr"] = "weight"
    g.graph["node_weight_attr"] = "weight"
    if friendly_name:
        save(friendly_name, g)
    return g


def blacklisted_topics(g):
    """
    removes blacklisted topics from a graph
    :param g: graph to modify
    :return: modified graph
    """
    g.remove_node(179)
    g.remove_node(245)
    g.remove_node(106)
    g.remove_node(13)
    g.remove_node(24)
#    g.remove_node(230)
    g.remove_node(59)
    g.remove_node(183)
    g.remove_node(234)
    g.remove_node(1)
    g.remove_node(14)
    return g


def recursive_partition(g, taxonomy_out, query_topic, k=4):
    """
    Based on a query topic and a vertex and edge-weighted graph, partition the graph into a query-based topical taxonomy
    :param g: source graph
    :param taxonomy_out: output graph (can be empty)
    :param query_topic: the head vertex to generate taxonomy from
    :param k: partition size for graph bisection
    :return: taxonomy graph (taxonomy_out)
    """
    from lib.subgraph import get_subgraph

    taxonomy_out.add_node(query_topic, weight=g.node[query_topic]["weight"])
    g_sub = get_subgraph(g, query_topic)
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
            if max_node is not None:
                recursive_partition(
                    g_sub.subgraph([g_sub.nodes()[i] for i, j in enumerate(parts) if j == part]),
                    taxonomy_out, max_node)
                taxonomy_out.add_node(max_node, weight=g_sub.node[max_node]["weight"])
                taxonomy_out.add_edge(query_topic, max_node)

    return taxonomy_out, query_topic

if __name__ == "__main__":
    with open("topic_words.tsv", "r") as infile:
        reader = csv.reader(infile, delimiter="\t")
        topic_words = {int(rows[0]): rows[1] for rows in reader}
    if True:
        with open("study_theta_matrix.pkl", "rb") as f:
            foo = pickle.load(f)
        g = build_graph(foo, topic_words, "indicator_topics")
    else:
        g = load("indicator_topics")

    t = Graph()

    t = recursive_partition(g, t, 36)
    for node in t[0].nodes():
        t[0].node[node]["label"] = topic_words[node]

    t = recursive_partition(g, t[0], 90)
    for node in t[0].nodes():
        t[0].node[node]["label"] = topic_words[node]

    t = recursive_partition(g, t[0], 230)
    for node in t[0].nodes():
        t[0].node[node]["label"] = topic_words[node]

    t = recursive_partition(g, t[0], 240)
    for node in t[0].nodes():
        t[0].node[node]["label"] = topic_words[node]

    #t = recursive_partition(g, t[0], 192)
    #for node in t[0].nodes():
    #    t[0].node[node]["label"] = topic_words[node]

    t[0].add_node("indicator", label="poisoning deaths")
    t[0].add_edge("indicator", 36)
    t[0].add_edge("indicator", 90)
    t[0].add_edge("indicator", 230)
    t[0].add_edge("indicator", 240)
    #t[0].add_edge("indicator", 192)
    topic_words["indicator"] = "poisoning deaths"
    save("breast_cancer_indicators", t[0])
    labels = {}
    for i in range(len(t[0].nodes())):
        for n, label in enumerate(t[0].nodes()):
            labels[i] = topic_words[label]
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()
    networkx.draw_networkx_labels(t[0], pos=networkx.spring_layout(t[0]), labels=labels)
    plt.show()
    pass
