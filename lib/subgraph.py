from lib.build_graph import get_rank
import networkx as nt
import numpy


def get_subgraph(g, topic_0):
    # get all nodes where rank(t_o | t_i) is <= r_max, k_i >= k_min and s_i >= s_min

    # the threshold rmax (the maximum rank) controls the relative-specificity with respect to topic_0.
    # With low value of rmax, only topics strongly related to the query topic will be included and vice versa.
    r_max = 3 # maybe?

    # lowering these increases specificity of topics that are included in the graph.
    # raising them results in broader topics being added to the taxonomy
    k_min = g.degree(topic_0) / 4
    s_min = g.node[topic_0]["weight"] / 2 #300

    num_topics = len(g)
    sub_graph = []
    for i in range(num_topics):
        if topic_0 != i:
            try:
                rank = get_rank(topic_0, i, g)
                topic_i_s = g.node[i]["weight"]
                if rank <= r_max and g.degree(i) >= k_min and topic_i_s >= s_min:
                    sub_graph.append(i)
            except KeyError:
                pass
    return g.subgraph(i for i in sub_graph).copy()

