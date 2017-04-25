from lib.build_graph import get_rank
import networkx as nt
import numpy

def get_subgraph(g, topic_i, cooccurrence_matrix):
    # get all nodes where rank(t_i | t_j) is <= r_max, k_i >= k_min and s_i >= s_min

    # the threshold rmax (the maximum rank) controls the relative-specificity with respect to topic_i.
    # With low value of rmax, only topics strongly related to the query topic will be included and vice versa.
    r_max = 50 # maybe?

    # lowering these increases specificity of topics that are included in the graph.
    # raising them results in large, unspecific topics being added to the taxonomy
    # lowering them increases the specificity of topics that are included (those with smaller degrees and strength)
    k_min = 50
    s_min = 3000
    num_topics = cooccurrence_matrix.shape[1]
    sub_graph = [topic_i]
    for j in range(num_topics):
        if topic_i != j:
            rank = get_rank(topic_i, j, cooccurrence_matrix)
            topic_j = cooccurrence_matrix[:, j]
            sum_i = numpy.nansum(topic_j)
            if rank <= r_max and g.degree(str(topic_i)) >= k_min and sum_i >= s_min:
                sub_graph.append(j)
    return g.subgraph(str(i) for i in sub_graph)

