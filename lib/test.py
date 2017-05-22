from lib.build_graph import build_graph, load, recursive_partition, save
from lib.test_ingestor import TestIngestor
import csv
import pickle
import networkx

if __name__ == "__main__":
    ingestor = TestIngestor()
    with ingestor.scan():
        pass

    with open("topic_words.tsv", "r") as infile:
        reader = csv.reader(infile, delimiter="\t")
        topic_words = {int(rows[0]): rows[1] for rows in reader}
    if True:
        with open("study_theta_matrix.pkl", "rb") as f:
            foo = pickle.load(f)
        g = build_graph(foo, topic_words, "indicator_topics")
    else:
        g = load("indicator_topics")

    t = networkx.Graph()

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
