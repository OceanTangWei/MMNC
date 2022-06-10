import random
import numpy as np


from Methods.RwBasedMethods.MMNC import mmnc_run
from Methods.RwBasedMethods.MMNC.mmnc_run import run_mmnc_align,run_immnc_align


import copy
import networkx as nx

from dataset.load_alignment_data import load_graph_alignment_datasets

np.random.seed(0)
random.seed(0)
def create_align_graph(g, remove_rate, add_rate=0.0):
    np.random.seed(0)

    max_deree = max([g.degree[i] for i in g.nodes()])
    edges = list(g.edges())
    nodes = list(g.nodes())
    remove_num = int(len(edges) * remove_rate)
    add_num = int(len(edges) * add_rate)
    random.shuffle(edges)
    random.shuffle(nodes)
    max_iters = (len(edges) + len(nodes)) * 2

    new_g = copy.deepcopy(g)

    r_edges = []
    while remove_num and max_iters:
        candidate_edge = edges.pop()
        if new_g.degree[candidate_edge[0]] > 1 and new_g.degree[candidate_edge[1]] > 1:
            new_g.remove_edge(candidate_edge[0], candidate_edge[1])
            r_edges.append([candidate_edge])
            remove_num -= 1
        max_iters -= 1

    max_iters = (len(edges) + len(nodes)) * 2
    while add_num and max_iters:
        n1 = random.choice(nodes)
        n2 = random.choice(nodes)
        if n1 != n2 and n1 not in new_g.neighbors(n2):
            if new_g.degree[n1] < max_deree - 1 or new_g.degree[n2] < max_deree - 1:
                new_g.add_edge(n1, n2)
                add_num -= 1
        max_iters -= 1
    return new_g
def shuffle_graph(g,features=None,shuffle=True):

    original_nodes = list(g.nodes())
    new_nodes = copy.deepcopy(original_nodes)
    if shuffle:
        random.shuffle(new_nodes)
    original_to_new = dict(zip(original_nodes, new_nodes))
    new_graph = nx.Graph()
    new_graph.add_nodes_from(new_nodes)
    for edge in g.edges():
        new_graph.add_edge(original_to_new[edge[0]], original_to_new[edge[1]])
    if features is not None:
        new_to_original = {original_to_new[i]: i for i in range(nx.number_of_nodes(g))}
        new_order = [new_to_original[i] for i in range(nx.number_of_nodes(g))]
        features = features[new_order,:]



        return new_graph, original_to_new, features
    return new_graph, original_to_new

train_ratio = 0.04
K_de = 3
K_nei =7
T = 5
fast_select = False
if __name__ == '__main__':
    np.random.seed(0)
    for dataname in ["Facebook-Twitter","DBLP1-DBLP2","Arxiv1-Arxiv2"]:
        G1, G2, ans_dict = load_graph_alignment_datasets(dataname)
        for r_rate in [0.00,0.02,0.04,0.06,0.08]:
            print("#################dataname:{}, remove_rate:{}###################".format(dataname, r_rate))
            PATH2 = r"dataset/{}/{}_G2_{}.edgelist".format(dataname, dataname.split('-')[-1], str(r_rate))
            G2 = nx.read_edgelist(PATH2, nodetype=int)
            metrics = ["hits1"]
            if dataname=="Facebook-Twitter":
                K_de =2
            if dataname =="Arxiv1-Arxiv2":
                fast_select = True
                # we do not perform robustness experiments on Arxiv1-Arxiv for time.
                if r_rate>0.01:
                    break
            run_mmnc_align(dataname,G1,G2,ans_dict,
                           train_ratio=train_ratio,
                           K_de=K_de,
                           K_nei=K_nei,
                           r_rate=r_rate,
                           metric=metrics,
                           fast=fast_select)
            run_immnc_align(dataname,G1,G2,ans_dict,
                            train_ratio=train_ratio,
                            K_de=K_de,
                            niters=T,
                            rate=train_ratio*0.5,
                            K_nei=K_nei,
                            r_rate=r_rate,
                            metric=metrics,
                            fast=fast_select)


