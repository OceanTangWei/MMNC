import os
import numpy as np
import networkx as nx




def load_graph_alignment_datasets(dataname,INFO=True):
    if dataname=="Facebook-Twitter":
        PATH_G1 = r"./dataset/Facebook-Twitter/facebook_G1.edgelist"
        PATH_G2 = r"./dataset/Facebook-Twitter/twitter_G2.edgelist"
        PATH_ANS_DICT = r"./dataset/Facebook-Twitter/facebook-twitter.npy"
    if dataname == "ACM-DBLP":
        PATH_G1 = r"./dataset/ACM-DBLP/ACM.edgelist"
        PATH_G2 = r"./dataset/ACM-DBLP/DBLP.edgelist"
        PATH_ANS_DICT = r"./dataset/ACM-DBLP/ACM-DBLP.npy"
    if dataname == "DBLP1-DBLP2":
        PATH_G1 = r"./dataset/DBLP1-DBLP2/DBLP_G1.edgelist"
        PATH_G2 = r"./dataset/DBLP1-DBLP2/DBLP_G2.edgelist"
        PATH_ANS_DICT = r"./dataset/DBLP1-DBLP2/DBLP1-DBLP2.npy"
    if dataname == "Arxiv1-Arxiv2":
        PATH_G1 = r"./dataset/Arxiv1-Arxiv2/Arxiv_G1.edgelist"
        PATH_G2 = r"./dataset/Arxiv1-Arxiv2/Arxiv_G2.edgelist"
        PATH_ANS_DICT = r"./dataset/Arxiv1-Arxiv2/Arxiv1-Arxiv2.npy"

    G1 = nx.read_edgelist(PATH_G1, nodetype=int)
    G2 = nx.read_edgelist(PATH_G2, nodetype=int)
    ans_dict = np.load(PATH_ANS_DICT, allow_pickle=True).item()

    if INFO:
        print("dataname: {}\n G1: {} nodes, {} edges\n G2:{} nodes, {} edges\n anchor links:{}".format(
            dataname,nx.number_of_nodes(G1),nx.number_of_edges(G1),
            nx.number_of_nodes(G2),nx.number_of_edges(G2),len(ans_dict)
        ))
    return G1,G2,ans_dict
