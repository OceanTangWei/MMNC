import networkx as nx
import pandas as pd
import numpy as np
from tools import cal_degree_dict


def CenaExtractNodeFeature(g,layers):
    g_degree_dict = cal_degree_dict(list(g.nodes()), g, layers)
    g_nodes = [i for i in range(len(g))]
    N1 = len(g_nodes)
    feature_mat = []
    for layer in range(layers + 1):
        L_max = [np.log( np.max(g_degree_dict[layer][x]) + 1) for x in g_nodes]
        L_med= [np.log(np.median(g_degree_dict[layer][x]) + 1) for x in g_nodes]
        L_min=  [np.log( np.min(g_degree_dict[layer][x]) + 1) for x in g_nodes]
        L_75 = [np.log(np.percentile(g_degree_dict[layer][x], 75) + 1) for x in g_nodes]
        L_25 = [np.log( np.percentile(g_degree_dict[layer][x], 25) + 1) for x in g_nodes]
        feature_mat.append(L_max)
        feature_mat.append(L_min)
        feature_mat.append(L_med)
        feature_mat.append(L_75)
        feature_mat.append(L_25)
    feature_mat = np.array(feature_mat).reshape((-1,N1))
    return feature_mat.transpose()



