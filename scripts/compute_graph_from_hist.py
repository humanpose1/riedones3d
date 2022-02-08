import pathlib
import re
import argparse
import glob
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import open3d
import os
import os.path as osp
import pandas as pd
import joblib


def parse_args():
    parser = argparse.ArgumentParser('visualize results'
                                     '')

    parser.add_argument('--path_histogram', dest='path_histogram',
                        help='path of X matrix containing the histogram (numpy file)',
                        type=str)
    parser.add_argument('-o', dest='output',
                        help='where we store the graph',
                        default='output')
    parser.add_argument('-m', dest='model',
                        help='path of the model',
                        default='../learning/logistic.pkl')
    parser.add_argument('--path_scaler', dest='path_scaler',
                        help='path of mean and std',
                        type=str)

    args = parser.parse_args()
    return args

def save_graph(graph, path_output):
    """
    a networkx graph
    """
    data = nx.readwrite.json_graph.node_link_data(graph)
    if(not os.path.exists(path_output)):
        pathlib.Path(path_output).mkdir(parents=True, exist_ok=True)
    with open(osp.join(path_output, 'graph.json'), 'w') as f:
        json.dump(data, f)

def unpack_scaler(path_scaler):
    assert os.path.exists(path_scaler)
    with open(path_scaler, "r") as f:
        dico_scaler = json.load(f)
    mean = np.asarray(dico_scaler['mean'])
    std = np.asarray(dico_scaler['std'])
    return mean, std



def main():
    graph = nx.Graph()
    args = parse_args()
    model = joblib.load(args.model)
    dico_histogram = np.load(args.path_histogram, allow_pickle=True).item()
    mean, std = unpack_scaler(args.path_scaler)
    for key, hist in dico_histogram.items():
        list_key = key.split("_")
        assert len(list_key) <= 4
        if(len(list_key) == 2):
            name_source, name_target = list_key
        else:
            middle = (len(list_key)-1)//2
            if len(re.findall('[0-9]+', list_key[0])) == 0:
                middle += 1
            name_source = "_".join(list_key[:middle])
            name_target = "_".join(list_key[middle:])
        scaled_hist = (hist - mean) / std
        prob = model.predict_proba([scaled_hist])
        graph.add_edge(name_source, name_target, weight=prob[0][1])
        print(name_source, name_target, prob)
    save_graph(graph, args.output)

if __name__ == "__main__":
    main()
