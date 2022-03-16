import joblib
import json
import networkx as nx
import numpy as np
import os
import os.path as osp
import pathlib
import re


def unpack_scaler(path_scaler):
    assert os.path.exists(path_scaler)
    with open(path_scaler, "r") as f:
        dico_scaler = json.load(f)
    mean = np.asarray(dico_scaler['mean'])
    std = np.asarray(dico_scaler['std'])
    return mean, std


def get_node_from_name(name):
    list_name = name.split("_")
    assert len(list_name) <= 4
    if(len(list_name) == 2):
        name_source, name_target = list_name
    else:
        middle = (len(list_name)-1)//2
        if len(re.findall('[0-9]+', list_name[0])) == 0:
            middle += 1
        name_source = "_".join(list_name[:middle])
        name_target = "_".join(list_name[middle:])
    return name_source, name_target

class HistClassifier:

    def __init__(self, path_model: str, path_scaler: str):
        self.model = joblib.load(path_model)
        self.mean, self.std = unpack_scaler(args.path_scaler)
        self.init_graph()

    def init_graph(self):
        self.graph = nx.Graph()

    def compute_graph(self, dico_histogram):
        for name, hist in dico_histogram.items():
            name_source, name_target = get_node_from_name(name)
            scaled_hist = (hist - self.mean) / self.std
            prob = self.model.predict_proba([scaled_hist])
            self.graph.add_edge(name_source, name_target, weight=prob[0][1])
            print(name_source, name_target, prob)

    def save_graph(self, path_output):
    """
    a networkx graph
    """
    data = nx.readwrite.json_graph.node_link_data(self.graph)
    if(not os.path.exists(path_output)):
        pathlib.Path(path_output).mkdir(parents=True, exist_ok=True)
    with open(osp.join(path_output, 'graph.json'), 'w') as f:
        json.dump(data, f)

