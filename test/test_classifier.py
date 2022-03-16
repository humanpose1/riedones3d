import unittest
import os
import os.path as osp
import sys
import numpy as np

from point_cloud.classifier import HistClassifier

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_clf():
    path_model = osp.join(ROOT, "classifiers", "logistic_part_droits_sym.pkl")
    path_scaler = osp.join(ROOT, "classifiers", "mean_std.json")
    clf = HistClassifier(path_model, path_scaler)
    return clf

def get_histogram():
    dico = {"L0001D_L0002D": np.random.rand(70),
    "L0457D_L0974D": np.random.rand(70)}
    return dico


class TestClassifier(unittest.TestCase):

    def test_compute_graph(self):
        clf = get_clf()
        dico_histogram = get_histogram()
        clf.compute_graph(dico_histogram)


if __name__ == "__main__":
    unittest.main()
