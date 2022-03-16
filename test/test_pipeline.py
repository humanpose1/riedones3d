import unittest
import sys
import os
import os.path as osp

from point_cloud.robust_estimator import SimpleFGREstimator
from point_cloud.pipeline import PretrainedRiedonesPipeline
from point_cloud.pipeline import ParallelPipeline

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def get_torch_data():
    path_data = osp.join(DIR_PATH, "data")
    list_path = [osp.join(path_data, f) for f in ["L0090D.pt", "L0093D.pt", "L0095D.pt", "L0096D.pt"]]
    return list_path

def get_ply_data():
    path_data = osp.join(DIR_PATH, "data")
    list_path = [osp.join(path_data, f) for f in ["L0775D.pt", "L0776D.pt"]]
    return list_path



class TestPipeline(unittest.TestCase):

    def test_pretrained_riedones(self):
        list_path = get_torch_data()
        pipeline = PretrainedRiedonesPipeline(SimpleFGREstimator())
        self.assertEqual(len(pipeline.list_transfo), 0)
        pipeline.compute_all(list_path)
        self.assertEqual(len(pipeline.list_transfo), (len(list_path) * (len(list_path)-1))//2)

    def test_paralllel_pretrained_riedones(self):
        list_path = get_torch_data()
        pipeline = ParallelPipeline(pipeline=PretrainedRiedonesPipeline(SimpleFGREstimator()), n_jobs=8)
        self.assertEqual(len(pipeline.list_transfo), 0)
        pipeline.compute_all(list_path)
        self.assertEqual(len(pipeline.list_transfo), (len(list_path) * (len(list_path)-1))//2)



if __name__ == "__main__":
    unittest.main()
