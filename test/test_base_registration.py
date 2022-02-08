import numpy as np
from omegaconf import OmegaConf
import torch
import unittest
import os
import os.path as osp
import sys

from point_cloud.base_registration import BaseRegistrator
from point_cloud import instantiate_registrator

from torch_points3d.datasets.registration.pair import Pair
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class MockRegistrator(BaseRegistrator):
    def registrate(self, s, t):
        return torch.eye(4)


class TestBaseRegistrator(unittest.TestCase):

    def test_track(self):
        pos0 = torch.arange(0, 3*150).view(150, 3).float()
        pos1 = torch.arange(0, 3*150).view(150, 3).float()
        ind = torch.tensor([0, 1, 2, 3, 4, 5]).reshape(1, -1).long()
        data = Pair.make_pair(Data(pos=pos0), Data(pos=pos1))
        data.pair_ind = torch.cat([ind, ind]).transpose(0, 1)
        registr = MockRegistrator()
        metric = registr.evaluate_pair(data)
        self.assertAlmostEqual(metric['sr'], 0.0, places=3)

    def test_instantiate_registrator(self):
        path_yaml = osp.join(ROOT, "benchmark", "conf", "registrator", "fpfh.yaml")
        cfg = OmegaConf.load(path_yaml)
        reg = instantiate_registrator(cfg)
        pos1 = torch.randn(102, 3)
        data1 = Data(pos=pos1, norm=pos1)
        pos2 = torch.randn(149, 3)
        data2 = Data(pos=pos2, norm=pos2)
        reg.registrate(data1, data2)

    def test_instantiate_registrator_wo_icp(self):
        path_yaml = osp.join(ROOT, "benchmark", "conf", "registrator", "fpfh_wo_icp.yaml")
        cfg = OmegaConf.load(path_yaml)
        reg = instantiate_registrator(cfg)
        pos1 = torch.randn(102, 3)
        data1 = Data(pos=pos1, norm=pos1)
        pos2 = torch.randn(149, 3)
        data2 = Data(pos=pos2, norm=pos2)
        reg.registrate(data1, data2)


if __name__ == "__main__":
    unittest.main()
