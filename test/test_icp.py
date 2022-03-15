import torch
import unittest
import os
import sys
from omegaconf import OmegaConf
from point_cloud.ICP import BaseICP
from point_cloud.ICP import instantiate_icp
from point_cloud.utils import eulerAnglesToRotationMatrix
from point_cloud.io import read_pcd
from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class TestICP(unittest.TestCase):

    def test_instantiate_icp(self):
        string = """
        class: BaseICP
        mode: "plane"
        num_iter: 10
        stopping_criterion: 1e-3
        max_dist: 0.1
        pre_transform_source:
          - transform: MultiRadiusPatchExtractor
            params:
              radius: 2
              max_num: 1000
        pre_transform_target:
          - transform: MultiRadiusPatchExtractor
            params:
              radius: 4
              max_num: 1000
        """
        conf = OmegaConf.create(string)
        icp = instantiate_icp(conf)

    def test_point_icp(self):

        source = read_pcd(os.path.join(DIR_PATH, "data/bunny.ply"))
        target = read_pcd(os.path.join(DIR_PATH, "data/bunny.ply"))
        # target.pos = target.pos[:20000]
        rand_theta = torch.rand(3) * 3.14 /6
        R = eulerAnglesToRotationMatrix(rand_theta)
        T = torch.eye(4)
        T[:3, :3] = R
        target.pos = target.pos @ R.T
        icp = BaseICP(mode="point", num_iter=100, stopping_criterion=1e-3)
        Id = torch.eye(4)
        T_est = icp(source, target)
        err = torch.norm(T @ torch.inverse(T_est) - Id)
        torch.testing.assert_allclose(err.item(), torch.tensor(0.0), atol=1e-3, rtol=1e-3)

    def test_plane_icp(self):
        source = read_pcd(os.path.join(DIR_PATH, "data/L775D.ply"))
        target = read_pcd(os.path.join(DIR_PATH, "data/L775D.ply"))
        source.pos = source.pos - source.pos.mean(0)
        target.pos = target.pos - target.pos.mean(0)
        rand_theta = torch.rand(3) * 3.14 / 12
        rand_theta[0] = 0
        rand_theta[1] = 0
        R = eulerAnglesToRotationMatrix(rand_theta)
        T = torch.eye(4)
        T[:3, :3] = R
        target.pos = target.pos @ R.T
        icp = BaseICP(mode="plane", num_iter=100, stopping_criterion=1e-3, max_dist=0.1)
        Id = torch.eye(4)
        T_est = icp(source, target)

        err = torch.norm(T @ torch.inverse(T_est) - Id)
        torch.testing.assert_allclose(err.item(), torch.tensor(0.0), atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    unittest.main()
