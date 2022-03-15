import os
import sys
import torch
import numpy as np
from omegaconf import OmegaConf
import random

import unittest
from point_cloud.utils import eulerAnglesToRotationMatrix
from point_cloud.robust_estimator import instantiate_robust_estimator
from point_cloud.robust_estimator import SimpleFGREstimator
from point_cloud.robust_estimator import TeaserEstimator
from point_cloud.robust_estimator import RansacEstimator
from point_cloud.robust_estimator import build_estimator


ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def generate_data():
    # Seed
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    xyz_s = torch.randn(1000, 3)
    xyz_t = xyz_s.clone()
    xyz_t[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], :] = 100000 * torch.rand(15, 3)
    T = torch.eye(4)
    T[:3, :3] = eulerAnglesToRotationMatrix(torch.tensor([20, 10, 50])*180 / np.pi)
    T[:3, 3] = torch.randn(3)
    xyz_t = xyz_t @ T[:3, :3].T + T[:3, 3]
    return xyz_s, xyz_t, T

def test_estimator(estimator):
    xyz_s, xyz_t, T_gt = generate_data()
    T_est = estimator.estimate(xyz_s, xyz_t)
    np.testing.assert_allclose(T_est[:3, :3], T_gt[:3, :3], rtol=1e-3)


class TestRobustEstimator(unittest.TestCase):

    def test_fgr(self):
        estimator = SimpleFGREstimator()
        test_estimator(estimator)

    def test_teaser(self):
        estimator = TeaserEstimator(noise_bound=1e-5)
        test_estimator(estimator)

    def test_ransac(self):
        estimator = RansacEstimator(distance_threshold=1e-8)
        test_estimator(estimator)

    def test_simple_build_estimator(self):
        estimator = build_estimator("fgr")
        test_estimator(estimator)

    def test_complex_build_estimator(self):
        estimator = build_estimator("fgr",
                                    noise_bound=1e-5,
                                    distance_threshold=1e-8,
        )
        test_estimator(estimator)

    def test_instantiation(self):
        string = """
        class: TeaserEstimator
        noise_bound: 1e-5
        cbar2: 1
        max_clique_time_limit: 1
        """
        conf = OmegaConf.create(string)
        estimator = instantiate_robust_estimator(conf)
        test_estimator(estimator)


if __name__ == "__main__":
    unittest.main()
