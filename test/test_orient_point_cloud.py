import torch
import unittest
import os
import sys

from torch_geometric.data import Data

from point_cloud.utils import compute_PCA
from point_cloud.utils import vec_2_transfo

from point_cloud.pre_transforms import RotateToAxis


ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class TestOrientPointCloud(unittest.TestCase):

    def test_compute_pca_assert_shape(self):
        data_input = dict(shape_3=torch.randn(1, 100, 3), dim_4=torch.randn(100, 4))
        for name, pos in data_input.items():
            with self.subTest(msg=f"we test {name} raise an error"):
                with self.assertRaises(AssertionError):
                    compute_PCA(pos)

    def test_compute_pca(self):
        pos = torch.randn(100, 3)
        pos[:, 2] = 0
        _, eigenvectors = compute_PCA(pos)
        self.assertTrue(torch.allclose(torch.abs(eigenvectors[:, -1]), torch.tensor([0.0, 0.0, 1.0])))


    def test_vec_2_transfo(self):
        vec1 = torch.tensor([1.0, 0.0, 0.0])
        vec2 = torch.tensor([0.0, 1.0, 0.0])

        gt_transfo = torch.tensor(
            [
                [0.0 ,-1.0, 0.0 , 0.0],
                [1.0 , 0.0, 0.0 , 0.0],
                [0.0 ,0.0, 1.0 , 0.0],
                [0.0 ,0.0, 0.0 , 1.0],
        ])
        expected_transfo = vec_2_transfo(vec1, vec2)
        self.assertTrue(torch.allclose(expected_transfo, gt_transfo))

    def test_orient_point_cloud(self):
        pos = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 0.5],
                [0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5]
            ])
        data = Data(pos=pos)
        transfo = RotateToAxis(axis_3d=torch.tensor([0.0, 0.0, 1.0]))
        data = transfo(data)
        _, eigenvectors = compute_PCA(data.pos)
        self.assertTrue(torch.allclose(torch.abs(eigenvectors[:, -1]), torch.tensor([0.0, 0.0, 1.0])))

    def test_polynomial_fitting(self):
        pass

if __name__ == "__main__":
    unittest.main()
