import torch
import unittest
import os
import sys

from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

from point_cloud.error_mesurer import PCDistance, compute_matches, pca_compute


class TestErrorMesurer(unittest.TestCase):

    def test_compute_match_1(self):
        ps = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]).float()
        pt = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 1, 1], [0, -0.5, -0.5]]).float()

        match = compute_matches(ps, pt, -1)
        match_gt = torch.tensor([[0, 3], [1, 0], [2, 1], [3, 2]]).long()
        for m in match_gt:
            assert m.tolist() in match.tolist()

    def test_compute_match_2(self):
        ps = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]).float()
        pt = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 1, 1], [0, -1, -1]]).float()

        match = compute_matches(ps, pt, 0.5)
        match_gt = torch.tensor([[1, 0], [2, 1], [3, 2]]).long()

        for m in match_gt:
            assert m.tolist() in match.tolist()
        self.assertEqual(len(match), len(match_gt))

    def test_pca_compute(self):
        ps = torch.tensor([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0.5, 0.7],
                           [0, -1, -0.2],
                           [0, 0, -1],
                           [0, -10, 0],
                           [0, 4.5, -2.7],
                           [0, 1, 6.2],
                           [0, -89, 12],
                           [0, -1, -1],
                           [0, 1, 1]]).float()
        normal = pca_compute(ps)
        torch.testing.assert_allclose(normal, torch.tensor([1, 0, 0]).float())

    def test_point_to_point_error_measurer(self):

        pos = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]).float()
        norm = torch.zeros_like(pos).float()
        norm[:, 0] = 1
        source = Data(pos=pos, norm=norm)
        pos_target = torch.tensor([[0, 0, 1e-1], [0, 1.1, 0], [0, 0, 0.985], [0, 1, 1.3]]).float()
        target = Data(pos=pos_target, norm=norm)
        distance = PCDistance(max_dist=0.2)
        
        _, _, dist = distance.compute(source, target)
        torch.testing.assert_allclose(dist, torch.tensor([0.1, 0.1, 0.015, 0.3]), atol=1e-3, rtol=1e-3)

    @unittest.skip("Need to redefine this test")
    def test_point_to_plane_error_measurer(self):

        pos = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]).float()
        norm = torch.zeros_like(pos).float()
        norm[:, 0] = 1
        source = Data(pos=pos, norm=norm)
        pos_target = torch.tensor([[1e-2, 0, 1e-1], [-1e-3, 1.05, 0], [-1e-1, 0, 0.985], [0, 1, 1.3]]).float()
        target = Data(pos=pos_target, norm=norm)
        distance = PCDistance(max_dist=0.2, plane_distance=True)

        _, _, dist = distance.compute(source, target)
        torch.testing.assert_allclose(dist, torch.tensor([1e-2, 1e-3, 0.1, 0.3]), atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    unittest.main()
