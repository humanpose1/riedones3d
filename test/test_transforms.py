import torch
import unittest
import os
import sys
from omegaconf import OmegaConf
from point_cloud.transforms import RadiusPatchExtractor, MultiRadiusPatchExtractor
from point_cloud.transforms import instantiate_radius_patch_extractor
from torch_geometric.data import Data
from torch_points_kernels import ball_query

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class TestTransfo(unittest.TestCase):

    def test_instantiate_extractor(self):
        string = """
        - transform: MultiRadiusPatchExtractor
          params:
            radius: 2.0
            max_num: 1000
        """
        conf = OmegaConf.create(string)
        cls = instantiate_radius_patch_extractor(conf)
        pos = 5 * torch.randn(100, 3).float()
        pair_ind = pos[torch.tensor([0, 4]).long()]

        data = Data(pos=pos, pair_ind=pair_ind)
        cls(data.clone())
        assert cls.radius == 2.0


    def test_rpe(self):
        pos = torch.tensor([[-0.5, -0.5, -0.5],
                            [-0.5, -0.5, -0.5],
                            [-0.5, -0.5, -0.5],
                            [-0.5, -0.5, -0.5],
                            [-0.5, -0.5, -0.5],
                            [-0.5, -0.5, -0.5],
                            [-0.5, -0.5, -0.5],
                            [-0.5, -0.5, -0.5],
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5],
                            [8, 8, 8]])
        data = Data(pos=pos)
        # print("norm", ((pos - 0.47).norm(dim=1) < 2).sum())

        transfo = RadiusPatchExtractor(radius=2, max_num=1000)

        new_data = transfo(data.clone())
        self.assertEqual(len(new_data.pos), 16)


    def test_multiple_rpe(self):
        pos = torch.tensor(
            [[-1, -1, -1], [-0.9, -1.1, -1.05], [-1.1, -1.2, -0.99], [-0.99, -0.97, -1.1],
             [1, 1, 1], [1.2, 1, 1.05], [0.99, 1.001, 1.02], [0.99, 0.97, 0.92],
             [42, 42, 42], [42, 42, 442], [1254, 42, -42], [0, 0, 5.0]])
        pair_ind = pos[torch.tensor([0, 4])]
        data = Data(pos=pos, pair_ind=pair_ind)
        transfo = MultiRadiusPatchExtractor(radius=0.5, max_num=1000)
        new_data = transfo(data.clone())
        self.assertEqual(len(new_data.pos), 8)



if __name__ == "__main__":
    unittest.main()
