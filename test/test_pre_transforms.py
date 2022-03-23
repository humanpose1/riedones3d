import torch
import unittest
import os
import sys

from point_cloud.pre_transforms import RotateToAxis
from point_cloud.pre_transforms import FixedScale

from torch_geometric.data import Data

ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class TestPretransforms(unittest.TestCase):

    def test_fixed_scale(self):
        pos = torch.tensor([[0,0,1], [0,1,1], [0,2,1]]).float()
        data = Data(pos=pos)
        data = FixedScale(2)(data)
        assert data.pos[0][2] == 2

    def test_rotate_to_axis(self):
        pos = torch.tensor([[0,0,1], [0,1,1], [0,2,1]]).float()
        data = Data(pos=pos)
        data = RotateToAxis()(data)


if __name__ == "__main__":
    unittest.main()
