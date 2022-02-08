import unittest
import os
import sys
import numpy as np


from point_cloud.io import read_pcd
from point_cloud.render import save_image
ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class TestRender(unittest.TestCase):
    """
    just test the render functions
    """
    def test_save_image(self):
        source = read_pcd(os.path.join(DIR_PATH, "data/L775D.ply"))
        pos = source.pos.numpy()
        norm = source.norm.numpy()  # N x 3
        z = np.asarray([0, 0, 1]).reshape(3, 1)
        bc = np.asarray([1, 0, 0]).reshape(1, 3)
        color = np.abs(norm.dot(z)**5).dot(bc)
        path_image = os.path.join(DIR_PATH, "data", "img.png")
        save_image(pos, norm, color, path_image)



if __name__ == "__main__":
    unittest.main()
