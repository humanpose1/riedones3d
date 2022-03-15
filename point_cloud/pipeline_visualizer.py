from typing import Optional, List
import open3d as o3d
import numpy as np
import os
import os.path as osp

from point_cloud.visu import torch2o3d, colorizer_v2

import matplotlib.pyplot as plt
import seaborn as sns

class BaseVisualizer(object):

    def __init__(self, list_colors: List[List[float]] = [[1, 0.7, 0.1], [0.1, 0.6, 0.8]], translate: float = 20, **kwargs):
        self.list_colors = list_colors
        self.translate = translate

    def torch2o3d(self, list_data):
        list_pcd = []
        for i, data in enumerate(list_data):
            
            pcd = torch2o3d(data, color=self.list_colors[i % len(self.list_colors)])
            if getattr(data, "dist", None) is not None:
                pcd.colors = o3d.utility.Vector3dVector(data.dist.cpu().numpy())
            list_pcd.append(pcd)
        return list_pcd

    def visualize(self, list_data, name: str = "", centered: bool = False, **kwargs):
        pass

    def visualize_hist(self, hist, bin_edges , name, **kwargs):
        pass


class Open3DVisualizer(BaseVisualizer):

    def __init__(self, list_colors: List[List[float]] = [[1, 0.7, 0.1], [0.1, 0.6, 0.8]],
                 translate: float = 20,
                 **kwargs):
        BaseVisualizer.__init__(self, list_colors, translate, **kwargs)

    def visualize(self, list_data, name: str = "", centered: bool = False, **kwargs):
        print(name)
        list_pcd = self.torch2o3d(list_data)
        if centered:
            list_pcd = [pcd.translate(pcd.get_center()) for pcd in list_pcd]
            list_pcd[-1] = list_pcd[-1].translate(np.array([self.translate, 0, 0]))
        o3d.visualization.draw_geometries(list_pcd)

    def visualize_hist(self, hist, bin_edges, name, **kwargs):
        plt.bar(bin_edges[:-1], hist, width=bin_edges[-1]/len(bin_edges), color=colorizer_v2(bin_edges[:-1]))
        plt.show()



