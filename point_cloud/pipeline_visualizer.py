from typing import Optional, List
import open3d as o3d
import numpy as np
import os
import os.path as osp
import pathlib

from point_cloud.visu import torch2o3d, colorizer_v2
from point_cloud.render import save_image

import matplotlib.pyplot as plt
import seaborn as sns

class BaseVisualizer(object):

    def __init__(self, list_colors: List[List[float]] = [[1, 0.7, 0.1], [0.1, 0.6, 0.8]], translate: float = 20, **kwargs):
        self.list_colors = list_colors
        self.translate = translate

    def torch2o3d(self, list_data, centered: bool = False):
        list_pcd = []
        for i, data in enumerate(list_data):
            pcd = torch2o3d(data, color=self.list_colors[i % len(self.list_colors)])
            if getattr(data, "dist", None) is not None:
                pcd.colors = o3d.utility.Vector3dVector(data.dist.cpu().numpy())
            list_pcd.append(pcd)
        if centered:
            list_pcd = [pcd.translate(pcd.get_center()) for pcd in list_pcd]
            list_pcd[-1] = list_pcd[-1].translate(np.array([self.translate, 0, 0]))
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
        list_pcd = self.torch2o3d(list_data, centered)
        o3d.visualization.draw_geometries(list_pcd)

    def visualize_hist(self, hist: np.ndarray, bin_edges: np.ndarray, name: str = "", **kwargs):
        plt.clf()
        plt.bar(bin_edges[:-1], hist, width=bin_edges[-1]/len(bin_edges), color=colorizer_v2(bin_edges[:-1]))
        plt.show() 


def get_np_from_pcd(list_pcd):
    pos = np.vstack([np.asarray(pcd.points) for pcd in list_pcd])
    norm = np.vstack([np.asarray(pcd.normals) for pcd in list_pcd])
    col = np.vstack([np.asarray(pcd.colors) for pcd in list_pcd]) 
    return pos, norm, col

def get_path_output(path_output: str, name: str, folder: Optional[str] = None):
    if folder is not None:
        path_img = osp.join(path_output, folder, f"{name}_{folder}.png")
    else:
        path_img = osp.join(path_output, f"{name}.png")
    pathlib.Path(osp.split(path_img)[0]).mkdir(parents=True,
                                               exist_ok=True)
    return path_img


class PyRenderVisualizer(BaseVisualizer):

    def __init__(self,
                 path_output,
                 list_colors: List[List[float]] = [[1, 0.7, 0.1], [0.1, 0.6, 0.8]],
                 translate: float = 20,
                 **kwargs):
        BaseVisualizer.__init__(self, list_colors, translate, **kwargs)
        self.path_output = path_output

    def visualize(self, list_data, name: str = "", centered: bool = False, folder: Optional[str] = None, **kwargs):
        print(folder, name)
        list_pcd = self.torch2o3d(list_data, centered)
        pos, norm, col = get_np_from_pcd(list_pcd)
        path_img = get_path_output(self.path_output, name, folder)
        save_image(pos, norm, col, path_img)

    def visualize_hist(self, hist: np.ndarray ,
                       bin_edges: np.ndarray,
                       name: str = "",
                        **kwargs):
        
        path_hist = get_path_output(self.path_output, name, "hist")
        plt.bar(bin_edges[:-1], hist, width=bin_edges[-1]/len(bin_edges), color=colorizer_v2(bin_edges[:-1]))
        plt.xlim(min(bin_edges), max(bin_edges)) 
        plt.savefig(path_hist)



