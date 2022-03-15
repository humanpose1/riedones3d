from typing import Callable, Optional
import numpy as np
import open3d
import torch
from torch_points_kernels import ball_query, knn
from point_cloud.visu import torch2o3d, colorizer, colorizer_v2
from torch_geometric.data import Data

from torch_points3d.core.data_transform import GridSampling3D

def compute_matches(ps, pt, max_dist):
    idx, dist = knn(pt.unsqueeze(0), ps.unsqueeze(0), k=1)
    rang = torch.arange(0, len(ps)).unsqueeze(-1)
    match = torch.cat((rang, idx[0]), 1)
    if (max_dist > 0):
        mask = torch.logical_and(dist.view(-1) >= 0, dist.view(-1) <= max_dist)
        match = match[mask]
    return match


def pca_compute(ps):
    #PCA compute
    ps_ = ps - ps.mean(0)
    cov_matrix = ps_.T.mm(ps_) / len(ps_)
    eig, v = torch.symeig(cov_matrix, eigenvectors=True)
    eigenvalues = eig
    normal = v[:, 0]
    z_axis = torch.tensor([0, 0, 1]).to(normal)
    if normal.dot(z_axis) < 0:
        return -normal
    else:
        return normal

def normal_mask(ps, ns, thresh_min, thresh_max):
    normal = pca_compute(ps) #  3
    
    mask_1 = torch.acos(torch.clamp(torch.abs((ns * normal).sum(1)), -1, 1)) > thresh_min
    mask_2 = torch.acos(torch.clamp(torch.abs((ns * normal).sum(1)), -1, 1)) < thresh_max
    return torch.logical_and(mask_1, mask_2)

def match2mask(match, len_ps):
    mask = torch.zeros(len_ps).bool()
    mask[match[:, 0]] = True
    return mask


class BaseDistance(object):

    def __init__(self,
                 max_dist=-1,
                 grid_size_source=0.05,
                 grid_size_target=0.1,
                 thresh_normal_max=1,
                 thresh_normal_min=-1,
                 transfo=None,
                 bins=70,
                 rang=(0, 1),
                 is_density=True):
        """
        Base Class for Distance
        grid_size_source: grid size for grid subsampling for the source
        grid_size_target: grid size for grid subsampling for the target
        transfo: other transformation
        bins: number of bins for the histogram
        rang: range of the histogram
        is_density: (is it density ?)
        """
        self.max_dist = max_dist
        self.grid_sampling_source = None
        self.grid_sampling_target = None
        if grid_size_source > 0:
            self.grid_sampling_source = GridSampling3D(mode="mean", size=grid_size_source)
        if grid_size_target > 0:
            self.grid_sampling_target = GridSampling3D(mode="mean", size=grid_size_target)
        self.transfo = transfo
        self.bins = bins
        self.rang = rang
        self.is_density = is_density
        self.thresh_normal_min = thresh_normal_min
        self.thresh_normal_max = thresh_normal_max

    def _apply_mask(self, data, dist, match):
        mask = torch.ones(len(data.pos)).bool()
        cloned_data = data.clone()
        if(self.max_dist > 0):
            mask1 = match2mask(match, len(data.pos))
            mask = torch.logical_and(mask, mask1)
        
        mask2 = normal_mask(data.pos, data.norm, self.thresh_normal_min, self.thresh_normal_max)
        mask = torch.logical_and(mask, mask2)
        
        cloned_data.pos = data.pos[mask]
        cloned_data.norm = data.norm[mask]
        new_dist = dist[mask]
        return cloned_data, new_dist

    def _preprocess(self, data: Data, grid_sampling_func: Optional[Callable]):
        cloned_data = data.clone()
        if grid_sampling_func is not None:
            cloned_data = grid_sampling_func(cloned_data)
        if(self.transfo is not None):
            cloned_data = self.transfo(cloned_data)
        return cloned_data

    def _compute(self, source, target, match):
        raise NotImplementedError

    def compute(self, source, target):
        s = self._preprocess(source, self.grid_sampling_source)
        t = self._preprocess(target, self.grid_sampling_target)
        match = compute_matches(s.pos.float(), t.pos.float(), self.max_dist)
        dist = self._compute(s, t, match)
        s, dist = self._apply_mask(s, dist, match)
        return s, t, dist

    def get_histogram(self, dist_map):
        hist, bin_edges = np.histogram(
            dist_map,
            bins=self.bins,
            range=self.rang,
            density=self.is_density)
        return hist, bin_edges

    def compute_histogram(self, source, target):
        _, _, dist_map = self.compute(source, target)
        dist_map = dist_map.detach().numpy().ravel()
        hist, bin_edges = self.get_histogram(dist_map)
        return hist, dist_map, bin_edges

    def compute_symmetric_histogram(self, source, target):

        hist_st, dist_map, bin_edges = self.compute_histogram(source, target)
        hist_ts, dist_map, bin_edges = self.compute_histogram(target, source)
        return (hist_st + hist_ts) * 0.5, dist_map, bin_edges

    def visualize_color(self, source, target):
        s, t, dist_map = self.compute(source, target)
        final_color = colorizer_v2(dist_map.detach().cpu().numpy())
        pcd_s = torch2o3d(s)
        pcd_s.colors = open3d.utility.Vector3dVector(final_color)
        open3d.visualization.draw_geometries([pcd_s])


class PCDistance(BaseDistance):

    def __init__(self,
                 max_dist=0.1,
                 plane_distance=False,
                 transfo=None,
                 thresh_normal_max=1,
                 thresh_normal_min=-1,
                 grid_size_source=0.05,
                 grid_size_target=0.1,
                 bins=70,
                 rang=(0, 1),
                 is_density=True):
        BaseDistance.__init__(self,
                              max_dist=max_dist,
                              grid_size_source=grid_size_source,
                              grid_size_target=grid_size_target,
                              thresh_normal_max=thresh_normal_max,
                              thresh_normal_min=thresh_normal_min,
                              transfo=transfo, bins=bins, rang=rang,
                              is_density=is_density)
        self.plane_distance = plane_distance


    def _compute(self, source, target, match):

        ps = source.pos.float()
        pt = target.pos.float()
        dist_map = self.max_dist * torch.ones(len(ps)) + 1
        if (self.plane_distance):
            assert pt.norm is not None
            nt = target.norm.float()
            d = ps[match[:, 0]] - pt[match[:, 1]]
            dist_map[match[:, 0]] = torch.abs((d * nt[match[:, 1]]).sum(1))
        else:
            d = torch.sqrt(((ps[match[:, 0]] - pt[match[:, 1]])**2).sum(1))
            dist_map[match[:, 0]] = d
        return dist_map



class NormalDistance(BaseDistance):

    def __init__(self,
                 gamma=1,
                 max_dist=0.1,
                 transfo=None,
                 thresh_normal_max=1,
                 thresh_normal_min=0,
                 grid_size_source=0.05,
                 grid_size_target=0.1,

                 bins=70,
                 rang=(0, 1),
                 is_density=True):
        BaseDistance.__init__(self,
                              max_dist=max_dist,
                              grid_size_source=grid_size_source,
                              grid_size_target=grid_size_target,
                              thresh_normal_max=thresh_normal_max,
                              thresh_normal_min=thresh_normal_min,
                              transfo=transfo, bins=bins, rang=rang,
                              is_density=is_density)
        self.gamma = gamma
    def _compute(self, source, target, match):

        ns = source.norm.float()
        nt = target.norm.float()
        dist_map = torch.ones(len(ns))
        d = 1 - torch.abs((ns[match[:, 0]]*nt[match[:, 1]]).sum(1))**self.gamma
        dist_map[match[:, 0]] = d
        return dist_map


class IoU(object):

    def __init__(self,
                 max_dist=0.06,
                 thresh_normal_max=1,
                 thresh_normal_min=0,
                 grid_size_source=0.05,
                 grid_size_target=0.05):
        """
        Compute Intersection over Union (IoU) for two point cloud
        """
        self.max_dist=max_dist
        self.thresh_normal_min = thresh_normal_min
        self.thresh_normal_max = thresh_normal_max
        self.grid_sampling_source = GridSampling3D(mode="mean", size=grid_size_source)
        self.grid_sampling_target = GridSampling3D(mode="mean", size=grid_size_target)

    def _preprocess(self, data: Data, grid_sampling_func: Callable) -> Data:
        cloned_data = data.clone()
        cloned_data = grid_sampling_func(data)
        mask = normal_mask(
            cloned_data.pos, cloned_data.norm,
            self.thresh_normal_min,
            self.thresh_normal_max
        )
        # mask_t = normal_mask(t.pos, t.norm, self.thresh_normal_min, self.thresh_normal_max)
        cloned_data.pos = cloned_data.pos[mask]
        cloned_data.norm = cloned_data.norm[mask]        
        return cloned_data

    def compute(self, source: Data, target: Data) -> float:

        s = self._preprocess(source, self.grid_sampling_source)
        t = self._preprocess(target, self.grid_sampling_target)
        match_source2target = compute_matches(s.pos.float(), t.pos.float(), self.max_dist)
        match_target2source = compute_matches(t.pos.float(), s.pos.float(), self.max_dist)

        inter = max(len(match_source2target), len(match_target2source))
        u = min(len(s.pos), len(t.pos))
        union = 2 * u - inter
        return inter / (union + 1e-20)
