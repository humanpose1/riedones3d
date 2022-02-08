iS_HANDCRAFT_DESC_IMPORT = True
try:
    import handcrafted_descriptor as hd
except ImportError:
    iS_HANDCRAFT_DESC_IMPORT = False

import open3d as o3d
import numpy as np
import torch

from point_cloud.base_registration import FeatureBasedRegistrator
from point_cloud.visu import torch2o3d



class FPFHRegistrator(FeatureBasedRegistrator):

    def __init__(self,
                 radius=0.5,
                 max_nn=30,
                 transform=None,
                 icp=None,
                 num_points=5000,
                 max_norm=0.5,
                 robust_estimator="teaser",
                 noise_bound_teaser=0.5,
                 num_iteration=80000,
                 max_num_matches=100,
                 min_num_matches=20,
                 verbose=True,
                 rot_thresh=5,
                 trans_thresh=0.3, **kwargs):
        FeatureBasedRegistrator.__init__(
            self,
            transform=transform,
            icp=icp,
            num_points=num_points,
            max_norm=max_norm,
            robust_estimator=robust_estimator,
            noise_bound_teaser=noise_bound_teaser,
            num_iteration=num_iteration,
            max_num_matches=max_num_matches,
            min_num_matches=min_num_matches,
            verbose=verbose,
            rot_thresh=rot_thresh,
            trans_thresh=trans_thresh, **kwargs)
        self.kd_tree = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)


    def _compute_features(self, data_s, data_t, rand_s, rand_t):
        pcd_s = torch2o3d(data_s)
        fpfh_s = np.asarray(o3d.pipelines.registration.compute_fpfh_feature(pcd_s, self.kd_tree).data).T
        output_s = torch.from_numpy(fpfh_s).to(data_s.pos)
        pcd_t = torch2o3d(data_t)
        fpfh_t = np.asarray(o3d.pipelines.registration.compute_fpfh_feature(pcd_t, self.kd_tree).data).T
        output_t = torch.from_numpy(fpfh_t).to(data_t.pos)
        return output_s[rand_s], output_t[rand_t]


class SHOTRegistrator(FeatureBasedRegistrator):

    def __init__(self,
                 radius=0.5,
                 transform=None,
                 icp=None,
                 num_points=5000,
                 max_norm=0.5,
                 robust_estimator="teaser",
                 noise_bound_teaser=0.5,
                 num_iteration=80000,
                 max_num_matches=100,
                 min_num_matches=20,
                 verbose=True,
                 rot_thresh=5,
                 trans_thresh=0.3, **kwargs):

        if not iS_HANDCRAFT_DESC_IMPORT:
            raise ImportError("Cannot import the lib to compute shot descriptor")
        FeatureBasedRegistrator.__init__(
            self,
            transform=transform,
            icp=icp,
            num_points=num_points,
            max_norm=max_norm,
            robust_estimator=robust_estimator,
            noise_bound_teaser=noise_bound_teaser,
            num_iteration=num_iteration,
            max_num_matches=max_num_matches,
            min_num_matches=min_num_matches,
            verbose=verbose,
            rot_thresh=rot_thresh,
            trans_thresh=trans_thresh, **kwargs)
        self.radius = radius

    def compute_shot_descriptor(self, data, rand):
        assert getattr(data, "norm", None) is not None
        assert getattr(data, "pos", None) is not None
        pos = data.pos.detach().cpu().numpy().astype(float)
        norm = data.norm.detach().cpu().numpy().astype(float)
        norm = np.asarray(norm).astype(float)
        small_pos = data.pos[rand].detach().cpu().numpy().astype(float)
        small_pos = np.asarray(small_pos).astype(float)
        small_norm = data.norm[rand].detach().cpu().numpy().astype(float)
        small_norm = np.asarray(small_norm).astype(float)
        feat = hd.compute_shot(pos, norm, small_pos, small_norm, self.radius)
        return torch.from_numpy(feat).to(data.pos)

    def _compute_features(self, data_s, data_t, rand_s, rand_t):
        feat_s = self.compute_shot_descriptor(data_s, rand_s)
        feat_t = self.compute_shot_descriptor(data_t, rand_t)
        return feat_s, feat_t




