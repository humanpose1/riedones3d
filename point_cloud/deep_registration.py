import open3d
import torch
import numpy as np
import time

from point_cloud.visu import torch2o3d
from point_cloud.utils import eulerAnglesToRotationMatrix
from sklearn.decomposition import PCA
from torch_geometric.data import Data

from torch_points3d.core.data_transform import Random3AxisRotation, SaveOriginalPosId
from torch_points3d.applications.pretrained_api import PretainedRegistry



from point_cloud.base_registration import BaseRegistrator, FeatureBasedRegistrator

def compute_color_from_features(list_feat):
    feats = np.vstack(list_feat)
    pca = PCA(n_components=3)
    pca.fit(feats)
    min_col = pca.transform(feats).min(axis=0)
    max_col = pca.transform(feats).max(axis=0)
    list_color = []
    for feat in list_feat:
        color = pca.transform(feat)
        color = (color - min_col) / (max_col - min_col)
        list_color.append(color)
    return list_color


class DeepRegistrationPipeline(FeatureBasedRegistrator):
    """
    Apply deep learning to compute feature and fast global registration to match them
    then refine ICP
    """
    def __init__(self, path_model,
                 device="cuda",
                 feat_dim=1,
                 transform=None,
                 icp=None,
                 num_points=5000, max_norm=0.5,
                 robust_estimator=None,
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
            max_num_matches=max_num_matches,
            min_num_matches=min_num_matches,
            verbose=verbose,
            rot_thresh=rot_thresh,
            trans_thresh=trans_thresh, **kwargs)
        self.path_model=path_model
        prop = {"feature_dimension": feat_dim}
        self.model = PretainedRegistry.from_file(self.path_model, mock_property=prop).to(device)
        self.device = device

    def _compute_features(self, data_s, data_t, rand_s, rand_t):
        with torch.no_grad():
            self.model.set_input(data_s, self.device)
            output_s = self.model.forward()
            self.model.set_input(data_t, self.device)
            output_t = self.model.forward()
        return output_s[rand_s], output_t[rand_t]

    def get_colored_features(self, source, target):
        data_s = self.transform(source.clone())
        data_t = self.transform(target.clone())
        with torch.no_grad():
            self.model.set_input(data_s, self.device)
            output_s = self.model.forward()
            self.model.set_input(data_t, self.device)
            output_t = self.model.forward()

        list_color = compute_color_from_features([output_s.detach().cpu().numpy(), output_t.detach().cpu().numpy()])
        data_s.colors = torch.from_numpy(list_color[0])
        data_t.colors = torch.from_numpy(list_color[1])
        return data_s, data_t




class DeepPatchRegistrationPipeline(FeatureBasedRegistrator):
    """
    Apply Patch deep learning method such as DIP or GeDI to compute feature and fast global registration to match them.
    then refine ICP
    """
    def __init__(self, path_model,
                 robust_estimator,
                 device="cuda",
                 feat_dim=3,
                 transform=None,
                 icp=None,
                 num_points=5000,
                 max_norm=0.5,
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
            num_iteration=num_iteration,
            max_num_matches=max_num_matches,
            min_num_matches=min_num_matches,
            verbose=verbose,
            rot_thresh=rot_thresh,
            trans_thresh=trans_thresh, **kwargs)
        self.path_model = path_model
        prop = {"feature_dimension": feat_dim}
        self.model = PretainedRegistry.from_file(self.path_model, mock_property=prop).to(device)
        assert hasattr(self.model, "whole_pipeline")
        self.model.whole_pipeline = True
        self.device = device

    def _compute_features(self, data_s, data_t, rand_s, rand_t):
        s = Data(pos=data_s.pos.unsqueeze(0))
        t = Data(pos=data_t.pos.unsqueeze(0))

        with torch.no_grad():
            self.model.set_input(s, self.device, rand_ind=rand_s.view(1, -1))
            output_s = self.model.forward().squeeze(0)
            self.model.set_input(t, self.device, rand_ind=rand_t.view(1, -1))
            output_t = self.model.forward().squeeze(0)
        return output_s[rand_s], output_t[rand_t]
