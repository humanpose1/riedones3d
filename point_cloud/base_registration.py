import torch
import open3d
import time

from torch_points3d.utils.registration import estimate_transfo
from torch_points3d.metrics.registration_metrics import (
    compute_transfo_error,
    compute_scaled_registration_error,
)
from torch_points3d.utils.registration import get_matches, fast_global_registration
from torch_points3d.utils.registration import teaser_pp_registration
from torch_points3d.utils.registration import ransac_registration

from point_cloud.robust_estimator import RobustEstimator
def unpack(data):
    s, t = data.to_data()
    matches_gt = data.pair_ind.long()
    return s, t, matches_gt




class BaseRegistrator(object):


    def __init__(self, icp=None, transform=None, rot_thresh=5, trans_thresh=0.3, **kwargs):
        self.rot_thresh = rot_thresh
        self.trans_thresh = trans_thresh
        self.icp = icp
        self.transform = transform

    def track(self, s, t, T_gt, T_est):
        xyz = s.pos
        trans_error, rot_error = compute_transfo_error(T_est, T_gt)
        res = dict()
        res["trans_error"] = trans_error.item()
        res["rot_error"] = rot_error.item()
        res["rre"] = float(rot_error.item() < self.rot_thresh)
        res["rte"] = float(trans_error.item() < self.trans_thresh)
        res["sr"] = compute_scaled_registration_error(xyz, T_gt, T_est).item()
        return res

    def _refine_registration(self, source, target):
        if(self.icp is None):
            return torch.eye(4)
        else:
            try:
                T = self.icp(source, target)
                return T
            except Exception:
                print("FAILURE")
                return torch.eye(4)

    def registrate(self, s, t):
        raise NotImplementedError

    def evaluate_pair(self, data):
        s, t, matches_gt = unpack(data)
        xyz = s.pos
        xyz_target = t.pos
        T_gt = estimate_transfo(xyz[matches_gt[:, 0]], xyz_target[matches_gt[:, 1]])
        t0 = time.time()
        T_est = self.registrate(s, t)
        t1 = time.time()
        metric = self.track(s, t, T_gt, T_est)
        metric["time"] = t1 - t0
        return metric


class FeatureBasedRegistrator(BaseRegistrator):

    def __init__(self,
                 robust_estimator,
                 transform=None,
                 icp=None,
                 num_points=5000,
                 max_norm_threshold=0.5,
                 max_num_matches=100,
                 min_num_matches=20,
                 verbose=True,
                 rot_thresh=5,
                 trans_thresh=0.3,
                 **kwargs):

        super(FeatureBasedRegistrator, self).__init__(icp=icp, transform=transform, rot_thresh=rot_thresh, trans_thresh=trans_thresh)
        self.num_points = num_points
        self.max_norm_threshold = max_norm_threshold
        assert robust_estimator is not None
        self.robust_estimator = robust_estimator
        self.verbose = verbose
        self.max_num_matches = max_num_matches
        self.min_num_matches = min_num_matches


    def _compute_features(self, data_s, data_t, rand_s, rand_t):
        raise NotImplementedError

    def _preprocess(self, data):
        if self.transform is not None:
            cloned_data = self.transform(data)
        else:
            cloned_data = data.clone()
        return cloned_data


    def _coarse_registration(self, source, target):
        t0 = time.time()
        data_s = self._preprocess(source)
        data_t = self._preprocess(target)
        delta_t = time.time() - t0
        t0 = time.time()
        rand_s = torch.randint(0, len(data_s.pos), (self.num_points,))
        rand_t = torch.randint(0, len(data_t.pos), (self.num_points,))
        output_s, output_t = self._compute_features(data_s, data_t, rand_s, rand_t)
        delta_t0 = time.time() - t0
        t0 = time.time()
        matches = get_matches(output_s, output_t, sym=True)
        delta_t1 = time.time() - t0
        t0 = time.time()
        if(len(matches) < 3):
            return torch.eye(4), torch.empty(0, 3), torch.empty(0, 3)
        T_est = self.robust_estimator.estimate(
            data_s.pos[rand_s][matches[:, 0]],
            data_t.pos[rand_t][matches[:, 1]]
        )
        mask = torch.norm(
            data_s.pos[rand_s][matches[:, 0]] @ T_est[:3, :3].T + T_est[:3, 3] - data_t.pos[rand_t][matches[:, 1]], dim=1) < self.max_norm_threshold


        matches = matches[mask, :]
        if(len(matches) > self.max_num_matches):
            rand = torch.randperm(len(matches))[:self.max_num_matches]
            matches = matches[rand, :]
        ps = torch.arange(len(source.pos))[rand_s][matches[:, 0]]
        pt = torch.arange(len(target.pos))[rand_t][matches[:, 1]]
        pair_ind_s = data_s.pos[ps]
        pair_ind_t = data_t.pos[pt]
        delta_t2 = time.time() - t0
        if(self.verbose):
            print("number of matches:", len(matches))
            print("preprocessing: {}".format(delta_t))
            print("compute feature: {}".format(delta_t0))
            print("compute matches: {}".format(delta_t1))
            print("compute robust estimation: {}".format(delta_t2))
        return T_est, pair_ind_s, pair_ind_t


    def registrate(self, s, t):
        source = s.clone()
        target = t.clone()
        assert(source.norm is not None)
        T_est, pair_ind_s, pair_ind_t = self._coarse_registration(source, target)
        source.pair_ind = pair_ind_s
        target.pair_ind = pair_ind_t
        if(len(source.pair_ind) < self.min_num_matches):
            print("Warning; try an other trans")
            print("num_matches:", len(source.pair_ind))
        source.pos = source.pos @ T_est[:3, :3].detach().cpu().T + T_est.detach().cpu()[:3, 3]
        source.pair_ind = source.pair_ind @ T_est[:3, :3].detach().cpu().T + T_est.detach().cpu()[:3, 3]
        source.norm = source.norm @ T_est[:3, :3].detach().cpu().T
        if(len(source.pair_ind) < self.min_num_matches):
            print("GROSS FAILURE, NO ICP")
            T_icp = torch.eye(4)
        else:
            T_icp = self._refine_registration(source, target)

        T_total = T_icp @ T_est
        return T_total



class PrecomputedFeatureBasedRegistrator(FeatureBasedRegistrator):

    def _compute_features(self, data_s, data_t, rand_s, rand_t):
        assert getattr(data_s, "feat", None) is not None
        assert getattr(data_t, "feat", None) is not None
        return data_s.feat[rand_s], data_t.feat[rand_t]


class LocalRegistrator(BaseRegistrator):

    def registrate(self, s, t):

        t0 = time.time()
        if self.transform is not None:
            data_s = self.transform(s.clone())
            data_t = self.transform(t.clone())
        else:
            data_s = s.clone()
            data_t = t.clone()
        delta_t = time.time() - t0
        T_icp = self._refine_registration(data_s, data_t)
        return T_icp


