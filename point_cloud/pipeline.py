import os
import os.path as osp
import torch
import numpy as np
import pathlib
from joblib import Parallel, delayed

from torch_points3d.core.data_transform import Random3AxisRotation, SaveOriginalPosId
from torch_points3d.core.data_transform import GridSampling3D, AddFeatByKey, AddOnes

from point_cloud.base_registration import BaseRegistrator
from point_cloud.base_registration import PrecomputedFeatureBasedRegistrator
from point_cloud.deep_registration import DeepRegistrationPipeline
from point_cloud.io import read_pcd
from point_cloud.transforms import MultiRadiusPatchExtractor
from point_cloud.error_mesurer import PCDistance
from point_cloud.robust_estimator import RobustEstimator
from point_cloud.ICP import BaseICP

from torch_geometric.transforms import Compose 


class BasePipeline(object):

    def __init__(self):
        self.init_output()

    def init_output(self):
        self.list_transfo = dict()
        self.list_hist = dict()

    def compute_pair(self, path_source, path_target):
        raise NotImplementedError("implement pair computation")

    def compute_all(self, list_path):
        for i in range(len(list_path)):
            for j in range(i+1, len(list_path)):
                T_f, hist, name = self.compute_pair(list_path[i], list_path[j])
                self.list_transfo[name] = T_f
                self.list_hist[name] = hist

    def get_name(self, path_source, path_target):
        name_s = osp.split(path_source)[1].split(".")[0]
        name_t = osp.split(path_target)[1].split(".")[0]
        name = name_s + "_" + name_t
        return name

    def read_point_cloud(self, path_source, path_target):
        raise NotImplementedError("What kind of point cloud")

    def save_final_results(self, path_output):
        path_output_transformation = osp.join(path_output, "transformation")
        pathlib.Path(path_output_transformation).mkdir(parents=True,
                                                       exist_ok=True)
        np.save(osp.join(path_output, "transfo.npy"), self.list_transfo)
        np.save(osp.join(path_output, "hist.npy"), self.list_hist)


class BaseRiedonesPipeline(BasePipeline):

    def __init__(self, estimator: RobustEstimator,
                 num_points: int = 5000,
                 sym: bool = False):
        """
        Pipeline adapted for Riedones3D
        """
        self.sym = sym
        self.num_points = num_points
        self.estimator = estimator
        self.icp = None
        self.registrator = None
        self.distance_computer = None
        self.init_icp()
        self.init_registrator()
        self.init_distance_computer()
        self.init_output()

    def init_icp(self):
        self.icp = BaseICP(
            mode="plane", num_iter=10, stopping_criterion=1e-3,
            max_dist=0.1,
            pre_transform_source=Compose([MultiRadiusPatchExtractor(radius=2, max_num=1000)]),
            pre_transform_target=Compose([MultiRadiusPatchExtractor(radius=4, max_num=1000)]),
            is_debug=False
        )

    def init_registrator(self):
        raise NotImplementedError("implement the registrator")

    def init_distance_computer(self):
        self.distance_computer = PCDistance(
            max_dist=0.6,
            thresh_normal_max=1,
            thresh_normal_min=0.0,
            rang=(0, 0.6), transfo=None)

    def registrate(self, source, target):
        assert self.registrator is not None
        return self.registrator.registrate(source, target)

    def compute_hist(self, data_s, data_t):
        if(self.sym):
            hist, dist_map, _ = self.distance_computer.compute_symmetric_histogram(data_s, data_t)
        else:
            hist, dist_map, _ = self.distance_computer.compute_histogram(data_s, data_t)
        return hist, dist_map

    def compute_pair(self, path_source, path_target):
        name = self.get_name(path_source, path_target)
        data_s, data_t = self.read_point_cloud(path_source, path_target)
        T_f = self.registrate(data_s, data_t)
        data_s.pos = data_s.pos @ T_f[:3, :3].T + T_f[:3, 3]
        data_s.norm = data_s.norm @ T_f[:3, :3].T
        hist, _ = self.compute_hist(data_s, data_t)
        return T_f.cpu().numpy(), hist, name


class PretrainedRiedonesPipeline(BaseRiedonesPipeline):

    def init_registrator(self):
        self.registrator = PrecomputedFeatureBasedRegistrator(
            max_norm=0.5,
            transform=None,
            icp=self.icp,
            robust_estimator=self.estimator,
            noise_bound_teaser=0.1,
            verbose=False,
            min_num_matches=3,
            num_points=self.num_points)

    def read_point_cloud(self, path_source, path_target):
        data_s = torch.load(path_source)
        data_t = torch.load(path_target)
        return data_s, data_t


class OnlineRiedonesPipeline(BaseRiedonesPipeline):

    def __init__(self, estimator: RobustEstimator,
                 path_model: str,
                 num_points: int = 5000,
                 sym: bool = False):
        """
        compute Features, transformation and then the histogram
        """
        self.path_model = path_model
        BaseRiedonesPipeline.__init__(self, estimator=estimator, num_points=num_points, sym=sym)

    def init_registrator(self):
        transform = Compose(
            [
                SaveOriginalPosId(),
                GridSampling3D(mode="last", size=0.1, quantize_coords=True),
                AddOnes(),
                AddFeatByKey(add_to_x=True, feat_name="ones"),
            ]
        )
        self.registrator = DeepRegistrationPipeline(
            path_model=self.path_model,
            num_points=self.num_points,
            max_norm=0.5,
            transform=transform,
            icp=self.icp,
            min_num_matches=3,
            robust_estimator=self.estimator)

    def read_point_cloud(self, path_source, path_target):
        return read_pcd(path_source), read_pcd(path_target)
