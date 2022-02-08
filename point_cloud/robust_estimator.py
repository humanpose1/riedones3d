import torch
import open3d
import numpy as np
import teaserpp_python
import sys

from torch_points3d.utils.registration import fast_global_registration

_custom_module = sys.modules[__name__]

def instantiate_robust_estimator(params):
    name = params.get("class")
    cls = getattr(_custom_module, name, None)
    if cls is None:
        raise NotImplementedError("%s is nowhere to be found" % name)
    instantiated = cls(**params)
    return instantiated


class RobustEstimator(object):

    def estimate(xyz_source: torch.Tensor, xyz_target: torch.Tensor):
        raise NotImplementedError


class SimpleFGREstimator(RobustEstimator):

    def __init__(*args, **kwargs):
        pass

    def estimate(self, xyz_source: torch.Tensor, xyz_target: torch.Tensor):
        assert xyz_source.shape == xyz_target.shape
        return fast_global_registration(xyz_source, xyz_target)



class TeaserEstimator(RobustEstimator):
    """
    a simple wrapper for teaser
    """

    def __init__(
            self,
            noise_bound:float,
            cbar2:float = 1.0,
            rotation_gnc_factor:float = 1.4,
            rotation_max_iterations:int = 100,
            rotation_cost_threshold:float = 1e-12,
            max_clique_time_limit: float = 1,
            **kwargs):
        self.cbar2 = cbar2
        self.noise_bound = noise_bound
        self.rotation_gnc_factor = rotation_gnc_factor
        self.rotation_max_iterations = rotation_max_iterations
        self.rotation_cost_threshold = rotation_cost_threshold
        self.max_clique_time_limit = max_clique_time_limit


    def prepare_solver(self):
        """
        the code must be run in multiprocessing
        """
        import teaserpp_python
        
        # Populating the parameters
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.cbar2 = self.cbar2
        solver_params.noise_bound = self.noise_bound
        solver_params.estimate_scaling = False
        solver_params.rotation_estimation_algorithm = (
            teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        )
        solver_params.rotation_gnc_factor = self.rotation_gnc_factor
        solver_params.rotation_max_iterations = self.rotation_max_iterations
        solver_params.rotation_cost_threshold = self.rotation_cost_threshold
        solver_params.max_clique_time_limit = self.max_clique_time_limit

        return teaserpp_python.RobustRegistrationSolver(solver_params)

    def estimate(self, xyz_source: torch.Tensor, xyz_target: torch.Tensor):
        assert xyz_source.shape == xyz_target.shape
        solver = self.prepare_solver()
        solver.solve(xyz_source.T.detach().cpu().numpy(), xyz_target.T.detach().cpu().numpy())
        solution = solver.getSolution()
        T_res = torch.eye(4, device=xyz_source.device)
        T_res[:3, :3] = torch.from_numpy(solution.rotation).to(xyz_source.device)
        T_res[:3, 3] = torch.from_numpy(solution.translation).to(xyz_source.device)
        return T_res


class RansacEstimator(RobustEstimator):

    def __init__(
            self,
            distance_threshold: float = 0.05,
            num_iterations: int = 80000,
            ransac_n: int = 4,
            **kwargs):
        self.distance_threshold = distance_threshold
        self.num_iterations = num_iterations
        self.ransac_n = ransac_n

    def estimate(self, xyz_source: torch.Tensor, xyz_target: torch.Tensor):
        assert xyz_source.shape == xyz_target.shape
        pcd_s = open3d.geometry.PointCloud()
        pcd_s.points = open3d.utility.Vector3dVector(xyz_source.detach().cpu().numpy())

        pcd_t = open3d.geometry.PointCloud()
        pcd_t.points = open3d.utility.Vector3dVector(xyz_target.detach().cpu().numpy())
        rang = np.arange(len(xyz_source))
        corres = np.stack((rang, rang), axis=1)
        corres = open3d.utility.Vector2iVector(corres)
        result = open3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcd_s,
            pcd_t,
            corres,
            self.distance_threshold,
            estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=self.ransac_n,
            criteria=open3d.pipelines.registration.RANSACConvergenceCriteria(4000000, self.num_iterations),
        )
        return torch.from_numpy(result.transformation).float()
