"""
implementation of class using torch data object
"""

from omegaconf import OmegaConf
import open3d
import scipy.linalg as sl
import sys
import torch
import probreg as preg

from torch_points_kernels import ball_query, knn

from point_cloud.utils import compute_T
from point_cloud.visu import torch2o3d
from point_cloud.transforms import instantiate_radius_patch_extractor

_custom_ICP = sys.modules[__name__]

def instantiate_icp(params):
    list_keys = []
    for k in params.keys():
        if(k != "pre_transform_source" and k != "pre_transform_target"):
            list_keys.append(k)
    icp_params = OmegaConf.masked_copy(params, list_keys)
    name = icp_params.get("class")
    cls = getattr(_custom_ICP, name, None)
    if cls is None:
        raise NotImplementedError("%s is nowhere to be found" % name)

    pre_transform_source_param = getattr(params, "pre_transform_source", None)
    if pre_transform_source_param is not None:
        pre_transform_source = instantiate_radius_patch_extractor(pre_transform_source_param)
    else:
        pre_transform_source = None

    pre_transform_target_param = getattr(params, "pre_transform_target", None)
    if pre_transform_target_param is not None:
        pre_transform_target = instantiate_radius_patch_extractor(pre_transform_target_param)
    else:
        pre_transform_target = None
    icp = cls(pre_transform_source=pre_transform_source,
              pre_transform_target=pre_transform_target, **icp_params)
    return icp




def point_to_point_solver(match, source, target):
    ps = source.pos[match[:, 0]]
    pt = target.pos[match[:, 1]]
    source_centered = ps - ps.mean(0)
    target_centered = pt - pt.mean(0)

    H = (source_centered.T @ target_centered) /len(source_centered)

    U, S, V = torch.svd(H)
    d = torch.det(V @ U.T)

    one = torch.ones_like(d)
    diag = torch.diag(torch.tensor([one, one, d]))
    R = V.mm(diag).mm(U.T)

    t = pt.mean(0) - ps.mean(0) @ R.T
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def compute_cn(ps, pt, nt):
    """
    compute the matrix cn used in the point2plane variant of ICP

    """
    c = torch.cross(ps, nt)
    cn = torch.cat((c, nt), 1)
    b = -((ps-pt)*nt).sum(axis=1)
    return cn, b


def point_to_plane_solver(match, source, target, tol=1e-2, use_scipy=False):
    assert hasattr(target, "norm")
    ps = source.pos[match[:, 0]]
    pt = target.pos[match[:, 1]]
    nt = target.norm[match[:, 1]]
    cn, b = compute_cn(ps, pt, nt)

    A = cn.T @ cn
    const = cn.T @ b.unsqueeze(-1)
    if(not use_scipy):
        sol = torch.solve(const, A)[0]
        T = compute_T(sol[:3, 0], sol[3:, 0])
        return T
    else:
        c, low = sl.cho_factor(A.cpu().detach().numpy())
        sol = sl.cho_solve((c, low), const.cpu().detach().numpy())
        sol = torch.from_numpy(sol)
        T = compute_T(sol[:3, 0], sol[3:, 0])

        return T



class BaseLocal(object):
    def __init__(self, pre_transform_source, pre_transform_target, num_iter=20, stopping_criterion=1e-2):
        self.num_iter = num_iter
        self.stopping_criterion = stopping_criterion
        self.pre_transform_source = pre_transform_source
        self.pre_transform_target = pre_transform_target

    def _preprocess(self, source, target):
        if(self.pre_transform_source is None):
            s = source.clone()
        else:
            s = self.pre_transform_source(source.clone())
        if(self.pre_transform_target is None):
            t = target.clone()
        else:
            t = self.pre_transform_target(target.clone())
        return s, t

    def __call__(self, source, target):
        raise NotImplementedError("define the main pipeline loop for registration")

class BaseICP(BaseLocal):

    """
    Implementation of ICP class
    """

    def __init__(self, pre_transform_source=None, pre_transform_target=None, num_iter=20, stopping_criterion=1e-2, mode="plane", max_dist=-1, is_debug=False, **kwargs):
        super(BaseICP, self).__init__(pre_transform_source, pre_transform_target, num_iter, stopping_criterion)
        self.mode = mode
        self.max_dist = max_dist
        self.is_debug = is_debug

    def _solver(self, match, source, target):
        if(self.mode == "point"):
            return point_to_point_solver(match, source, target)
        elif(self.mode == "plane"):
            return point_to_plane_solver(match, source, target)

    def _compute_match(self, source, target):
        ps = source.pos.float()
        pt = target.pos.float()
        bs = torch.zeros(len(ps)).to(ps).long()
        bt = torch.zeros(len(pt)).to(pt).long()
        if(self.max_dist < 0):
            idx, dist = knn(ps.unsqueeze(0), pt.unsqueeze(0), k=1)

        else:
            idx, dist = ball_query(self.max_dist, 1,
                                   pt, ps, mode="partial_dense",
                                   batch_x=bt, batch_y=bs)
            idx = idx.unsqueeze(0)
        rang = torch.arange(0, len(ps)).unsqueeze(-1)
        match = torch.cat((rang, idx[0]), 1)
        match = match[match[:, 1] >= 0, :]
        return match


    def check_criterion(self, T, T_pred):
        Id = torch.eye(4)
        err = torch.norm(T @ torch.inverse(T_pred) - Id)

        return err.item() < self.stopping_criterion

    def __call__(self, source, target):

        s, t = self._preprocess(source, target)

        if(self.is_debug):
            pcd_s = torch2o3d(s, color=[0.9,0.7,0.1])
            pcd_t = torch2o3d(t, color=[0.1,0.9,0.7])
            open3d.visualization.draw_geometries([pcd_s, pcd_t])
        T_final = torch.eye(4)
        for i in range(self.num_iter):

            match = self._compute_match(s, t)

            T = self._solver(match, s, t)
            s.pos = s.pos @ T.t()[:3, :3] + T[:3, 3]
            if(getattr(s, "norm", None) is not None):
                s.norm = s.norm @ T.t()[:3, :3]

            if self.check_criterion(T_final, T @ T_final):
                return T_final
            else:
                T_final = T @ T_final
        return T_final


class CPDProbReg(BaseLocal):
    """
    Probabilistic Registration(CPD) using the probreg library

    """

    def __init__(self, pre_transform_source, pre_transform_target, num_iter=20, stopping_criterion=1e-4, use_cuda=False, w=0.2):
        super(CPDProbReg, self).__init__(pre_transform_source, pre_transform_target, num_iter, stopping_criterion)
        self.w = w
        self.registrator = None
    def __call__(self, source, target):
        s, t = self._preprocess(source, target)
        pcd_s = torch2o3d(s, color=[0.9,0.7,0.1])
        pcd_t = torch2o3d(t, color=[0.1,0.9,0.7])
        if self.use_cuda:
            import cupy as cp
            to_cpu = cp.asnumpy
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        else:
            cp = np
            to_cpu = lambda x: x
        source = cp.asarray(pcd_s.points, dtype=cp.float32)
        target = cp.asarray(pcd_t.points, dtype=cp.float32)
        rcpd = preg.cpd.RigidCPD(source, use_cuda=use_cuda, update_scale=False)
        tf_params, _ = rcpd.registration(target, maxiter=self.num_iter, tol=self.stopping_criterion, w=self.w)
        T_final = torch.from_numpy(to_cpu(tf_params.transformation))
        return T_final

class GMMTreeProbReg(BaseLocal):

    def __init__(self, pre_transform_source, pre_transform_target, num_iter=20, stopping_criterion=1e-4, lambda_s=0.001, lambda_c=0.01, tree_level=2):
        super(GMMTreeProbReg, self).__init__(pre_transform_source, pre_transform_target, num_iter, stopping_criterion)



    def __call__(self, source, target):
        s, t = self._preprocess(source, target)
        pcd_s = torch2o3d(s, color=[0.9,0.7,0.1])
        pcd_t = torch2o3d(t, color=[0.1,0.9,0.7])
        res = preg.gmmtree.registration_gmmtree(
            source, target,
            maxiter=self.num_iter,
            tol=self.stopping_criterion,
            lambda_s=self.lambda_s,
            lambda_c=self.lambda_c,
            tree_level=self.tree_level,
            update_scale=False)

class FilterRegProbReg(BaseLocal):
    def __init__(self, pre_transform_source, pre_transform_target, num_iter=20, stopping_criterion=1e-4):
        super(GMMTreeProbReg, self).__init__(pre_transform_source, pre_transform_target, num_iter, stopping_criterion)





