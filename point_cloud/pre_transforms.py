import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso

from torch_geometric.data import Data

from point_cloud.utils import vec_2_transfo, compute_PCA


class RotateToAxis(object):
    """
    Perform a rotation to reorient a point cloud using a Principal component analysis
    """

    def __init__(self, axis_3d: torch.Tensor = torch.tensor([0.0, 0.0, 1.0])):
        self.axis_3d = axis_3d

    def __call__(self, data: Data, **kwargs):
        # First define the rotation
        _, eigenvectors = compute_PCA(data.pos)
        global_normal = eigenvectors[:, -1]
        if (global_normal.dot(self.axis_3d) < 0):
            global_normal = -global_normal
        transfo = vec_2_transfo(global_normal, self.axis_3d)
        data.pos = data.pos @ transfo[:3, :3].T
        if getattr(data, "norm", None) is not None:
            data.norm = data.norm @ transfo[:3, :3].T
        return data

    def __repr__(self):
        return f"OrientPointCloud(axis_3d={self.axis_3d})"


class BendingByPolynomialRegression(object):
    """
    fit the point cloud with a polynomial function of type
    find the coefficient a_{ij} such as:
    z = f(x, y) = \sum_{i=0}^d \sum_{j=0}^d a_{ij} x^{i} y^{j}
    """

    def __init__(self, deg: int = 2, alpha: float = 1e-5):
        self.deg = deg
        if(alpha == 0.0):
            self.regressor = LinearRegression()
        else:
            self.regressor = Lasso(alpha=alpha)

    def compute_normals_polynomial(self, xyz: torch.Tensor): 
        norm = torch.zeros_like(xyz)
        # TODO: implement the norm computation
        
        return norm

    def augment_input(self, X: torch.Tensor, Y: torch.Tensor):
        feat = torch.ones((X.shape[0], 1))
        for i in range(self.deg+1):
            for j in range(self.deg+1):
                if(i == 0 and j==0):
                    continue
                feat = torch.cat([feat, X**i * Y**j], 1)
        return feat

    def fit_polynomial(self, xyz:torch.Tensor): 
        X = xyz[:, 0].reshape(-1, 1)
        Y = xyz[:, 1].reshape(-1, 1)
        feat = self.augment_input(X, Y)
        z_gt = xyz[:, 2]
        reg = self.regressor.fit(feat.detach().cpu().numpy(), z_gt.detach().cpu().numpy())
        z_pred = feat @ torch.from_numpy(reg.coef_.reshape(-1))
        self.coef_ = torch.from_numpy(reg.coef_.reshape(-1))
        self.intercept_ = reg.intercept_
        new_xyz = xyz.clone()
        new_xyz[:, 2] = z_gt - z_pred
        return new_xyz

    def __call__(self, data: Data, **kwargs):
        data.pos = self.fit_polynomial(data.pos)
        if getattr(data, "norm", None) is not None:
            pass
            # normal_polynomial = compute_normals_polynomial(data.pos)
            # TODO: implement the batch_vec_2_transfo in utils
            # R = batch_vec_2_transfo(data.norm, normal_polynomial)
            # data.norm = torch.einsum("cab, cb -> ca", R, data.norm)
            # data.norm = self.correct_normal(data.norm, normal_polynomial)
        return data

    def __repr__(self):
        return f"BendingByPolynomialRegression(deg={self.deg})"

