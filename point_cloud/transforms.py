import sys
import torch
from torch_points_kernels import ball_query
from torch_geometric.data import Data

_custom_transforms = sys.modules[__name__]

def instantiate_radius_patch_extractor(cfg_transfo, attr="transform"):
    if len(cfg_transfo) < 1:
        return None
    tr_name = getattr(cfg_transfo[0], attr, None)
    try:
        tr_params = cfg_transfo[0].get("params")
    except KeyError:
        tr_params = None
    cls = getattr(_custom_transforms, tr_name, None)
    if cls is None:
        raise notImplementedError("this transform does not exist")
    if tr_params is not None:
        return cls(**tr_params)
    return cls()


class BaseRadiusPatchExtractor(object):
    """
    extract multiple patch
    """
    def __init__(self, radius: float = 1, max_num: int = 10000, **kwargs):
        self.radius = radius
        self.max_num = max_num

    def query(self, data: Data, centers: torch.Tensor):
        pos = data.pos
        b = torch.zeros(len(pos)).to(pos).long()
        bc = torch.zeros(len(centers)).to(centers).long()
        idx, dist = ball_query(radius=self.radius, nsample=self.max_num,
                               x=pos, y=centers, mode="partial_dense",
                               batch_x=b, batch_y=bc)
        for key in data.keys:
            if(len(data[key]) == len(pos)):
                data[key] = data[key][idx[idx >= 0].view(-1)]
        return data

    def __call__(self, data: Data, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return "MultiRadiusPatchExtractor(radius={}, max_num={})".format(self.radius, self.max_num)


class RadiusPatchExtractor(BaseRadiusPatchExtractor):
    """
    extract a patch.
    """
    def __call__(self, data: Data, **kwargs):
        center = data.pos.mean(0).unsqueeze(0)
        return self.query(data, center)

    def __repr__(self):
        return "RadiusPatchExtractor(radius={}, max_num={})".format(self.radius, self.max_num)


class MultiRadiusPatchExtractor(BaseRadiusPatchExtractor):
    """
    extract a patch.
    """
    def __call__(self, data: Data, **kwargs):
        assert hasattr(data, "pair_ind")
        centers = data.pair_ind
        return self.query(data, centers)

    def __repr__(self):
        return "RadiusPatchExtractor(radius={}, max_num={})".format(self.radius, self.max_num)
