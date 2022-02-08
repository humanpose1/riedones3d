try:
    import handcrafted_descriptor
except ImportError:
    pass
import importlib
from omegaconf import OmegaConf
import sys
from point_cloud.ICP import instantiate_icp
from point_cloud.robust_estimator import instantiate_robust_estimator
from torch_points3d.core.data_transform import instantiate_transforms


def instantiate_registrator(cfg_registrator):

    registrator_class = cfg_registrator.get("class")
    registrator_path = registrator_class.split('.')
    module = ".".join(registrator_path[:-1])
    class_name = registrator_path[-1]
    reg_module = ".".join(["point_cloud", module])
    reglib = importlib.import_module(reg_module)

    reg_cls = None
    for name, cls in reglib.__dict__.items():
        if name.lower() == class_name.lower():
            reg_cls = cls

    list_keys = []
    exclude_key = ["icp_params", "transform_params"]
    for k in cfg_registrator.keys():
        if(k not in exclude_key):
            list_keys.append(k)
    transform_params = cfg_registrator.get("transform_params")
    icp_params = cfg_registrator.get("icp_params")
    robust_estimator_params = cfg_registrator.get("robust_estimator_params")
    if transform_params is not None:
        transform = instantiate_transforms(transform_params)
    else:
        transform = None
    if icp_params is not None:
        icp = instantiate_icp(icp_params)
    else:
        icp = None
    if robust_estimator_params is not None:
        robust_estimator = instantiate_robust_estimator(robust_estimator_params)
    else:
        robust_estimator = None
    params = OmegaConf.masked_copy(cfg_registrator, list_keys)
    return reg_cls(robust_estimator=robust_estimator, icp=icp, transform=transform, **params)
