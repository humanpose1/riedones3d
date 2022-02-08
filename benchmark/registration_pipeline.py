try:
    import handcrafted_descriptor
except ImportError:
    pass
import open3d as o3d
import torch
import hydra
import pandas as pd
from omegaconf import OmegaConf
import os
import os.path as osp
import time

from point_cloud.visu import torch2o3d
from point_cloud import instantiate_registrator

from torch_points3d.datasets.dataset_factory import instantiate_dataset
from torch_points3d.models import model_interface

from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq

from torch_points3d.utils.registration import estimate_transfo


class MockModel(model_interface.DatasetInterface):
    def __init__(self, conv_type="SPARSE"):
        self._conv_type = conv_type

    @property
    def conv_type(self):
        return self._conv_type


def main_pipeline(dataset, registrator, name="Test"):
    # instantiate dataset
    # for loop
    # list of dict with name of scene and name of coin

    loader = dataset.test_dataloaders[0] 
    list_res = []
    with Ctq(loader) as tq_test_loader:
        for i, data in enumerate(tq_test_loader):
            s, t = data.to_data()

            name_scene, name_pair_source, name_pair_target = dataset.test_dataset[0].get_name(i)
            res = dict(name_scene=name_scene, name_pair_source=name_pair_source, name_pair_target=name_pair_target)
            metric = registrator.evaluate_pair(data)
            res = dict(**res, **metric)
            list_res.append(res)
            tq_test_loader.set_postfix(**res)

    df = pd.DataFrame(list_res)
    output_path = os.path.join(name, registrator.__class__.__name__)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    df.to_csv(osp.join(output_path, "final_res_{}.csv".format(time.strftime("%Y%m%d-%H%M%S"))))
    print(df.groupby("name_scene").mean())

@hydra.main(config_path="conf", config_name="config")
def run(cfg):
    OmegaConf.set_struct(cfg, False)

    # cfg_data = OmegaConf.load("/media/admincaor/DataHDD2To/mines/code/deeppointcloud-benchmarks/conf/data/registration/testliffre.yaml").data
    # cfg_data.dataroot = "/media/admincaor/DataHDD2To/mines/code/deeppointcloud-benchmarks/data"
    cfg_data = cfg.data
    dataset = instantiate_dataset(cfg_data)
    dataset.create_dataloaders(
        MockModel(), 1, False, cfg.num_workers, False,
    )
    registrator = instantiate_registrator(cfg.registrator)
    main_pipeline(dataset, registrator, cfg_data.name)


if __name__ == "__main__":
    run()
