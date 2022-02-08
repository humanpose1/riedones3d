import argparse
import torch
import time
import os
import os.path as osp
import pandas as pd
import pathlib
import tqdm

from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torch_points3d.core.data_transform import SaveOriginalPosId
from torch_points3d.core.data_transform import GridSampling3D, AddFeatByKey, AddOnes
from torch_points3d.applications.pretrained_api import PretainedRegistry

from point_cloud.io import read_pcd

def generate_list_path(path_coin, df):

    list_path = []
    for name in df["name"]:
        path = osp.join(path_coin, name + ".ply")
        list_path.append(path)
    return list_path


def parse_args():
    parser = argparse.ArgumentParser("compute feature and save points"
                                     "")
    parser.add_argument('--path_coin', dest='path_coin',
                        help='path coin',
                        type=str)
    parser.add_argument('--path_output', dest='path_output',
                        help='path of the output',
                        type=str)
    parser.add_argument('--list_coin', dest='list_coin',
                        help='list containing the name of the coin',
                        type=str)
    parser.add_argument('-m',
                        dest='path_model',
                        help="path of the model",
                        type=str)
    parser.add_argument('--name', dest='name',
                        help='name of the file',
                        type=str, default="test")
    parser.add_argument('--device', dest='device',
                        help='device cuda or cpu',
                        type=str, default="cuda")
    args = parser.parse_args()
    return args

def main(args):
    """
    main loop for computing features (without a dataloader)
    """
    df = pd.read_csv(args.list_coin)
    list_path = generate_list_path(args.path_coin, df)
    path_res = osp.join(args.path_output, args.name)
    transform = Compose(
        [
            SaveOriginalPosId(),
            GridSampling3D(mode="last", size=0.1, quantize_coords=True),
            AddOnes(),
            AddFeatByKey(add_to_x=True, feat_name="ones"),
        ]
    )
    prop = {"feature_dimension": 1}
    model = PretainedRegistry.from_file(args.path_model, mock_property=prop).to(args.device)
    start = time.time()
    for i in range(len(list_path)):
        # read pcd
        print(list_path[i])
        data = transform(read_pcd(list_path[i]))
        with torch.no_grad():
            model.set_input(data, args.device)
            output_s = model.forward()
        data.feat = output_s
        path_output = osp.join(path_res, "feature")
        pathlib.Path(path_output).mkdir(parents=True,
                                        exist_ok=True)

        name = osp.split(list_path[i])[1].split(".")[0] + ".pt"
        torch.save(data.cpu(), osp.join(path_output, name))
    end = time.time()
    print("It takes {}s to compute features".format(end - start))


if __name__ == "__main__":
    args = parse_args()
    main(args)
