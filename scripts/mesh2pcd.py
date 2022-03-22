import open3d
import argparse
import numpy as np
import os
import os.path as osp
import pathlib
import glob
import torch

from torch_geometric.data import Data

from point_cloud.io import read_mesh
from point_cloud.visu import torch2o3d
from point_cloud.pre_transforms import RotateToAxis

def parse_args():
    parser = argparse.ArgumentParser("save coins in ply format (orient it)")

    parser.add_argument('--path_coin',
                        dest='path_coin',
                        help='path cointaining the coins in stl or ply',
                        nargs='+')
    parser.add_argument('-o',
                        dest='path_output',
                        help='path of the res')
    parser.add_argument("--no-keep-folder", dest="is_keep_folder", help="do we keep the name of the last folders ?", action="store_false")
    parser.add_argument("--keep-folder", dest="is_keep_folder", help="do we keep the name of the last folders ?", action="store_true")

    parser.add_argument("--no-orient_pcd", dest="orient_pcd", help="no orientation using PCA", action="store_false")
    parser.add_argument("--orient_pcd", dest="orient_pcd", help="orientation using PCA", action="store_true")

    parser.set_defaults(is_keep_folder=True)
    parser.set_defaults(orient_pcd=True)
    args = parser.parse_args()
    return args

def orient_to_z(data):
    orienter = RotateToAxis()
    data = orienter(data)
    return data

def torch2mesh(data):
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(data.pos.detach().cpu().numpy())
    mesh.vertex_normals = open3d.utility.Vector3dVector(data.norm.detach().cpu().numpy())
    mesh.triangles = open3d.utility.Vector3iVector(data.faces.detach().cpu().numpy())
    return mesh


def main():
    args = parse_args()
    for path_coin in args.path_coin:
        data = read_mesh(path_coin)
        if args.orient_pcd:
            data = orient_to_z(data)
        pcd = torch2mesh(data)
        path_res_coin, namefile = osp.split(path_coin)
        name = namefile.split(".")[0]
        if args.path_output is not None:
            if args.is_keep_folder:
                name_folder = osp.split(osp.split(path_coin)[0])[1]
                path_res_coin = osp.join(args.path_output, name_folder)
            else:
                path_res_coin = args.path_output
            pathlib.Path(path_res_coin).mkdir(exist_ok=True, parents=True)
        filename = osp.join(path_res_coin, name + ".ply")
        open3d.io.write_triangle_mesh(filename, pcd)


if __name__ == "__main__":
    main()

