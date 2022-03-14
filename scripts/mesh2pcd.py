import open3d
import argparse
import numpy as np
import os
import os.path as osp
import pathlib
import glob

from point_cloud.io import read_mesh_vertices

def parse_args():
    parser = argparse.ArgumentParser("save coins in point cloud in ply format")

    parser.add_argument('--path_coin',
                        dest='path_coin',
                        help='path cointaining the coins in stl or ply',
                        nargs='+')
    parser.add_argument('-o',
                        dest='path_output',
                        help='path of the res')
    parser.add_argument("--no-keep-folder", dest="is_keep_folder", help="do we keep the name of the last folders ?", action="store_false")
    parser.add_argument("--keep-folder", dest="is_keep_folder", help="do we keep the name of the last folders ?", action="store_true")
    parser.set_defaults(is_keep_folder=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    for path_coin in args.path_coin:
        vertices, normals = read_mesh_vertices(path_coin)
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(vertices)
        pcd.normals = open3d.utility.Vector3dVector(normals)
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
        open3d.io.write_point_cloud(filename, pcd)


if __name__ == "__main__":
    main()

