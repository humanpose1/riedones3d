import open3d
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import os.path as osp
import pathlib
import trimesh
import pyrender
import sys

from point_cloud.render import render_offline_mesh




ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_ply(path, gamma=25, color=[0.9, 0.7, 0.1]):
    fuze_trimesh = trimesh.load(path)
    z = np.array([0, 0, 1])
    intensity = (np.abs(fuze_trimesh.vertex_normals.dot(z))**gamma).reshape(-1, 1)
    # intensity = np.ones(len(fuze_trimesh.vertices)).reshape(-1, 1)
    vertices = fuze_trimesh.vertices
    colors = intensity.dot(np.array(color).reshape(1, 3))
    normals = fuze_trimesh.vertex_normals
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(vertices)
    pcd.normals = open3d.utility.Vector3dVector(normals)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd




def parse_args():
    parser = argparse.ArgumentParser("save all coin in image")

    parser.add_argument('--path_coin',
                        dest='path_coin',
                        help='path cointaining the coin. Warning, it only works with stl')
    parser.add_argument('--path_tr',
                        dest='path_tr',
                        help='path cointaining the transfo')
    parser.add_argument('--path_graph',
                        dest='path_graph',
                        help='path containing the graph')
    parser.add_argument('--path_output',
                        dest='path_output',
                        help='where we store the images, the depth projection matrix and pose')
    parser.add_argument('-t', dest='thresh',
                        help='threshold of probability', default=0.80,
                        type=float)
    parser.add_argument('--clustered', dest='clustered',
                        help='save every images  with its clustered part', action='store_true')
    parser.add_argument('--no-clustered', dest='clustered',
                        help='do not save every images  with its clustered part', action='store_false')

    parser.add_argument('--save-3d', dest="save_3d", help="store 3D data (mesh and ply)", action='store_true')
    parser.add_argument('--no-save-3d', dest="save_3d", help="do not store 3D data (mesh and ply)", action='store_false')
    parser.set_defaults(clustered=True)
    parser.set_defaults(save_3d=True)
    args = parser.parse_args()
    return args

class LinkException(Exception):
    """
    exception raise when a link is missing in the graph
    """
    def __init__(self, message: str):
        self.message: str = message
        super().__init__(self.message)
    def __str__(self) -> str:
        return str(self.message)

def is_np_file(path_tr):
    return path_tr[-3:] == "npy"
        

def get_final_transfo_from_numpy_file(path, path_tr):
    T_res = np.eye(4)
    assert len(path) > 1
    dico_trans = np.load(path_tr, allow_pickle=True).item()
    for i in range(len(path)-1):
        reverse = False
        key = "{}_{}".format(path[i], path[i+1])
        if key not in dico_trans.keys():
            reverse = True
            key = "{}_{}".format(path[i+1], path[i])
            if key not in dico_trans.keys():
                raise LinkException(f"WARNING: BAD TRANSFORMATION FOR THE path {path[i+1]} and {path[i]}")
        T = dico_trans[key]
        if(reverse):
            T = np.linalg.inv(T)
        T_res = T.dot(T_res)
    return np.linalg.inv(T_res)


def extract_graph(filename='results/graph.json', thresh=0.45):
    with open(filename, 'r') as f:
        data = json.load(f)

    old_graph = nx.readwrite.json_graph.node_link_graph(data)

    remain_edge = [e for e in old_graph.edges() if old_graph[e[0]][e[1]]['weight']>thresh]
    graph = nx.Graph()
    graph.add_nodes_from(old_graph.nodes())
    graph.add_edges_from(remain_edge)
    return old_graph, graph

class PoseComputer:
    def __init__(self, graph, path_transfo):
        self.graph = graph
        self.path_transfo = path_transfo

    def get_pose_from_graph(self, first_coin, current_coin):
        shortest = nx.shortest_path(self.graph, source=first_coin, target=current_coin)
        T_0 = np.eye(4)
        if(len(shortest) > 1):
            T_f = get_final_transfo_from_numpy_file(shortest, self.path_transfo)
            pose = T_0.dot(T_f)
        else:
            pose = T_0
        return pose
    


class FileSaver:

    def __init__(self, path_dir_coin, pose_computer, save_3d):
        self.path_dir_coin = path_dir_coin
        self.save_3d = save_3d
        self.pose_computer = pose_computer
        self.path_res_coin = None

    def set_path_res_coin(self, path_res_coin):
        self.path_res_coin = path_res_coin

    def get_path_coin(self, coin):
        path_coin = None
        for ext in ['STL', 'stl', 'PLY', 'ply']:
            path_coin = osp.join(self.path_dir_coin, coin+"."+ext)
            if(osp.exists(path_coin)):
                return path_coin
        if not osp.exists(path_coin):
            print("WARNING this file does not exist")
            return None
        return path_coin

    def save_color(self, coin, pose):
        path_coin = self.get_path_coin(coin)
        color, depth, projection = render_offline_mesh(path_coin, pose=pose)
        pathlib.Path(osp.join(self.path_res_coin, "color")).mkdir(
            parents=True, exist_ok=True)
        path_color = osp.join(self.path_res_coin, 'color',
                              '{}_color.png'.format(coin))
        plt.imsave(path_color, color)

    def save_stl(self, coin, pose):
        path_coin = self.get_path_coin(coin)
        pathlib.Path(osp.join(self.path_res_coin, "mesh")).mkdir(
            parents=True, exist_ok=True)
        path_mesh = osp.join(self.path_res_coin, 'mesh',
                             '{}.stl'.format(coin))
        stl = open3d.io.read_triangle_mesh(path_coin)
        stl.transform(pose)
        stl.compute_vertex_normals()
        open3d.io.write_triangle_mesh(path_mesh, stl)

    def save_ply(self, coin, pose, color=[0.9, 0.7, 0.1]):
        path_coin = self.get_path_coin(coin)
        pathlib.Path(osp.join(self.path_res_coin, "pcd")).mkdir(
            parents=True, exist_ok=True)
        path_ply = osp.join(self.path_res_coin, 'pcd',
                            '{}.ply'.format(coin))
        ply = get_ply(path_coin, gamma=25, color=color)
        ply.transform(pose)
        open3d.io.write_point_cloud(path_ply, ply)

    def save_all(self, sorted_group):
        for coin in list(sorted_group):
            print(coin)
            pose = self.pose_computer.get_pose_from_graph(sorted_group[0], coin)
            self.save_color(coin, pose)
            if self.save_3d:
                self.save_ply(coin, pose)
                self.save_stl(coin, pose)

def main():
    args = parse_args()
    old_graph, graph = extract_graph(filename=args.path_graph,
                                     thresh=args.thresh)
    groups = list(nx.connected_components(graph))
    pose_computer = PoseComputer(graph, args.path_tr)
    filesaver = FileSaver(args.path_coin, pose_computer, args.save_3d)
    for ind, group in enumerate(groups):
        if(args.clustered):
            path_res_coin = osp.join(args.path_output, 'coin_{:04d}'.format(ind))
        else:
            path_res_coin = args.path_output
        pathlib.Path(path_res_coin).mkdir(parents=True, exist_ok=True)
        sorted_group = sorted(group)
        filesaver.set_path_res_coin(path_res_coin)
        filesaver.save_all(sorted_group)
        
if __name__ == "__main__":
    main()
