from plyfile import PlyData

import os
import os.path as osp
import numpy as np
import torch
import trimesh

from torch_geometric.data import Data, Batch


def read_mesh_vertices_ply(filename):
    """ read XYZ for each vertex.
    """
    assert os.path.isfile(filename)
    normals = None
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count

        vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]

        try:
            normals = np.zeros(shape=[num_verts, 3], dtype=np.float32)
            normals[:, 0] = plydata["vertex"].data["nx"]
            normals[:, 1] = plydata["vertex"].data["ny"]
            normals[:, 2] = plydata["vertex"].data["nz"]
        except Exception:
            normals = None

    return vertices, normals

def read_mesh_vertices_stl(filename):

    fuze_trimesh = trimesh.load(filename)
    vertices = fuze_trimesh.vertices
    normals = fuze_trimesh.vertex_normals
    return vertices, normals

def read_pcd(filename):
    parent, f = osp.split(filename)
    ext = f.split(".")[1]
    if(ext.lower() == "ply"):
        vertices, normals = read_mesh_vertices_ply(filename)
    elif (ext.lower() == "stl"):
        vertices, normals = read_mesh_vertices_stl(filename)
    else:
        raise NotImplementedError
    pos = torch.from_numpy(vertices)
    if(normals is not None):
        norm = torch.from_numpy(normals)
    else:
        norm = None
    batch = torch.zeros(len(pos)).long()
    data = Batch(pos=pos, norm=norm, batch=batch).to(torch.float)
    return data
