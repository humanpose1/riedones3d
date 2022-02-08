import numpy as np
import os
import pyrender
import trimesh
import torch
import matplotlib.pyplot as plt
from point_cloud.utils import eulerAnglesToRotationMatrix

def save_image(xyz, norm, color, path_image, pose=np.eye(4), t=50):
    """
    Use pyrender to save an image of the pointcloud
    input:
    xyz numpy array N x 3 point cloud
    norm: numpy array N x 3 normal
    color: numpy array N x 3 color between 0 and 1 of the point cloud
    path_image: str path where we save the image
    pose: pose of the scene
    """
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    mesh = pyrender.Mesh.from_points(xyz, normals=norm, colors=color)

    centroid = np.median(xyz, axis=0)
    T = np.eye(4)
    T[:3, 3] = pose[:3, :3].dot(centroid) + pose[:3, 3]
    T[2, 3] += t

    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0],
                           bg_color=[1.0, 1.0, 1.0])
    scene.add(mesh, pose=pose)
    oc = pyrender.OrthographicCamera(xmag=35.0, ymag=35.0)
    scene.add(oc, pose=T)

    # pyrender.Viewer(scene, use_raymond_lighting=False)
    r = pyrender.OffscreenRenderer(500, 500, point_size=0.1)

    color, depth = r.render(scene)
    r.delete()
    plt.imsave(path_image, color)


def render_offline_mesh(path, pose=np.eye(4)):

    # os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    fuze_trimesh = trimesh.load(path)

    z = np.array([0, 0, 1])
    intensity = (np.abs(fuze_trimesh.vertex_normals.dot(z))**25).reshape(-1, 1)
    # intensity = np.ones(len(fuze_trimesh.vertices)).reshape(-1, 1)
    vertices = fuze_trimesh.vertices

    one_color = intensity.dot(np.array([0.95, 0.75, 0.1]).reshape(1, 3))
    fuze_trimesh.visual.vertex_colors = one_color

    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    # centroid = mesh.centroid
    centroid = np.median(vertices, axis=0)
    std = np.std(vertices, axis=0)
    T = np.eye(4)
    T[:3, 3] = pose[:3, :3].dot(centroid) + pose[:3, 3]
    T[2, 3] += 30

    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0],
                           bg_color=[1.0, 1.0, 1.0])
    scene.add(mesh, pose=pose)
    oc = pyrender.OrthographicCamera(xmag=15.0, ymag=15.0)
    scene.add(oc, pose=T)

    # pyrender.Viewer(scene, use_raymond_lighting=False)
    r = pyrender.OffscreenRenderer(600, 600, point_size=0.1)
    color, depth = r.render(scene)
    r.delete()
    return color, depth, oc.get_projection_matrix()

def render_online_mesh(path, T=np.eye(4), run_in_thread=True):
    fuze_trimesh = trimesh.load(path)

    z = np.array([0, 0, 1])
    intensity = (np.abs(fuze_trimesh.vertex_normals.dot(z))**25).reshape(-1, 1)
    # intensity = np.ones(len(fuze_trimesh.vertices)).reshape(-1, 1)
    vertices = fuze_trimesh.vertices

    one_color = intensity.dot(np.array([0.95, 0.75, 0.1]).reshape(1, 3))
    fuze_trimesh.visual.vertex_colors = one_color

    mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
    # centroid = mesh.centroid
    centroid = np.median(vertices, axis=0)
    std = np.std(vertices, axis=0)
    print(std)
    T = np.eye(4)
    T[:3, 3] = centroid
    T[2, 3] += 30
    scene = pyrender.Scene()
    scene.add(mesh)
    oc = pyrender.OrthographicCamera(xmag=15.0, ymag=15.0)
    scene.add(oc, pose=T)
    return scene, pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=run_in_thread)

def rotate_meshes(angle, scene, viewer):
    """
    rotate the mesh wrt the euler angle.
    """
    node = [n for n in scene.nodes if n.mesh is not None][0]
    T = np.eye(4)
    R = eulerAnglesToRotationMatrix(torch.tensor(angle)).detach().cpu().numpy()
    t = (np.eye(3) - R).dot(np.median(node.mesh.primitives[0].positions, axis=0))
    T[:3, :3] = R
    T[:3, 3] = t
    viewer.render_lock.acquire()
    scene.set_pose(node, T)
    viewer.render_lock.release()
    return T

if __name__ == "__main__":

    path = '../../../dataset_archeology/'
    '3D/pièces de monnaies liffré/'
    'Droits/L372D.stl'
    render_online_mesh(path)
