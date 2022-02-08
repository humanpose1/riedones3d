import open3d
import numpy as np
import torch
import matplotlib.pyplot as plt

def colorise_flat_pc(data, power=25, color=[1,1,0], ax=[0, 0, 1]):
    c = torch.tensor(color).to(data.pos).unsqueeze(0)
    ez = torch.tensor(ax).to(data.pos).unsqueeze(-1)
    gray = torch.abs(data.norm @ ez)**25
    color_g = gray @ c
    data.colors = color_g
    return data



def torch2o3d(data, color=[1, 1, 0], is_flat=True):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(data.pos.detach().cpu().numpy())
    if(data.norm is not None):
        pcd.normals = open3d.utility.Vector3dVector(data.norm.detach().cpu().numpy())
        if(is_flat):
            data = colorise_flat_pc(data, color=color)
            pcd.colors = open3d.utility.Vector3dVector(data.colors.detach().cpu().numpy())
        else:
            pcd.paint_uniform_color(color)
    else:
        pcd.paint_uniform_color(color)

    return pcd


def colorizer(field, color1=[237/255, 248/255, 177/255],
              color2=[44/255,127/255,184/255],
              color3=None,
              gamma=0.4, max_f=2.0):
    """
    input scalar field for each point cloud
    output: a color from color1 to color2
    gamma : when the error is equal to gamma the point is yellow
    """

    field_norm = (field - 0)/(max_f-0) # according to Matteo
    field_norm[field_norm > 1] = 1
    field_norm = field_norm
    field_norm = field_norm.T.dot(np.ones((1, 3)))
    c1 = np.ones((field.shape[0], 1)).dot(np.asarray([color1]))
    c2 = np.ones((field.shape[0], 1)).dot(np.asarray([color2]))
    if(color3 is None):
        final_color = (1-field_norm) * c1 + field_norm * c2
    else:
        c3 = np.ones((field.shape[0], 1)).dot(np.asarray([color3]))
        coef1 = field_norm/gamma
        coef2 = (field_norm-gamma)/(1-gamma)
        final_color = ((1-coef1) * c1 + coef1 * c2 ) * (field_norm<=gamma) + \
            ((1-coef2) * c2 + coef2*c3 )*(field_norm>gamma)
    return final_color


def colorizer_v2(field, inter_dist=0.4):
    """
    input: scalar field
interdist (put 0.5 at interdistt)
    output color with respect to the size
    """
    alpha = np.log(0.5)/np.log(inter_dist)
    return plt.cm.Spectral_r(field ** alpha)[:, :3]
