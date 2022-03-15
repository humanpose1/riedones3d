import argparse
import glob
import numpy as np
import open3d
import os
import os.path as osp
import joblib
import json
import sys
import torch
import time

import matplotlib.pyplot as plt
import seaborn as sns


from torch_geometric.data import Data
from torch_geometric.transforms import Compose


from torch_points3d.core.data_transform import Random3AxisRotation, SaveOriginalPosId
from torch_points3d.core.data_transform import GridSampling3D, AddFeatByKey, AddOnes
from torch_points3d.applications.pretrained_api import PretainedRegistry
from torch_points3d.utils.registration import get_matches

from point_cloud.pre_transforms import BendingByPolynomialRegression, RotateToAxis
from point_cloud.transforms import RadiusPatchExtractor, MultiRadiusPatchExtractor
from point_cloud.ICP import BaseICP
from point_cloud.utils import eulerAnglesToRotationMatrix
from point_cloud.io import read_pcd
from point_cloud.deep_registration import DeepRegistrationPipeline

from point_cloud.error_mesurer import PCDistance, NormalDistance, IoU

from point_cloud.visu import torch2o3d, colorizer, colorizer_v2


from point_cloud.robust_estimator import SimpleFGREstimator
from point_cloud.robust_estimator import TeaserEstimator
from point_cloud.robust_estimator import RansacEstimator

sns.set()
ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
sys.path.insert(0, ROOT)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def parse_args():

    parser = argparse.ArgumentParser("evaluate the whole pipeline"
                                     "1) point cloud registration using FCGF(or variant) and ICP to refine results)"
                                     "2) compute the point 2 point distance and compute the histogram"
                                     "3) use the logistic regression to make the decision")

    parser.add_argument('--path', dest='path_coin',
                        help='path coin',
                        type=str, nargs='+')
    parser.add_argument('--path_scaler', dest='path_scaler',
                        help='path of mean and std',
                        type=str)
    parser.add_argument('--min_num_matches', dest="min_num_matches", type=int, default=20)
    parser.add_argument('--num_points', dest="num_points", type=int, default=5000)
    parser.add_argument('-m', dest='path_model', help="path of the model")
    parser.add_argument('--angle', dest='angle', help="rotation angle", default=0.0, type=float)
    parser.add_argument('--deg', dest='deg', help="polynomial degree for regression", default=0, type=int)
    parser.add_argument('--alpha', dest='alpha', help="alpha for ridge for regression", default=0.0, type=float)
    parser.add_argument('--trans', dest='trans', help="translation", default=20.0, type=float)
    parser.add_argument('--clf', dest="path_clf", help="path of the classifier")
    parser.add_argument('--est', dest='robust_estimator',
                        help='robust estimator (either ransac, fgr or teaser) ',
                        type=str, default="teaser")
    
    args = parser.parse_args()
    return args


def registrate(s, t, args, visu=True):
    source = s.clone()
    target = t.clone()
    t0 = time.time()
    
    print("read pointcloud: {}".format(time.time()-t0))
    if(visu):
        pcd_s = torch2o3d(source, color=[0.9,0.7,0.1])
        pcd_t = torch2o3d(target, color=[0.1,0.9,0.7])
        open3d.visualization.draw_geometries([pcd_s, pcd_t])
    t0 = time.time()
    ts = Compose([MultiRadiusPatchExtractor(radius=2, max_num=1000)])
    tt = Compose([MultiRadiusPatchExtractor(radius=4, max_num=1000)])
    icp = BaseICP(
        mode="plane", num_iter=10, stopping_criterion=1e-3,
        max_dist=0.1, pre_transform_source=ts,
        pre_transform_target=tt, is_debug=False)

    transform = Compose(
        [
            SaveOriginalPosId(),
            GridSampling3D(mode="last", size=0.1, quantize_coords=True),
            AddOnes(),
            AddFeatByKey(add_to_x=True, feat_name="ones"),
        ]
    )
    if args.robust_estimator == "teaser":
        estimator = TeaserEstimator(0.5,  max_clique_time_limit=0.5)
    elif args.robust_estimator == "fgr":
        estimator = SimpleFGREstimator()
    elif args.robust_estimator == "ransac":
        estimator = RansacEstimator(distance_threshold=0.5, num_iterations=10000)
    else:
        raise NotImplementedError
    

    registrator = DeepRegistrationPipeline(path_model=args.path_model,
                                           num_points=args.num_points,
                                           max_norm=0.5,
                                           transform=transform,
                                           icp=icp,
                                           min_num_matches=args.min_num_matches,
                                           robust_estimator=estimator)
    print("load model: {}".format(time.time()-t0))
    t0 = time.time()
    T_f = registrator.registrate(source, target)
    print(T_f.detach().cpu().numpy())
    
    print("registrate: {}".format(time.time()-t0))
    if(visu):
        pcd_s.transform(T_f.detach().cpu().numpy())
        open3d.visualization.draw_geometries([pcd_s, pcd_t])
    return T_f


def main():

    args = parse_args()
    list_path = []
    for path in args.path_coin:
        p = glob.glob(path)
        list_path = list_path + p
    print(len(list_path))
    assert(len(list_path) >= 2)
    source = read_pcd(list_path[0])
    target = read_pcd(list_path[1])
    if (args.deg > 0):
        bender = BendingByPolynomialRegression(deg=args.deg, alpha=args.alpha)
        rotator = RotateToAxis() 
        source = bender(rotator(source))
        target = bender(rotator(target))
    
    rand_theta = torch.zeros(3)
    
    rand_theta[2] = args.angle * np.pi / 180.0
    t = torch.tensor([args.trans, 0, 0]).float()

    R = eulerAnglesToRotationMatrix(rand_theta)
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    R = T[:3, :3]
    t = T[:3, 3]

    target.pos = target.pos @ R.T + t
    target.norm = target.norm @ R.T

    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print("Step 1) Registration")
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    T_f = registrate(source, target, args, visu=True)
    np.savetxt("temp.txt", T_f.cpu().numpy())

    source.pos = source.pos @ T_f[:3, :3].T + T_f[:3, 3]
    source.norm = source.norm @ T_f[:3, :3].T
    pcd_s = torch2o3d(source)
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print("Step 2) Point2Point distance")
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    rang = (0, 0.6)
    distance_computer = PCDistance(max_dist=0.6, thresh_normal_max=1, thresh_normal_min=0.3, rang=rang, transfo=None)
    hist, dist_map, bin_edges = distance_computer.compute_histogram(source, target)


    distance_computer = PCDistance(max_dist=0.6, thresh_normal_max=1, thresh_normal_min=0.0, rang=rang, transfo=None)
    distance_computer.visualize_color(source, target)
    distance_computer.visualize_color(target, source)
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    print("Step 3) classification")
    print("--------------------------------------------------------------------------------")
    print("--------------------------------------------------------------------------------")
    with open(args.path_scaler, "r") as f:
        dico_scaler = json.load(f)
    mean = np.asarray(dico_scaler['mean'])
    std = np.asarray(dico_scaler['std'])
    plt.clf()
    plt.bar(bin_edges[:-1], hist, width=rang[1]/70, color=colorizer_v2(bin_edges[:-1]))
    # plt.xlim(min(bin_edges), max(bin_edges))
    plt.show()
    x = (hist - mean)/std
    plt.plot(x)
    plt.show()
    clf = joblib.load(args.path_clf)
    prob = clf.predict_proba([x])
    print(prob)

if __name__ == "__main__":
    main()
