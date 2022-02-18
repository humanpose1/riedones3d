import argparse
from functools import partial
import torch
import time
import os
import os.path as osp
import pandas as pd
import pathlib
import tqdm
import numpy as np
from joblib import Parallel, delayed


from torch_geometric.data import Data
from torch_geometric.transforms import Compose

from point_cloud.transforms import MultiRadiusPatchExtractor
from point_cloud.base_registration import PrecomputedFeatureBasedRegistrator
from point_cloud.ICP import BaseICP
from point_cloud.error_mesurer import PCDistance
from point_cloud.robust_estimator import SimpleFGREstimator
from point_cloud.robust_estimator import TeaserEstimator
from point_cloud.robust_estimator import RansacEstimator


def generate_list_path(directory, dataframe):

    list_path = []
    for name in dataframe["name"]:
        path = osp.join(directory, name + ".pt")
        list_path.append(path)
    return list_path


def parse_args():
    parser = argparse.ArgumentParser("compute transformations"
                                     "it is a bit naive because it will compare every pairs")
    parser.add_argument('--path_feature', dest='path_feature',
                        help='path of the feature .pt files (Data(feat=..., pos=...))',
                        type=str)
    parser.add_argument('--path_output', dest='path_output',
                        help='path of the output',
                        type=str)
    parser.add_argument('--list_coin', dest='list_coin',
                        help='list containing the name of the coin a csv',
                        type=str)
    parser.add_argument('--est', dest='robust_estimator',
                        help='robust estimator (either ransac, fgr or teaser) ',
                        type=str, default="teaser")
    parser.add_argument('--num_points', dest='num_points',
                        help='number of points for estimation ',
                        type=int, default=5000)
    parser.add_argument('--n_jobs', dest="n_jobs", help="number of job for parallel computing", type=int, default=1)
    parser.add_argument('--sym', dest='sym',
                        help='symetric histogram',
                        action='store_true')
    parser.add_argument('--no-sym', dest='sym',
                        help='no symetric histogram',
                        action='store_false')
    parser.set_defaults(sym=False)

    args = parser.parse_args()
    return args

def registrate_and_hist(i, j, list_path, distance_computer, registrator, sym=False):
    name_s = osp.split(list_path[i])[1].split(".")[0]
    name_t = osp.split(list_path[j])[1].split(".")[0]
    name = name_s + "_" + name_t
    print(name)
    data_s = torch.load(list_path[i])
    data_t = torch.load(list_path[j])
    T_f = registrator.registrate(data_s, data_t)
    data_s.pos = data_s.pos @ T_f[:3, :3].T + T_f[:3, 3]
    data_s.norm = data_s.norm @ T_f[:3, :3].T
    if(sym):
        hist = distance_computer.compute_symmetric_histogram(data_s, data_t)
    else:
        hist, _, _ = distance_computer.compute_histogram(data_s, data_t)
    return T_f.cpu().numpy(), hist, name

def sequential_loop_for_trans_and_hist(list_path, loop_func):
    list_transfo = dict()
    list_hist = dict()
    for i in range(len(list_path)):
        for j in range(len(list_path)):
            if(i < j):
                T_f, hist, name = loop_func(i, j)
                list_transfo[name] = T_f
                list_hist[name] = hist
    return list_transfo, list_hist

def parallel_loop_for_trans_and_hist(list_path, loop_func, n_jobs):
    list_transfo = dict()
    list_hist = dict()
    res = Parallel(n_jobs=n_jobs)(delayed(loop_func)(i, j) for i in range(len(list_path)) for j in range(i+1, len(list_path)))
    for T_f, hist, name in res:
        list_transfo[name] = T_f
        list_hist[name] = hist
    return list_transfo, list_hist

def save_final_results(list_transfo, list_hist, path_output):
    path_output_transformation = osp.join(path_output, "transformation")
    pathlib.Path(path_output_transformation).mkdir(parents=True,
                                    exist_ok=True)
    np.save(osp.join(path_output, "transfo.npy"), list_transfo)
    np.save(osp.join(path_output, "hist.npy"), list_hist)
  

def main(args):
    
    dataframe = pd.read_csv(args.list_coin)
    list_path_feature = generate_list_path(args.path_feature, dataframe)
    
    start = time.time()
    icp = BaseICP(
        mode="plane", num_iter=10, stopping_criterion=1e-3,
        max_dist=0.1,
        pre_transform_source=Compose([MultiRadiusPatchExtractor(radius=2, max_num=1000)]),
        pre_transform_target=Compose([MultiRadiusPatchExtractor(radius=4, max_num=1000)]),
        is_debug=False
        )
    if args.robust_estimator == "teaser":
        estimator = TeaserEstimator(noise_bound=0.5,  max_clique_time_limit=0.1)
    elif args.robust_estimator == "fgr":
        estimator = SimpleFGREstimator()
    elif args.robust_estimator == "ransac":
        estimator = RansacEstimator(distance_threshold=0.5, num_iterations=10000)
    else:
        raise NotImplementedError
    registrator = PrecomputedFeatureBasedRegistrator(
        max_norm=0.5,
        transform=None,
        icp=icp,
        robust_estimator=estimator,
        noise_bound_teaser=0.1,
        verbose=False,
        min_num_matches=3,
        num_points=args.num_points)
    distance_computer = PCDistance(
        max_dist=0.6,
        thresh_normal_max=1,
        thresh_normal_min=0.0,
        rang=(0, 0.6), transfo=None)
    
    start = time.time()
    loop_func = partial(registrate_and_hist,
                        list_path=list_path_feature,
                        registrator=registrator,
                        distance_computer=distance_computer,
                        sym=args.sym)
    n_jobs = args.n_jobs
    if n_jobs == 1:
        list_transfo, list_hist = sequential_loop_for_trans_and_hist(list_path_feature, loop_func)
    else:
        list_transfo, list_hist = parallel_loop_for_trans_and_hist(list_path_feature, loop_func, n_jobs)
    end = time.time()
    print("It takes {}s to registrate".format(end - start))
    save_final_results(list_transfo, list_hist, args.path_output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
