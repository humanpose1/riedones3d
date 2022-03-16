import argparse
import time
import os
import os.path as osp
import pandas as pd
import pathlib
import numpy as np

from point_cloud.robust_estimator import build_estimator
from point_cloud.pipeline_visualizer import PyRenderVisualizer
from point_cloud.pipeline_visualizer import BaseVisualizer
from point_cloud.pipeline import PretrainedRiedonesPipeline
from point_cloud.pipeline import ParallelPipeline

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
    parser.add_argument('--trans', dest='trans',
                        help='translation of initiialisation()',
                        type=float, default=20.0)
    parser.add_argument('--n_jobs', dest="n_jobs", help="number of job for parallel computing", type=int, default=1)

    parser.add_argument('--sym', dest='sym',
                        help='symetric histogram',
                        action='store_true')
    parser.add_argument('--no-sym', dest='sym',
                        help='no symetric histogram',
                        action='store_false')

    parser.add_argument('--save_image', dest='save_image',
                        help='save the images of registration in output directory',
                        action='store_true')
    parser.add_argument('--np-save_image', dest='save_image',
                        help='do not save the images of registration in output directory',
                        action='store_false')
    
    parser.set_defaults(sym=False)
    parser.set_defaults(save_image=False)

    args = parser.parse_args()
    return args


  

def main(args):
    
    dataframe = pd.read_csv(args.list_coin)
    list_path_feature = generate_list_path(args.path_feature, dataframe)
    
    start = time.time()
    estimator = build_estimator(args.robust_estimator,
                                noise_bound=0.5,
                                max_clique_time_limit=0.5,
                                distance_threshold=0.5,
                                num_iterations=10000
    )
    if args.save_image:
        visualizer = PyRenderVisualizer(path_output=args.path_output, translate=args.trans)
    else:
        visualizer = BaseVisualizer(translate=args.trans)
    pipeline = PretrainedRiedonesPipeline(
        estimator=estimator,
        visualizer=visualizer,
        num_points=args.num_points,
        sym=args.sym)
    if args.n_jobs > 1:
        pipeline = ParallelPipeline(pipeline=pipeline, n_jobs=args.n_jobs)
    pipeline.compute_all(list_path_feature)
    pipeline.save_final_results(args.path_output)
    end = time.time()
    print("It takes {}s to registrate".format(end - start))

if __name__ == "__main__":
    args = parse_args()
    main(args)
