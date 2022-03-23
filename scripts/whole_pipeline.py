import argparse
import glob

from point_cloud.robust_estimator import build_estimator
from point_cloud.pipeline_visualizer import Open3DVisualizer
from point_cloud.pipeline_visualizer import PyRenderVisualizer
from point_cloud.pipeline import OnlineRiedonesPipeline
from point_cloud.classifier import HistClassifier

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
    parser.add_argument('--path_output', dest='path_output',
                        help='path of the output (if None it display online)',
                        type=str, default=None)
    
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    list_path = []
    for path in args.path_coin:
        p = glob.glob(path)
        list_path = list_path + p
    
    estimator = build_estimator(args.robust_estimator,
                                noise_bound=0.5,
                                max_clique_time_limit=0.5,
                                distance_threshold=0.5,
                                num_iterations=10000
    )
    if args.path_output is None:
        visualizer = Open3DVisualizer(translate=args.trans)
    else:
        visualizer = PyRenderVisualizer(path_output=args.path_output, translate=args.trans)
    pipeline = OnlineRiedonesPipeline(estimator=estimator,
                                      path_model=args.path_model,
                                      visualizer=visualizer,
                                      num_points=args.num_points,
                                      sym=True)
    clf = HistClassifier(args.path_clf, args.path_scaler)
    pipeline.compute_all(list_path)
    clf.compute_graph(pipeline.get_dico_histogram())
    if args.path_output is not None:
        clf.save_graph(args.path_output)

if __name__ == "__main__":
    main()
