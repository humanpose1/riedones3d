import argparse
import os
import os.path as osp
import numpy as np

from point_cloud.classifier import HistClassifier



def parse_args():
    parser = argparse.ArgumentParser('visualize results'
                                     '')

    parser.add_argument('--path_histogram', dest='path_histogram',
                        help='path of X matrix containing the histogram (numpy file)',
                        type=str)
    parser.add_argument('--path_output', dest='path_output',
                        help='where we store the graph',
                        default='output')
    parser.add_argument('--clf', dest='path_clf',
                        help='path of the model',
                        default='../learning/logistic.pkl')
    parser.add_argument('--path_scaler', dest='path_scaler',
                        help='path of mean and std',
                        type=str)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dico_histogram = np.load(args.path_histogram, allow_pickle=True).item()
    clf = HistClassifier(args.path_clf, args.path_scaler)
    clf.compute_graph(dico_histogram)
    clf.save_graph(args.path_output)

if __name__ == "__main__":
    main()
