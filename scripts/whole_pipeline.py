import argparse



def main():

    estimator = build_estimator(args.robust_estimator,
                                noise_bound=0.5,
                                max_clique_time_limit=0.5,
                                distance_threshold=0.5,
                                num_iterations=10000
    )
    visualizer = Open3DVisualizer()
    pipeline = OnlineRiedonesPipeline(estimator=estimator,
                                      path_model=args.path_model,
                                      visualizer=visualizer,
                                      num_points=args.num_points)

if __name__ == "__main__":
    main()
