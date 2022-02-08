# Riedones3D

It is the code to perform coin die recognition and die clustering

## Installation
requirements:
```
hydra-core==1.0.0
omegaconf==2.0.6
scikit-learn==0.24.2
torch-cluster==1.5.9
torch-scatter==2.0.7
torch-geometric==1.7.2
torch==1.8.1
torch_points_kernels
torch-points3d==1.3.0
MinkowskiEngine==0.5.2

omegaconf==2.0.6
hydra-core==1.0.0
open3d==0.12.0
```
First install anaconda or miniconda (use this [website](https://docs.conda.io/en/latest/miniconda.html) for example)

then open a terminal.

create a new environnement (for example using conda)
 ```
 conda create -n "riedones_3d"
 ```
 ```
 conda activate riedones_3d
 ```
 execute this script to install the correct packages(here we install with cuda102)
 ```
 pip install scikit-learn==0.24.2 
 pip install torch==1.8.1
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.1+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.8.1+cu102.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.8.1+cu102.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.8.1+cu102.html
pip install torch-geometric==1.7.2
pip install git+https://github.com/nicolas-chaulet/torch-points3d.git@e090530eab5e3e5798c5abc764088d6a1f9827c3
```
install [minkowski engine](https://github.com/NVIDIA/MinkowskiEngine)
```
apt-get install build-essential python3-dev libopenblas-dev
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine@9f81ae66b33b883cd08ee4f64d08cf633608b118 --no-deps
 ```
 install [torch-sparse](https://github.com/mit-han-lab/torchsparse)
 ```
 apt-get install libsparsehash-dev
 pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
 ```

## Preprocessing
As input, we need point cloud in ply format. We propose a simple script to convert from stl to ply with normals(if you already have point cloud in ply, you do not need this step).

```
python experiment/mesh2pcd.py --path_coin mymesh.stl
```

## Register a pair of coin

It will register the pair and it will also compute the histogram of distance. It displays the results with open3D.
```
python experiment/whole_pipeline.py --path ../coin_die_recognition/results/create_unsupervised_dataset/Revers/L0014R.ply ../coin_die_recognition/results/create_unsupervised_dataset/Revers/L0013R.ply -m /media/admincaor/DataHDD2To/mines/code/deeppointcloud-benchmarks/outputs/benchmark/benchmark-MinkUNet_Fragment-20210429_193357/MinkUNet_Fragment --angle 0 --trans 20  --clf learning/logistic_revers_pc.pkl --path_scaler learning/mean_std_pc_edge.json --est ransac
```


## Coin die Clustering

### Compute the features
First you need to compute the features:
```
python experiment/compute_feature.py --path_coin ../coin_die_recognition/results/create_unsupervised_dataset/Droits --list_coin Coins_et_Monnaies_Droits_all.csv -m /media/admincaor/DataHDD2To/mines/code/deeppointcloud-benchmarks/outputs/benchmark/benchmark-MinkUNet_Fragment-20210429_193357/MinkUNet_Fragment --path_output results --name Droits_v2
```
It takes few minutes to compute every features
### compute a pair similarity comparison
```
python experiment/compute_transformation.py --path_feature results/Droits_v2/feature/  --path_output results/Droits_v2/transformation --list_coin Coins_et_Monnaies_Droits_all.csv --num_points 5000 --est teaser --clf learning/logistic_part_droits_sym.pkl --path_scaler learning/mean_std.json --n_jobs 8 --sym
```

`--sym` means the histogram is symmetric.
`--num_points` is the number of points
`--est` is the robust estimator to compute the transformation
It takes few days to compute every similarities
it will generate two files:
- a file containing every transformations
- a file containing every histograms of distance

### Compute the Graph

We compute the graph of similarity between the pairs of coins
```
python experiment/compute_graph_from_hist.py --path_histogram results/Droits_v2/transformation/hist.npy -m learning/logistic_part_droits_sym.pkl -o results/Droits_v2/graph --path_scaler learning/mean_std.json
```
### Clean the graph using graph visualizer

You can select links nodes, remove/add links, search for a node cluster the results, save the graph.

TODO: Tutorial about how to use graph visualizer

### generate images and 3D models



## How to register all coin you can see in experiment

### automatic registration




## Clustering (old way)

### compute the histograms
```
python experiment/create_histogram.py --path_coin ../coin_die_recognition/results/create_unsupervised_dataset/Revers/ --path_output ./results/ --list_coin Coins_et_Monnaies_Revers_all.csv --path_transfo ../../dataset_archeology/all_transformation/Revers --name Revers --start 0
```



### create the graph
```
python graph/matrix2graph.py --path_x_deep results/Part_Revers/X_deep.txt  --path_y_name results/Part_revers_normal/Y_names.txt -m learning/logistic_part_droits_normal.pkl -o results/Part_Revers --path_scaler learning/mean_std_normal_5_edge.json 
```
this script allow to save the graph into a json format

### visualize the graph (old visualization using Plotly)
```
python graph/visu_graph.py -p results/Part_Revers/graph.json -t 0.1
```
Plot the dendrogram:
```
python graph/plot_dendrogram.py -p results/Part_Revers/graph.json -o results/Part_Revers/
```


### register first transformation manually
```
python render/rotate_coin_manually.py --path_coin ../../dataset_archeology/3D/pièces\ de\ monnaies\ liffré/Revers/L075R.STL --tz 60 --path_first results/Part_Revers/transformation_first
```

### render the coin
```
python render/save_all_coin.py --path_coin ../../dataset_archeology/3D/pièces\ de\ monnaies\ liffré/Revers --path_tr results/Part_Revers/transformation/ --path_first results/Part_Revers/transformation_first --path_graph results/Part_Revers/graph.json --path_output results/Part_Revers/data -t 0.1 --clustered
```
