# Tutorial pour tester la méthode sur une paire de monnaie

1) Ouvrir la session linux

2) Ouvrir un terminal (appuyer en même temps sur Ctrl + Alt + T)
% Screenshot du terminal
3) activer l'environnement conda (c'est pour charger les bonnes bibliothèques python qui font les calculs)
```
conda activate riedones_3d
```

exporter PYTHONPATH.

```
export PYTHONPATH=$PYTHONPATH:/home/sofiane/Documents/riedones3d
```

% screenshot de la commande
Maintenant tout est prêt pour classer les monnaies

4) Se rendre dans le bon dossier
```
cd /home/sofiane/Documents/riedones3d
```

Si vous tapez `ls`, vous devez voir quelques chose comme:
% capture d'écran des dossiers

5) convertir vos mesh (STL ou PLY) en mesh PLY avec les normales et orienté selon l'axe z avec ce script.
voici un script qui permet de convertir des mesh (stl ou ply)

```
python scripts/preprocessing.py --path_coin mymesh.stl -o endroit où les ply sont sauvegardées
```
Pour faire une conversion (par exemple la L0082D):
```
python scripts/preprocessing.py  --path_coin /media/sofiane/DATA/sofiane/data/Part_droits_Riedones3D/L0082D.STL -o /media/sofiane/DATA/sofiane/data/PLY/
```

pour en traiter plusieurs:
```
python scripts/preprocessing.py  --path_coin /media/sofiane/DATA/sofiane/data/Part_droits_Riedones3D/*.STL -o /media/sofiane/DATA/sofiane/data/PLY/
```
NOTE: le *.STL permet de selectionner tous les STL d'un dossier. 
ATTENTION: les monnaies ou autres motifs vont être orienté selon l'axe z. 
Il est possible de rajouter un scale en modifiant le paramètre `--scale 1.5`

## Comparaison deux à deux
6) Supposons nous voulons savoir si deux monnaies ont été frappées avec le même coin. le scripts `whole_pipeline.py` permet de voir toutes les étapes de la comparaison.
Maintenant, nous pouvons appliquer le script. 
```
python scripts/whole_pipeline.py --path COIN1.ply COIN2.ply -m PATH OF THE MODEL --angle 0 --trans 20  --clf classifiers/logistic_part_droits_sym.pkl --path_scaler classifiers/mean_std.json --est ransac
```

`--path` il faut indiquer les chemins des deux monnaies (stl ou ply) (IMPORTANT)

`-m` le chemin du modèle, c'est le réseau de neurones à charger !!! (IMPORTANT)

`--angle` et `--trans` ne sont pas important, c'est des paramètres pour bien voir les monnaies au début.

`--clf` c'est un classifieur ie un outil qui permet de prendre la desision à partir des info calculées précédemment. lui donner un format .pkl(modèle statistique)  (IMPORTANT)
`--path_scaler` json qui contient des valeurs importantes pour le classifieur format .json (IMPORTANT)

`--est` c'est l'estimateur (ransac ou teaser)

`--path_output` endroit où on sauvegarde des captures d'écran des résultats (optionel)

Par exemple:
```
python scripts/whole_pipeline.py --path /media/sofiane/DATA/sofiane/data/PLY/Part_droits_Riedones3D/L0093D.ply /media/sofiane/DATA/sofiane/data/PLY/Part_droits_Riedones3D/L0098D.ply -m /media/sofiane/DATA/sofiane/data/benchmark-MinkUNet_Fragment-20210429_193357/MinkUNet_Fragment.pt --angle 0 --trans 20  --clf classifiers/logistic_part_droits_sym.pkl --path_scaler classifiers/mean_std.json --est ransac
```


## Que faire si nous voulons faire le classement de plusieurs monnaies ?

7) il faut faire un csv qui liste les monnaies que nous souhaitons classer selon leur coin.
Ouvrir libreoffice

8) sur une cellule de la première ligne, écrire "name" (sans les guillemets).
Lister les monnaies à comparer en dessous de "name"
% capture d'écran

9) sauvegarder en CSV (IMPORTANT) en csv (le délimiteur c'est ",").

10) Utiliser le réseau de neurones pour calculer les "features": c'est pour le recalage
```
python scripts/compute_feature.py --path_coin dossier contenant les monnaies --list_coin le csv qui contient les monnaies -m PATH OF THE MODEL --path_output results --name test
```
`--path_coin` chemin vers le dossier
`--list_coin` le csv créé ci-dessus qui liste les monnaie à classer
`-m` le chemin du réseau de neurones
 `--path_output` chemin où on sauvegarde les "features".
 `--name` nom du dossier où tout est sauvegardé. Ici on a choisi test mais il peut s'agir de ce que vous voulez.

Par exemple
```
python scripts/compute_feature.py --path_coin /media/sofiane/DATA/sofiane/data/PLY/Part_droits_Riedones3D/ --list_coin /media/sofiane/DATA/sofiane/data/list_coins.csv -m /media/sofiane/DATA/sofiane/data/benchmark-MinkUNet_Fragment-20210429_193357/MinkUNet_Fragment.pt  --path_output results --name test
```

Verifier avec `ls` que le réseau a créé les bon fichier (des .pt).

```
ls results/test/feature/
```

vous devriez voir des fichier .pt c'est des fichiers intermédiaires pour aligner correctement les motifs

11) Finaliser le recalage avec l'estimateur

```
python scripts/compute_transformation.py --path_feature results/test/feature/  --path_output results/test/transformation --list_coin fichier csv --num_points 5000 --est ransac --n_jobs 8 --sym
```

`--path_feature` chemin du dossier qui contient les .pt.
`--path_output` dossier où seront sauvegardé les transformations
`--num_points` parametre pour controler le nombre de points utilisé pour le recalage des monnnaies. 5000 est un bon choix mais la contrepartie est que les comparaisons sont lentes
`--est` estimateur. vous pouvez choisir "ransac" "fgr" ou "teaser" (ransac fonctionne bien mais il est lent).

--clf` c'est un classifieur ie un outil qui permet de prendre la desision à partir des info calculées précédemment. lui donner un format .pkl(modèle statistique)  (IMPORTANT)

`--path_scaler` json qui contient des valeurs importantes pour le classifieur format .json (IMPORTANT)
`--n_jobs` c'est pour aller plus vite !! (8 c'est très bien)
`--sym` paramètre important à mettre (calcul symétrique de la similarité)


Par exemple:
```
python scripts/compute_transformation.py --path_feature results/test/feature/  --path_output results/test/transformation --list_coin /media/sofiane/DATA/sofiane/data/PLY/Part_droits_Riedones3D/ --list_coin /media/sofiane/DATA/sofiane/data/list_coins.csv --num_points 5000 --est ransac --n_jobs 8 --sym
```

NB: C'est la partie la plus lente du programme(peut prendre une semaine pour 1000 monnaies). L'ordinateur risque de faire du bruit !!
NB2: il vaut mieux classer les monnaies petit à petit 500 puis 500 que faire tout d'un coup. C'est plus rapide.


vous pouvez faire `ls` pour vérifier
```
ls results/test/transformation/
```
normalement vous devriez voir deux fichiers `hist.npy`  `transfo.npy`  et un dossier `transformation`

12) Maintenant, vous pouvez créer le graphe de similarité
```
python scripts/compute_graph_from_hist.py --path_histogram le fichier hist.npy --clf classifiers/logistic_part_droits_sym.pkl --path_output results/test/graph --path_scaler classifiers/mean_std.json
```

`--path_histogram` c'est le chemin vers le fichier hist.npy qu'on a vu précédemment.
`--clf` c'est le classifieur .plk
`--path_scaler` json qui contient des valeurs importantes pour le classifieur format .json (IMPORTANT)
`--path_output` c'est le dossier où on sauvegarde la sortie du script qui est le graphe de correspondance

par exemple
```
python scripts/compute_graph_from_hist.py --path_histogram results/test/transformation/hist.npy --path_clf classifiers/logistic_part_droits_sym.pkl --path_output results/test/graph --path_scaler classifiers/mean_std.json
```

13) visualiser le graph avec graph_visualiser

Ouvrir un terminal Ctrl + Alt + T

Aller sur graph_visualiser
```
cd Documents/graph_visualizer/
```

ouvrir l'outil de visualisation de graphe avec firefox
```
firefox main.html &
```
