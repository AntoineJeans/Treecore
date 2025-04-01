# Treecore (forgetting)
### But original
Rechercher la possibilité de trouver un sous-ensemble des données représentatif et qui permet d'obtenir des arbres de décision plus simples et plus performants.

La réflexion est que les points les plus critiques, ceux les plus près de la frontière de décision, sont ceux qui sont le plus probable d'être oubliés à travers l'apprentissage. Ils devraient donc être compressés et utilisés dans

### Stratégies de compression
- [DropUnforgettable](CompressionStrategies/DropUnforgettable.py): cette stratégie de compression garde les points qui ont été oubliés au moins une fois à travers l'apprentissage. Ce fichier contient la stratégie pour un modèle de classification et de régression. 
- [DropNForgets](./CompressionStrategies/DropNForgets.py): similaire à DropUnforgettable pour n=1, celle-ci garde seulement les points dont le résultat varie au fil de l'apprentissage (les points oubliés). Seulement le modèle de classification est inclus. 

### Libraires et outils python
- numpy
- matplotlib
- pandas
- itertools
- seaborn
- sklearn
- openml (utilisé pour avoir le dataset de MNIST [id:554](https://openml.org/search?type=data&sort=runs&id=554&status=active))
- abc
- os
- pathlib

### Fonctionnalités et idées 
Random_keeps: l'idée est de "booster" l'ensemble des données des points compressés par une quantité déterminée de points pris au hasard dans ceux qui ne sont pas dans l'ensemble compressé.

Contraire de forgetting: les points qui sont non-oubliés sont ceux qui se font compresser dans l'ensemble. 

Variance: calculer la variance *de ...* entre le point et la frontière de décision pour calculer la probabilité de ce point à être oublié par la machine.

### Contributeurs
Antoine Jean et Émylie-Rose Desmarais
