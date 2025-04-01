# Treecore (forgetting)
### But original
Rechercher la possibilite de trouver un sous-ensemble des donnees representatif et qui permet d'obtenir des arbres de decision plus simples et plus performants

### Strategies de compression
- [DropNForgets](./CompressionStrategies/DropNForgets.py): garder seulement les points dont le resultat varie au fil des epoques. La reflexion est que ceux-cisont probablement plus pres de la frontiere de decision.
- [DropUnforgettable](CompressionStrategies/DropUnforgettable.py): 

### Libraires et outils python
- numpy
- matplotlib
- pandas
- itertools
- seaborn
- sklearn
- openml (utilise pour avoir le dataset de mnist [id:554](https://openml.org/search?type=data&sort=runs&id=554&status=active))
- abc
- os
- pathlib

### Fonctionnalites et idees 
Random_keeps: l'idée est de booster l'ensemble des données des points compressés par une quantité déterminée de points pris au hasard dans ceux qui ne sont pas dans l'ensemble rappelé.

Variance: calculer la variance entre ...

Contraire de forgetting: les points qui sont non-oubliés sont ceux qui se font compresser. 

### Contributeurs
Antoine Jean
