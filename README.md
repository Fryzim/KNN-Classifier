# k-NN From Scratch — Implémentation et Optimisation

Implémentation du classifieur k plus proches voisins sans bibliothèque de machine learning, sur le dataset Waveform (5000 instances, 21 variables continues), avec une progression du calcul naïf vers des versions optimisées.

## Démarche

- **k-NN naïf** : distance euclidienne calculée point par point.
- **Réduction de dimension** : ACP en deux passes pour visualiser l'effet de la réduction sur la séparabilité des classes.
- **Optimisations de calcul** : distances précalculées (`scipy.spatial.distance.cdist`), élagage par inégalité triangulaire, et structure kd-tree, comparées en temps d'exécution sur le même jeu de test.
- **Comparatif final** : k-NN naïf vs. inégalité triangulaire vs. kd-tree, en précision et en temps.

## Contenu du dépôt

- `projet.ipynb` — notebook complet (implémentation + comparatifs + visualisations)
- `waveform.data.csv` — dataset utilisé
- `report_knn_analysis.pdf` — rapport d'analyse

## Stack

Python — `numpy`, `pandas`, `scipy` (uniquement pour `cdist`, pas pour l'algorithme lui-même), `matplotlib`
