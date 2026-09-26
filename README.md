# k-NN From Scratch — Implémentation et Optimisation

Implémentation du classifieur k plus proches voisins sans bibliothèque de machine learning, sur le dataset Waveform (5000 instances, 21 variables continues), avec une progression du calcul naïf vers des versions optimisées.

## Démarche

- **k-NN naïf** : distance euclidienne calculée point par point.
- **Réduction de dimension** : ACP en deux passes pour visualiser l'effet de la réduction sur la séparabilité des classes.
- **Optimisations de calcul** : distances précalculées (`scipy.spatial.distance.cdist`), élagage par inégalité triangulaire, et structure kd-tree, comparées en temps d'exécution sur le même jeu de test.
- **Comparatif final** : k-NN naïf vs. inégalité triangulaire vs. kd-tree, en précision et en temps.

## Reporting

- **Live dashboard :** [fryzim.github.io/KNN-Classifier](https://fryzim.github.io/KNN-Classifier/) — accuracy vs k, effet de la réduction du dataset, comparatif des 4 approches (source dans `docs/index.html`). Chiffres obtenus en exécutant réellement `src/` sur un échantillon de 1300 lignes du dataset (voir `generate_report_data.py`), pas de valeurs inventées.
- **Power Query :** `reporting/power_query.m` — charge et type `waveform.data.csv` (fichier sans en-tête) pour Power BI/Excel.

**À noter :** la cross-validation du kd-tree (`src/kdtree.py`) retient le **maximum** des folds pour chaque k plutôt que la moyenne (comportement du notebook original, conservé tel quel) — son score de CV n'est donc pas directement comparable à celui des 3 autres approches, qui utilisent la moyenne. L'accuracy sur le test set, elle, reste comparable entre les 4.

## Contenu du dépôt

```
src/
  data.py                  chargement + split train/test
  knn.py                    k-NN naïf (boucle) et vectorisé (cdist), évaluation, cross-validation
  knn_precomputed.py         variante : matrice de distances calculée une seule fois
  triangle_inequality.py      variante : élagage par inégalité triangulaire
  kdtree.py                   variante : structure kd-tree
  reduction.py                réduction du dataset (suppression de la zone de biais + condensing)
  pca_viz.py                  ACP manuelle (calcul explicite des vecteurs propres) pour visualiser la réduction
  visualize.py                 graphiques partagés (accuracy vs k, comparatif entre pipelines)
  compare_svc.py               comparaison du meilleur pipeline k-NN avec des SVC scikit-learn
projet.ipynb              notebook de démo : charge les données, appelle src/, affiche les résultats
waveform.data.csv         dataset utilisé
report_knn_analysis.pdf   rapport d'analyse
docs/index.html           dashboard de reporting (GitHub Pages)
reporting/power_query.m   script Power Query (M) pour Power BI / Excel
generate_report_data.py   génère les données réelles utilisées par le dashboard
```

**Remarque :** le notebook original contenait aussi des cellules d'expérimentation répétée (boucles sur plusieurs seeds pour comparer les 3 optimisations) qui référençaient des variables jamais définies dans le notebook et plantaient telles quelles. Ces cellules mortes ont été laissées de côté lors de la restructuration ; toute la logique qui fonctionnait réellement (les 4 variantes de k-NN, la réduction de dataset, l'ACP, la comparaison avec SVC) est reprise fidèlement dans `src/`.

## Stack

Python — `numpy`, `pandas`, `scipy` (uniquement pour `cdist`, pas pour l'algorithme lui-même), `matplotlib`, `scikit-learn` (comparaison SVC)
