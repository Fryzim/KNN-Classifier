"""Réduction du dataset d'entraînement par deux règles vues en cours,
toutes deux basées sur une évaluation 1-NN entre sous-ensembles :

- `reduction_bias_removal` : supprime la "zone de biais" (points mal
  classés à la frontière entre classes) en faisant converger deux
  sous-ensembles qui se classent mutuellement.
- `reduction_condensing` : ne garde que les points nécessaires à une
  classification correcte (condensed nearest neighbor) — les points
  déjà bien classés par le sous-ensemble courant sont jetés.
"""
import random

import numpy as np
from scipy.spatial.distance import cdist


def un_NN(p, X, y):
    """Renvoie le label du plus proche voisin de p dans (X, y) (1-NN)."""
    dmin = cdist(p.reshape(1, -1), X[0].reshape(1, -1), metric="euclidean")[0][0]
    label = y[0]

    for i in range(1, X.shape[0]):
        d = cdist(p.reshape(1, -1), X[i].reshape(1, -1), metric="euclidean")[0][0]
        if d < dmin:
            dmin = d
            label = y[i]

    return label


def _sep_data(X):
    """Mélange les indices de X et renvoie le point de coupure médian."""
    l_id = list(range(X.shape[0]))
    random.shuffle(l_id)
    milieu = len(l_id) // 2
    return l_id, milieu


def reduction_bias_removal(X, y):
    """Sépare X en deux sous-ensembles S1/S2, puis retire itérativement de
    chacun les points mal classés par l'autre (1-NN), jusqu'à stabilisation.
    """
    ordre, milieu = _sep_data(X)
    S1_X, S1_y = X[ordre[:milieu]], y[ordre[:milieu]]
    S2_X, S2_y = X[ordre[milieu:]], y[ordre[milieu:]]

    changement = True
    while changement:
        changement = False

        nv_S1_X, nv_S1_y = [], []
        for xi, yi in zip(S1_X, S1_y):
            if un_NN(xi, S2_X, S2_y) == yi:
                nv_S1_X.append(xi)
                nv_S1_y.append(yi)
            else:
                changement = True
        S1_X, S1_y = np.array(nv_S1_X), np.array(nv_S1_y)

        nv_S2_X, nv_S2_y = [], []
        for xi, yi in zip(S2_X, S2_y):
            if un_NN(xi, S1_X, S1_y) == yi:
                nv_S2_X.append(xi)
                nv_S2_y.append(yi)
            else:
                changement = True
        S2_X, S2_y = np.array(nv_S2_X), np.array(nv_S2_y)

    X_cleaned = np.concatenate((S1_X, S2_X), axis=0)
    y_cleaned = np.concatenate((S1_y, S2_y))
    return X_cleaned, y_cleaned


def reduction_condensing(X, y):
    """Condensed Nearest Neighbor : part d'un point au hasard dans STORAGE,
    puis ajoute à STORAGE tout point mal classé par le STORAGE courant
    (1-NN), jusqu'à ce que plus rien ne change.
    """
    storage_X, storage_y = [], []

    idx0 = random.randint(0, X.shape[0] - 1)
    storage_X.append(X[idx0])
    storage_y.append(y[idx0])

    changement = True
    while changement:
        changement = False

        for xi, yi in zip(X, y):
            liste_storage_X = [p.tolist() for p in storage_X]
            if xi.tolist() in liste_storage_X:
                continue

            if un_NN(xi, np.array(storage_X), np.array(storage_y)) != yi:
                storage_X.append(xi)
                storage_y.append(yi)
                changement = True

    return np.array(storage_X), np.array(storage_y)
