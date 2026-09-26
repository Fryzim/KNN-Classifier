"""k-NN de base : version naïve (boucle Python, gardée pour comparaison) et
version vectorisée (scipy.spatial.distance.cdist), utilisée dans le reste
du pipeline.
"""
import numpy as np
from scipy.spatial.distance import cdist


def distance_euclidienne(point1, point2):
    """Distance euclidienne entre deux points, calculée manuellement."""
    somme_carres = 0.0
    for i in range(len(point1)):
        difference = point1[i] - point2[i]
        somme_carres += difference ** 2
    return somme_carres ** 0.5


def knn_predict_naive(x, k, metrique, X_train, y_train):
    """Prédit la classe de x en calculant sa distance à CHAQUE point
    d'entraînement dans une boucle Python.

    Gardée pour comparer sa vitesse à la version vectorisée ci-dessous ;
    non utilisée dans les pipelines (trop lente au-delà de quelques
    centaines de points d'entraînement).
    """
    list_distances = []
    x_2d = np.array(x).reshape(1, -1)

    for i in range(len(X_train)):
        point_2d = np.array(X_train.iloc[i]).reshape(1, -1)
        dist = cdist(x_2d, point_2d, metric=metrique)[0][0]
        list_distances.append((dist, i))

    list_distances.sort(key=lambda x: x[0])
    k_nearest_neighbors = list_distances[:k]

    votes = {}
    for dist, index in k_nearest_neighbors:
        label = y_train.iloc[index]
        votes[label] = votes.get(label, 0) + 1

    return max(votes, key=votes.get)


def knn_predict(x, k, metrique, X_train, y_train):
    """Prédit la classe de x : une seule passe `cdist` vectorisée contre
    tout X_train, au lieu d'une boucle Python point par point."""
    x_2d = np.array(x).reshape(1, -1)
    X_2d = np.array(X_train)

    distances = cdist(x_2d, X_2d, metric=metrique)[0]
    k_indices = np.argsort(distances)[:k]
    votes = y_train.iloc[k_indices].value_counts()
    return votes.idxmax()


def evaluer_modele(X_test, y_test, k, metrique, X_train, y_train):
    """Précision du k-NN vectorisé sur un ensemble de test."""
    nombre_correct = 0
    total = len(X_test)

    for i in range(total):
        point_test = X_test.iloc[i].values
        vrai_label = y_test.iloc[i]
        prediction = knn_predict(point_test, k, metrique, X_train, y_train)
        if prediction == vrai_label.item():
            nombre_correct += 1

    return nombre_correct / total


def cross_validation_knn(X, y, k_values, n_folds=5, metric='euclidean'):
    """Cross-validation pour choisir le meilleur k.

    Returns:
        best_k, best_accuracy, mean_accuracies (dict k -> accuracy moyenne)
    """
    results = {k: [] for k in k_values}

    fold_size = len(X) // n_folds
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    print(f"Début de la cross-validation ({n_folds} folds) pour les k: {k_values}")

    for fold in range(n_folds):
        print(f"fold {fold + 1}/{n_folds}...")

        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

        X_train_fold = X.iloc[train_indices].reset_index(drop=True)
        y_train_fold = y.iloc[train_indices].reset_index(drop=True)
        X_val_fold = X.iloc[val_indices].reset_index(drop=True)
        y_val_fold = y.iloc[val_indices].reset_index(drop=True)

        for k in k_values:
            accuracy = evaluer_modele(X_val_fold, y_val_fold, k, metric, X_train_fold, y_train_fold)
            results[k].append(accuracy)

    best_accuracy = 0
    best_k = None
    mean_accuracies = {}

    for k in k_values:
        mean_accuracy = np.mean(results[k])
        mean_accuracies[k] = mean_accuracy
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_k = k

    return best_k, best_accuracy, mean_accuracies
