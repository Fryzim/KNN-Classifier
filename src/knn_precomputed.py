"""k-NN avec matrice de distances précalculée.

Au lieu de recalculer les distances à chaque évaluation (comme dans
`knn.py`), on calcule une seule fois la matrice de distances complète
`D = cdist(X, X)`, puis chaque prédiction ne fait qu'un accès à cette
matrice. Optimisation payante dès qu'on répète beaucoup d'évaluations
sur le même dataset (cross-validation, comparaisons multiples).
"""
import numpy as np
from scipy.spatial.distance import cdist


def knn_predict_precomputed(point_indice, k, indices_train, D, y_np):
    """Prédit la classe du point d'indice `point_indice` à partir de la
    matrice de distances précalculée D."""
    distances = D[point_indice, indices_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest = [indices_train[i] for i in k_indices]
    votes = y_np[k_nearest].astype(int)
    return np.bincount(votes).argmax()


def evaluer_modele_precomputed(y_np, indices_test, k, indices_train, D):
    nombre_correct = 0
    total = len(indices_test)

    for i in indices_test:
        vrai_label = y_np[i]
        prediction = knn_predict_precomputed(i, k, indices_train, D, y_np)
        if prediction == vrai_label:
            nombre_correct += 1

    return nombre_correct / total


def cross_validation_precomputed(X_np, y_np, k_values, n_folds=5):
    """Calcule la matrice de distances complète une seule fois, puis fait
    la cross-validation dessus pour chaque valeur de k."""
    results = {k: [] for k in k_values}
    fold_size = X_np.shape[0] // n_folds

    print(f"Début de la cross-validation ({n_folds} folds) pour les k: {k_values}")

    D = cdist(X_np, X_np, metric='euclidean')

    for fold in range(n_folds):
        print(f"fold {fold + 1}/{n_folds}")

        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        indices_test = list(range(val_start, val_end))
        indices_train = [i for i in range(X_np.shape[0]) if i not in indices_test]

        for k in k_values:
            accuracy = evaluer_modele_precomputed(y_np, indices_test, k, indices_train, D)
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


def run_pipeline(X_np_train, y_np_train, X_np_test, y_np_test, k_values=None):
    """CV sur le train set pour choisir k, puis évaluation sur le test set.

    Returns:
        (best_k, best_accuracy_cv, mean_accuracies, accuracy_test)
    """
    if k_values is None:
        k_values = list(range(1, 51))

    n = X_np_train.shape[0]

    best_k, best_accuracy, mean_accuracies = cross_validation_precomputed(
        X_np_train, y_np_train, k_values, n_folds=5
    )
    print(f"\nMeilleur k: {best_k}, précision de {best_accuracy:.4f}")

    indices_train = list(range(n))
    indices_test = list(range(n, n + X_np_test.shape[0]))

    tout_X = np.concatenate((X_np_train, X_np_test), axis=0)
    tout_y = np.concatenate((y_np_train, y_np_test), axis=0)

    D = cdist(tout_X, tout_X, metric='euclidean')
    final_accuracy = evaluer_modele_precomputed(tout_y, indices_test, best_k, indices_train, D)
    print(f"Accuracy finale sur le test set: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%) avec k={best_k}")

    return best_k, best_accuracy, mean_accuracies, final_accuracy
