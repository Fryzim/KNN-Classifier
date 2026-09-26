"""k-NN optimisé par élagage via l'inégalité triangulaire.

Idée : si on connaît déjà un voisin proche X' et sa distance à un
candidat Xi, l'inégalité triangulaire donne une borne inférieure sur
d(point_test, Xi) sans avoir à la calculer explicitement — ce qui permet
d'éliminer Xi sans jamais mesurer sa vraie distance au point testé.
"""
import numpy as np
from scipy.spatial.distance import cdist


def knn_predict_triangle_inequality(point_test, k, metrique, X_train, y_train):
    x_2d = np.array(point_test).reshape(1, -1)
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    distances = cdist(x_2d, X_train_np, metric=metrique)[0]
    sorted_indices = np.argsort(distances)

    first_idx = sorted_indices[0]
    d_min = distances[first_idx]

    k_nearest_indices = [first_idx]
    k_nearest_distances = [d_min]

    for idx in sorted_indices[1:]:
        if len(k_nearest_indices) >= k:
            if distances[idx] > max(k_nearest_distances):
                break

        d_X_Xi = distances[idx]
        can_eliminate = False

        if d_X_Xi > d_min:
            # Distance entre le voisin le plus proche actuel (X') et le candidat Xi
            x_prime_idx = k_nearest_indices[0]
            d_Xprime_Xi = cdist(
                X_train_np[x_prime_idx].reshape(1, -1),
                X_train_np[idx].reshape(1, -1),
                metric=metrique,
            )[0][0]

            # Si Xi est dans la sphère centrée en X' de rayon d(X,Xi) - d_min, on peut l'éliminer
            if d_Xprime_Xi + d_min <= d_X_Xi:
                can_eliminate = True

        if not can_eliminate:
            k_nearest_indices.append(idx)
            k_nearest_distances.append(d_X_Xi)

            if d_X_Xi < d_min:
                d_min = d_X_Xi
                k_nearest_indices[0], k_nearest_indices[-1] = k_nearest_indices[-1], k_nearest_indices[0]
                k_nearest_distances[0], k_nearest_distances[-1] = k_nearest_distances[-1], k_nearest_distances[0]

    if len(k_nearest_indices) > k:
        sorted_k = np.argsort(k_nearest_distances)[:k]
        k_nearest_indices = [k_nearest_indices[i] for i in sorted_k]

    votes = {}
    for idx in k_nearest_indices:
        label = y_train_np[idx]
        votes[label] = votes.get(label, 0) + 1

    return max(votes, key=votes.get)


def evaluer_modele_triangle_inequality(X_test, y_test, k, metrique, X_train, y_train):
    X_test_np = np.array(X_test)
    y_test_np = np.array(y_test)
    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)

    nombre_correct = 0
    total = len(X_test_np)

    for i in range(total):
        prediction = knn_predict_triangle_inequality(X_test_np[i], k, metrique, X_train_np, y_train_np)
        if prediction == y_test_np[i]:
            nombre_correct += 1

    return nombre_correct / total


def cross_validation_triangle_inequality(X, y, k_values, n_folds=5, metric='euclidean'):
    X_np = np.array(X)
    y_np = np.array(y)

    results = {k: [] for k in k_values}
    fold_size = len(X_np) // n_folds
    indices = np.arange(len(X_np))
    np.random.shuffle(indices)

    print(f"Début de la cross-validation ({n_folds} folds) avec inégalité triangulaire pour les k: {k_values}")

    for fold in range(n_folds):
        print(f"fold {fold + 1}/{n_folds}...")

        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

        X_train_fold, y_train_fold = X_np[train_indices], y_np[train_indices]
        X_val_fold, y_val_fold = X_np[val_indices], y_np[val_indices]

        for k in k_values:
            accuracy = evaluer_modele_triangle_inequality(X_val_fold, y_val_fold, k, metric, X_train_fold, y_train_fold)
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


def run_pipeline(X_train, y_train, X_test, y_test, k_values=None):
    """CV sur le train set pour choisir k, puis évaluation sur le test set."""
    if k_values is None:
        k_values = list(range(1, 51))

    best_k, best_accuracy, mean_accuracies = cross_validation_triangle_inequality(
        X_train, y_train, k_values, n_folds=5, metric='euclidean'
    )
    print(f"\nMeilleur k: {best_k}, précision de {best_accuracy:.4f}")

    final_accuracy = evaluer_modele_triangle_inequality(X_test, y_test, best_k, 'euclidean', X_train, y_train)
    print(f"Accuracy finale sur le test set: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%) avec k={best_k}")

    return best_k, best_accuracy, mean_accuracies, final_accuracy
