"""k-NN optimisé par structure kd-tree : recherche des k plus proches
voisins en élaguant les branches qui ne peuvent pas contenir de point
plus proche que ceux déjà trouvés (au lieu de comparer à tous les points).
"""
from collections import Counter, defaultdict
from heapq import heappush, heappop
from random import shuffle

import numpy as np


class KDNode:
    def __init__(self, point, label, left=None, right=None, axis=0):
        self.point = point
        self.label = label
        self.left = left
        self.right = right
        self.axis = axis


def build_kdtree(points, prof=0):
    """Construit récursivement le kd-tree en alternant l'axe de coupure à
    chaque niveau (coupure sur la médiane)."""
    if not points:
        return None
    k = len(points[0][0])
    axis = prof % k
    points.sort(key=lambda x: x[0][axis])
    milieu = len(points) // 2
    return KDNode(
        point=points[milieu][0],
        label=points[milieu][1],
        left=build_kdtree(points[:milieu], prof + 1),
        right=build_kdtree(points[milieu + 1:], prof + 1),
        axis=axis,
    )


def distance_carre(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2))


def knn_search(racine, target, k):
    """Recherche des k plus proches voisins dans le kd-tree, avec élagage
    des branches trop éloignées (max-heap de taille k sur -distance²)."""
    heap = []

    def rec_recherche(node):
        if node is None:
            return

        dist2 = distance_carre(target, node.point)
        heappush(heap, (-dist2, node.label))
        if len(heap) > k:
            heappop(heap)

        axis = node.axis
        diff = target[axis] - node.point[axis]

        close_branch = node.left if diff < 0 else node.right
        far_branch = node.right if diff < 0 else node.left

        rec_recherche(close_branch)

        if len(heap) < k or diff ** 2 < -heap[0][0]:
            rec_recherche(far_branch)

    rec_recherche(racine)
    return [label for _, label in sorted(heap, reverse=True)]


def predict_kdtree(tree, point, k=3):
    """Prédiction par vote majoritaire parmi les k plus proches voisins."""
    neighbors = knn_search(tree, point, k)
    vote = Counter(neighbors)
    return vote.most_common(1)[0][0]


def cross_validation_kdtree(X, y, k_values, n_splits=5):
    """Cross-validation stratifiée par classe (chaque fold reçoit une part
    égale de chaque classe)."""
    X = list(X)
    y = list(y)

    class_indices = defaultdict(list)
    for i, label in enumerate(y):
        class_indices[label].append(i)

    folds = [[] for _ in range(n_splits)]
    for label, indices in class_indices.items():
        shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % n_splits].append(idx)

    k_to_accuracies = defaultdict(list)

    print(f"Début de la cross-validation sur {n_splits} folds\n")

    for k in k_values:
        print(f"Test de k = {k}")
        for i in range(n_splits):
            print(f"-> Fold {i + 1}/{n_splits}", end=' ', flush=True)

            test_idx = folds[i]
            train_idx = [idx for j in range(n_splits) if j != i for idx in folds[j]]

            X_train = [X[j] for j in train_idx]
            y_train = [y[j] for j in train_idx]
            X_test = [X[j] for j in test_idx]
            y_test = [y[j] for j in test_idx]

            train_data = list(zip(X_train, y_train))
            kdtree = build_kdtree(train_data)

            y_pred = [predict_kdtree(kdtree, x_test, k=k) for x_test in X_test]
            correct = sum(1 for pred, true in zip(y_pred, y_test) if pred == true)
            accuracy = correct / len(y_test)

            print(f"Précision : {accuracy:.3f}")
            k_to_accuracies[k].append(accuracy)

        avg_acc = np.mean(k_to_accuracies[k])
        print(f"Moyenne pour k = {k} : {avg_acc:.3f}\n")

    k_to_avg_accuracy = {k: np.max(accs) for k, accs in k_to_accuracies.items()}
    best_k = max(k_to_avg_accuracy, key=k_to_avg_accuracy.get)
    best_accuracy = k_to_avg_accuracy[best_k]

    print("Résultats fin :")
    for k, acc in sorted(k_to_avg_accuracy.items()):
        print(f"  k = {k} → précision moyenne : {acc:.3f}")
    print(f"\nMeilleur k trouvé : {best_k}\n")

    return best_k, best_accuracy, k_to_avg_accuracy


def run_pipeline(X_train, y_train, X_test, y_test, k_values=None):
    """CV sur le train set pour choisir k, puis évaluation sur le test set."""
    if k_values is None:
        k_values = list(range(1, 51))

    best_k, best_accuracy, mean_accuracies = cross_validation_kdtree(X_train, y_train, k_values, n_splits=5)
    print(f"\nMeilleur k: {best_k}, précision de {best_accuracy:.4f}")

    train_data = list(zip(X_train, y_train))
    kdtree = build_kdtree(train_data)
    y_pred = [predict_kdtree(kdtree, xi, k=best_k) for xi in X_test]
    correct = sum(1 for pred, true in zip(y_pred, y_test) if pred == true)
    final_accuracy = correct / len(y_test)
    print(f"Accuracy finale sur le test set: {final_accuracy:.4f} ({final_accuracy * 100:.2f}%) avec k={best_k}")

    return best_k, best_accuracy, mean_accuracies, final_accuracy
