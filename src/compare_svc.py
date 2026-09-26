"""Comparaison du meilleur pipeline k-NN avec des SVM scikit-learn
(linéaire, RBF, polynomial) : temps, accuracy, F1, precision, recall."""
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from .triangle_inequality import knn_predict_triangle_inequality

MODELES = ['knn', 'svc_linear', 'svc_rbf', 'svc_poly']
MODELES_LABELS = ['kNN\n(Triangle inequality)', 'SVC\nLinear', 'SVC\nRBF', 'SVC\nPoly']
COLORS = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']


def comparer_knn_svc(X_train, y_train, X_test, y_test, knn_pipeline_fn, nb_exp=5):
    """Compare un pipeline k-NN (fonction (X_train,y_train,X_test,y_test)
    -> (best_k, best_accuracy_cv, mean_accuracies, accuracy_test)) à trois
    SVC scikit-learn, moyenné sur `nb_exp` répétitions.

    Returns:
        resultats: dict brut par modèle et par métrique
        moyennes: dict moyenne/écart-type par modèle et par métrique
    """
    resultats = {m: {'temps': [], 'accuracy_cv': [], 'accuracy_test': [],
                      'f1_test': [], 'precision_test': [], 'recall_test': []}
                 for m in MODELES}

    print("=" * 70)
    print("COMPARAISON kNN vs SVC")
    print("=" * 70)

    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    X_test_np = np.array(X_test)
    y_test_np = np.array(y_test)

    for exp in range(nb_exp):
        print(f"\n--- Expérience {exp + 1}/{nb_exp} ---")

        print("\n1. kNN avec optimisation")
        debut = time.time()
        best_k, best_accuracy_cv, mean_accuracies, accuracy_test = knn_pipeline_fn(
            X_train_np, y_train_np, X_test_np, y_test_np
        )
        temps_knn = time.time() - debut

        y_pred_knn = np.array([
            knn_predict_triangle_inequality(X_test_np[i], best_k, 'euclidean', X_train_np, y_train_np)
            for i in range(len(X_test_np))
        ])

        _record(resultats['knn'], temps_knn, best_accuracy_cv, accuracy_test, y_test_np, y_pred_knn)
        _print_metrics("kNN avec optimisation", resultats['knn'])

        for name, kernel, extra in [('svc_linear', 'linear', {}),
                                     ('svc_rbf', 'rbf', {}),
                                     ('svc_poly', 'poly', {'degree': 3})]:
            print(f"\n{name}: SVC noyau {kernel}")
            debut = time.time()
            svc = SVC(kernel=kernel, random_state=42, **extra)
            accuracy_cv = np.mean(cross_val_score(svc, X_train_np, y_train_np, cv=5))
            svc.fit(X_train_np, y_train_np)
            accuracy_t = svc.score(X_test_np, y_test_np)
            y_pred = svc.predict(X_test_np)
            temps = time.time() - debut

            _record(resultats[name], temps, accuracy_cv, accuracy_t, y_test_np, y_pred)
            _print_metrics(name, resultats[name])

    print("\n" + "=" * 70)
    print("RÉSULTATS MOYENS")
    print("=" * 70)

    moyennes = {}
    for modele in MODELES:
        moyennes[modele] = {
            f'{metric}_moyen' if metric != 'temps' else 'temps_moyen': np.mean(resultats[modele][metric])
            for metric in resultats[modele]
        }
        moyennes[modele].update({
            f'{metric}_std' if metric != 'temps' else 'temps_std': np.std(resultats[modele][metric])
            for metric in resultats[modele]
        })
        print(f"\n{modele.upper().replace('_', ' ')}:")
        print(f"  Temps: {moyennes[modele]['temps_moyen']:.2f}s (±{moyennes[modele]['temps_std']:.2f}s)")
        print(f"  Accuracy Test: {moyennes[modele]['accuracy_test_moyen']:.4f} "
              f"(±{moyennes[modele]['accuracy_test_std']:.4f})")
        print(f"  F1 Score: {moyennes[modele]['f1_test_moyen']:.4f} (±{moyennes[modele]['f1_test_std']:.4f})")

    return resultats, moyennes


def _record(bucket, temps, accuracy_cv, accuracy_test, y_true, y_pred):
    bucket['temps'].append(temps)
    bucket['accuracy_cv'].append(accuracy_cv)
    bucket['accuracy_test'].append(accuracy_test)
    bucket['f1_test'].append(f1_score(y_true, y_pred, average='weighted'))
    bucket['precision_test'].append(precision_score(y_true, y_pred, average='weighted'))
    bucket['recall_test'].append(recall_score(y_true, y_pred, average='weighted'))


def _print_metrics(label, bucket):
    print(f"   Temps: {bucket['temps'][-1]:.2f}s")
    print(f"   Accuracy CV: {bucket['accuracy_cv'][-1]:.4f}")
    print(f"   Accuracy Test: {bucket['accuracy_test'][-1]:.4f}")
    print(f"   F1 Score: {bucket['f1_test'][-1]:.4f}")


def plot_svc_comparison(moyennes):
    """Grille de 4 barres (temps, accuracy CV, accuracy test, F1) comparant
    kNN et les 3 SVC, avec écart-type en barre d'erreur."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ('temps_moyen', 'temps_std', 'Execution Time', 'Time (s)', axes[0, 0]),
        ('accuracy_cv_moyen', 'accuracy_cv_std', 'Accuracy Cross-Validation', 'Accuracy', axes[0, 1]),
        ('accuracy_test_moyen', 'accuracy_test_std', 'Accuracy Test Set', 'Accuracy', axes[1, 0]),
        ('f1_test_moyen', 'f1_test_std', 'F1 Score (weighted)', 'F1 Score', axes[1, 1]),
    ]

    for mean_key, std_key, title, ylabel, ax in metrics:
        values = [moyennes[m][mean_key] for m in MODELES]
        errors = [moyennes[m][std_key] for m in MODELES]
        bars = ax.bar(MODELES_LABELS, values, yerr=errors, color=COLORS, alpha=0.7,
                       capsize=5, edgecolor='black', linewidth=1.5)
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if 'accuracy' in mean_key or 'f1' in mean_key:
            ax.set_ylim([0, 1])
            ax.axhline(y=0.86, color='red', linestyle='--', linewidth=2, alpha=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{val:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.show()
