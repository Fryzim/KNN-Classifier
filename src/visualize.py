"""Graphiques partagés par les trois pipelines k-NN (naïf/précalculé,
inégalité triangulaire, kd-tree)."""
import matplotlib.pyplot as plt


def plot_accuracy_vs_k(mean_accuracies, titre="Performance du k-NN en fonction de k (Cross-Validation)"):
    """Précision moyenne (CV) en fonction de k, avec le meilleur k annoté."""
    k_values = list(mean_accuracies.keys())
    accuracies = list(mean_accuracies.values())

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Valeur de k')
    plt.ylabel('Accuracy moyenne')
    plt.title(titre)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)

    best_k = max(mean_accuracies, key=mean_accuracies.get)
    best_acc = mean_accuracies[best_k]
    min_k = min(mean_accuracies, key=mean_accuracies.get)
    min_acc = mean_accuracies[min_k]
    plt.plot(best_k, best_acc, 'ro', markersize=10, label=f'Meilleur k: {best_k} ({best_acc:.3f})')
    plt.plot([k_values[0], best_k], [best_acc, best_acc], "r:")
    plt.plot([best_k, best_k], [min_acc, best_acc], "r:")
    plt.legend()
    plt.show()


def plot_pipeline_comparison(labels, best_k_lists, best_accuracy_lists, final_accuracy_lists, time_lists):
    """4 graphiques comparant plusieurs pipelines k-NN (un point/barre par
    pipeline) : meilleur k, accuracy finale (test), meilleure accuracy (CV),
    et temps d'exécution."""
    import numpy as np

    valeurs = [np.mean(v) for v in best_k_lists]
    plt.bar(labels, valeurs, color='skyblue', edgecolor='black')
    for i, valeur in enumerate(valeurs):
        plt.text(i, valeur + 0.05, f"{valeur:.2f}", ha='center', va='bottom')
    plt.title("Best k")
    plt.xlabel("Pipeline")
    plt.ylabel("Best k mean")
    plt.show()

    valeurs = [np.mean(v) for v in final_accuracy_lists]
    plt.bar(labels, valeurs)
    for i, valeur in enumerate(valeurs):
        plt.text(i, valeur + 0.001, f"{valeur:.4f}", ha='center', va='bottom')
    plt.ylim(0, max(valeurs) + 0.05)
    plt.title("Final accuracy (test time)")
    plt.xlabel("Pipeline")
    plt.ylabel("Final accuracy mean")
    plt.show()

    valeurs = [np.mean(v) for v in best_accuracy_lists]
    plt.bar(labels, valeurs, edgecolor='black')
    for i, valeur in enumerate(valeurs):
        plt.text(i, valeur + 0.001, f"{valeur:.4f}", ha='center', va='bottom')
    plt.title("Best accuracy (train time)")
    plt.xlabel("Pipeline")
    plt.ylabel("Best accuracy mean")
    plt.show()

    valeurs = [np.mean(v) for v in time_lists]
    plt.bar(labels, valeurs, color='green', edgecolor='black')
    for i, valeur in enumerate(valeurs):
        plt.text(i, valeur + 0.05, f"{valeur:.2f}", ha='center', va='bottom')
    plt.title("Execution time")
    plt.xlabel("Pipeline")
    plt.ylabel("Time (s)")
    plt.show()
