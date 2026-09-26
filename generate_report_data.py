"""Runs the 4 k-NN pipelines on a real (subsampled) slice of waveform.data.csv
and writes reporting/results.json, consumed by docs/index.html.

Run from the repo root: python generate_report_data.py
"""
import json, time, sys
sys.path.insert(0, '.')
from src.data import load_dataset, split_train_test
from src.knn import cross_validation_knn, evaluer_modele
from src.knn_precomputed import run_pipeline as run_precomputed
from src.triangle_inequality import run_pipeline as run_triangle
from src.kdtree import run_pipeline as run_kdtree
from src.reduction import reduction_bias_removal, reduction_condensing

t_start = time.time()

dataset = load_dataset("waveform.data.csv", seed=42)
dataset = dataset.iloc[:1300].reset_index(drop=True)  # real subsample: 1000 train / 300 test
X_train, y_train, X_test, y_test = split_train_test(dataset, n_train=1000)
print(f"split: train={len(X_train)} test={len(X_test)}", flush=True)

k_values = list(range(1, 50, 2))  # 25 values

print("Running baseline CV...", flush=True)
t0 = time.time()
best_k, best_acc, mean_accs = cross_validation_knn(X_train, y_train, k_values, n_folds=3)
baseline_final = evaluer_modele(X_test, y_test, best_k, 'euclidean', X_train, y_train)
baseline_time = time.time() - t0
print(f"baseline done in {baseline_time:.1f}s: best_k={best_k} cv={best_acc:.4f} test={baseline_final:.4f}", flush=True)

X_np_train, y_np_train = X_train.to_numpy(), y_train.to_numpy()
X_np_test, y_np_test = X_test.to_numpy(), y_test.to_numpy()

print("Running reduction...", flush=True)
t0 = time.time()
Xc, yc = reduction_bias_removal(X_np_train, y_np_train)
Xc2, yc2 = reduction_condensing(Xc, yc)
reduction_time = time.time() - t0
print(f"reduction done in {reduction_time:.1f}s: {X_np_train.shape[0]} -> {Xc.shape[0]} -> {Xc2.shape[0]}", flush=True)

results = {}
for name, fn in [("precomputed", run_precomputed), ("triangle", run_triangle), ("kdtree", run_kdtree)]:
    print(f"Running {name} pipeline...", flush=True)
    t0 = time.time()
    if name == "kdtree":
        bk, ba, ma, fa = fn(list(map(tuple, X_np_train)), list(y_np_train), list(map(tuple, X_np_test)), list(y_np_test), k_values=k_values)
    else:
        bk, ba, ma, fa = fn(X_np_train, y_np_train, X_np_test, y_np_test, k_values=k_values)
    elapsed = time.time() - t0
    results[name] = {"best_k": int(bk), "best_accuracy_cv": float(ba), "final_accuracy": float(fa), "time_s": round(elapsed, 2)}
    print(f"{name} done in {elapsed:.1f}s: best_k={bk} cv={ba:.4f} test={fa:.4f}", flush=True)

out = {
    "sample": {"n_train": len(X_train), "n_test": len(X_test), "n_full_dataset": 5000},
    "baseline": {
        "k_values": k_values,
        "mean_accuracies": {str(k): round(v, 4) for k, v in mean_accs.items()},
        "best_k": int(best_k), "best_accuracy_cv": round(float(best_acc), 4),
        "final_accuracy": round(float(baseline_final), 4), "time_s": round(baseline_time, 2),
    },
    "reduction": {
        "n_before": int(X_np_train.shape[0]),
        "n_after_bias_removal": int(Xc.shape[0]),
        "n_after_condensing": int(Xc2.shape[0]),
        "time_s": round(reduction_time, 2),
    },
    "pipelines": results,
}

with open('reporting/results.json', 'w') as f:
    json.dump(out, f, indent=1)

print(f"TOTAL TIME: {time.time() - t_start:.1f}s", flush=True)
