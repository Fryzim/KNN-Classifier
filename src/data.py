"""Chargement du dataset Waveform et découpage train/test."""
import pandas as pd


def load_dataset(path="waveform.data.csv", seed=42):
    """Charge le CSV (sans en-tête) et mélange les lignes de façon reproductible."""
    dataset = pd.read_csv(path, sep=',', header=None)
    return dataset.sample(frac=1.0, random_state=seed)


def split_train_test(dataset, n_train=4000):
    """Sépare features (toutes les colonnes sauf la dernière) et label
    (dernière colonne), puis découpe en train/test sur les n_train premières lignes."""
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    X_train = X.iloc[:n_train].reset_index(drop=True)
    y_train = y.iloc[:n_train].reset_index(drop=True)
    X_test = X.iloc[n_train:].reset_index(drop=True)
    y_test = y.iloc[n_train:].reset_index(drop=True)

    return X_train, y_train, X_test, y_test
