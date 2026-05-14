import numpy as np
import matplotlib.pyplot as plt

def log_loss(y_true, y_prob):
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(
        y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
    )

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def reliability_diagram(y_true, probs, n_bins=10, label=None):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1   #trasforma e ti dice in quale intervallo cade

    bin_true = [] #freq reale
    bin_pred = [] # freq predetta

    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) > 0:
            bin_true.append(np.mean(y_true[mask]))
            bin_pred.append(np.mean(probs[mask]))

    plt.plot(bin_pred, bin_true, marker='o', label=label)