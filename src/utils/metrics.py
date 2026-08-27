import numpy as np
import matplotlib.pyplot as plt

def log_loss(y_true, y_prob):
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps) #to avoid log(0) [array, inferior_limit, superior_limit]
    return -np.mean(
        y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
    )

def brier_score(y_true, y_prob):  #bounded in [0,1]
    return np.mean((y_prob - y_true) ** 2) #still how much prob. differs from reality

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean() #confronting elements and corrected predictions


def reliability_diagram(y_true, probs, n_bins=10, label=None, strategy="uniform"):

    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    # turning the array into a np array because next
    # we will use it in operations like mean, sum...


    if strategy == "uniform":
        bins = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":

        # borders calculated on the same data -> every bin has almost the same number of points
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(probs, quantiles)
        bins = np.unique(bins)  # avoids duplicated borders if there are more repeated values
        bins[0] = 0.0
        bins[-1] = 1.0 #last element of array
    else:
        raise ValueError("strategy has to be 'uniform' o 'quantile'")


    # calculating number of bins
    n_bins_eff = len(bins) - 1

    bin_ids = np.digitize(probs, bins) - 1

    # without this clip, points with prob == bins [-1] like 1.0 end up
    # in an out of range bin and silently discarded

    bin_ids = np.clip(bin_ids, 0, n_bins_eff - 1)

    bin_true = []
    bin_pred = []

    for i in range(n_bins_eff):
        # boolean mask for true/false of bin
        mask = bin_ids == i
        if np.sum(mask) > 0:

            #mean on true mask
            bin_true.append(np.mean(y_true[mask]))

            #mean on predicted probabilities
            bin_pred.append(np.mean(probs[mask]))

    plt.plot(bin_pred, bin_true, marker='o', label=label)