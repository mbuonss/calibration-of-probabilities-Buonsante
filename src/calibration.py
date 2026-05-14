from src.logistic_regression import LogisticRegression
import numpy as np

class PlattScal:

    def __init__(self):
        self.model = LogisticRegression(lr=0.1, num_iter=1000)

    def fit(self, probs, y):
        probs = probs.reshape(-1, 1)
        self.model.fit(probs, y)

    def pred_probab(self, probs):
        probs = probs.reshape(-1, 1)
        return self.model.pred_probab(probs)


class Isotonic:

    def __init__(self):
        self.thresholds = None
        self.values = None

    def fit(self, probs, y):
        probs = np.asarray(probs)
        y = np.asarray(y)

        # 1. Ordina in base alle probabilità predette
        order = np.argsort(probs)
        probs_sorted = probs[order]
        y_sorted = y[order]

        # 2. Crea blocchi iniziali
        blocks = []
        for p, target in zip(probs_sorted, y_sorted):
            blocks.append({
                "sum_y": target,
                "count": 1,
                "min_prob": p,
                "max_prob": p
            })

            # 3. Pool Adjacent Violators
            while len(blocks) >= 2:
                last = blocks[-1]
                prev = blocks[-2]

                last_value = last["sum_y"] / last["count"]
                prev_value = prev["sum_y"] / prev["count"]

                if prev_value <= last_value:
                    break

                merged = {
                    "sum_y": prev["sum_y"] + last["sum_y"],
                    "count": prev["count"] + last["count"],
                    "min_prob": prev["min_prob"],
                    "max_prob": last["max_prob"]
                }

                blocks.pop() #rimuovo gli ultimi due dalla lista
                blocks.pop()
                blocks.append(merged) #aggiungo il blocco unito

        # 4. Salva soglie e valori calibrati
        self.thresholds = np.array([b["max_prob"] for b in blocks])
        self.values = np.array([b["sum_y"] / b["count"] for b in blocks]) #salvo il valore calibrato di ogni blocco (media etichette vere)

    def pred_probab(self, probs):
        probs = np.asarray(probs)

        indices = np.searchsorted(self.thresholds, probs, side="left")
        indices = np.clip(indices, 0, len(self.values) - 1)

        return self.values[indices]