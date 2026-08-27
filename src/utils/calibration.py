from sklearn.linear_model import LogisticRegression
import numpy as np

class PlattScal:

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000) #only to correct probabilities

    def fit(self, probs, y):
        probs = probs.reshape(-1, 1) #logistic regression wants a matrix and not a vector
        self.model.fit(probs, y)

    def pred_probab(self, probs):
        probs = probs.reshape(-1, 1)
        return self.model.predict_proba(probs)[:, 1]


class Isotonic:

    def __init__(self):
        self.thresholds = None
        self.values = None

    def fit(self, probs, y):
        probs = np.asarray(probs)
        y = np.asarray(y)

        # orders based on predicted probabilities
        order = np.argsort(probs) #returns index for probs order
        probs_sorted = probs[order]
        y_sorted = y[order]

        # creation of initial blocks
        blocks = []
        for p, target in zip(probs_sorted, y_sorted):
            blocks.append({
                "sum_y": target, #sum of true labels
                "count": 1, #num of elements in the block
                "min_prob": p,
                "max_prob": p
            })

            # pool Adjacent Violators
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

                # removing last two values from the list

                blocks.pop()
                blocks.pop()

                # adding the merged block

                blocks.append(merged)

        # saves thresholds and calibrated values
        self.thresholds = np.array([b["max_prob"] for b in blocks])
        self.values = np.array([b["sum_y"] / b["count"] for b in blocks]) # i save the calibrated value of every block (avg of true labels)

    def pred_probab(self, probs):
        probs = np.asarray(probs)

        indices = np.searchsorted(self.thresholds, probs, side="left")
        indices = np.clip(indices, 0, len(self.values) - 1)

        return self.values[indices]