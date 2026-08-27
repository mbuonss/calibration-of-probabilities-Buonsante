from matplotlib import pyplot as plt

from src.utils.data_loader import load_adult_dataset, load_breast_cancer_dataset
from sklearn.linear_model import LogisticRegression
from src.utils.metrics import accuracy, log_loss, brier_score, reliability_diagram
from src.utils.calibration import Isotonic, PlattScal
from sklearn.ensemble import RandomForestClassifier


# evaluation metrics
def evaluate_model(y_test, probs_test, preds_test):
    acc = accuracy(y_test, preds_test)
    ll = log_loss(y_test, probs_test)
    bs = brier_score(y_test, probs_test)
    return acc, ll, bs


# plotting
def plot_reliability_single(y_test, probs, label, title, filename):
    plt.figure()
    reliability_diagram(y_test, probs, label=label)
    plt.plot([0, 1], [0, 1], "--", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("True frequency")
    plt.legend()
    plt.title(title)
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()

def print_results(title, preds, acc, ll, bs):
    print(f"- {title} -")
    print("Predictions:", preds[:10])
    print("Accuracy:", acc)
    print("Log Loss:", ll)
    print("Brier Score:", bs)
    print()


def evaluate_calibrated_model(
        model,
        model_name,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
):

    # training only on training data
    model.fit(X_train, y_train)

    #probabilities on validation set with only second column (positive)
    probs_val = model.predict_proba(X_val)[:, 1]

    #non calibrated probabilities on positive class
    probs_test = model.predict_proba(X_test)[:, 1]

    #final prediction on default threshold
    preds_test = model.predict(X_test)

    acc, ll, bs = evaluate_model(y_test, probs_test, preds_test)


    #results before calibration
    print_results(
        f"{model_name} (not calibrated)",
        preds_test,
        acc,
        ll,
        bs
    )

    # platt
    platt = PlattScal()
    platt.fit(probs_val, y_val)

    probs_platt = platt.pred_probab(probs_test)

    #conversion of binary entries
    preds_platt = (probs_platt >= 0.5).astype(int)

    #re-calculation after platt's prob and predictions
    acc_platt, ll_platt, bs_platt = evaluate_model(
        y_test,
        probs_platt,
        preds_platt
    )

    print_results(
        f"{model_name} + Platt",
        preds_platt,
        acc_platt,
        ll_platt,
        bs_platt
    )

    # isotonic
    isotonic = Isotonic()
    isotonic.fit(probs_val, y_val)

    probs_iso = isotonic.pred_probab(probs_test)
    preds_iso = (probs_iso >= 0.5).astype(int)

    acc_iso, ll_iso, bs_iso = evaluate_model(
        y_test,
        probs_iso,
        preds_iso
    )

    print_results(
        f"{model_name} + Isotonic",
        preds_iso,
        acc_iso,
        ll_iso,
        bs_iso
    )

    return probs_test, probs_platt, probs_iso


def run_experiment(dataset_name, loader):

    print("-" * 60)
    print(f"DATASET: {dataset_name}")
    print("-" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = loader()




    probs_test, prob_test_platt, prob_test_iso = (
        evaluate_calibrated_model(
            LogisticRegression(max_iter=1000),
            "Logistic Regression",
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )
    )

    plot_reliability_single(
        y_test, probs_test,
        "Logistic",
        f"Reliability Diagram - Logistic - {dataset_name}",
        f"Figure_Logistic_{dataset_name}.png"
    )

    plot_reliability_single(
        y_test, prob_test_iso,
        "Logistic + Isotonic",
        f"Reliability Diagram - Logistic + Isotonic - {dataset_name}",
        f"Figure_Logistic_Iso_{dataset_name}.png"
    )

    plot_reliability_single(
        y_test, prob_test_platt,
        "Logistic + Platt",
        f"Reliability Diagram - Logistic + Platt - {dataset_name}",
        f"Figure_Logistic_Platt_{dataset_name}.png"
    )

    plt.figure()
    reliability_diagram(y_test, probs_test, label="Logistic")
    reliability_diagram(y_test, prob_test_iso, label="Isotonic")
    reliability_diagram(y_test, prob_test_platt, label="Platt")
    plt.plot([0, 1], [0, 1], "--", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("True frequency")
    plt.legend()
    plt.title(f"Reliability Diagram - {dataset_name}")
    plt.savefig(f"Figure_1_{dataset_name}.png", dpi=300)
    plt.show()
    plt.close()




    probs_test_rf, prob_test_platt_rf, prob_test_iso_rf = (
        evaluate_calibrated_model(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=2,
                max_features=20,
                random_state=42
            ),
            "Random Forest",
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )
    )


    plot_reliability_single(
        y_test, probs_test_rf,
        "Random Forest",
        f"Reliability Diagram - Random Forest - {dataset_name}",
        f"Figure_RF_{dataset_name}.png"
    )

    plot_reliability_single(
        y_test, prob_test_iso_rf,
        "RF + Isotonic",
        f"Reliability Diagram - RF + Isotonic - {dataset_name}",
        f"Figure_RF_Iso_{dataset_name}.png"
    )

    plot_reliability_single(
        y_test, prob_test_platt_rf,
        "RF + Platt",
        f"Reliability Diagram - RF + Platt - {dataset_name}",
        f"Figure_RF_Platt_{dataset_name}.png"
    )

    # Reliability diagram Random Forest
    plt.figure()
    reliability_diagram(y_test, probs_test_rf, label="Random Forest")
    reliability_diagram(y_test, prob_test_iso_rf, label="RF + Isotonic")
    reliability_diagram(y_test, prob_test_platt_rf, label="RF + Platt")
    plt.plot([0, 1], [0, 1], "--", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("True frequency")
    plt.legend()
    plt.title(f"Reliability Diagram - Random Forest - {dataset_name}")
    plt.savefig(f"Figure_2_{dataset_name}.png", dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":

    datasets = [
        ("Adult", load_adult_dataset),
        ("BreastCancer", load_breast_cancer_dataset),
    ]

    for dataset_name, loader in datasets:
        run_experiment(dataset_name, loader)