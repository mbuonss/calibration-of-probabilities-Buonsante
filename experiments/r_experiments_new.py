from matplotlib import pyplot as plt

from src.data_loader import load_adult_dataset, load_breast_cancer_dataset
from src.logistic_regression import LogisticRegression
from src.metrics import accuracy, log_loss, brier_score, reliability_diagram
from src.calibration import Isotonic, PlattScal
from src.Random_Forest import RandomForest


def evaluate_model(y_test, probs_test, preds_test):
    acc = accuracy(y_test, preds_test)
    ll = log_loss(y_test, probs_test)
    bs = brier_score(y_test, probs_test)
    return acc, ll, bs

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


def run_experiment(dataset_name, loader):

    print("-" * 450)
    print(f"DATASET: {dataset_name}")
    print("-" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = loader()


    #Logistic Regression
    model = LogisticRegression(lr=0.1, num_iter=1000)
    model.fit(X_train, y_train)

    probs_val = model.pred_probab(X_val)
    probs_test = model.pred_probab(X_test)
    preds_test = model.predict(X_test)

    acc, ll, bs = evaluate_model(y_test, probs_test, preds_test)
    print_results("Logistic Regression (not calibrated)", preds_test, acc, ll, bs)

    #Platt Logistic
    platt = PlattScal()
    platt.fit(probs_val, y_val)

    probTestPlatt = platt.pred_probab(probs_test)
    predsTestPlatt = (probTestPlatt >= 0.5).astype(int)

    acc_platt, ll_platt, bs_platt = evaluate_model(y_test, probTestPlatt, predsTestPlatt)
    print_results("Logistic + Platt", predsTestPlatt, acc_platt, ll_platt, bs_platt)


    #Isotonic Logistic
    isotonic = Isotonic()
    isotonic.fit(probs_val, y_val)

    probTestIso = isotonic.pred_probab(probs_test)
    predsTestIso = (probTestIso >= 0.5).astype(int)

    acc_iso, ll_iso, bs_iso = evaluate_model(y_test, probTestIso, predsTestIso)
    print_results("Logistic + Isotonic", predsTestIso, acc_iso, ll_iso, bs_iso)

    plot_reliability_single(
        y_test, probs_test,
        "Logistic",
        f"Reliability Diagram - Logistic - {dataset_name}",
        f"Figure_Logistic_{dataset_name}.png"
    )

    plot_reliability_single(
        y_test, probTestIso,
        "Logistic + Isotonic",
        f"Reliability Diagram - Logistic + Isotonic - {dataset_name}",
        f"Figure_Logistic_Iso_{dataset_name}.png"
    )

    plot_reliability_single(
        y_test, probTestPlatt,
        "Logistic + Platt",
        f"Reliability Diagram - Logistic + Platt - {dataset_name}",
        f"Figure_Logistic_Platt_{dataset_name}.png"
    )

    plt.figure()
    reliability_diagram(y_test, probs_test, label="Logistic")
    reliability_diagram(y_test, probTestIso, label="Isotonic")
    reliability_diagram(y_test, probTestPlatt, label="Platt")
    plt.plot([0, 1], [0, 1], "--", label="Perfect")
    plt.xlabel("Predicted probability")
    plt.ylabel("True frequency")
    plt.legend()
    plt.title(f"Reliability Diagram - {dataset_name}")
    plt.savefig(f"Figure_1_{dataset_name}.png", dpi=300)
    plt.show()
    plt.close()


    #RF
    rf = RandomForest(n_trees=10, max_depth=8, min_samples_split=2, n_features=20)
    rf.fit(X_train, y_train)

    probs_val_rf = rf.pred_probab(X_val)
    probs_test_rf = rf.pred_probab(X_test)
    preds_test_rf = rf.predict(X_test)

    acc_rf, ll_rf, bs_rf = evaluate_model(y_test, probs_test_rf, preds_test_rf)
    print_results("Random Forest (not calibrated)", preds_test_rf, acc_rf, ll_rf, bs_rf)

    #Platt Random Forest
    platt_rf = PlattScal()
    platt_rf.fit(probs_val_rf, y_val)

    probTestPlattRf = platt_rf.pred_probab(probs_test_rf)
    predsTestPlattRf = (probTestPlattRf >= 0.5).astype(int)

    acc_platt_rf, ll_platt_rf, bs_platt_rf = evaluate_model(
        y_test, probTestPlattRf, predsTestPlattRf
    )
    print_results("Random Forest + Platt", predsTestPlattRf, acc_platt_rf, ll_platt_rf, bs_platt_rf)

    # Isotonic on Random Forest
    isotonic_rf = Isotonic()
    isotonic_rf.fit(probs_val_rf, y_val)

    probTestIsoRf = isotonic_rf.pred_probab(probs_test_rf)
    predsTestIsoRf = (probTestIsoRf >= 0.5).astype(int)

    acc_iso_rf, ll_iso_rf, bs_iso_rf = evaluate_model(
        y_test, probTestIsoRf, predsTestIsoRf
    )
    print_results("Random Forest + Isotonic", predsTestIsoRf, acc_iso_rf, ll_iso_rf, bs_iso_rf)

    plot_reliability_single(
        y_test, probs_test_rf,
        "Random Forest",
        f"Reliability Diagram - Random Forest - {dataset_name}",
        f"Figure_RF_{dataset_name}.png"
    )

    plot_reliability_single(
        y_test, probTestIsoRf,
        "RF + Isotonic",
        f"Reliability Diagram - RF + Isotonic - {dataset_name}",
        f"Figure_RF_Iso_{dataset_name}.png"
    )

    plot_reliability_single(
        y_test, probTestPlattRf,
        "RF + Platt",
        f"Reliability Diagram - RF + Platt - {dataset_name}",
        f"Figure_RF_Platt_{dataset_name}.png"
    )

    # Reliability diagram Random Forest
    plt.figure()
    reliability_diagram(y_test, probs_test_rf, label="Random Forest")
    reliability_diagram(y_test, probTestIsoRf, label="RF + Isotonic")
    reliability_diagram(y_test, probTestPlattRf, label="RF + Platt")
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