from matplotlib import pyplot as plt

from src.data_loader import load_adult_dataset, load_breast_cancer_dataset
from src.logistic_regression import LogisticRegression
from src.metrics import accuracy, log_loss, brier_score, reliability_diagram
from src.calibration import Isotonic, PlattScal
from src.random_forest import RandomForest


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

    model.fit(X_train, y_train)

    probs_val = model.pred_probab(X_val)
    probs_test = model.pred_probab(X_test)
    preds_test = model.predict(X_test)

    acc, ll, bs = evaluate_model(y_test, probs_test, preds_test)
    print_results(
        f"{model_name} (not calibrated)",
        preds_test,
        acc,
        ll,
        bs
    )

    # Platt
    platt = PlattScal()
    platt.fit(probs_val, y_val)

    probs_platt = platt.pred_probab(probs_test)
    preds_platt = (probs_platt >= 0.5).astype(int)

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

    # Isotonic
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


    # #Logistic Regression
    # model = LogisticRegression(lr=0.1, num_iter=1000)
    # model.fit(X_train, y_train)
    #
    # probs_val = model.pred_probab(X_val)
    # probs_test = model.pred_probab(X_test)
    # preds_test = model.predict(X_test)
    #
    # acc, ll, bs = evaluate_model(y_test, probs_test, preds_test)
    # print_results("Logistic Regression (not calibrated)", preds_test, acc, ll, bs)
    #
    # #Platt Logistic
    # platt = PlattScal()
    # platt.fit(probs_val, y_val)
    #
    # prob_test_platt = platt.pred_probab(probs_test)
    # preds_test_platt = (prob_test_platt >= 0.5).astype(int)
    #
    # acc_platt, ll_platt, bs_platt = evaluate_model(y_test, prob_test_platt, preds_test_platt)
    # print_results("Logistic + Platt", preds_test_platt, acc_platt, ll_platt, bs_platt)
    #
    # #Isotonic Logistic
    # isotonic = Isotonic()
    # isotonic.fit(probs_val, y_val)
    #
    # prob_test_iso = isotonic.pred_probab(probs_test)
    # preds_test_iso = (prob_test_iso >= 0.5).astype(int)
    #
    # acc_iso, ll_iso, bs_iso = evaluate_model(y_test, prob_test_iso, preds_test_iso)
    # print_results("Logistic + Isotonic", preds_test_iso, acc_iso, ll_iso, bs_iso)

    probs_test, prob_test_platt, prob_test_iso = (
        evaluate_calibrated_model(
            LogisticRegression(lr=0.1, num_iter=1000),
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


    # #RF
    # rf = RandomForest(n_trees=10, max_depth=8, min_samples_split=2, n_features=20)
    # rf.fit(X_train, y_train)
    #
    # probs_val_rf = rf.pred_probab(X_val)
    # probs_test_rf = rf.pred_probab(X_test)
    # preds_test_rf = rf.predict(X_test)
    #
    # acc_rf, ll_rf, bs_rf = evaluate_model(y_test, probs_test_rf, preds_test_rf)
    # print_results("Random Forest (not calibrated)", preds_test_rf, acc_rf, ll_rf, bs_rf)
    #
    # #Platt Random Forest
    # platt_rf = PlattScal()
    # platt_rf.fit(probs_val_rf, y_val)
    #
    # prob_test_platt_rf = platt_rf.pred_probab(probs_test_rf)
    # preds_test_platt_rf = (prob_test_platt_rf >= 0.5).astype(int)
    #
    # acc_platt_rf, ll_platt_rf, bs_platt_rf = evaluate_model(
    #     y_test, prob_test_platt_rf, preds_test_platt_rf
    # )
    # print_results("Random Forest + Platt", preds_test_platt_rf, acc_platt_rf, ll_platt_rf, bs_platt_rf)
    #
    # # Isotonic on Random Forest
    # isotonic_rf = Isotonic()
    # isotonic_rf.fit(probs_val_rf, y_val)
    #
    # prob_test_iso_rf = isotonic_rf.pred_probab(probs_test_rf)
    # preds_test_iso_rf = (prob_test_iso_rf >= 0.5).astype(int)
    #
    # acc_iso_rf, ll_iso_rf, bs_iso_rf = evaluate_model(
    #     y_test, prob_test_iso_rf, preds_test_iso_rf
    # )
    # print_results("Random Forest + Isotonic", preds_test_iso_rf, acc_iso_rf, ll_iso_rf, bs_iso_rf)

    probs_test_rf, prob_test_platt_rf, prob_test_iso_rf = (
        evaluate_calibrated_model(
            RandomForest(
                n_trees=10,
                max_depth=8,
                min_samples_split=2,
                n_features=20
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