import numpy as np
from matplotlib import pyplot as plt
from src.data_loader import load_adult_dataset, load_breast_cancer_dataset
from src.logistic_regression import LogisticRegression
from src.metrics import accuracy, log_loss, brier_score, reliability_diagram
from src.calibration import Isotonic, PlattScal
from src.Random_Forest import RandomForest

# Carica dati
X_train, X_val, X_test, y_train, y_val, y_test = load_breast_cancer_dataset()

model = LogisticRegression(lr=0.1, num_iter=1000) # base model

model.fit(X_train, y_train) # base model training

# not calibrated predictions
probs_val = model.pred_probab(X_val)
probs_test = model.pred_probab(X_test)
preds_test = model.predict(X_test)

# not calibrated model's metrics
acc = accuracy(y_test, preds_test)
ll = log_loss(y_test, probs_test)
bs = brier_score(y_test, probs_test)

print("- Logistic Regression (not calibrated) -")
print("Predictions:", preds_test[:10])
print("Accuracy:", acc)
print("Log Loss:", ll)
print("Brier Score:", bs)



# Platt
platt = PlattScal()
platt.fit(probs_val, y_val)
probTestPlatt = platt.pred_probab(probs_test)
predsTestPlatt = (probTestPlatt >= 0.5).astype(int)

acc_platt = accuracy(y_test, predsTestPlatt)
ll_platt = log_loss(y_test, probTestPlatt)
bs_platt = brier_score(y_test, probTestPlatt)

print("\n- Logistic + Platt -")
print("Predictions:", predsTestPlatt[:10])
print("Accuracy:", acc_platt)
print("Log Loss:", ll_platt)
print("Brier Score:", bs_platt)



# Isotonic
isotonic = Isotonic()
isotonic.fit(probs_val, y_val)
probTestIso = isotonic.pred_probab(probs_test)
predsTestIso = (probTestIso >= 0.5).astype(int)

acc_iso = accuracy(y_test, predsTestIso)
ll_iso = log_loss(y_test, probTestIso)
bs_iso = brier_score(y_test, probTestIso)

print("\n- Logistic + Isotonic -")
print("Predictions:", predsTestIso[:10])
print("Accuracy:", acc_iso)
print("Log Loss:", ll_iso)
print("Brier Score:", bs_iso)


plt.figure()

reliability_diagram(y_test, probs_test, label="Logistic")
reliability_diagram(y_test, probTestIso, label="Isotonic")
reliability_diagram(y_test, probTestPlatt, label="Platt")

plt.plot([0, 1], [0, 1], '--', label="Perfect")

plt.xlabel("Predicted probability")
plt.ylabel("True frequency")
plt.legend()
plt.title("Reliability Diagram - Bank")



rf = RandomForest(n_trees=5, max_depth=5, min_samples_split=2, n_features=10)

rf.fit(X_train, y_train)

probs_test_rf = rf.pred_probab(X_test)
preds_test_rf = rf.predict(X_test)

acc_rf = accuracy(y_test, preds_test_rf)
ll_rf = log_loss(y_test, probs_test_rf)
bs_rf = brier_score(y_test, probs_test_rf)

print("\n- Random Forest (not calibrated) -")
print("Predictions:", preds_test_rf[:10])
print("Accuracy:", acc_rf)
print("Log Loss:", ll_rf)
print("Brier Score:", bs_rf)

probs_val_rf = rf.pred_probab(X_val)

platt_rf = PlattScal()
platt_rf.fit(probs_val_rf, y_val)

probTestPlattRf = platt_rf.pred_probab(probs_test_rf)
predsTestPlattRf = (probTestPlattRf >= 0.5).astype(int)

print("\n- Random Forest + Platt -")
print("Accuracy:", accuracy(y_test, predsTestPlattRf))
print("Log Loss:", log_loss(y_test, probTestPlattRf))
print("Brier Score:", brier_score(y_test, probTestPlattRf))


isotonic_rf = Isotonic()
isotonic_rf.fit(probs_val_rf, y_val)

probTestIsoRf = isotonic_rf.pred_probab(probs_test_rf)
predsTestIsoRf = (probTestIsoRf >= 0.5).astype(int)

print("\n- Random Forest + Isotonic -")
print("Accuracy:", accuracy(y_test, predsTestIsoRf))
print("Log Loss:", log_loss(y_test, probTestIsoRf))
print("Brier Score:", brier_score(y_test, probTestIsoRf))

plt.figure()

reliability_diagram(y_test, probs_test_rf, label="Random Forest")
reliability_diagram(y_test, probTestIsoRf, label="RF + Isotonic")
reliability_diagram(y_test, probTestPlattRf, label="RF + Platt")

plt.plot([0, 1], [0, 1], '--', label="Perfect")

plt.xlabel("Predicted probability")
plt.ylabel("True frequency")
plt.legend()
plt.title("Reliability Diagram - Random Forest - Bank")

print("Distribuzione y_test:", np.bincount(y_test))
plt.show()