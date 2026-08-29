# Calibration of Probabilities — Buonsante

*Statistical Methods for Machine Learning — A.Y. 2025/2026*  
*Università degli Studi di Milano*  
*Student:* Michele Davide Buonsante  

---

## Project Overview

This repository contains the implementation of the project for the Statistical Methods for Machine Learning course, focused on the *Calibration of Probabilities* in binary classification tasks.

The objective of the project is to evaluate and improve the reliability of probabilistic predictions produced by machine learning classifiers through post-processing calibration techniques.

### Key Highlights & Methodology:
- **Base Models:** Logistic Regression and Random Forest (trained via `scikit-learn`).
- **Custom Post-Processing Calibration (From Scratch):**
  - **Platt Scaling:** Logistic transformation on raw output probabilities.
  - **Isotonic Regression:** Non-parametric calibration using the **Pool Adjacent Violators (PAV)** algorithm implemented from scratch.


The project includes custom implementations of classification models, calibration methods, evaluation metrics, and reliability diagrams.

---

## Implemented Models

### Classification Algorithms
- Logistic Regression
- Random Forest built on top of custom Decision Trees

### Calibration Techniques
- Platt Scaling
- Isotonic Regression using the *Pool Adjacent Violators (PAV)* algorithm

### Evaluation Metrics
- Accuracy
- Log-Loss
- Brier Score
- Reliability Diagrams

---

## Project Structure

```text
├── main/
│   └── main.py
│       # Main script used to run the experiments
│
├── src/
│   ├── calibration.py
│   │   # Platt Scaling and Isotonic Regression implementations
│   │
│   ├── data_loader.py
│   │   # Dataset loading and preprocessing
│   │
│   └── metrics.py
│       # Evaluation metrics and plotting utilities
│
├── Report - Calibration_Project_Buonsante.pdf
│   # Final project report
│
└── README.md
```

---

## Requirements

The project requires *Python 3.x* and the following libraries:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## Running the Experiments

To execute the project, run the following command from the repository root:

```bash
python -m main.main
```

---

## Output

### Console Output
During execution, the script prints the following metrics for each model configuration:

- Accuracy
- Log-Loss
- Brier Score

### Generated Visualizations
The program automatically generates *Reliability Diagrams* for calibrated and non-calibrated models.

The plots are:

- displayed during execution
- saved as .png files inside the project directory

Example output files:

```text
Figure_Logistic_Adult.png
Figure_RF_Iso_BreastCancer.png
```

---

## Datasets

The experiments are performed on:

- Adult Dataset
- Breast Cancer Dataset

using preprocessing utilities implemented inside the project.

---

## Report

The repository also contains the final PDF report describing:

- theoretical background
- implementation details
- experimental setup
- calibration results and analysis
