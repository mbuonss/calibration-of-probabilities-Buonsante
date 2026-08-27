import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_adult_dataset(random_state=42):

    # loading dataset

    adult = fetch_openml(name='adult', version=2, as_frame=True)
    X = adult.data
    y = adult.target.astype(str).str.contains('>50K').astype(int)


    # dividing train/val/test

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )


    # identifying columns

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns


    # preprocessing

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols), #transforming numbers (-1, 0, 1)
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols) #dividing binary columns (001, 010, 100)
        ]
    )

    # fit only on train
    X_train = preprocessor.fit_transform(X_train)

    # transform on val and test
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)


    # best method to convert pandas into numpy array
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_breast_cancer_dataset(random_state=42):

    from sklearn.datasets import load_breast_cancer

    # load

    breast = load_breast_cancer(as_frame=True)

    X = breast.data
    y = breast.target


    # train/value/test division

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )


    # columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols)
    ])

    X_train = preprocessor.fit_transform(X_train)

    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    #in order to guarantee the return
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)


    return X_train, X_val, X_test, y_train, y_val, y_test


def analyze_raw_datasets():
    print("=" * 60)
    print(" ADULT DATASET ANALYSIS ")
    print("=" * 60)
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    X_adult = adult.data
    y_adult_raw = adult.target
    y_adult_num = y_adult_raw.astype(str).str.contains('>50K').astype(int)

    num_cols_adult = X_adult.select_dtypes(include=['int64', 'float64']).columns
    cat_cols_adult = X_adult.select_dtypes(include=['object', 'category']).columns
    missing_adult = X_adult.isna().sum().sum()

    print(f"Total Samples (N): {X_adult.shape[0]}")
    print(f"Original Features: {X_adult.shape[1]} (Numeric: {len(num_cols_adult)}, Categorical: {len(cat_cols_adult)})")
    print(f"Missing Values (NaN): {missing_adult}")
    print(f"Target Balance (y=1 ratio): {np.mean(y_adult_num)*100:.2f}%\n")

    print("=" * 60)
    print(" BREAST CANCER DATASET ANALYSIS ")
    print("=" * 60)
    from sklearn.datasets import load_breast_cancer
    bc = load_breast_cancer(as_frame=True)
    X_bc = bc.data
    y_bc = bc.target
    missing_bc = X_bc.isna().sum().sum()

    print(f"Total Samples (N): {X_bc.shape[0]}")
    print(f"Original Features: {X_bc.shape[1]}")
    print(f"Missing Values (NaN): {missing_bc}")
    print(f"Target Balance (y=1 ratio): {np.mean(y_bc)*100:.2f}%\n")


if __name__ == "__main__":
    analyze_raw_datasets()