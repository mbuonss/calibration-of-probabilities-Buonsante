import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_adult_dataset(random_state=42):

    # 1. Caricamento dataset

    adult = fetch_openml(name='adult', version=2, as_frame=True)
    X = adult.data
    y = (adult.target == '>50K').astype(int)


    #divisione train/val/test

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )


    # identifica colonne

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns


    # preprocessing

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols), #numeri trasformati in -1, 0, 1...
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols) #colonne divise non in 1,2,3 ma 100 010 001 es.
        ]
    )

    # fit solo su train
    X_train = preprocessor.fit_transform(X_train)

    # transform su val e test
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)


    # conversione in numpy

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


    # divisione train/value/test

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )


    # colonne
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols)
    ])

    X_train = preprocessor.fit_transform(X_train)

    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    #labels to numpy arrays
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)


    return X_train, X_val, X_test, y_train, y_val, y_test