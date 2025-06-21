# ML Interview Craft Exercise Template
# Author: Qimeng Chen
# Purpose: Reusable pipeline for classification problems (90-minute exercise)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1. Load and Explore Data
# -----------------------------
def load_data(filepath, sep= ','):
    df = pd.read_csv(filepath, sep = sep)
    print(df.head())
    print(df.info())
    print(df.describe())
    return df


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def visualize_data(df, target_col, max_categories=10):
    print("\n=== Starting Data Visualization ===")
    # Target Distribution
    print("\n--- Visualizing Target Distribution ---")
    sns.countplot(data=df, x=target_col)
    plt.title("Target Variable Distribution")
    plt.show()

    # Correlation Heatmap
    print("\n--- Correlation Heatmap ---")
    plt.figure(figsize=(12, 8))
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation")
    plt.show()

    # Numerical Feature Distributions
    print("\n--- Numerical Feature Distributions ---")
    num_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(target_col, errors='ignore')
    df[num_features].hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    # Target vs Numerical Feature Boxplots
    print("\n--- Target vs Numerical Feature Boxplots ---")
    for col in num_features:
        sns.boxplot(data=df, x=target_col, y=col)
        plt.title(f'{col} vs. {target_col}')
        plt.tight_layout()
        plt.show()

    # Target vs Numerical Feature KDEs
    print("\n--- KDE Plot of Numerical Features by Target ---")
    for col in num_features:
        plt.figure(figsize=(8, 5))
        for cls in df[target_col].unique():
            subset = df[df[target_col] == cls]
            sns.kdeplot(subset[col], label=f"{target_col} = {cls}", fill=True, common_norm=False, alpha=0.5)
        plt.title(f'Distribution of {col} by {target_col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.show()
    # Categorical Feature Distributions
    print("\n--- Categorical Feature Distributions ---")
    cat_features = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_features:
        top_cats = df[col].value_counts().nlargest(max_categories)
        sns.barplot(x=top_cats.index, y=top_cats.values)
        plt.title(f'{col} (Top {max_categories} Categories)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Target vs Categorical Feature
    print("\n--- Target vs Categorical Feature Breakdown ---")

    for col in cat_features:
        if df[col].nunique() <= max_categories:
            sns.countplot(data=df, x=col, hue=target_col)
            plt.title(f'{col} by {target_col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# -----------------------------
# 2. Preprocess Data
# -----------------------------
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def create_categorical_encoder(X_train, encoding_type="onehot", drop_first=False):
    """
    Create and fit a categorical encoder.

    Parameters:
        X_train (DataFrame): Input features
        encoding_type (str): 'onehot' or 'label'
        drop_first (bool): Whether to drop the first category (for one-hot)

    Returns:
        encoder: fitted encoder(s)
        cat_cols: list of categorical columns
    """
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    if encoding_type == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", drop="first" if drop_first else None)
        encoder.fit(X_train[cat_cols])
        return encoder, cat_cols

    elif encoding_type == "label":
        encoder = {}
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(X_train[col].astype(str))  # Ensure consistent str type
            encoder[col] = le
        return encoder, cat_cols

    else:
        raise ValueError("encoding_type must be 'onehot' or 'label'")


from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_numeric_scaler(X_train, method="standard"):
    """
    Fit a scaler on numeric columns.

    method: 'standard' or 'minmax'
    """
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")

    scaler.fit(X_train[num_cols])
    return scaler, num_cols


def transform_numeric(scaler, X, num_cols):
    return scaler.transform(X[num_cols])

from sklearn.model_selection import train_test_split

def split_data(df, target_col, method="random", date_col=None,
               train_size=0.7, val_size=0.15, test_size=0.15,
               stratify=False, time_boundaries=None, random_state=42):
    """
    Split data into train/val/test using either random or time-based logic.

    Parameters:
        df (pd.DataFrame): Full dataset including target
        target_col (str): Name of the target column
        method (str): "random" or "time"
        date_col (str): Name of datetime column (required for time-based split)
        train_size, val_size, test_size: Must sum to 1.0
        stratify (bool): Stratify target in random split
        time_boundaries (tuple): (train_end_date, val_end_date) if using specific cutoff dates
        random_state (int): Random seed

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

    if method == "random":
        X = df.drop(columns=[target_col])
        y = df[target_col]
        stratify_y = y if stratify else None

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_y)

        val_fraction = val_size / (train_size + val_size)
        stratify_y = y_train_val if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_fraction,
            random_state=random_state, stratify=stratify_y)

    elif method == "time":
        assert date_col is not None, "date_col is required for time-based split"
        df_sorted = df.sort_values(date_col).reset_index(drop=True)

        if time_boundaries:  # Manual date boundaries
            train_end, val_end = time_boundaries

            df_train = df_sorted[df_sorted[date_col] <= train_end]
            df_val   = df_sorted[(df_sorted[date_col] > train_end) & (df_sorted[date_col] <= val_end)]
            df_test  = df_sorted[df_sorted[date_col] > val_end]
        else:  # Percentile-based
            n = len(df_sorted)
            train_idx = int(train_size * n)
            val_idx   = int((train_size + val_size) * n)

            df_train = df_sorted.iloc[:train_idx]
            df_val   = df_sorted.iloc[train_idx:val_idx]
            df_test  = df_sorted.iloc[val_idx:]

        X_train = df_train.drop(columns=[target_col, date_col])
        y_train = df_train[target_col]
        X_val   = df_val.drop(columns=[target_col, date_col])
        y_val   = df_val[target_col]
        X_test  = df_test.drop(columns=[target_col, date_col])
        y_test  = df_test[target_col]

    else:
        raise ValueError("method must be 'random' or 'time'")

    return X_train, y_train, X_val, y_val, X_test, y_test

def build_preprocessor(X_train):
    cat_encoder, cat_cols = create_categorical_encoder(X_train, encoding_type="onehot", drop_first=True)
    num_scaler, num_cols = create_numeric_scaler(X_train, method="standard")

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols)
    ])

    return preprocessor

# -----------------------------
# 3. Train Model
# -----------------------------
def train_model(preprocessor, X_train, y_train, X_val, y_val, model_type="xgb", params=None):
    if model_type == "xgb":
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **(params or {}))
    elif model_type == "logit":
        model = LogisticRegression(max_iter=1000, **(params or {}))
    elif model_type == "lgbm":
        model = lgb.LGBMClassifier(**(params or {}))
    else:
        raise ValueError("Unsupported model_type. Choose from 'xgb', 'logit', 'lgbm'.")

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    y_val_prob = clf.predict_proba(X_val)[:, 1]

    print("\n--- Validation Performance ---")
    print(classification_report(y_val, y_val_pred))
    print("Validation ROC AUC:", roc_auc_score(y_val, y_val_prob))

    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n=== Final Test Evaluation ===")
    print(classification_report(y_test, y_pred))
    print("Test ROC AUC:", roc_auc_score(y_test, y_prob))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Test Confusion Matrix")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("Test ROC Curve")
    plt.legend()
    plt.show()

def run_pipeline(filepath, target_col, sep=',', split_method="random", model_type="xgb", date_col=None, params=None):
    df = load_data(filepath, sep)
    visualize_data(df, target_col)

    df = df.dropna(subset=[target_col])
    if df[target_col].dtype == "object":
        df[target_col] = df[target_col].map({'yes': 1, 'no': 0}).fillna(0).astype(int)

    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        df, target_col=target_col, method=split_method, date_col=date_col)

    X_combined = pd.concat([X_train, X_val])
    preprocessor = build_preprocessor(X_combined)

    clf = train_model(preprocessor, X_train, y_train, X_val, y_val, model_type=model_type, params=params)
    evaluate_model(clf, X_test, y_test)
    # show_feature_importance(clf, X_test)

    return clf



# -----------------------------
# 5. Run All
# -----------------------------
if __name__ == "__main__":
    filepath = "bank-full.csv"
    target_col = "y"
    sep = ";"
    model = run_pipeline(filepath, target_col, sep=sep, model_type="xgb")


    # df = load_data(filepath, ';')
    # visualize_data(df, target_col)
