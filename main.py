# ML Interview Craft Exercise Template (Jupyter Friendly)
# Author: Qimeng Chen

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   OneHotEncoder, LabelEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb
import shap
import optuna

warnings.filterwarnings("ignore")

# ----------------------------------------
# Data Loading and Visualization
# ----------------------------------------
def load_data(filepath, sep=','):
    df = pd.read_csv(filepath, sep=sep)
    display(df.head())
    display(df.info())
    display(df.describe())
    return df

def visualize_data(df, target_col, max_categories=10):
    print("\n=== Data Visualization ===")

    sns.countplot(data=df, x=target_col)
    plt.title("Target Distribution")
    plt.show()

    plt.figure(figsize=(12, 8))
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    num_features = df.select_dtypes(include=['int64', 'float64']).columns.drop(target_col, errors='ignore')
    df[num_features].hist(bins=30, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    for col in num_features:
        sns.boxplot(data=df, x=target_col, y=col)
        plt.title(f'{col} vs {target_col}')
        plt.tight_layout()
        plt.show()

    for col in num_features:
        plt.figure(figsize=(8, 5))
        for cls in df[target_col].unique():
            subset = df[df[target_col] == cls]
            sns.kdeplot(subset[col], label=f"{target_col}={cls}", fill=True, common_norm=False, alpha=0.5)
        plt.title(f'{col} Distribution by {target_col}')
        plt.legend()
        plt.tight_layout()
        plt.show()

    cat_features = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_features:
        top_cats = df[col].value_counts().nlargest(max_categories)
        sns.barplot(x=top_cats.index, y=top_cats.values)
        plt.title(f'{col} (Top {max_categories})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        if df[col].nunique() <= max_categories:
            sns.countplot(data=df, x=col, hue=target_col)
            plt.title(f'{col} by {target_col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# ----------------------------------------
# Preprocessing and Utilities
# ----------------------------------------
def create_categorical_encoder(X_train, encoding_type="onehot", drop_first=False):
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if encoding_type == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", drop="first" if drop_first else None)
        encoder.fit(X_train[cat_cols])
        return encoder, cat_cols
    elif encoding_type == "label":
        encoder = {}
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(X_train[col].astype(str))
            encoder[col] = le
        return encoder, cat_cols
    else:
        raise ValueError("encoding_type must be 'onehot' or 'label'")

def transform_with_label_encoding(X, encoder_dict):
    X_copy = X.copy()
    for col, encoder in encoder_dict.items():
        X_copy[col] = X_copy[col].astype(str).map(lambda val: encoder.transform([val])[0] if val in encoder.classes_ else -1)
    return X_copy

def create_numeric_scaler(X_train, method="standard"):
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    scaler.fit(X_train[num_cols])
    return scaler, num_cols

def split_data(df, target_col, method="random", date_col=None,
               train_size=0.7, val_size=0.15, test_size=0.15,
               stratify=False, time_boundaries=None, random_state=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    if method == "random":
        X = df.drop(columns=[target_col])
        y = df[target_col]
        stratify_y = y if stratify else None
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size,
                                                                    random_state=random_state,
                                                                    stratify=stratify_y)
        val_fraction = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_fraction,
                                                          random_state=random_state,
                                                          stratify=y_train_val if stratify else None)
    else:
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        if time_boundaries:
            train_end, val_end = time_boundaries
            df_train = df_sorted[df_sorted[date_col] <= train_end]
            df_val = df_sorted[(df_sorted[date_col] > train_end) & (df_sorted[date_col] <= val_end)]
            df_test = df_sorted[df_sorted[date_col] > val_end]
        else:
            n = len(df_sorted)
            train_idx = int(train_size * n)
            val_idx = int((train_size + val_size) * n)
            df_train = df_sorted.iloc[:train_idx]
            df_val = df_sorted.iloc[train_idx:val_idx]
            df_test = df_sorted.iloc[val_idx:]

        X_train = df_train.drop(columns=[target_col, date_col])
        y_train = df_train[target_col]
        X_val = df_val.drop(columns=[target_col, date_col])
        y_val = df_val[target_col]
        X_test = df_test.drop(columns=[target_col, date_col])
        y_test = df_test[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test

def build_preprocessor(X_train):
    cat_encoder, cat_cols = create_categorical_encoder(X_train, encoding_type="onehot", drop_first=True)
    num_scaler, num_cols = create_numeric_scaler(X_train)
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols)
    ])

# ----------------------------------------
# Model Training & Evaluation
# ----------------------------------------
def train_xgb_with_label_encoding(X_train, y_train, X_val, y_val, use_optuna=False, n_trials=30):
    cat_encoder, cat_cols = create_categorical_encoder(X_train, encoding_type="label")
    X_train_enc = transform_with_label_encoding(X_train, cat_encoder)
    X_val_enc = transform_with_label_encoding(X_val, cat_encoder)

    if use_optuna:
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "n_estimators": 1000,
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "use_label_encoder": True,
                "eval_metric": "aucpr",
                "early_stopping_rounds": 20
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train_enc, y_train,
                      eval_set=[(X_val_enc, y_val)],
                      verbose=False)
            y_pred = model.predict_proba(X_val_enc)[:, 1]
            return roc_auc_score(y_val, y_pred)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_params["use_label_encoder"] = True
        best_params["eval_metric"] = "aucpr"
        model = xgb.XGBClassifier(**best_params)
    else:
        model = xgb.XGBClassifier(use_label_encoder=True, eval_metric='aucpr', early_stopping_rounds = 20)

    model.fit(X_train_enc, y_train, eval_set=[(X_val_enc, y_val)], verbose=False)
    y_val_pred = model.predict(X_val_enc)
    y_val_prob = model.predict_proba(X_val_enc)[:, 1]

    print("--- Validation Performance (Label Encoded) ---")
    print(classification_report(y_val, y_val_pred))
    print("Validation ROC AUC:", roc_auc_score(y_val, y_val_prob))

    return model, cat_encoder

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n=== Final Test Evaluation ===")
    print(classification_report(y_test, y_pred))
    print("Test ROC AUC:", roc_auc_score(y_test, y_prob))

    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Test Confusion Matrix")
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

def explain_with_shap(clf, X_sample):
    import shap
    print("--- SHAP Explanation ---")

    model = clf
    data = X_sample
    explainer = shap.Explainer(model)
    shap_values = explainer(data)
    feature_names = model.get_booster().feature_names
    shap.summary_plot(shap_values, features=data, feature_names=feature_names)

# 1. Load Data
df = load_data("bank-full.csv", sep=";")
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# 2. Optional EDA
visualize_data(df, target_col="y")

# 3. Split
X_train, y_train, X_val, y_val, X_test, y_test = split_data(df, target_col="y", method="random", stratify=True)

# 4. Preprocessor
# X_combined = pd.concat([X_train, X_val])  # to avoid unseen categories
# preprocessor = build_preprocessor(X_combined)

# 5. Train Model
# clf = train_model(preprocessor, X_train, y_train, X_val, y_val, model_type="xgb")
clf, encoder = train_xgb_with_label_encoding(X_train, y_train, X_val, y_val, use_optuna=True, n_trials=30)

# 6. Evaluate
X_test_enc = transform_with_label_encoding(X_test, encoder)
evaluate_model(clf, X_test_enc, y_test)

# 7. SHAP Explanation (optional but informative)
explain_with_shap(clf, X_test_enc.sample(100, random_state=42))  # limit to 100 for speed
