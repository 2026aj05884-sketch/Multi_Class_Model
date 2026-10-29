"""Train, tune, evaluate and save the five classifiers required by ML Assignment 2.

Workflow
--------
1. Load the UCI "Online Shoppers Purchasing Intention" data (downloaded on first run).
2. Drop exact duplicate sessions, then make a stratified 80/20 train/test split.
   The untouched test rows are written to ``test_data.csv`` for the Streamlit app.
3. For every model, tune hyper-parameters with 5-fold stratified cross-validation
   on the training split only, selecting the configuration with the best F1 score
   for the Purchase class (the classes are imbalanced: ~15.6% purchases).
4. Refit the winning configuration on the full training split, evaluate it once on
   the held-out test split and save the fitted pipeline to ``model/<name>.joblib``.

Every saved file is a complete scikit-learn Pipeline (imputation, log scaling,
one-hot encoding and the classifier), so the app only has to call ``predict``.

Usage:
    python train_models.py            # full grid search (~3 min on a 22-core laptop)
    python train_models.py --quick    # refit the stored best hyper-parameters (seconds)
"""
from __future__ import annotations

import argparse
import inspect
import io
import json
import platform
import time
import urllib.request
import warnings
import zipfile
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    matthews_corrcoef,
    precision_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import shopper_core as core

TEST_SIZE = 0.20
CV_FOLDS = 5
MIN_CATEGORY_COUNT = 30  # category levels seen fewer times in training share one "infrequent" column
SELECTION_METRIC = "f1"

CV_SCORING = {
    "accuracy": "accuracy",
    "roc_auc": "roc_auc",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "mcc": make_scorer(matthews_corrcoef),
}
CV_KEY_TO_METRIC = {
    "accuracy": "Accuracy", "roc_auc": "AUC", "precision": "Precision",
    "recall": "Recall", "f1": "F1", "mcc": "MCC",
}


def load_raw_data() -> pd.DataFrame:
    if not core.RAW_DATA_PATH.exists():
        print(f"Downloading dataset from {core.UCI_ZIP_URL}")
        with urllib.request.urlopen(core.UCI_ZIP_URL, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        core.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        core.RAW_DATA_PATH.write_bytes(archive.read(csv_name))
    return pd.read_csv(core.RAW_DATA_PATH)


def make_one_hot_encoder() -> OneHotEncoder:
    # scikit-learn 1.2 renamed `sparse` to `sparse_output`; support both so the
    # script also runs on older lab images.
    dense_flag = "sparse_output" if "sparse_output" in inspect.signature(OneHotEncoder).parameters else "sparse"
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=MIN_CATEGORY_COUNT,
        **{dense_flag: False},
    )


def build_preprocessor() -> ColumnTransformer:
    """Shared feature engineering: every model sees the same feature matrix."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        # Counts, durations and PageValues are heavily right-skewed (skew 2-7.6).
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, core.NUMERIC_FEATURES),
            ("cat", categorical, core.CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def model_search_spaces() -> dict[str, tuple[object, dict[str, list]]]:
    seed = core.RANDOM_STATE
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000),
            {
                "clf__C": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=seed),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [3, 4, 5, 6, 8, 10, None],
                "clf__min_samples_leaf": [5, 10, 25, 50, 100, 200, 400],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "clf__n_neighbors": [5, 9, 15, 21, 31, 45, 61, 81],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [float(v) for v in np.logspace(-9, 2, 23)]},
        ),
        "Support Vector Machine": (
            SVC(kernel="rbf", cache_size=1000, random_state=seed),
            {
                "clf__C": [0.3, 1, 3, 10, 30, 100],
                "clf__gamma": ["scale", 0.0003, 0.001, 0.003, 0.01],
                "clf__class_weight": [None, "balanced"],
            },
        ),
    }


def stored_search_spaces() -> dict[str, tuple[object, dict[str, list]]]:
    """One-candidate grids holding the hyper-parameters chosen by an earlier full run."""
    if not core.METADATA_PATH.exists():
        raise SystemExit(f"{core.METADATA_PATH} not found - run once without --quick first.")
    tuned = json.loads(core.METADATA_PATH.read_text(encoding="utf-8"))["models"]
    return {
        name: (estimator, {f"clf__{key}": [value] for key, value in tuned[name]["best_params"].items()})
        for name, (estimator, _) in model_search_spaces().items()
    }


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "|".join(["---"] + [":---:"] * (len(frame.columns) - 1)) + "|"
    rows = [
        "| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, divider, *rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel workers for GridSearchCV (default: all cores)")
    parser.add_argument(
        "--quick", action="store_true",
        help="skip the grid search and refit the hyper-parameters stored in model/metadata.json "
             "(same models and metrics in a fraction of the time)",
    )
    args = parser.parse_args()
    search_spaces = stored_search_spaces() if args.quick else model_search_spaces()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    started = time.perf_counter()
    raw = load_raw_data()
    n_raw = len(raw)
    raw = raw.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {n_raw} sessions; removed {n_raw - len(raw)} exact duplicates -> {len(raw)} rows, "
          f"{len(core.FEATURE_COLUMNS)} features.")

    features, issues = core.prepare_features(raw)
    target = core.prepare_target(raw).astype(int)
    if any(issues.values()):
        print(f"Input repairs applied: {issues}")

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, stratify=target, random_state=core.RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows ({y_train.mean():.2%} purchases) | "
          f"Test: {len(X_test)} rows ({y_test.mean():.2%} purchases)")

    test_rows = raw.loc[X_test.index].sort_index()
    test_rows.to_csv(core.TEST_DATA_PATH, index=False)
    print(f"Saved held-out test split -> {core.TEST_DATA_PATH.name}")

    core.MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=core.RANDOM_STATE)

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_metrics = core.compute_metrics(
        y_test, baseline.predict(X_test), core.positive_class_scores(baseline, X_test)
    )

    model_records, table_rows = {}, []
    for name, (estimator, grid) in search_spaces.items():
        pipeline = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
        search = GridSearchCV(
            pipeline, grid, scoring=CV_SCORING, refit=SELECTION_METRIC,
            cv=cv, n_jobs=args.n_jobs, error_score="raise",
        )
        tic = time.perf_counter()
        search.fit(X_train, y_train)
        search_seconds = time.perf_counter() - tic

        # SVC's `probability` option is deprecated in scikit-learn 1.9, so the SVM's AUC
        # comes from decision_function scores - exactly as during the search.
        model = search.best_estimator_
        y_pred = model.predict(X_test)
        y_score = core.positive_class_scores(model, X_test)
        test_metrics = core.compute_metrics(y_test, y_pred, y_score)

        best = search.best_index_
        cv_metrics = {
            CV_KEY_TO_METRIC[key]: {
                "mean": float(search.cv_results_[f"mean_test_{key}"][best]),
                "std": float(search.cv_results_[f"std_test_{key}"][best]),
            }
            for key in CV_SCORING
        }
        best_params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

        model_path = core.MODEL_DIR / core.MODEL_FILES[name]
        joblib.dump(model, model_path, compress=3)

        model_records[name] = {
            "file": f"model/{model_path.name}",
            "estimator": type(estimator).__name__,
            "best_params": best_params,
            "candidates_evaluated": len(search.cv_results_["params"]),
            "search_seconds": round(search_seconds, 2),
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=core.CLASS_NAMES, output_dict=True, zero_division=0
            ),
        }
        table_rows.append({"ML Model Name": name, **test_metrics})
        print(f"\n[{name}] {len(search.cv_results_['params'])} candidates x {CV_FOLDS} folds in {search_seconds:.1f}s")
        print(f"  best params : {best_params}")
        print(f"  CV F1       : {cv_metrics['F1']['mean']:.4f} +/- {cv_metrics['F1']['std']:.4f}")
        print("  test        : " + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))

    table = pd.DataFrame(table_rows)
    table.round(4).to_csv(core.TEST_METRICS_PATH, index=False)

    engineered_features = build_preprocessor().fit(X_train).get_feature_names_out()
    metadata = {
        "project": "Online shoppers purchase-intent classification (ML Assignment 2)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "Online Shoppers Purchasing Intention Dataset",
            "source": core.UCI_PAGE_URL,
            "raw_rows": n_raw,
            "duplicates_removed": n_raw - len(raw),
            "rows_used": len(raw),
            "n_features": len(core.FEATURE_COLUMNS),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "target": {"column": core.TARGET_COLUMN, "classes": core.CLASS_NAMES, "positive_class": core.CLASS_NAMES[1]},
        "features": {"numeric": core.NUMERIC_FEATURES, "categorical": core.CATEGORICAL_FEATURES},
        "engineered_features": [str(name) for name in engineered_features],
        "protocol": {
            "search": "stored hyper-parameters (--quick)" if args.quick else "full grid search",
            "random_state": core.RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "selection_metric": "F1 (Purchase class)",
            "min_category_count": MIN_CATEGORY_COUNT,
        },
        "baseline_majority_class": baseline_metrics,
        "models": model_records,
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    core.METADATA_PATH.write_text(json.dumps(jsonable(metadata), indent=2), encoding="utf-8")

    print("\nMajority-class baseline on test: " + "  ".join(f"{k}={v:.4f}" for k, v in baseline_metrics.items()))
    print("\nComparison table (held-out test split):\n")
    print(markdown_table(table))
    print(f"\nSaved 5 pipelines + metadata to {core.MODEL_DIR.name}/ in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
