"""Streamlit front-end for ML Assignment 2 - online shopper purchase intent.

Upload the held-out test CSV (the 17 input features plus the Revenue target),
pick one of the five saved classifiers and the app reports the six evaluation
metrics, the confusion matrix and the classification report, and compares all
five models side by side. The fitted pipelines in model/ are only loaded and
used for prediction - nothing is re-trained on the uploaded data.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
from html import escape
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import sklearn  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

import shopper_core as core  # noqa: E402

MODEL_COLOURS = {
    "Logistic Regression": "#0E7C66",
    "Decision Tree": "#E07A1F",
    "KNN": "#3D5A98",
    "Naive Bayes": "#B23A48",
    "Support Vector Machine": "#7A4EAB",
}
METRIC_HELP = {
    "Accuracy": "Share of all sessions that were classified correctly.",
    "AUC": "Area under the ROC curve: how well the model ranks buyers above non-buyers, independent of any threshold.",
    "Precision": "Of the sessions predicted as purchases, the share that really ended in a purchase.",
    "Recall": "Of the sessions that really ended in a purchase, the share the model caught.",
    "F1": "Harmonic mean of precision and recall for the Purchase class.",
    "MCC": "Matthews correlation coefficient (-1 to 1): summarises the whole confusion matrix and stays honest under class imbalance.",
}
STYLE_SHEET = Path(__file__).resolve().parent / "assets" / "app.css"
HOW_IT_WORKS = [
    ("Upload", "Drop the test CSV with the 17 input features and <code>Revenue</code> (TRUE/FALSE, Yes/No or 1/0)."),
    ("Validate", "Required columns are checked, unreadable values are repaired or flagged and extra columns ignored."),
    ("Predict", "The five saved scikit-learn pipelines score every session. Nothing is retrained."),
    ("Evaluate", "Accuracy, AUC, Precision, Recall, F1 and MCC with the confusion matrix, report and ROC curves."),
]

st.set_page_config(
    page_title="Shopper Purchase-Intent Classifier",
    page_icon=":material/shopping_cart:",
    layout="wide",
)
sns.set_theme(style="whitegrid", context="notebook")


# --------------------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading the five saved pipelines...")
def load_models() -> dict[str, object]:
    return {name: joblib.load(core.MODEL_DIR / file) for name, file in core.MODEL_FILES.items()}


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    return json.loads(core.METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_csv_bytes(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    if frame.shape[1] == 1 and ";" in str(frame.columns[0]):  # semicolon-separated export (e.g. Excel)
        frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig", sep=";")
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


@st.cache_data(show_spinner="Scoring the rows with all five models...")
def score_with_all_models(features: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    return {
        name: {
            "pred": np.asarray(model.predict(features)).astype(int),
            "score": np.asarray(core.positive_class_scores(model, features), dtype=float),
        }
        for name, model in load_models().items()
    }


# --------------------------------------------------------------------------- figures
def confusion_matrix_figure(cm: np.ndarray, title: str) -> plt.Figure:
    row_totals = cm.sum(axis=1, keepdims=True)
    shares = np.divide(cm, row_totals, out=np.zeros(cm.shape, dtype=float), where=row_totals > 0)
    annotations = np.array(
        [[f"{count:,}\n{share:.1%} of row" for count, share in zip(c_row, s_row)] for c_row, s_row in zip(cm, shares)]
    )
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    sns.heatmap(
        shares, annot=annotations, fmt="", cmap="Greens", vmin=0, vmax=1, cbar=False,
        xticklabels=core.CLASS_NAMES, yticklabels=core.CLASS_NAMES,
        linewidths=2, linecolor="white", annot_kws={"fontsize": 10}, ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def roc_figure(y_true: np.ndarray, scored: dict[str, dict], highlight: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    for name, outputs in scored.items():
        fpr, tpr, _ = roc_curve(y_true, outputs["score"])
        emphasised = highlight is None or name == highlight
        ax.plot(
            fpr, tpr, color=MODEL_COLOURS[name], lw=2.6 if name == highlight else 1.6,
            alpha=1.0 if emphasised else 0.45,
            label=f"{name} (AUC {roc_auc_score(y_true, outputs['score']):.3f})",
        )
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#7B8794", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def metric_bars_figure(table: pd.DataFrame) -> plt.Figure:
    long = table.reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score").dropna()
    fig, ax = plt.subplots(figsize=(10, 3.9))
    sns.barplot(data=long, x="Metric", y="Score", hue="ML Model Name", palette=MODEL_COLOURS, ax=ax)
    ax.set_ylim(min(0.0, long["Score"].min() - 0.05), 1.0)
    ax.set_xlabel("")
    ax.legend(ncol=5, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout()
    return fig


def show_figure(fig: plt.Figure) -> None:
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- helpers
def format_params(params: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


def report_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=core.CLASS_NAMES, output_dict=True, zero_division=0
    )
    rows = {label: report[label] for label in [*core.CLASS_NAMES, "macro avg", "weighted avg"]}
    frame = pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]]
    frame["support"] = frame["support"].astype(int)
    return frame


def render_hero(metadata: dict, compact: bool) -> None:
    """Dark banner at the top of the page; the KPI tiles are only shown on the landing page."""
    models, dataset = metadata["models"], metadata["dataset"]
    best_f1 = max(models, key=lambda name: models[name]["test_metrics"]["F1"])
    best_auc = max(models, key=lambda name: models[name]["test_metrics"]["AUC"])
    pills = "".join(
        f'<span class="pill">{label}</span>'
        for label in [
            f'<span class="dot"></span>{len(models)} models online',
            f"scikit-learn {escape(metadata['environment']['scikit-learn'])}",
            "inference only &middot; no retraining",
            "UCI dataset #468",
        ]
    )
    kpis = ""
    if not compact:
        tiles = [
            (f"{dataset['rows_used']:,}", "sessions", f"{dataset['duplicates_removed']} duplicates removed"),
            (f"{dataset['n_features']} &rarr; {len(metadata['engineered_features'])}", "features &rarr; inputs", "after one-hot encoding"),
            (str(len(models)), "trained pipelines", "LR &middot; DT &middot; KNN &middot; NB &middot; SVM"),
            (f"{dataset['train_rows']:,} / {dataset['test_rows']:,}", "train / test", "stratified 80 / 20 split"),
            (f"{models[best_f1]['test_metrics']['F1']:.3f}", "best test F1", escape(best_f1)),
            (f"{models[best_auc]['test_metrics']['AUC']:.3f}", "best test AUC", escape(best_auc)),
        ]
        kpis = '<div class="kpis">' + "".join(
            f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div><div class="s">{sub}</div></div>'
            for value, label, sub in tiles
        ) + "</div>"
    st.html(
        f'<div class="hero{" compact" if compact else ""}">'
        '<div class="hero-eyebrow">ML Assignment 2 &middot; BITS Pilani WILP &middot; M.Tech AIML</div>'
        "<h1>Online Shopper Purchase-Intent Classifier</h1>"
        '<p class="lead">Will this browsing session end in a <b>purchase</b>? Five classifiers trained on the UCI '
        "<i>Online Shoppers Purchasing Intention</i> dataset are benchmarked live on the test file you provide.</p>"
        f'<div class="pills">{pills}</div>{kpis}</div>'
    )


def leaderboard_html(metadata: dict) -> str:
    ranked = sorted(metadata["models"].items(), key=lambda item: item[1]["test_metrics"]["F1"], reverse=True)
    rows = []
    for rank, (name, record) in enumerate(ranked):
        scores = record["test_metrics"]
        badge = '<span class="badge">TOP F1</span>' if rank == 0 else ""
        rows.append(
            f'<div class="lb-row"><div class="lb-name">{escape(name)}{badge}</div>'
            f'<div class="lb-track"><div class="lb-fill" style="width:{scores["F1"] * 100:.1f}%"></div></div>'
            f'<div class="lb-val">F1 {scores["F1"]:.3f} &middot; AUC {scores["AUC"]:.3f}</div></div>'
        )
    return (
        '<div class="section-title">Leaderboard &middot; held-out test split</div>'
        f'<div class="board">{"".join(rows)}'
        f'<div class="board-note">Stored results for {metadata["dataset"]["test_rows"]:,} test sessions '
        "(model/metadata.json). Load the CSV to recompute every metric live.</div></div>"
    )


def schema_html() -> str:
    groups = [
        (f"numeric &middot; {len(core.NUMERIC_FEATURES)}", core.NUMERIC_FEATURES, ""),
        (f"categorical &middot; {len(core.CATEGORICAL_FEATURES)}", core.CATEGORICAL_FEATURES, "cat"),
        ("target", [core.TARGET_COLUMN], "target"),
    ]
    blocks = []
    for title, columns, css_class in groups:
        chips = "".join(
            f'<span class="chip {css_class}" title="{escape(core.FEATURE_DESCRIPTIONS[col])}">{escape(col)}</span>'
            for col in columns
        )
        blocks.append(f'<div class="chip-group"><div class="g">{title}</div><div class="chips">{chips}</div></div>')
    return '<div class="section-title">Expected input schema</div>' + "".join(blocks)


def render_landing(metadata: dict) -> None:
    st.html(
        '<div class="cta"><span class="arrow">&larr;</span><div><b>Start in the sidebar:</b> upload the test CSV, '
        "or switch on <b>Use bundled test_data.csv</b>. The held-out split "
        f"({metadata['dataset']['test_rows']:,} sessions) can be downloaded there as well.</div></div>"
    )
    steps = "".join(
        f'<div class="step"><div class="n">{index:02d}</div><div class="t">{title}</div><div class="d">{text}</div></div>'
        for index, (title, text) in enumerate(HOW_IT_WORKS, start=1)
    )
    st.html(f'<div class="section-title">How it works</div><div class="steps">{steps}</div>')
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.html(leaderboard_html(metadata))
    with right:
        st.html(schema_html())


def render_selected_model(name: str, y_true: np.ndarray, outputs: dict, record: dict, raw_rows: pd.DataFrame) -> None:
    metrics = core.compute_metrics(y_true, outputs["pred"], outputs["score"])
    st.subheader(name)
    st.caption(
        f"`{record['estimator']}` | tuned hyper-parameters: {format_params(record['best_params'])} | "
        f"cross-validated F1 on the training split: {record['cv_metrics']['F1']['mean']:.4f} "
        f"+/- {record['cv_metrics']['F1']['std']:.4f}"
    )

    for column, metric in zip(st.columns(len(core.METRIC_NAMES)), core.METRIC_NAMES):
        value = metrics[metric]
        column.metric(metric, "n/a" if np.isnan(value) else f"{value:.4f}", help=METRIC_HELP[metric], border=True)
    if np.isnan(metrics["AUC"]):
        st.warning("The uploaded rows contain only one class, so AUC and the ROC curve are undefined.")

    cm = confusion_matrix(y_true, outputs["pred"], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Confusion matrix")
        show_figure(confusion_matrix_figure(cm, f"{name} - {len(y_true):,} sessions"))
        st.caption(
            f"{tp:,} buyers caught, {fn:,} buyers missed, {fp:,} false alarms, {tn:,} non-buyers correctly ignored."
        )
    with right:
        st.markdown("##### Classification report")
        st.dataframe(
            report_frame(y_true, outputs["pred"]).style.format(
                {"precision": "{:.4f}", "recall": "{:.4f}", "f1-score": "{:.4f}", "support": "{:,}"}
            ),
        )
        if not np.isnan(metrics["AUC"]):
            show_figure(roc_figure(y_true, {name: outputs}, highlight=name))

    with st.expander("Row-level predictions"):
        is_probability = hasattr(load_models()[name], "predict_proba")
        score_label = "P(Purchase)" if is_probability else "SVM decision score"
        predictions = raw_rows.copy()
        predictions["Actual"] = np.where(y_true == 1, "Purchase", "No purchase")
        predictions["Predicted"] = np.where(outputs["pred"] == 1, "Purchase", "No purchase")
        predictions[score_label] = outputs["score"]
        predictions["Correct"] = predictions["Actual"] == predictions["Predicted"]
        score_column = (
            st.column_config.ProgressColumn(score_label, min_value=0.0, max_value=1.0, format="%.3f")
            if is_probability
            else st.column_config.NumberColumn(score_label, format="%.3f", help="Signed distance to the margin; > 0 means Purchase.")
        )
        st.dataframe(
            predictions[["Actual", "Predicted", score_label, "Correct", *core.FEATURE_COLUMNS]],
            column_config={score_label: score_column},
            height=320,
        )
        st.download_button(
            "Download predictions as CSV",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{core.MODEL_FILES[name].removesuffix('.joblib')}.csv",
            mime="text/csv",
            on_click="ignore",
            icon=":material/download:",
        )


def render_comparison(y_true: np.ndarray, scored: dict[str, dict], selected: str) -> None:
    table = pd.DataFrame(
        {name: core.compute_metrics(y_true, out["pred"], out["score"]) for name, out in scored.items()}
    ).T[core.METRIC_NAMES]
    table.index.name = "ML Model Name"

    st.subheader("All five models on the same file")
    st.dataframe(
        table.style.format("{:.4f}", na_rep="n/a").highlight_max(
            axis=0, props="font-weight: 700; background-color: #CDEBE1;"
        ),
    )
    leaders = "  |  ".join(
        f"**{metric}**: {table[metric].idxmax()} ({table[metric].max():.4f})"
        for metric in core.METRIC_NAMES
        if table[metric].notna().any()
    )
    st.caption(f"Best per metric (highlighted) - {leaders}")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("##### Metric comparison")
        show_figure(metric_bars_figure(table))
    with right:
        st.markdown("##### ROC curves")
        if np.unique(y_true).size == 2:
            show_figure(roc_figure(y_true, scored, highlight=selected))
        else:
            st.info("ROC curves need both classes in the uploaded rows.")


def render_details(metadata: dict) -> None:
    dataset, protocol = metadata["dataset"], metadata["protocol"]
    st.subheader("Dataset")
    st.markdown(
        f"""
**[{dataset['name']}]({dataset['source']})** (UCI Machine Learning Repository, Sakar et al., 2018).
Each row is one browsing session on an online retailer's website over a one-year period;
the target `Revenue` says whether the session ended in a purchase.

* {dataset['raw_rows']:,} sessions in the source file, {dataset['duplicates_removed']} exact duplicates removed -> {dataset['rows_used']:,} rows
* {dataset['n_features']} input features ({len(core.NUMERIC_FEATURES)} numeric, {len(core.CATEGORICAL_FEATURES)} categorical) - {len(metadata['engineered_features'])} columns after one-hot encoding
* Purchases: {dataset['train_positive_rate']:.2%} of the training split, {dataset['test_positive_rate']:.2%} of the test split
* Stratified split: {dataset['train_rows']:,} training rows / {dataset['test_rows']:,} test rows (`random_state={protocol['random_state']}`)
"""
    )
    st.dataframe(
        pd.DataFrame(
            {"Column": list(core.FEATURE_DESCRIPTIONS), "Description": list(core.FEATURE_DESCRIPTIONS.values())}
        ),
        hide_index=True,
    )

    st.subheader("Training protocol")
    baseline = metadata["baseline_majority_class"]
    st.markdown(
        f"""
* One shared preprocessing pipeline: median / most-frequent imputation, `log1p` + standardisation of the
  skewed numeric features, one-hot encoding with categories rarer than {protocol['min_category_count']} rows pooled.
* Hyper-parameters chosen by {protocol['cv_folds']}-fold stratified cross-validation on the training split only,
  maximising **{protocol['selection_metric']}**; the test split was used once, for the final report.
* Reference point: always predicting "No purchase" already scores Accuracy {baseline['Accuracy']:.4f}
  but F1 = MCC = 0 - which is why F1, MCC and AUC matter more than accuracy here.
"""
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Model": name,
                    "Estimator": record["estimator"],
                    "Tuned hyper-parameters": format_params(record["best_params"]),
                    "Configurations tried": record["candidates_evaluated"],
                    "CV F1 (mean)": record["cv_metrics"]["F1"]["mean"],
                    "CV F1 (std)": record["cv_metrics"]["F1"]["std"],
                }
                for name, record in metadata["models"].items()
            ]
        ).style.format({"CV F1 (mean)": "{:.4f}", "CV F1 (std)": "{:.4f}"}),
        hide_index=True,
    )
    env = metadata["environment"]
    st.caption(
        f"Trained with Python {env['python']}, scikit-learn {env['scikit-learn']}, numpy {env['numpy']}, "
        f"pandas {env['pandas']} on {metadata['created_utc']} UTC."
    )


# --------------------------------------------------------------------------- page
metadata = load_metadata()
st.html(STYLE_SHEET)

with st.sidebar:
    st.html(
        '<div class="side-brand">SHOPPER<span>/</span>INTENT'
        "<small>model evaluation console</small></div>"
    )
    st.markdown("### 1 · Test data")
    uploaded = st.file_uploader(
        "Upload a test CSV",
        type=["csv"],
        help="Needs the 17 input features and the Revenue target column. Upload test data only - "
        "the models are already trained.",
    )
    use_bundled = st.toggle(
        "Use bundled test_data.csv",
        value=False,
        disabled=uploaded is not None,
        help="Evaluate on the held-out 20% split that ships with the repository.",
    )
    st.download_button(
        "Download test_data.csv",
        data=core.TEST_DATA_PATH.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        on_click="ignore",
        icon=":material/download:",
        width="stretch",
    )
    st.markdown("### 2 · Model")
    selected_model = st.selectbox(
        "Classifier",
        list(core.MODEL_FILES),
        help="All five models were trained once on the same 80% training split.",
        filter_mode=None,  # disables typing, so the dropdown is read-only
    )
    st.divider()
    st.caption(f"[GitHub repository]({core.GITHUB_REPO_URL}) · [Dataset on UCI]({core.UCI_PAGE_URL})")
    st.caption("BITS Pilani WILP · M.Tech AIML · Machine Learning, Assignment 2")

render_hero(metadata, compact=uploaded is not None or use_bundled)

if sklearn.__version__ != metadata["environment"]["scikit-learn"]:
    st.warning(
        f"Models were saved with scikit-learn {metadata['environment']['scikit-learn']} but "
        f"{sklearn.__version__} is installed - install the pinned versions from requirements.txt."
    )

if uploaded is not None:
    payload, source_name = uploaded.getvalue(), uploaded.name
elif use_bundled:
    payload, source_name = core.TEST_DATA_PATH.read_bytes(), "test_data.csv (bundled)"
else:
    render_landing(metadata)
    st.stop()

try:
    raw = read_csv_bytes(payload)
except Exception as exc:  # malformed / non-CSV uploads
    st.error(f"Could not read **{source_name}** as CSV: {exc}")
    st.stop()

if raw.empty:
    st.error(f"**{source_name}** has a header but no data rows.")
    st.stop()

absent = core.missing_columns(raw)
if absent:
    st.error(
        f"**{source_name}** is missing {len(absent)} required column(s): `{'`, `'.join(absent)}`. "
        "The file must contain all 17 input features and the Revenue target."
    )
    st.stop()

target = core.prepare_target(raw)
unreadable_target = target.isna()
if unreadable_target.all():
    st.error("None of the Revenue values could be read - use TRUE/FALSE, True/False, Yes/No or 1/0.")
    st.stop()
raw = raw.loc[~unreadable_target].reset_index(drop=True)
y_true = target.loc[~unreadable_target].astype(int).to_numpy()
features, repairs = core.prepare_features(raw)
ignored_columns = [col for col in raw.columns if col not in core.FEATURE_COLUMNS + [core.TARGET_COLUMN]]

with st.container(border=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source", source_name)
    c2.metric("Sessions evaluated", f"{len(y_true):,}")
    c3.metric("Actual purchases", f"{int(y_true.sum()):,} ({y_true.mean():.1%})")
    c4.metric("Extra columns ignored", len(ignored_columns))
    notes = []
    if unreadable_target.any():
        notes.append(f"{int(unreadable_target.sum())} row(s) dropped because Revenue was unreadable")
    if repairs["unparseable_values"]:
        notes.append(f"{repairs['unparseable_values']} unreadable feature value(s) imputed")
    if repairs["negative_values_clipped"]:
        notes.append(f"{repairs['negative_values_clipped']} negative value(s) clipped to 0")
    if notes:
        st.warning("Data checks: " + "; ".join(notes) + ".")
    with st.expander("Preview uploaded rows"):
        st.dataframe(raw.head(50), height=260)

scored = score_with_all_models(features)

tab_selected, tab_all, tab_details = st.tabs(
    ["Selected model", "All five models", "Dataset & training details"]
)
with tab_selected:
    render_selected_model(selected_model, y_true, scored[selected_model], metadata["models"][selected_model], raw)
with tab_all:
    render_comparison(y_true, scored, selected_model)
with tab_details:
    render_details(metadata)
