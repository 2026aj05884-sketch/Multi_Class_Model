# Predicting Online Shopper Purchase Intent with Five Classifiers

**Machine Learning - Assignment 2** | M.Tech (AIML), BITS Pilani WILP

Five classification models are trained on the UCI *Online Shoppers Purchasing Intention* dataset,
compared on six evaluation metrics and served through an interactive Streamlit app that evaluates
the saved models on an uploaded test CSV.

* **Live Streamlit app:** <https://2026aj05884.streamlit.app/>
* **GitHub repository:** <https://github.com/2026aj05884-sketch/Multi_Class_Model>

---

## a. Problem Statement

An online retailer wants to know, from what a visitor does during a browsing session, whether that
session will end in a **purchase**. Being able to predict this in real time lets the business target
likely buyers (offers, chat support, free shipping prompts) and understand what drives conversion.

The task is framed as **binary classification**: predict `Revenue` (True = the session ended with a
transaction) from 17 session-level features. Only about 15.5% of sessions end in a purchase, so the
problem is **imbalanced**: a model that always answers "no purchase" is already ~84% accurate while
being useless. Models are therefore judged on AUC, Precision, Recall, F1 and MCC as well as Accuracy.

Objectives:

1. Train Logistic Regression, Decision Tree, K-Nearest Neighbours, Naive Bayes (Gaussian) and
   Support Vector Machine classifiers on the same dataset.
2. Evaluate every model on the same held-out test set with Accuracy, AUC, Precision, Recall, F1 and MCC.
3. Deploy an interactive Streamlit app that loads the saved models (no retraining), accepts an
   uploaded test CSV and reports the metrics, the confusion matrix and the classification report.

## b. Dataset Description

| Property | Value |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | [UCI Machine Learning Repository, dataset #468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) (donated 30-Aug-2018) |
| Introductory paper | Sakar, Polat, Katircioglu & Kastro (2019), *Neural Computing and Applications* |
| Instances | 12,330 sessions (12,205 after removing 125 exact duplicate rows) |
| Features | 17 input features: 10 numeric + 7 categorical (minimum required: 12) |
| Target | `Revenue`: False = 10,422 sessions (84.5%), True = 1,908 sessions (15.5%) |
| Missing values | None |
| Task | Binary classification on imbalanced classes |

Each row is one session on an e-commerce site. The data was collected so that every session
belongs to a different user over a one-year period, which avoids any bias towards a specific
campaign, special day, user profile or period.

| Feature | Type | Description |
|---|---|---|
| `Administrative` | numeric | Number of account / administrative pages visited in the session |
| `Administrative_Duration` | numeric | Seconds spent on administrative pages |
| `Informational` | numeric | Number of informational pages (contact, about, ...) visited |
| `Informational_Duration` | numeric | Seconds spent on informational pages |
| `ProductRelated` | numeric | Number of product pages visited |
| `ProductRelated_Duration` | numeric | Seconds spent on product pages |
| `BounceRates` | numeric | Average Google Analytics bounce rate of the visited pages |
| `ExitRates` | numeric | Average Google Analytics exit rate of the visited pages |
| `PageValues` | numeric | Average value of the pages visited before completing a transaction |
| `SpecialDay` | numeric | Closeness of the visit to a special day such as Valentine's Day (0 to 1) |
| `Month` | categorical | Month of the visit (10 distinct values; no January or April) |
| `OperatingSystems` | categorical | Operating-system code (8 levels) |
| `Browser` | categorical | Browser code (13 levels) |
| `Region` | categorical | Geographic region code (9 levels) |
| `TrafficType` | categorical | Traffic-source code (20 levels) |
| `VisitorType` | categorical | Returning_Visitor, New_Visitor or Other |
| `Weekend` | categorical | Whether the visit happened on a weekend |
| **`Revenue`** | **target** | Whether the session ended in a purchase |

What the exploratory analysis showed, and how it shaped the preprocessing:

* **`PageValues` is the dominant signal.** Sessions with `PageValues > 0` (22% of all sessions)
  convert 56% of the time, compared with 3.9% when `PageValues = 0`.
* **The numeric features are heavily right-skewed** (skewness between 2 and 7.6, many zeros), so they
  are `log1p`-transformed and standardised.
* **Several features are strongly correlated.** `BounceRates` and `ExitRates` have r = 0.90, and each
  page count has r = 0.60-0.86 with its duration. This matters for Naive Bayes (see the observations).
* **Several categorical codes have rare levels** (some Browser and TrafficType codes appear only once).
  One-hot encoding pools levels seen fewer than 30 times in training into one "infrequent" column
  (for OperatingSystems, Browser and TrafficType). A code never seen during training is mapped to that
  column, or ignored for features that have no rare levels, so new codes cannot crash the app.

**Split and preprocessing.** After removing the 125 exact duplicates, the data is split 80/20 with
stratification (`random_state = 5884`) into **9,764 training** and **2,441 test** sessions (15.6%
purchases in both). The untouched test rows are saved as [`test_data.csv`](test_data.csv). One shared
preprocessing pipeline is fitted on the training split only and stored inside every model file:
median / most-frequent imputation, `log1p` and standard scaling for the numeric columns, and one-hot
encoding for the categorical columns, giving 66 model inputs.

## c. GitHub Repository Link

**<https://github.com/2026aj05884-sketch/Multi_Class_Model>**

The repository contains the complete source code (`train_models.py`, `shopper_core.py`, `app.py`),
`requirements.txt`, this README, the test data (`test_data.csv`), the raw dataset (`data/`) and the
five trained pipelines with their metadata (`model/`). The full layout is listed under
[Repository Structure](#repository-structure) below.

## d. Models Used

All five models use the same preprocessing and the same train/test split. Hyper-parameters were tuned
with **5-fold stratified cross-validation on the training split only**, selecting the configuration with
the highest **F1 score for the Purchase class** (accuracy would reward ignoring buyers). The best
configuration was refitted on the full training split and evaluated **once** on the held-out test split.

| ML Model Name | scikit-learn estimator | Search space (configurations tried) | Selected hyper-parameters | CV F1 (mean +/- std) |
|---|---|---|---|:---:|
| Logistic Regression | `LogisticRegression` (L2) | C, class_weight (20) | C = 0.003, class_weight = balanced | 0.6723 +/- 0.0110 |
| Decision Tree | `DecisionTreeClassifier` | criterion, max_depth, min_samples_leaf, class_weight (196) | entropy, max_depth = 4, min_samples_leaf = 200, no class weights | 0.6573 +/- 0.0296 |
| KNN | `KNeighborsClassifier` | n_neighbors, weights, p (32) | k = 31, distance weighting, Euclidean (p = 2) | 0.6161 +/- 0.0291 |
| Naive Bayes | `GaussianNB` | var_smoothing (23) | var_smoothing = 1.0 | 0.5888 +/- 0.0095 |
| Support Vector Machine | `SVC` (RBF kernel) | C, gamma, class_weight (60) | C = 30, gamma = 0.001, class_weight = balanced | 0.6724 +/- 0.0086 |

### Comparison Table of Evaluation Metrics

Held-out test set of 2,441 sessions (382 purchases). The best value in each column is in bold.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.8738 | 0.9090 | 0.5717 | 0.7723 | 0.6570 | 0.5917 |
| Decision Tree | 0.8964 | 0.9112 | **0.7295** | 0.5366 | 0.6184 | 0.5689 |
| KNN | **0.8972** | 0.9039 | 0.7205 | 0.5602 | 0.6303 | 0.5778 |
| Naive Bayes | 0.8562 | 0.8672 | 0.5303 | 0.7094 | 0.6069 | 0.5294 |
| Support Vector Machine | 0.8730 | **0.9153** | 0.5687 | **0.7801** | **0.6578** | **0.5932** |

*Precision, Recall and F1 refer to the positive class (Purchase). AUC is computed from `predict_proba`
(from the decision function for the SVM). For reference, a majority-class baseline that always predicts
"No purchase" scores Accuracy 0.8435, AUC 0.5000 and Precision = Recall = F1 = MCC = 0.
Confusion matrices on the test set ([TN, FP], [FN, TP]): Logistic Regression [1838, 221], [87, 295];
Decision Tree [1983, 76], [177, 205]; KNN [1976, 83], [168, 214]; Naive Bayes [1819, 240], [111, 271];
SVM [1833, 226], [84, 298].*

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Joint-best model. After log-scaling, the log-odds of a purchase are close to linear in the features: `PageValues` has by far the largest coefficient (+1.17), followed by November (+0.51), while `ExitRates`, `BounceRates`, May and March push towards "no purchase". Cross-validation chose strong L2 regularisation (C = 0.003) and balanced class weights, which favour buyers and give Recall 0.772 at Precision 0.572 (87 buyers missed, 221 false alarms). Its CV F1 (0.672) is close to its test F1 (0.657), so it generalises well, and it is the cheapest model to train, store (3 KB) and explain. |
| Decision Tree | A depth-4 tree with only 14 leaves. Its root split is `PageValues` (78% of the total feature importance), followed by November and `BounceRates`, so it reads as a short rule set. Cross-validation preferred no class weighting and large leaves (min_samples_leaf = 200), which makes the tree conservative: it has the highest Precision (0.730) and the second-best Accuracy (0.896) but the lowest Recall (0.537). It misses 177 of the 382 buyers, which drags F1 (0.618) and MCC (0.569) down. It was also the least stable model across CV folds (F1 std 0.030, nearly three times that of Logistic Regression) and lost 0.04 F1 between CV and test, the usual high variance of a single tree. |
| KNN | Highest Accuracy (0.897) and high Precision (0.721), but Recall is only 0.560. KNN cannot re-weight classes, and with 84% non-buyers most neighbourhoods are dominated by non-buyers, so borderline sessions are labelled "no purchase". Cross-validation chose a large, distance-weighted neighbourhood (k = 31) to smooth out noise in a 66-dimensional space of scaled numeric and one-hot columns, where distances carry less information. Its AUC (0.904) is the second lowest. As a lazy learner it stores all 9,764 training rows (the largest model file, 0.5 MB) and predicts about 12 times slower than the tree. |
| Naive Bayes | The weakest model, with the lowest Accuracy (0.856), AUC (0.867), Precision (0.530), F1 (0.607) and MCC (0.529). Both of its assumptions are broken on this data. The features are not independent: `BounceRates` and `ExitRates` are about 0.90 correlated, and each page count overlaps with its duration, so the same evidence is counted twice. The zero-inflated counts and 0/1 one-hot columns are also far from Gaussian. Cross-validation chose heavy smoothing (var_smoothing = 1.0, which adds a variance of 1 to every feature) to flatten these poorly fitted likelihoods. It still clearly beats the majority-class baseline (MCC 0.53 vs 0) and trains in milliseconds, so it is a useful baseline. |
| Support Vector Machine | Best on four of the six metrics: AUC (0.915), Recall (0.780), F1 (0.658) and MCC (0.593), missing only 84 of the 382 buyers. Cross-validation chose an RBF kernel with a very small gamma (0.001), C = 30 and balanced class weights, which produce a smooth, almost linear decision boundary, so its scores are within 0.01 of Logistic Regression on every metric. The price is cost: it keeps 3,510 support vectors (36% of the training rows), had the slowest grid search (118 s for 60 configurations) and predicts about 20 times slower than Logistic Regression. It also outputs margin scores rather than probabilities, so its AUC is computed from the decision function. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (RBF kernel).** It has the best F1, MCC, AUC and Recall, so it finds the most buyers while keeping the best balance across all four cells of the confusion matrix. **Logistic Regression is a very close runner-up**: the two tie on cross-validated F1 (0.6723 vs 0.6724), and Logistic Regression would be the pragmatic choice for production because it is about 20 times faster, tiny and fully interpretable. Judging by Accuracy alone would wrongly favour KNN and the Decision Tree. The "never buys" baseline already reaches 0.844 accuracy, and those two models gain their extra accuracy while missing about 45% of the buyers. |

---

## Streamlit App

**Live app:** <https://2026aj05884.streamlit.app/> (deployed on Streamlit Community Cloud from branch
`main`, main file `app.py`)

The app (`app.py`) loads the five saved pipelines from `model/` and **never retrains** them. Features:

* **Landing page** with a dark dashboard-style banner (live model status, dataset and score KPIs), a
  "how it works" strip, a leaderboard of the stored test scores and the expected input schema, plus a
  dark control-panel sidebar.
* **CSV upload** for test data only (25 MB limit), plus a one-click option to use the bundled
  `test_data.csv` and a button to download it. Uploads are validated: missing columns are reported,
  extra columns are ignored, `Revenue` may be TRUE/FALSE, True/False, Yes/No or 1/0, and unreadable
  values are flagged.
* **Model selection dropdown** covering the five classifiers.
* **Evaluation metrics** shown as cards: Accuracy, AUC, Precision, Recall, F1 and MCC.
* **Confusion matrix** heatmap with counts and row percentages, the **classification report** and the
  ROC curve for the selected model, plus row-level predictions that can be downloaded as CSV.
* **All five models** tab: the metric table for every model on the uploaded file (best value
  highlighted), a grouped bar chart and overlaid ROC curves.
* **Dataset & training details** tab: the feature dictionary, the training protocol, the tuned
  hyper-parameters and the library versions.

## Repository Structure

```
ML2/
├── app.py                        # Streamlit front-end (loads the saved models, no retraining)
├── train_models.py               # data split, tuning, evaluation and saving of the five models
├── shopper_core.py               # shared schema, input cleaning and metric helpers
├── requirements.txt              # pinned dependencies (Python 3.11 - 3.14)
├── README.md
├── test_data.csv                 # held-out test split: 2,441 rows, 17 features + Revenue
├── assets/
│   └── app.css                   # styles for the landing page and banner
├── data/
│   └── online_shoppers_intention.csv   # raw UCI dataset (12,330 rows)
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib                # each file = preprocessing + classifier pipeline
│   ├── metadata.json             # schema, tuned parameters, CV and test metrics, versions
│   └── test_metrics.csv          # the comparison table above
└── .streamlit/
    └── config.toml               # app theme (fonts, dark sidebar) and upload limit
```

## How to Run

```bash
git clone https://github.com/2026aj05884-sketch/Multi_Class_Model.git
cd ML2
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train_models.py             # full grid search, rewrites model/ and test_data.csv (about 3 min on 22 cores)
python train_models.py --quick     # reuse the tuned hyper-parameters in model/metadata.json (seconds)
streamlit run app.py               # opens http://localhost:8501
```

The dependencies are pinned to the versions used for training (scikit-learn 1.9.1), because a pickled
scikit-learn model is only guaranteed to load with the same release. They install on Python 3.11 to 3.14.

**Deployment (Streamlit Community Cloud):** the app is live at <https://2026aj05884.streamlit.app/>
and redeploys automatically on every push to `main`. It was deployed by signing in at
<https://streamlit.io/cloud> with GitHub, choosing **New app**, selecting the repository
`2026aj05884-sketch/ML2`, branch `main` and main file `app.py` (any Python version from 3.11 to 3.14
works), and clicking **Deploy**.

## References

1. C. O. Sakar, S. O. Polat, M. Katircioglu and Y. Kastro, "Real-time prediction of online shoppers'
   purchasing intention using multilayer perceptron and LSTM recurrent neural networks",
   *Neural Computing and Applications*, 31, 2019.
2. UCI Machine Learning Repository, *Online Shoppers Purchasing Intention Dataset*,
   <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>.
3. scikit-learn user guide, <https://scikit-learn.org/stable/user_guide.html>; Streamlit documentation,
   <https://docs.streamlit.io>.
