"""
ML engine — tabular pipeline, deep learning, agentic decisions, run history.
"""
import difflib
import json
import re
import warnings
import time
import datetime
import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, KFold)
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import (LinearRegression, LogisticRegression,
                                   Ridge, RidgeClassifier, Lasso)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier,
                              ExtraTreesRegressor, ExtraTreesClassifier,
                              AdaBoostRegressor, AdaBoostClassifier,
                              BaggingRegressor, BaggingClassifier)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
import joblib

from llm import ask

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

try:
    import torch
    import torch.nn as tnn
    import torch.optim as topt
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================ 1. LOADER
ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")


def _read_csv_robust(path):
    notes = []
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            bad = []
            def _skip(line):
                bad.append(line); return None
            try:
                df = pd.read_csv(path, encoding=enc, engine="python",
                                 on_bad_lines=_skip)
            except UnicodeDecodeError:
                continue
            notes.append(f"Skipped {len(bad)} malformed rows")
        if enc != "utf-8":
            notes.append(f"File was read using the '{enc}' text encoding")
        return df, notes
    raise ValueError("Could not read this CSV file with any known encoding")


def load_table_with_notes(path):
    p = str(path).lower()
    if p.endswith(".csv"):
        df, notes = _read_csv_robust(path)
    elif p.endswith((".xlsx", ".xls")):
        df, notes = pd.read_excel(path), []
    else:
        raise ValueError("Unsupported file type. Use .csv, .xlsx or .xls")
    stripped = [c for c in df.columns if isinstance(c, str) and c != c.strip()]
    if stripped:
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
        notes.append(f"Trimmed extra spaces from {len(stripped)} column name(s)")
    return df, notes


def load_table(path):
    return load_table_with_notes(path)[0]


# ============================================================ 2. PROFILER
ID_WORDS = {"id", "uuid", "isbn", "isbn10", "isbn13", "sku", "zip", "zipcode", "phone"}


def outlier_count(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def looks_like_id(s, name):
    tokens = set(re.split(r"[^a-z0-9]+", str(name).lower()))
    if tokens & ID_WORDS:
        return True
    n = len(s)
    if n < 20 or pd.api.types.is_bool_dtype(s):
        return False
    ratio = s.nunique() / n
    if pd.api.types.is_integer_dtype(s):
        return ratio >= 0.99
    if not pd.api.types.is_numeric_dtype(s):
        return ratio >= 0.95
    return False


def profile_df(df):
    n = max(len(df), 1)
    info = {"rows": len(df), "duplicate_rows": int(df.duplicated().sum()),
            "columns": {}}
    for col in df.columns:
        s = df[col]
        item = {"type": str(s.dtype),
                "missing_pct": round(s.isna().mean() * 100, 1),
                "unique": int(s.nunique()),
                "unique_pct": round(s.nunique() / n * 100, 1),
                "looks_like_id": bool(looks_like_id(s, col))}
        if (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
                and s.notna().any()):
            item["min"] = float(s.min()); item["max"] = float(s.max())
            item["skew"] = round(float(s.skew()), 2)
            item["outliers"] = outlier_count(s)
            item["outlier_pct"] = round(item["outliers"] / n * 100, 1)
        info["columns"][col] = item
    return info


def profile(path):
    return profile_df(load_table(path))


# ============================================================ 3. FEATURES
TOP_CATS = 15
TOP_ITEMS = 10
TOP_WORDS = 15
NUM_UNIT = r"^\s*(-?\d[\d,]*(?:\.\d+)?)\s*([A-Za-z][A-Za-z .%/]*)?\s*$"

NOTES = {
    "date": "date -> year, month, weekday",
    "numunit": "number with a unit -> number (and unit)",
    "list": "list of values -> item count + common items",
    "text": "free text -> length, word count, common keywords",
    "topcat": f"many categories -> top {TOP_CATS} + 'Other'",
}


def _name_is_id(name):
    tokens = set(re.split(r"[^a-z0-9]+", str(name).lower()))
    return bool(tokens & ID_WORDS)


def _to_datetime(s):
    t = s.astype(str).str.strip().where(s.notna())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast = pd.to_datetime(t, errors="coerce")
        if fast.notna().sum() >= 0.8 * max(int(t.notna().sum()), 1):
            return fast
        try:
            return pd.to_datetime(t, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return fast


def column_kind(s, name=""):
    present = s.dropna()
    if len(present) == 0:
        return "empty"
    if _name_is_id(name):
        return "id"
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        n = len(present)
        if pd.api.types.is_integer_dtype(s) and n >= 20 and present.nunique() / n >= 0.99:
            return "id"
        return "num"
    v = present.astype(str).str.strip()
    v = v[v != ""]
    if len(v) == 0:
        return "empty"
    n, k = len(v), v.nunique()
    sample = v.sample(min(n, 2000), random_state=0)
    if sample.str.match(NUM_UNIT).mean() >= 0.8:
        return "numunit"
    dates = _to_datetime(sample)
    if dates.notna().mean() >= 0.8 and sample.str.contains(r"[-/ ,]").mean() >= 0.8:
        return "date"
    words = sample.str.split().str.len()
    if sample.str.contains(",").mean() >= 0.15:
        items = sample.str.split(",").explode().str.strip()
        items = items[items != ""]
        if items.str.split().str.len().mean() <= 3 and items.str.len().mean() <= 30:
            return "list"
    if k / n >= 0.95:
        return "text" if words.mean() >= 2 else "id"
    if words.mean() >= 6 or sample.str.len().mean() >= 60:
        return "text"
    return "cat" if k <= 50 else "topcat"


def conversion_note(s, name=""):
    return NOTES.get(column_kind(s, name))


def usable_columns(df, target):
    kept, dropped = [], []
    n = max(len(df), 1)
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        kind = column_kind(s, c)
        if kind == "id":
            dropped.append((c, "looks like an ID or code"))
        elif kind == "empty":
            dropped.append((c, "has no values"))
        elif s.nunique(dropna=True) <= 1:
            dropped.append((c, "constant — no variation to learn from"))
        elif s.isna().mean() > 0.60:
            dropped.append((c, f"{s.isna().mean():.0%} missing, too empty to use"))
        elif s.nunique(dropna=True) == n and n >= 50 and kind not in ("num", "numunit", "date"):
            dropped.append((c, "unique on every row, so it cannot generalise"))
        else:
            kept.append(c)
    return kept, dropped


def _unit_norm(u):
    u = re.sub(r"\s+", " ", str(u).strip().lower())
    return u[:-1] if len(u) > 3 and u.endswith("s") else u


def _split_numunit(s):
    m = s.astype(str).str.extract(NUM_UNIT)
    num = pd.to_numeric(m[0].astype(str).str.replace(",", "", regex=False), errors="coerce")
    unit = m[1].map(_unit_norm, na_action="ignore")
    return num, unit


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.specs_, self.report_, self.dropped_ = {}, [], []
        self.date_pairs_ = []
        for c in X.columns:
            s = X[c]
            kind = column_kind(s, c)
            spec = {"kind": kind}
            if kind == "id":
                self.dropped_.append((c, "looks like an ID or code")); continue
            if kind == "empty":
                self.dropped_.append((c, "has no values")); continue
            miss = float(s.isna().mean())
            if 0.05 <= miss <= 0.60:
                spec["missing_indicator"] = True
                self.report_.append(f"'{c}' is {miss:.0%} empty -> added 'is_missing_{c}' flag")
            if kind == "date":
                self.report_.append(f"'{c}' (date) -> year, month, weekday")
            elif kind == "numunit":
                _, unit = _split_numunit(s)
                u = unit.dropna(); u = u[u != ""]
                counts = u.value_counts()
                if 2 <= len(counts) <= 20:
                    spec["units"] = list(counts.index)
                self.report_.append(f"'{c}' (number with a unit) -> number"
                                    + (" and unit" if "units" in spec else ""))
            elif kind == "num":
                v = pd.to_numeric(s, errors="coerce").dropna()
                if len(v) >= 20:
                    skew = float(v.skew())
                    if skew > 2 and float(v.min()) >= 0:
                        spec["log"] = True
                        self.report_.append(f"'{c}' heavily skewed (skew={skew:.1f}) -> added log1p")
            elif kind == "list":
                items = s.dropna().astype(str).str.split(",").explode().str.strip()
                items = items[items != ""]
                spec["items"] = list(items.value_counts().head(TOP_ITEMS).index)
                self.report_.append(f"'{c}' (comma list) -> count + {len(spec['items'])} items")
            elif kind == "text":
                texts = s.fillna("").astype(str)
                spec["vocab"] = []
                try:
                    cv = CountVectorizer(stop_words="english", max_features=TOP_WORDS,
                                         binary=True, min_df=5)
                    cv.fit(texts)
                    spec["cv"] = cv
                    spec["vocab"] = list(cv.get_feature_names_out())
                except ValueError:
                    pass
                self.report_.append(f"'{c}' (free text) -> length + keywords")
            elif kind == "topcat":
                v = s.dropna().astype(str).str.strip()
                spec["top"] = list(v.value_counts().head(TOP_CATS).index)
                self.report_.append(f"'{c}' ({v.nunique()} values) -> top {TOP_CATS}")
            self.specs_[c] = spec
        dates = [c for c, sp in self.specs_.items() if sp["kind"] == "date"]
        for i in range(len(dates) - 1):
            a, b = dates[i], dates[i + 1]
            self.date_pairs_.append((a, b))
            self.report_.append(f"'{a}' and '{b}' -> days_{a}_minus_{b}")
        self.feature_names_ = list(self.transform(X.head(20)).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X); out = {}
        for c, spec in self.specs_.items():
            if c not in X.columns:
                continue
            s = X[c]; present = s.notna(); kind = spec["kind"]
            if spec.get("missing_indicator"):
                out[f"is_missing_{c}"] = s.isna().astype(float)
            if kind == "num":
                v = pd.to_numeric(s, errors="coerce"); out[c] = v
                if spec.get("log"):
                    out[f"{c}_log"] = np.log1p(v.clip(lower=0))
            elif kind == "bool":
                out[c] = pd.to_numeric(s.map({True: 1.0, False: 0.0}), errors="coerce")
            elif kind == "cat":
                out[c] = s.astype(str).str.strip().where(present)
            elif kind == "topcat":
                t = s.astype(str).str.strip()
                out[c] = t.where(t.isin(spec["top"]), "Other").where(present)
            elif kind == "date":
                d = _to_datetime(s)
                out[f"{c}_year"] = d.dt.year.astype(float)
                out[f"{c}_month"] = d.dt.month.astype(float)
                out[f"{c}_weekday"] = d.dt.dayofweek.astype(float)
            elif kind == "numunit":
                num, unit = _split_numunit(s)
                out[f"{c}_number"] = num.where(present)
                if "units" in spec:
                    out[f"{c}_unit"] = unit.where(unit.isin(spec["units"]), "other").where(unit.notna())
            elif kind == "list":
                sets = s.map(lambda v: set() if pd.isna(v)
                             else {i.strip() for i in str(v).split(",") if i.strip()})
                out[f"{c}_count"] = sets.map(len).astype(float).where(present)
                for item in spec["items"]:
                    out[f"{c}={item}"] = sets.map(lambda st, it=item: float(it in st)).where(present)
            elif kind == "text":
                t = s.fillna("").astype(str)
                out[f"{c}_length"] = t.str.len().astype(float).where(present)
                out[f"{c}_words"] = t.str.split().str.len().astype(float).where(present)
                cv = spec.get("cv")
                if cv is not None:
                    arr = cv.transform(t).toarray()
                    for j, w in enumerate(spec["vocab"]):
                        out[f"{c}~{w}"] = pd.Series(arr[:, j].astype(float), index=X.index).where(present)
        for a, b in getattr(self, "date_pairs_", []):
            if a in X.columns and b in X.columns:
                da = _to_datetime(X[a]); db = _to_datetime(X[b])
                out[f"days_{a}_minus_{b}"] = (da - db).dt.days.astype(float)
        return pd.DataFrame(out, index=X.index)


def engineering_summary(df):
    try:
        fe = FeatureEngineer().fit(df)
        return fe.report_
    except Exception as e:
        return [f"(Could not summarise: {type(e).__name__})"]


# ============================================================ 4. CLEANER
ALLOWED_IMPUTE = {"median", "mean", "mode"}
SYNONYMS = {"most_frequent": "mode", "most frequent": "mode", "average": "mean"}
MAX_OUTLIER_SHARE = 0.10
YEAR_WORDS = {"year", "yr", "date", "time", "day", "month"}


def looks_like_year(s, name):
    tokens = set(re.split(r"[^a-z0-9]+", str(name).lower()))
    if tokens & YEAR_WORDS:
        return True
    if pd.api.types.is_integer_dtype(s) and s.notna().any():
        return bool(s.min() >= 1000 and s.max() <= 2200 and s.nunique() <= 300)
    return False


def validate_plan(plan, df):
    warnings_ = []
    n = max(len(df), 1)
    clean = {"drop_duplicates": bool(plan.get("drop_duplicates", False)),
             "impute": {}, "outlier_columns": [], "drop_high_missing": []}
    impute = plan.get("impute") or {}
    if not isinstance(impute, dict):
        warnings_.append("Ignored 'impute': not in the expected format")
        impute = {}
    for col, method in impute.items():
        method = SYNONYMS.get(str(method).lower(), str(method).lower())
        if col not in df.columns:
            warnings_.append(f"Skipped '{col}': column does not exist")
        elif looks_like_id(df[col], col):
            warnings_.append(f"Skipped '{col}': looks like an ID or free text")
        elif method not in ALLOWED_IMPUTE:
            warnings_.append(f"Skipped '{col}': method '{method}' is not allowed")
        elif method in ("median", "mean") and not pd.api.types.is_numeric_dtype(df[col]):
            warnings_.append(f"Skipped '{col}': {method} needs a numeric column")
        elif df[col].isna().sum() == 0:
            warnings_.append(f"Skipped '{col}': no missing values to fill")
        else:
            clean["impute"][col] = method
    for col in plan.get("outlier_columns") or []:
        if (col not in df.columns or not pd.api.types.is_numeric_dtype(df[col])
                or pd.api.types.is_bool_dtype(df[col])):
            warnings_.append(f"Skipped outlier column '{col}': not numeric")
        elif looks_like_id(df[col], col):
            warnings_.append(f"Skipped '{col}': looks like an ID")
        elif looks_like_year(df[col], col):
            warnings_.append(f"Skipped '{col}': years and dates are real values")
        else:
            share = outlier_count(df[col].dropna()) / n
            if share > MAX_OUTLIER_SHARE:
                warnings_.append(f"Skipped '{col}': {share:.0%} flagged, so skewed")
            else:
                clean["outlier_columns"].append(col)
    for col in plan.get("drop_high_missing") or []:
        if col not in df.columns:
            warnings_.append(f"Skipped drop '{col}': column does not exist")
        elif looks_like_id(df[col], col):
            continue
        elif df[col].isna().mean() < 0.40:
            warnings_.append(f"Kept '{col}': only {df[col].isna().mean():.0%} missing")
        else:
            clean["drop_high_missing"].append(col)
    return clean, warnings_


def restrict_plan(plan, features):
    keep = set(features)
    return {"drop_duplicates": bool(plan.get("drop_duplicates", False)),
            "impute": {c: m for c, m in (plan.get("impute") or {}).items() if c in keep},
            "outlier_columns": [c for c in (plan.get("outlier_columns") or []) if c in keep],
            "drop_high_missing": [c for c in (plan.get("drop_high_missing") or []) if c in keep]}


def apply_plan(df, plan, log):
    df = df.copy()
    if plan["drop_duplicates"]:
        before = len(df); df = df.drop_duplicates()
        log.append(f"Removed {before - len(df)} duplicate rows")
    for col, method in plan["impute"].items():
        n = int(df[col].isna().sum())
        if n == 0 or df[col].notna().sum() == 0:
            continue
        if method == "median": value = df[col].median()
        elif method == "mean": value = df[col].mean()
        else: value = df[col].mode().iloc[0]
        df[col] = df[col].fillna(value)
        log.append(f"Filled {n} missing values in '{col}' with the {method} ({value})")
    for col in plan["outlier_columns"]:
        low, high = df[col].quantile(0.01), df[col].quantile(0.99)
        n = int(((df[col] < low) | (df[col] > high)).sum())
        df[col] = df[col].clip(low, high)
        log.append(f"Capped {n} extreme values in '{col}'")
    return df


# ============================================================ 5. PLANNER
def heuristic_plan(df):
    """Rule-based cleaning plan used when the LLM is down or returns junk."""
    plan = {"drop_duplicates": bool(df.duplicated().sum() > 0),
            "impute": {}, "outlier_columns": [], "drop_high_missing": []}
    n = max(len(df), 1)
    for col in df.columns:
        s = df[col]
        if looks_like_id(s, col):
            continue
        miss = float(s.isna().mean())
        if miss >= 0.60:
            plan["drop_high_missing"].append(col)
            continue
        if miss >= 0.02:
            if (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
                    and s.notna().any()):
                skew = float(pd.to_numeric(s, errors="coerce").skew() or 0)
                plan["impute"][col] = "median" if abs(skew) > 1 else "mean"
            else:
                plan["impute"][col] = "mode"
        if (pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
                and not looks_like_year(s, col) and s.notna().sum() >= 20):
            share = outlier_count(s.dropna()) / n
            if 0 < share <= MAX_OUTLIER_SHARE:
                plan["outlier_columns"].append(col)
    return plan


def get_plan(df):
    fallback = heuristic_plan(df)
    p = profile_df(df)
    prompt = (
        "You are a careful data cleaning planner. Given this dataset profile, return "
        "ONLY JSON with keys: drop_duplicates (true/false), impute (column -> 'median', "
        "'mean' or 'mode'), outlier_columns (list of column names), drop_high_missing "
        "(list of column names that are mostly empty). Rules: be conservative; only "
        "fill missing values in columns that actually have missing values; never touch "
        "columns with looks_like_id=true; only list a column in outlier_columns if it "
        "is a continuous measurement with outlier_pct under 10; never list years, dates, "
        "ratings, scores, counts or codes unless clearly needed; put a column in "
        "drop_high_missing only if missing_pct is 60 or higher. Profile: " + json.dumps(p)
    )
    for _ in range(2):
        try:
            reply = ask(prompt) or ""
        except Exception:
            break
        start, end = reply.find("{"), reply.rfind("}")
        try:
            data = json.loads(reply[start:end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return fallback


# ============================================================ 6. CATALOG
REASON = {
    "linear": "simple and explainable; shows how far straight-line patterns go",
    "ridge": "linear with L2 regularization; better when inputs are correlated",
    "lasso": "linear with L1 regularization; drops weak coefficients to zero",
    "ridge_classifier": "regularized linear classifier; strong baseline for many classes",
    "tree": "reads as a set of yes/no questions; good for small data",
    "knn": "predicts from the most similar past examples",
    "random_forest": "many trees voting; robust on mixed data",
    "extra_trees": "like random forest but with extra randomness; sometimes stronger",
    "gradient_boosting": "trees that fix each other's mistakes; often the strongest on tables",
    "adaboost": "boosting with shallow stumps; strong on simple signal",
    "bagging": "many weak learners on subsets; reduces variance cheaply",
    "gnb": "fast probabilistic classifier; great on roughly independent features",
    "mlp": "a neural network that can learn curved patterns",
    "torch_nn": "a deep learning network with GPU support and full hyperparameter control",
    "svm": "widest margin between classes; strong on small clean data",
}


def baseline(task):
    if task == "regression":
        return "Baseline (just guesses the average)", DummyRegressor()
    return "Baseline (always the most common class)", DummyClassifier(strategy="most_frequent")


def hyperparams(key, task):
    """Editable hyperparameters for each model key."""
    reg = task == "regression"
    if key == "linear":
        if reg:
            return []
        return [{"name": "C", "kind": "float", "default": 1.0, "min": 0.001,
                 "max": 100.0, "step": 0.1,
                 "help": "Inverse regularization strength. Smaller = stronger penalty."}]
    if key in ("ridge", "ridge_classifier"):
        return [{"name": "alpha", "kind": "float", "default": 1.0, "min": 0.001,
                 "max": 100.0, "step": 0.1,
                 "help": "L2 penalty. Higher = smoother model."}]
    if key == "lasso":
        return [{"name": "alpha", "kind": "float", "default": 0.001, "min": 0.0001,
                 "max": 10.0, "step": 0.0005,
                 "help": "L1 penalty. Higher = more coefficients forced to zero."}]
    if key == "tree":
        return [
            {"name": "max_depth", "kind": "int", "default": 6, "min": 1, "max": 40, "step": 1,
             "help": "Maximum depth of the tree."},
            {"name": "min_samples_leaf", "kind": "int", "default": 5, "min": 1, "max": 100,
             "step": 1, "help": "Minimum rows per leaf."},
        ]
    if key == "knn":
        return [
            {"name": "n_neighbors", "kind": "int", "default": 5, "min": 1, "max": 50, "step": 1,
             "help": "Number of neighbours to average."},
            {"name": "weights", "kind": "choice", "default": "uniform",
             "choices": ["uniform", "distance"],
             "help": "Weight neighbours equally or by distance."},
        ]
    if key == "random_forest":
        return [
            {"name": "n_estimators", "kind": "int", "default": 100, "min": 10, "max": 500,
             "step": 10, "help": "Number of trees."},
            {"name": "max_depth", "kind": "int", "default": 12, "min": 1, "max": 40, "step": 1,
             "help": "Maximum depth of each tree."},
            {"name": "min_samples_leaf", "kind": "int", "default": 2, "min": 1, "max": 50,
             "step": 1, "help": "Minimum rows per leaf."},
        ]
    if key == "extra_trees":
        return [
            {"name": "n_estimators", "kind": "int", "default": 100, "min": 10, "max": 500,
             "step": 10, "help": "Number of trees."},
            {"name": "max_depth", "kind": "int", "default": 12, "min": 1, "max": 40, "step": 1,
             "help": "Maximum depth of each tree."},
        ]
    if key == "gradient_boosting":
        return [
            {"name": "learning_rate", "kind": "float", "default": 0.1, "min": 0.001,
             "max": 1.0, "step": 0.01, "help": "Step size."},
            {"name": "max_iter", "kind": "int", "default": 100, "min": 10, "max": 1000,
             "step": 10, "help": "Number of boosting rounds."},
            {"name": "max_depth", "kind": "int", "default": 0, "min": 0, "max": 20, "step": 1,
             "help": "Tree depth. 0 = unlimited."},
        ]
    if key == "adaboost":
        return [
            {"name": "n_estimators", "kind": "int", "default": 100, "min": 10, "max": 500,
             "step": 10, "help": "Number of boosting rounds."},
            {"name": "learning_rate", "kind": "float", "default": 1.0, "min": 0.01,
             "max": 2.0, "step": 0.05, "help": "Learning rate for each stump."},
        ]
    if key == "bagging":
        return [
            {"name": "n_estimators", "kind": "int", "default": 50, "min": 5, "max": 300,
             "step": 5, "help": "Number of base estimators."},
            {"name": "max_samples", "kind": "float", "default": 1.0, "min": 0.1,
             "max": 1.0, "step": 0.05, "help": "Fraction of rows each estimator sees."},
        ]
    if key == "gnb":
        return [{"name": "var_smoothing", "kind": "float", "default": 1e-9, "min": 1e-12,
                 "max": 1e-2, "step": 1e-9,
                 "help": "Portion of the largest variance added for stability."}]
    if key == "svm":
        return [
            {"name": "C", "kind": "float", "default": 1.0, "min": 0.01, "max": 100.0,
             "step": 0.1, "help": "Regularization strength."},
            {"name": "kernel", "kind": "choice", "default": "rbf",
             "choices": ["rbf", "linear", "poly"],
             "help": "Kernel function."},
        ]
    if key == "mlp":
        return [
            {"name": "alpha", "kind": "float", "default": 0.0001, "min": 0.0,
             "max": 1.0, "step": 0.0001, "help": "L2 penalty on weights."},
            {"name": "learning_rate_init", "kind": "float", "default": 0.001, "min": 1e-5,
             "max": 1.0, "step": 0.0001, "help": "Initial step size."},
        ]
    return []


def make_with_params(key, task, params):
    cat = catalog(task, 10000)
    if key not in cat:
        return None
    m = cat[key]["make"]()
    p = params or {}
    safe = {}
    if key == "linear":
        if "C" in p: safe["C"] = p["C"]
    elif key in ("ridge", "ridge_classifier", "lasso"):
        if "alpha" in p: safe["alpha"] = p["alpha"]
    elif key == "tree":
        if "max_depth" in p: safe["max_depth"] = int(p["max_depth"])
        if "min_samples_leaf" in p: safe["min_samples_leaf"] = int(p["min_samples_leaf"])
    elif key == "knn":
        if "n_neighbors" in p: safe["n_neighbors"] = int(p["n_neighbors"])
        if "weights" in p: safe["weights"] = p["weights"]
    elif key in ("random_forest", "extra_trees"):
        if "n_estimators" in p: safe["n_estimators"] = int(p["n_estimators"])
        if "max_depth" in p and int(p["max_depth"]) > 0:
            safe["max_depth"] = int(p["max_depth"])
        if "min_samples_leaf" in p: safe["min_samples_leaf"] = int(p["min_samples_leaf"])
    elif key == "gradient_boosting":
        if "learning_rate" in p: safe["learning_rate"] = float(p["learning_rate"])
        if "max_iter" in p: safe["max_iter"] = int(p["max_iter"])
        if "max_depth" in p: safe["max_depth"] = int(p["max_depth"])
    elif key == "adaboost":
        if "n_estimators" in p: safe["n_estimators"] = int(p["n_estimators"])
        if "learning_rate" in p: safe["learning_rate"] = float(p["learning_rate"])
    elif key == "bagging":
        if "n_estimators" in p: safe["n_estimators"] = int(p["n_estimators"])
        if "max_samples" in p: safe["max_samples"] = float(p["max_samples"])
    elif key == "gnb":
        if "var_smoothing" in p: safe["var_smoothing"] = float(p["var_smoothing"])
    elif key == "svm":
        if "C" in p: safe["C"] = float(p["C"])
        if "kernel" in p: safe["kernel"] = p["kernel"]
    elif key == "mlp":
        if "alpha" in p: safe["alpha"] = float(p["alpha"])
        if "learning_rate_init" in p: safe["learning_rate_init"] = float(p["learning_rate_init"])
    if safe:
        try:
            m.set_params(**safe)
        except Exception:
            pass
    return m


def _sklearn_mlp(task, nn_config):
    layers = tuple(nn_config.get("hidden_layers", (64, 32)))
    activation = nn_config.get("activation", "relu")
    if activation not in ("relu", "tanh", "logistic", "identity"):
        activation = "relu"
    lr = nn_config.get("learning_rate", 0.001)
    l2 = nn_config.get("weight_decay", 1e-4)
    max_epochs = nn_config.get("max_epochs", 100)
    early = nn_config.get("early_stopping", True)
    if task == "regression":
        return MLPRegressor(hidden_layer_sizes=layers, activation=activation,
                            learning_rate_init=lr, alpha=l2, max_iter=max_epochs,
                            early_stopping=early, random_state=42)
    return MLPClassifier(hidden_layer_sizes=layers, activation=activation,
                         learning_rate_init=lr, alpha=l2, max_iter=max_epochs,
                         early_stopping=early, random_state=42)


def _torch_mlp(task, nn_config):
    return TorchNN(
        task=task,
        hidden_layers=tuple(nn_config.get("hidden_layers", (64, 32))),
        activation=nn_config.get("activation", "relu"),
        dropout=nn_config.get("dropout", 0.0),
        optimizer=nn_config.get("optimizer", "adam"),
        learning_rate=nn_config.get("learning_rate", 0.001),
        weight_decay=nn_config.get("weight_decay", 1e-4),
        batch_size=nn_config.get("batch_size", 32),
        max_epochs=nn_config.get("max_epochs", 200),
        early_stopping=nn_config.get("early_stopping", True),
        patience=nn_config.get("patience", 20),
        use_gpu=nn_config.get("use_gpu", True))


def catalog(task, n_rows, nn_config=None, mode="all"):
    nn_config = nn_config or {}
    framework = nn_config.get("framework", "sklearn")
    if framework == "torch" and not TORCH_AVAILABLE:
        framework = "sklearn"
        nn_config = {**nn_config, "framework": "sklearn"}
    reg = task == "regression"
    c = {}
    if mode in ("all", "ml"):
        c["linear"] = {
            "name": "Linear Regression" if reg else "Logistic Regression",
            "make": (lambda: LinearRegression()) if reg else (lambda: LogisticRegression(max_iter=1000)),
            "blurb": REASON["linear"]}
        if reg:
            c["ridge"] = {"name": "Ridge Regression", "make": lambda: Ridge(alpha=1.0), "blurb": REASON["ridge"]}
            c["lasso"] = {"name": "Lasso Regression", "make": lambda: Lasso(alpha=0.001, max_iter=5000), "blurb": REASON["lasso"]}
        else:
            c["ridge_classifier"] = {"name": "Ridge Classifier", "make": lambda: RidgeClassifier(alpha=1.0), "blurb": REASON["ridge_classifier"]}
        c["tree"] = {"name": "Decision Tree",
                     "make": ((lambda: DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)) if reg
                              else (lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=42))),
                     "blurb": REASON["tree"]}
        c["knn"] = {"name": "KNN",
                    "make": (lambda: KNeighborsRegressor()) if reg else (lambda: KNeighborsClassifier()),
                    "blurb": REASON["knn"]}
        c["random_forest"] = {"name": "Random Forest",
                              "make": ((lambda: RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)) if reg
                                       else (lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))),
                              "blurb": REASON["random_forest"]}
        c["extra_trees"] = {"name": "Extra Trees",
                            "make": ((lambda: ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)) if reg
                                     else (lambda: ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1))),
                            "blurb": REASON["extra_trees"]}
        c["gradient_boosting"] = {"name": "Gradient Boosting",
                                  "make": ((lambda: HistGradientBoostingRegressor(random_state=42)) if reg
                                           else (lambda: HistGradientBoostingClassifier(random_state=42))),
                                  "blurb": REASON["gradient_boosting"]}
        c["adaboost"] = {"name": "AdaBoost",
                         "make": ((lambda: AdaBoostRegressor(n_estimators=100, random_state=42)) if reg
                                  else (lambda: AdaBoostClassifier(n_estimators=100, random_state=42))),
                         "blurb": REASON["adaboost"]}
        c["bagging"] = {"name": "Bagging",
                        "make": ((lambda: BaggingRegressor(n_estimators=50, random_state=42, n_jobs=-1)) if reg
                                 else (lambda: BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1))),
                        "blurb": REASON["bagging"]}
        if not reg:
            c["gnb"] = {"name": "Gaussian Naive Bayes", "make": lambda: GaussianNB(), "blurb": REASON["gnb"]}
        if n_rows <= 5000:
            c["svm"] = {"name": "SVM",
                        "make": (lambda: SVR()) if reg else (lambda: SVC()),
                        "blurb": REASON["svm"]}
    if mode in ("all", "dl"):
        if framework == "torch" and TORCH_AVAILABLE:
            c["mlp"] = {"name": "PyTorch Neural Network",
                        "make": lambda: _torch_mlp(task, nn_config), "blurb": REASON["torch_nn"]}
        else:
            c["mlp"] = {"name": "Neural Network (MLP)",
                        "make": lambda: _sklearn_mlp(task, nn_config), "blurb": REASON["mlp"]}
    return c


def recommend(task, n_rows, n_features, available, n_classes=None, imbalance_ratio=1.0):
    if not available:
        return [], "no models available"
    if len(available) == 1:
        return [available[0]], "only one model available in this mode"
    imbalanced = task == "classification" and imbalance_ratio >= 4
    many_classes = task == "classification" and (n_classes or 0) >= 8
    if imbalanced:
        order = ["gradient_boosting", "random_forest", "extra_trees",
                 "ridge_classifier", "linear", "gnb"]
        regime = (f"classes are unbalanced ({imbalance_ratio:.1f}x), so tree ensembles "
                  "and regularized linear models are safer than nearest-neighbours")
    elif n_rows < 500:
        order = ["linear", "ridge", "ridge_classifier", "random_forest", "knn", "tree",
                 "gradient_boosting"]
        regime = f"small data ({n_rows} rows), so simpler models are less likely to memorize it"
    elif many_classes:
        order = ["random_forest", "extra_trees", "gradient_boosting", "ridge_classifier",
                 "linear", "knn"]
        regime = f"{n_classes} classes, so ensembles hold up better than a single linear cut"
    elif n_rows < 3000 or n_features < 10:
        order = ["linear", "ridge", "ridge_classifier", "random_forest", "extra_trees",
                 "gradient_boosting", "knn", "tree"]
        regime = f"medium data ({n_rows} rows, {n_features} prepared columns)"
    else:
        order = ["random_forest", "extra_trees", "gradient_boosting", "mlp",
                 "linear", "ridge", "knn"]
        regime = (f"larger data ({n_rows} rows, {n_features} prepared columns), "
                  "so flexible models have enough examples to learn from")
    keys = [k for k in order if k in available][:4]
    return keys, regime


def ask_user_models(task, n_rows):
    cat = catalog(task, n_rows)
    keys = list(cat)
    print("\nAvailable models (baseline included for comparison):")
    for i, k in enumerate(keys, 1):
        print(f"  {i} = {cat[k]['name']}: {cat[k]['blurb']}")
    print("Type numbers separated by commas, 'all', or Enter to auto-pick.")
    ans = input("> ").strip().lower()
    if not ans:
        return None
    if ans == "all":
        return keys
    chosen = []
    for part in re.split(r"[,\s]+", ans):
        if part.isdigit() and 1 <= int(part) <= len(keys) and keys[int(part) - 1] not in chosen:
            chosen.append(keys[int(part) - 1])
    if not chosen:
        print("Not understood; picking for you."); return None
    return chosen


# ============================================================ 7. TRAINING
STRATEGY = {"median": "median", "mean": "mean", "mode": "most_frequent"}


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, low=0.01, high=0.99):
        self.low, self.high = low, high
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.low, axis=0)
        self.hi_ = np.nanquantile(X, self.high, axis=0)
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


class NumPicker:
    def __init__(self, strat, capped_flag, impute, capped):
        self.strat, self.capped_flag = strat, capped_flag
        self.impute, self.capped = impute, capped
    def __call__(self, X):
        return [c for c in X.columns
                if pd.api.types.is_numeric_dtype(X[c])
                and ((c in self.capped) == self.capped_flag)
                and STRATEGY.get(self.impute.get(c), "median") == self.strat]


def _cat_picker(X):
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def detect_task(y):
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        return "regression"
    return "classification"


def task_evidence(y):
    y = y.dropna(); n, k = len(y), y.nunique()
    if k < 2:
        return "classification", "low", "the target has fewer than 2 different values"
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return "classification", "high", "the target is text or yes/no, so it is a category"
    if k == 2:
        return "classification", "high", "it has only 2 different values (yes/no type)"
    is_whole = bool((y % 1 == 0).all())
    if is_whole and n >= 20 and k / n >= 0.95:
        return "regression", "low", "almost every value is different, which looks like an ID"
    if not is_whole:
        if k > 15:
            return "regression", "high", "decimal values with many different values"
        return "regression", "low", f"decimal values but only {k} different ones"
    if k <= 10:
        return "classification", "low", f"whole numbers with only {k} different values"
    if k <= 50:
        return "regression", "low", f"whole numbers with {k} different values"
    return "regression", "high", f"whole numbers with {k} different values, so a quantity"


def select_features(df, target):
    return usable_columns(df, target)


def rank_features(df, features, target, task):
    """Mutual-information ranking so prune/retry keeps the columns that actually inform y."""
    cols = [c for c in features if c in df.columns and c != target]
    if len(cols) < 2:
        return cols
    X = df[cols].copy()
    y = df[target]
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0)
        else:
            codes, _ = pd.factorize(X[c].astype(str), sort=False)
            X[c] = codes.astype(float)
    X = X.fillna(0)
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        fn = mutual_info_regression if task == "regression" else mutual_info_classif
        scores = fn(X, y, random_state=42, n_neighbors=min(3, max(1, len(X) - 1)))
        return list(pd.Series(scores, index=X.columns).sort_values(ascending=False).index)
    except Exception:
        return cols


def prune_features(df, features, target, task, min_keep=3):
    ranked = rank_features(df, features, target, task)
    n = max(min_keep, len(ranked) // 2)
    return ranked[:n], ranked


def build_preprocessor(plan):
    impute = dict(plan.get("impute") or {})
    capped = set(plan.get("outlier_columns") or [])
    transformers = []
    for do_cap in (False, True):
        for strat in ("median", "mean", "most_frequent"):
            steps = [("cap", Winsorizer())] if do_cap else []
            steps += [("imp", SimpleImputer(strategy=strat)), ("sc", StandardScaler())]
            transformers.append((f"num_{strat}_{'cap' if do_cap else 'plain'}",
                                 Pipeline(steps), NumPicker(strat, do_cap, impute, capped)))
    transformers.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), _cat_picker))
    return ColumnTransformer(transformers)


def _metric(task, y_true, pred):
    if task == "regression":
        return r2_score(y_true, pred)
    return f1_score(y_true, pred, average="macro")


def _cv_splitter(task, y_tr, folds):
    n = len(y_tr)
    folds = max(2, min(folds, n // 5 if n >= 10 else 2))
    if task == "classification":
        min_count = int(pd.Series(y_tr).value_counts().min())
        splits = min(folds, min_count)
        if splits >= 2:
            return StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    return KFold(n_splits=folds, shuffle=True, random_state=42)


def _fit_all(models, plan, X_tr, X_te, y_tr, y_te, task, use_nlp=False, scoring=None):
    if use_nlp:
        from nlp import TextPipeline
    if scoring is None:
        scoring = "r2" if task == "regression" else "f1_macro"
    rows, preds, pipes, skipped = [], {}, {}, []
    for name, model in models.items():
        try:
            t0 = time.perf_counter()
            pipe = (TextPipeline(model) if use_nlp
                    else Pipeline([("fe", FeatureEngineer()),
                                   ("prep", build_preprocessor(plan)),
                                   ("model", model)]))
            folds = 3 if type(model).__name__ in ("MLPRegressor", "MLPClassifier",
                                                   "TorchNN", "SVC", "SVR") else 5
            splitter = _cv_splitter(task, y_tr, folds)
            cv = cross_val_score(pipe, X_tr, y_tr, cv=splitter, scoring=scoring).mean()
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            secs = round(time.perf_counter() - t0, 1)
        except Exception as e:
            skipped.append(f"{name} ({type(e).__name__})"); continue
        preds[name], pipes[name] = pred, pipe
        if task == "regression":
            rows.append({"Model": name, "CV R2": round(cv, 3),
                         "Test R2": round(r2_score(y_te, pred), 3),
                         "Test MAE": round(mean_absolute_error(y_te, pred), 3),
                         "Test RMSE": round(mean_squared_error(y_te, pred) ** 0.5, 3),
                         "Time (s)": secs})
        else:
            rows.append({"Model": name, "CV F1": round(cv, 3),
                         "Accuracy": round(accuracy_score(y_te, pred), 3),
                         "Precision": round(precision_score(y_te, pred, average="macro", zero_division=0), 3),
                         "Recall": round(recall_score(y_te, pred, average="macro", zero_division=0), 3),
                         "F1": round(f1_score(y_te, pred, average="macro"), 3),
                         "Time (s)": secs})
    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values(results.columns[1], ascending=False).reset_index(drop=True)
    return results, preds, pipes, skipped


def diagnose(results, task, y_tr, X_tr, pipes):
    findings, fixes = [], set()
    pcol = "Test R2" if task == "regression" else "F1"
    best = results.iloc[0]
    if best["Model"].startswith("Baseline"):
        return ["no model did better than the simple baseline"], fixes
    if task == "classification":
        vc = y_tr.value_counts()
        if len(vc) >= 2 and vc.iloc[0] / vc.iloc[-1] >= 5 and best[pcol] < 0.6:
            findings.append(f"classes are very unbalanced (biggest {int(vc.iloc[0])}, smallest {int(vc.iloc[-1])})")
            fixes.add("balance")
    try:
        train_score = _metric(task, y_tr, pipes[best["Model"]].predict(X_tr))
        if train_score - best[pcol] > 0.25:
            findings.append(f"train score {train_score:.2f} vs test {best[pcol]:.2f} — memorizing")
            fixes.add("simplify")
    except Exception:
        pass
    if best[pcol] > 0.98:
        findings.append("score is almost perfect, which is suspicious")
    return findings, fixes


def _maybe_balance(model, task, y_tr):
    if task != "classification":
        return model
    try:
        vc = pd.Series(y_tr).value_counts()
        if len(vc) < 2 or vc.iloc[0] / max(int(vc.iloc[-1]), 1) < 4:
            return model
        params = model.get_params()
        if "class_weight" in params and params.get("class_weight") is None:
            model.set_params(class_weight="balanced")
    except Exception:
        pass
    return model


def apply_fixes(model, fixes):
    m = clone(model)
    try:
        params = m.get_params()
    except Exception:
        return m, {}
    kind = type(m).__name__; changes = {}
    if "balance" in fixes and "class_weight" in params:
        changes["class_weight"] = "balanced"
    if "simplify" in fixes:
        if kind.startswith("RandomForest") or kind.startswith("ExtraTrees"):
            changes.update(max_depth=12, min_samples_leaf=5)
        elif kind.startswith("HistGradientBoosting"):
            changes.update(max_depth=4, l2_regularization=1.0)
        elif kind.startswith("AdaBoost"):
            changes.update(n_estimators=50, learning_rate=0.5)
        elif kind.startswith("Bagging"):
            changes.update(max_samples=0.7)
        elif kind.startswith("MLP"):
            changes["alpha"] = 0.01
        elif kind == "TorchNN":
            layers = params.get("hidden_layers") or (64, 32)
            changes["hidden_layers"] = tuple(max(8, l // 2) for l in layers[:2])
            changes["dropout"] = min(0.4, params.get("dropout", 0.0) + 0.1)
        elif kind.startswith("KNeighbors"):
            changes["n_neighbors"] = 15
        elif kind.startswith("LogisticRegression"):
            changes["C"] = 0.1
        elif kind.startswith("Ridge"):
            changes["alpha"] = 10.0
        elif kind.startswith("LinearSVC"):
            changes["C"] = 0.1
        elif kind.startswith("ComplementNB"):
            changes["alpha"] = 1.0
        elif kind.startswith("DecisionTree"):
            changes.update(max_depth=4, min_samples_leaf=10)
    if changes:
        try: m.set_params(**changes)
        except Exception: pass
    return m, changes


def _self_check(results, preds, pipes, selected, plan, task, X_tr, X_te, y_tr, y_te,
                use_nlp=False, scoring=None):
    lines = []
    decision = agent_diagnose(results, task, y_tr, X_tr, pipes, use_nlp=use_nlp)
    findings = decision.get("findings", [])
    action = decision.get("action", "none")
    reason = decision.get("reason", "")
    if findings:
        lines.append("Agent noticed: " + "; ".join(findings) + ".")
    if reason:
        lines.append("Decision: " + reason)
    if action == "none":
        if not findings:
            lines.append("Checked for imbalances, overfitting and suspicious perfect "
                         "scores — nothing worth fixing.")
        return results, preds, pipes, lines
    fixes = {action}
    new_models, changed = {}, {}
    for name, model in selected.items():
        m2, ch = apply_fixes(model, fixes)
        if ch:
            new_models[name] = m2; changed[name] = ch
    if not new_models:
        lines.append("No adjustable settings for the chosen models; keeping current results.")
        return results, preds, pipes, lines
    parts = [name + " (" + ", ".join(k + "=" + str(v) for k, v in ch.items()) + ")"
             for name, ch in changed.items()]
    lines.append("Agent tried: " + "; ".join(parts) + ".")
    res2, preds2, pipes2, _ = _fit_all(new_models, plan, X_tr, X_te, y_tr, y_te, task,
                                        use_nlp=use_nlp, scoring=scoring)
    if res2.empty:
        lines.append("Retry failed; original results kept.")
        return results, preds, pipes, lines
    cvcol = results.columns[1]
    kept, rejected = [], []
    for _, row in res2.iterrows():
        name = row["Model"]
        idx = results.index[results["Model"] == name][0]
        old, new = float(results.at[idx, cvcol]), float(row[cvcol])
        if new > old + 0.003:
            for col in results.columns:
                results.at[idx, col] = row[col]
            preds[name], pipes[name] = preds2[name], pipes2[name]
            kept.append(f"{name} ({cvcol} {old:.3f} -> {new:.3f})")
        else:
            rejected.append(f"{name} ({cvcol} {old:.3f} vs {new:.3f})")
    if kept:
        lines.append("Kept new settings for: " + "; ".join(kept) + ".")
    if rejected:
        lines.append("Did not help: " + "; ".join(rejected) + ".")
    results = results.sort_values(cvcol, ascending=False).reset_index(drop=True)
    return results, preds, pipes, lines


def explain(results, task):
    lines = []
    base = results[results["Model"].str.startswith("Baseline")].iloc[0]
    best = results.iloc[0]
    if best["Model"].startswith("Baseline"):
        return ["No model did better than the simple baseline. The columns provided "
                "probably do not contain enough information to predict this target. "
                "That is a finding about the data, not a bug."]
    if task == "regression":
        r2 = best["Test R2"]
        lines.append(f"Best model: {best['Model']}. It explains about {max(r2, 0) * 100:.0f}% "
                     f"of the ups and downs in the target (R2 = {r2}).")
        lines.append(f"On average predictions are off by {best['Test MAE']} "
                     f"(baseline {base['Test MAE']}). RMSE = {best['Test RMSE']}.")
        if r2 < 0.2:
            lines.append("Weak result: the columns do not predict the target well.")
        elif r2 > 0.7:
            lines.append("This is a strong result.")
        if abs(best["CV R2"] - r2) > 0.1:
            lines.append("Warning: CV and test scores differ a lot (overfitting or too little data).")
    else:
        lines.append(f"Best model by F1: {best['Model']}. It gets {best['Accuracy'] * 100:.0f}% of "
                     f"predictions right (baseline {base['Accuracy'] * 100:.0f}%).")
        lines.append(f"Precision = {best['Precision']}, Recall = {best['Recall']}, "
                     f"F1 = {best['F1']} (baseline F1 = {base['F1']}).")
        if best["Accuracy"] < base["Accuracy"]:
            lines.append("Note: accuracy below baseline but F1 above — the model spreads guesses.")
        if base["Accuracy"] >= 0.7:
            lines.append("Warning: accuracy alone is misleading here (class imbalance).")
        if best["F1"] < 0.3:
            lines.append("Weak result: the chosen columns do not predict the target well.")
        if abs(best["CV F1"] - best["F1"]) > 0.1:
            lines.append("Warning: CV and test scores differ a lot.")
    return lines


def _score_from_preference(task, metric_preference):
    if task == "regression":
        return "r2"
    return {"balanced": "f1_macro",
            "precision": "precision_macro",
            "recall": "recall_macro"}.get(metric_preference, "f1_macro")


def run(df, target, plan, task=None, model_keys=None, self_check=True, nn_config=None,
        mode="all", metric_preference="balanced",
        interaction_specs=None, source_name="", hyperparams_overrides=None):
    notes = []
    df = df.copy()
    if plan.get("drop_duplicates"):
        before = len(df); df = df.drop_duplicates()
        notes.append(f"Removed {before - len(df)} duplicate rows before splitting")
    before = len(df)
    df = df.dropna(subset=[target])
    if len(df) < before:
        notes.append(f"Dropped {before - len(df)} rows with a missing target")

    applied_interactions = []
    if interaction_specs:
        df, applied_interactions = apply_interactions(df, interaction_specs)
        for spec in applied_interactions:
            notes.append(f"Added interaction feature '{spec['name']}' "
                         f"({spec['a']} {spec['op']} {spec['b']}) — {spec.get('reason', '')}")

    task = task or detect_task(df[target])
    if task == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(f"'{target}' is not numeric; choose classification.")
    if (task == "classification" and pd.api.types.is_numeric_dtype(df[target])
            and df[target].nunique() > 15):
        med = df[target].median()
        df[target] = np.where(df[target] >= med, "High", "Low")
        notes.append(f"Turned '{target}' into two classes: High (>= {med:g}) and Low (< {med:g})")
    notes.append(f"Task: {task} (target: '{target}')  |  metric: {metric_preference}")
    if task == "classification":
        counts = df[target].value_counts()
        rare = counts[counts < 10].index
        if len(rare) > 0:
            df = df[~df[target].isin(rare)]
            notes.append(f"Removed {len(rare)} very rare class(es)")
        if df[target].nunique() < 2:
            raise ValueError("The target has fewer than 2 classes.")
    if task == "regression":
        y = df[target]
        q1, q3 = y.quantile(0.25), y.quantile(0.75); iqr = q3 - q1
        keep = (y >= q1 - 3 * iqr) & (y <= q3 + 3 * iqr)
        if (~keep).sum() > 0:
            notes.append(f"Removed {int((~keep).sum())} rows with extreme target values")
            df = df[keep]

    drop_missing = set(plan.get("drop_high_missing") or [])
    kept, dropped = select_features(df, target)
    kept = [c for c in kept if c not in drop_missing]
    for c in drop_missing:
        if c != target:
            dropped.append((c, "cleaning plan: too many missing values"))
    for c, why in dropped:
        notes.append(f"Not used as a feature: '{c}' ({why})")
    for spec in applied_interactions:
        if spec["name"] not in kept:
            kept.append(spec["name"])
    if not kept:
        raise ValueError("No usable feature columns are left")

    X, y = df[kept], df[target]
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                                               stratify=stratify)
    notes.append(f"Split: {len(X_tr)} train, {len(X_te)} test")

    from nlp import (is_text_column, nlp_catalog,
                     top_words_per_class, top_bigrams_per_class)
    text_cols = [c for c in kept if is_text_column(df[c], c)]
    use_nlp = task == "classification" and len(text_cols) > 0

    scoring = _score_from_preference(task, metric_preference)

    if use_nlp:
        notes.append(f"Text input in {len(text_cols)} column(s) ({', '.join(text_cols)}); "
                     "using the NLP pipeline.")
        cat = nlp_catalog(len(X_tr))
    else:
        fe_preview = FeatureEngineer().fit(X_tr)
        for line in fe_preview.report_:
            notes.append("Converted " + line)
        for c, why in fe_preview.dropped_:
            notes.append(f"Not used as a feature: '{c}' ({why})")
        n_feat = len(fe_preview.feature_names_)
        if not n_feat:
            raise ValueError("None of the input columns could be turned into usable features")
        notes.append(f"The models use {n_feat} prepared input columns")
        notes.append(f"Mode: {mode.upper()}")
        cat = catalog(task, len(X_tr), nn_config=nn_config, mode=mode)

    keys = [k for k in (model_keys or []) if k in cat]
    user_manual_mode = bool(keys)
    if keys:
        notes.append("Models you chose: " + ", ".join(cat[k]["name"] for k in keys))
    else:
        if use_nlp:
            keys = list(cat)[:3]
            notes.append("Text pipeline picked: " + ", ".join(cat[k]["name"] for k in keys))
            for k in keys:
                notes.append("Why " + cat[k]["name"] + ": " + cat[k]["blurb"])
        else:
            if mode == "dl":
                keys = ["mlp"] if "mlp" in cat else list(cat)[:1]
                notes.append("DL mode: training the neural network")
            else:
                vc = y_tr.value_counts() if task == "classification" else None
                n_classes = int(len(vc)) if vc is not None else None
                imbalance = (float(vc.iloc[0] / max(vc.iloc[-1], 1))
                             if vc is not None and len(vc) >= 2 else 1.0)
                rule_keys, rule_reason = recommend(
                    task, len(X_tr), n_feat, list(cat),
                    n_classes=n_classes, imbalance_ratio=imbalance)
                prior = summarize_improvement(source_name, target) if source_name else None
                summary = (f"{len(X_tr)} rows, {n_feat} prepared features, "
                           f"target is {task}, mode is {mode}"
                           + (f", {n_classes} classes, imbalance {imbalance:.1f}x"
                              if n_classes else "")
                           + ". "
                           + (f"Prior runs: {prior}" if prior else "No prior runs on this dataset."))
                keys, reason = agent_pick_models(task, len(X_tr), n_feat, list(cat),
                                                  summary, fallback=rule_keys)
                if not keys:
                    keys, reason = rule_keys, rule_reason
                notes.append("Agent picked: " + ", ".join(cat[k]["name"] for k in keys)
                             + " — " + reason)

    overrides = hyperparams_overrides or {}
    selected = {}
    applied_hp = {}
    for k in keys:
        name = cat[k]["name"]
        if k in overrides and overrides[k]:
            m = make_with_params(k, task, overrides[k])
            if m is None:
                m = cat[k]["make"]()
        else:
            m = cat[k]["make"]()
        selected[name] = _maybe_balance(m, task, y_tr)
        
        # Capture hyperparameters for the results page
        hp_def = hyperparams(k, task)
        params = {}
        for p in hp_def:
            try:
                params[p["name"]] = m.get_params().get(p["name"], p["default"])
            except Exception:
                params[p["name"]] = p["default"]
        applied_hp[name] = params

    specific_reasons = {}
    if (not use_nlp and keys and not user_manual_mode
            and not cat[keys[0]]["name"].startswith("Baseline")):
        try:
            specific_reasons = agent_reason_models(
                task, len(X_tr), n_feat, [cat[k]["name"] for k in keys],
                {k: REASON.get(k, "") for k in keys})
        except Exception:
            specific_reasons = {}

    base_name, base_model = baseline(task)
    all_models = {base_name: base_model}
    all_models.update(selected)

    results, preds, pipes, skipped = _fit_all(all_models, plan, X_tr, X_te, y_tr, y_te,
                                                task, use_nlp=use_nlp, scoring=scoring)
    for s in skipped:
        notes.append("Skipped because of an error: " + s)
    if results.empty or len(results) < 2:
        raise ValueError("No model could be trained on this data")

    selfcheck = []
    if self_check:
        results, preds, pipes, selfcheck = _self_check(
            results, preds, pipes, selected, plan, task, X_tr, X_te, y_tr, y_te,
            use_nlp=use_nlp, scoring=scoring)

    best_name = results.iloc[0]["Model"]

    cm, per_class, top_words, top_bigrams = None, None, None, None
    if task == "classification":
        labels = sorted(y.unique())
        if len(labels) <= 8:
            cm = pd.DataFrame(confusion_matrix(y_te, preds[best_name], labels=labels),
                              index=[f"Actual {l}" for l in labels],
                              columns=[f"Predicted {l}" for l in labels])
        rep = classification_report(y_te, preds[best_name], output_dict=True, zero_division=0)
        rows = {k: v for k, v in rep.items() if isinstance(v, dict)
                and k not in ("macro avg", "weighted avg")}
        per_class = pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]]
        per_class[["precision", "recall", "f1-score"]] = per_class[["precision", "recall", "f1-score"]].round(3)
        per_class["support"] = per_class["support"].astype(int)
        per_class = per_class.sort_values("support", ascending=False)
        if use_nlp and not best_name.startswith("Baseline"):
            top_words = top_words_per_class(pipes[best_name], n=10)
            try:
                top_bigrams = top_bigrams_per_class(pipes[best_name], n=8)
            except Exception:
                top_bigrams = None

    nn_info = None
    for name, pipe in pipes.items():
        final = None
        if hasattr(pipe, "pipe_"):
            final = pipe.pipe_.named_steps.get("model")
        elif hasattr(pipe, "named_steps"):
            final = pipe.named_steps.get("model")
        if isinstance(final, TorchNN):
            nn_info = {"model_name": name, "device": getattr(final, "device_used_", "unknown"),
                       "epochs_trained": getattr(final, "epochs_trained_", None),
                       "hidden_layers": list(final.hidden_layers), "activation": final.activation,
                       "optimizer": final.optimizer, "learning_rate": final.learning_rate,
                       "dropout": final.dropout}
            break

    correct_samples, wrong_samples = [], []
    try:
        correct_samples, wrong_samples = sample_predictions(pipes[best_name], X_te, y_te,
                                                              n_correct=6, n_wrong=6)
    except Exception:
        pass

    best_row = results.iloc[0]
    best_score_col = results.columns[1]
    best_score = float(best_row[best_score_col])

    if source_name:
        try:
            log_run(source_name, target, task, kept, keys,
                    best_name, best_score, mode, metric_preference,
                    interaction_specs or [])
        except Exception:
            pass

    extras = {"task": task, "explanations": explain(results, task), "confusion": cm,
              "per_class": per_class, "selfcheck": selfcheck, "best_name": best_name,
              "best_pipeline": pipes[best_name], "features": kept, "target": target,
              "n_train": len(X_tr), "n_test": len(X_te),
              "mode": "nlp" if use_nlp else "tabular",
              "learning_mode": mode,
              "metric_preference": metric_preference,
              "top_words": top_words,
              "top_bigrams": top_bigrams,
              "text_columns": text_cols if use_nlp else [],
              "nn_info": nn_info,
              "X_test": X_te, "y_test": y_te,
              "correct_samples": correct_samples,
              "wrong_samples": wrong_samples,
              "interactions": applied_interactions,
              "specific_reasons": specific_reasons,
              "model_keys": keys,
              "applied_hyperparams": applied_hp} # Added here
    return results, notes, extras


# ============================================================ 8. INTERACTIONS
def agent_propose_interactions(df, features, target, max_n=3):
    numeric_feats = [f for f in features
                     if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    if len(numeric_feats) < 2:
        return []
    prompt = (
        "You are a feature engineer. Propose up to "
        f"{max_n} interaction features between these numeric columns that could "
        f"help predict '{target}'.\n\n"
        f"Available numeric columns: {numeric_feats}\n\n"
        "Interaction types you may use:\n"
        '- "ratio"     a / b   (when relative scale matters)\n'
        '- "product"   a * b   (when a combined effect matters)\n'
        '- "difference" a - b  (when distance matters)\n\n'
        "Rules: use only columns from the list. Do not repeat pairs. Give a one-line "
        "reason per interaction. Return ONLY JSON:\n"
        '{"interactions": [{"op": "ratio", "a": "col1", "b": "col2", '
        '"reason": "why this could help"}]}'
    )
    try:
        reply = ask(prompt, temperature=0.2) or ""
        start, end = reply.find("{"), reply.rfind("}")
        data = json.loads(reply[start:end + 1])
        out = []
        for item in data.get("interactions", [])[:max_n]:
            op = item.get("op"); a = item.get("a"); b = item.get("b")
            if op not in ("ratio", "product", "difference"): continue
            if a not in numeric_feats or b not in numeric_feats: continue
            if a == b: continue
            sym = {"ratio": "per", "product": "x", "difference": "minus"}[op]
            out.append({"op": op, "a": a, "b": b,
                        "name": f"{a}_{sym}_{b}",
                        "reason": item.get("reason", "")})
        return out
    except Exception:
        return []


def apply_interactions(df, specs):
    df = df.copy()
    applied = []
    for spec in specs:
        a, b, op = spec["a"], spec["b"], spec["op"]
        if a not in df.columns or b not in df.columns:
            continue
        try:
            if op == "ratio":
                denom = df[b].replace(0, np.nan)
                new = df[a] / denom
            elif op == "product":
                new = df[a] * df[b]
            else:
                new = df[a] - df[b]
            df[spec["name"]] = new
            applied.append(spec)
        except Exception:
            continue
    return df, applied


def agent_reason_models(task, n_rows, n_features, model_names, static_reasons):
    """Ask the LLM for dataset-specific reasons why each model was chosen."""
    prompt = (
        "You are a machine learning engineer explaining your picks to a colleague.\n"
        f"Dataset: {n_rows} rows, {n_features} prepared features, task: {task}.\n"
        f"Models chosen: {model_names}\n\n"
        "For EACH model, write ONE concise sentence (max 22 words) explaining why "
        "this specific model is a good fit for THIS dataset. Reference the number "
        "of rows, the number of features, or the task if it helps. Do NOT just "
        "repeat generic model descriptions. Be specific and confident.\n\n"
        "Return ONLY JSON: {\"reasons\": {\"<model name>\": \"<sentence>\", ...}}"
    )
    try:
        reply = ask(prompt, temperature=0.3) or ""
        start, end = reply.find("{"), reply.rfind("}")
        data = json.loads(reply[start:end + 1])
        raw = data.get("reasons") or {}
        out = {}
        for name in model_names:
            v = raw.get(name)
            if isinstance(v, str) and 5 < len(v) <= 260:
                out[name] = v.strip()
        return out
    except Exception:
        return {}


# ============================================================ 9. ADVISE
def suggest_features(df, target, used_features, task, leaderboard_summary):
    unused = [c for c in df.columns if c not in used_features and c != target]
    if not unused:
        return ["All columns are already in use."]
    prompt = f"""You are a data science advisor. The model below performed poorly.
Suggest which UNUSED columns might help predict the target, and why.

Target: {target}
Task: {task}
Currently used features: {used_features}
Unused columns: {unused}
Leaderboard (best scores): {leaderboard_summary}

For each column you suggest, give ONE short line: "column_name — why it might help".
Only suggest columns from the unused list. If none look useful, say so honestly.
Return ONLY a JSON object: {{"suggestions": ["col — reason", ...], "note": "one-sentence note"}}
"""
    try:
        reply = ask(prompt) or ""
        start, end = reply.find("{"), reply.rfind("}")
        data = json.loads(reply[start:end + 1])
        lines = list(data.get("suggestions", []))
        if data.get("note"):
            lines.append("Overall: " + data["note"])
        return lines or ["The LLM had no concrete suggestions."]
    except Exception as e:
        return [f"(Advice unavailable: {type(e).__name__})"]


# ============================================================ 10. EXPORT
def _safe(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "run"


def export_all(source_path, target, task, plan, warnings_list, load_notes, results, notes, extras):
    stem = os.path.splitext(os.path.basename(source_path))[0]
    folder = os.path.join("outputs", f"{_safe(stem)}__{_safe(target)}")
    os.makedirs(folder, exist_ok=True)
    results.to_csv(os.path.join(folder, "leaderboard.csv"), index=False)
    joblib.dump({"pipeline": extras["best_pipeline"], "features": extras["features"],
                 "target": target, "task": task, "best_model": extras["best_name"]},
                os.path.join(folder, "model.joblib"))
    lines = [
        "ML IN A BOX - RUN REPORT",
        "Created: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Source file: " + os.path.basename(source_path),
        f"Predicting: {target} ({task})",
        "Learning mode: " + extras.get("learning_mode", "all").upper(),
        "Metric preference: " + extras.get("metric_preference", "balanced"),
        "Input columns: " + ", ".join(extras["features"]),
        f"Training rows: {extras['n_train']}   Test rows: {extras['n_test']}",
        "",
        "LOAD NOTES: " + (", ".join(load_notes) if load_notes else "none"),
        "CLEANING PLAN (validated): " + str(plan),
        "WARNINGS: " + (", ".join(warnings_list) if warnings_list else "none"),
        "",
    ]
    if extras.get("interactions"):
        lines += ["INTERACTION FEATURES"]
        for spec in extras["interactions"]:
            lines += [f" - {spec['name']} = {spec['a']} {spec['op']} {spec['b']} "
                      f"({spec.get('reason', '')})"]
        lines += [""]
    lines += ["PIPELINE NOTES"] + [" - " + n for n in notes]
    lines += ["", "LEADERBOARD", results.to_string(index=False)]
    if extras.get("per_class") is not None:
        lines += ["", "PER-CLASS RESULTS", extras["per_class"].to_string()]
    if extras.get("confusion") is not None:
        lines += ["", "CONFUSION MATRIX", extras["confusion"].to_string()]
    if extras.get("nn_info"):
        nn = extras["nn_info"]
        lines += ["", "NEURAL NETWORK CONFIG"]
        lines += [f" - Model: {nn['model_name']}"]
        lines += [f" - Device: {nn['device']}"]
        lines += [f" - Architecture: {nn['hidden_layers']}  activation={nn['activation']}"]
        lines += [f" - Optimizer: {nn['optimizer']}  lr={nn['learning_rate']}  dropout={nn['dropout']}"]
    lines += ["", "SELF-CHECK"] + [" - " + s for s in extras["selfcheck"]]
    lines += ["", "IN PLAIN WORDS"] + [" - " + s for s in extras["explanations"]]
    lines += ["", f"Best model: {extras['best_name']} (saved as model.joblib)"]
    with open(os.path.join(folder, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return folder


def export_pdf(folder, source_path, target, task, plan, warnings_list,
               load_notes, results, notes, extras):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle)
    except ImportError:
        return None

    pdf_path = os.path.join(folder, "report.pdf")
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="ML in a Box — Run Report",
        author="Anas Albakoush, Muad Himeedah")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=20,
                        textColor=colors.HexColor("#0a2540"), spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                        textColor=colors.HexColor("#0a2540"),
                        spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14)
    meta = ParagraphStyle("meta", parent=styles["BodyText"], fontSize=9,
                          textColor=colors.HexColor("#6b7a8c"))
    story = []
    story.append(Paragraph("ML in a Box — Run Report", h1))
    story.append(Paragraph(
        f"Created {datetime.datetime.now():%Y-%m-%d %H:%M} &nbsp;·&nbsp; "
        f"Source: {os.path.basename(source_path)}", meta))
    story.append(Spacer(1, 14))
    story.append(Paragraph("Summary", h2))
    summary_rows = [
        ["Target", str(target)],
        ["Task", str(task)],
        ["Learning mode", extras.get("learning_mode", "all").upper()],
        ["Metric preference", extras.get("metric_preference", "balanced")],
        ["Input columns", ", ".join(extras["features"])],
        ["Training rows", str(extras.get("n_train", "?"))],
        ["Test rows", str(extras.get("n_test", "?"))],
    ]
    t = Table(summary_rows, colWidths=[4.2 * cm, 12.3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f8")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0a2540")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5d1de")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)
    story.append(Paragraph("Leaderboard", h2))
    lb_cols = list(results.columns)
    lb_rows = [lb_cols] + [list(r) for r in results.astype(str).values.tolist()]
    lbt = Table(lb_rows, repeatRows=1)
    lbt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0a2540")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5d1de")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(lbt)
    interactions = extras.get("interactions") or []
    if interactions:
        story.append(Paragraph("Interaction features", h2))
        for s in interactions:
            story.append(Paragraph(
                f"<b>{s['name']}</b> = {s['a']} {s['op']} {s['b']} — "
                f"{s.get('reason', '')}", body))
    nn = extras.get("nn_info")
    if nn:
        story.append(Paragraph("Neural network", h2))
        story.append(Paragraph(
            f"Model: {nn['model_name']} &nbsp;·&nbsp; Device: {nn['device']}", body))
        story.append(Paragraph(
            f"Architecture: {nn['hidden_layers']} &nbsp;·&nbsp; "
            f"activation: {nn['activation']} &nbsp;·&nbsp; "
            f"dropout: {nn['dropout']}", body))
        story.append(Paragraph(
            f"Optimizer: {nn['optimizer']} &nbsp;·&nbsp; "
            f"learning rate: {nn['learning_rate']}", body))
    story.append(Paragraph("Agent self-check", h2))
    for line in extras.get("selfcheck", []):
        story.append(Paragraph("• " + str(line), body))
    story.append(Paragraph("In plain words", h2))
    for line in extras.get("explanations", []):
        story.append(Paragraph("• " + str(line), body))
    if notes:
        story.append(Paragraph("Pipeline notes", h2))
        for n in notes[:40]:
            story.append(Paragraph("• " + str(n), body))
    story.append(Paragraph("Cleaning plan (validated)", h2))
    story.append(Paragraph(str(plan), body))
    if warnings_list:
        story.append(Paragraph("Warnings", h2))
        for w in warnings_list:
            story.append(Paragraph("• " + str(w), body))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Engr. Anas Yousef Albakoush &nbsp;·&nbsp; Engr. Muad Salah Himeedah "
        "&nbsp;·&nbsp; ML in a Box", meta))
    doc.build(story)
    return pdf_path


# ============================================================ 11. HISTORY
HISTORY_FILE = os.path.join("outputs", "_run_history.json")


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def log_run(source_name, target, task, features, model_keys, best_model,
            best_score, learning_mode, metric_preference, interactions):
    os.makedirs("outputs", exist_ok=True)
    history = load_history()
    history.append({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": source_name, "target": target, "task": task,
        "n_features": len(features), "features": list(features),
        "model_keys": list(model_keys or []),
        "best_model": best_model, "best_score": float(best_score),
        "learning_mode": learning_mode, "metric_preference": metric_preference,
        "interactions": [s.get("name") for s in (interactions or [])],
    })
    history = history[-200:]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return history


def find_prior_runs(source_name, target):
    return [h for h in load_history()
            if h.get("source") == source_name and h.get("target") == target]


def summarize_improvement(source_name, target):
    runs = find_prior_runs(source_name, target)
    if not runs:
        return None
    if len(runs) == 1:
        r = runs[0]
        return f"1 prior run. Best: {r['best_model']} ({r['best_score']:.3f})."
    best_ever = max(runs, key=lambda r: r["best_score"])
    last = runs[-1]
    return (f"{len(runs)} prior runs. Best ever: {best_ever['best_model']} "
            f"({best_ever['best_score']:.3f}). Latest: {last['best_model']} "
            f"({last['best_score']:.3f}).")


def history_dataframe(source_name=None, target=None):
    rows = load_history()
    if source_name:
        rows = [r for r in rows if r.get("source") == source_name]
    if target:
        rows = [r for r in rows if r.get("target") == target]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# ============================================================ 12. INTERVIEW (CLI)
MAX_ROUNDS = 12
CONFIRM_PHRASES = {"", "y", "yes", "ok", "okay", "go", "run", "continue", "start",
                   "done", "go ahead", "looks good", "correct", "right"}
ADD_WORDS = {"add", "include", "also", "plus"}
REMOVE_WORDS = {"remove", "without", "exclude", "drop", "skip", "except"}
TARGET_WORDS = {"predict", "target", "forecast"}
GENERIC = {"num", "number", "count", "code", "id", "date", "no", "total",
           "avg", "average", "the", "of"}


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _sing(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def resolve(name, columns):
    if not isinstance(name, str):
        return None
    if name in columns:
        return name
    n = _norm(name); norms = {_norm(c): c for c in columns}
    if n in norms:
        return norms[n]
    close = difflib.get_close_matches(n, list(norms), n=1, cutoff=0.75)
    return norms[close[0]] if close else None


def target_problem(df, target, task):
    s = df[target]
    if looks_like_id(s, target):
        return f"'{target}' looks like an ID, so there is nothing meaningful to predict."
    if (task == "classification" and not pd.api.types.is_numeric_dtype(s)
            and s.nunique() > 30):
        return f"'{target}' has {s.nunique()} different values — too many to predict."
    return None


def _columns_info(df, target):
    kept, dropped = select_features(df, target)
    why = dict(dropped)
    lines = []
    for c in df.columns:
        if c == target:
            status = "current target"
        elif c in kept:
            note = conversion_note(df[c], c)
            status = "usable as input" + (f" (converted: {note})" if note else "")
        else:
            status = "not usable: " + why.get(c, "unsupported")
        lines.append(f"- {c} (type {df[c].dtype}, {df[c].nunique()} distinct; {status})")
    return "\n".join(lines)


def _interpret(text, df, state):
    prompt = (
        "You are the setup assistant of a tool that trains ML models. You only see "
        "column metadata, never real rows. Translate the user's message into actions.\n\n"
        "Return ONLY JSON: {\"actions\": [...], \"reply\": string or null}\n"
        "Allowed actions:\n"
        "  {\"op\": \"set_target\", \"column\": name, \"task\": \"regression\"|\"classification\"|null}\n"
        "  {\"op\": \"set_task\", \"task\": \"regression\"|\"classification\"}\n"
        "  {\"op\": \"set_features\", \"columns\": [names]}\n"
        "  {\"op\": \"add_features\", \"columns\": [names]}\n"
        "  {\"op\": \"remove_features\", \"columns\": [names]}\n"
        "  {\"op\": \"confirm\"}\n\n"
        "Use exact column names. Never invent columns.\n\n"
        f"Current target: {state['target']} (task: {state['task']})\n"
        f"Current input columns: {state['features']}\n"
        f"Columns:\n{_columns_info(df, state['target'])}\n\n"
        f"User message: {text}"
    )
    for _ in range(2):
        try:
            reply = ask(prompt, temperature=0.1) or ""
        except Exception:
            return None
        start, end = reply.find("{"), reply.rfind("}")
        try:
            data = json.loads(reply[start:end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            acts = data.get("actions")
            data["actions"] = acts if isinstance(acts, list) else []
            r = data.get("reply")
            data["reply"] = r.strip() if isinstance(r, str) and r.strip() else None
            return data
    return None


def _default_features(df, target):
    return list(select_features(df, target)[0])


def _apply(ops, df, state, say):
    columns = list(df.columns)
    confirm = False
    for op in ops:
        if not isinstance(op, dict):
            continue
        kind = op.get("op")
        if kind == "confirm":
            confirm = True
        elif kind == "set_task":
            if state["target"] is None:
                say("Tell me what you want to predict first.")
            elif op.get("task") in ("regression", "classification"):
                state["task"] = op["task"]
                say(f"OK, treating this as {state['task']}.")
        elif kind == "set_target":
            col = resolve(op.get("column"), columns)
            if col is None:
                say(f"I could not find a column called '{op.get('column')}'.")
                continue
            if col == state["target"]:
                continue
            task = op.get("task")
            if task not in ("regression", "classification"):
                task = task_evidence(df[col])[0]
            if task == "regression" and not pd.api.types.is_numeric_dtype(df[col]):
                task = "classification"
            state["target"], state["task"] = col, task
            kept = select_features(df, col)[0]
            keep = [f for f in (state["features"] or []) if f in kept and f != col]
            if state["custom"] and keep:
                state["features"] = keep
            else:
                state["features"] = _default_features(df, col); state["custom"] = False
            say(f"Now predicting '{col}'.")
        elif kind in ("add_features", "remove_features", "set_features"):
            if state["target"] is None:
                say("Tell me what you want to predict first."); continue
            kept, dropped = select_features(df, state["target"])
            why = dict(dropped); good = []
            for name in op.get("columns") or []:
                col = resolve(name, columns)
                if col is None:
                    say(f"Column '{name}' not found.")
                elif col == state["target"]:
                    say(f"'{col}' is the target, not an input.")
                elif kind != "remove_features" and col not in kept:
                    say(f"Cannot use '{col}': {why.get(col, 'not supported')}.")
                elif col not in good:
                    good.append(col)
            if not good:
                continue
            if kind == "add_features":
                new = [c for c in good if c not in state["features"]]
                if new:
                    state["features"] = state["features"] + new
                    state["custom"] = True
                    say("Added: " + ", ".join(new))
            elif kind == "remove_features":
                rest = [c for c in state["features"] if c not in good]
                if rest:
                    state["features"] = rest; state["custom"] = True
                    say("Removed: " + ", ".join(good))
            else:
                state["features"] = good; state["custom"] = True
                say("Using only: " + ", ".join(good))
    return confirm


def _show_summary(state, df):
    kind = "a number" if state["task"] == "regression" else "a category"
    print("\nHere is what I understood:")
    print(f"  Predict: '{state['target']}' ({kind})")
    for c in state["features"]:
        note = conversion_note(df[c], c)
        print(f"    - {c}" + (f"  [{note}]" if note else ""))


def interview(df):
    state = {"target": None, "task": None, "features": None,
             "custom": False, "last_reply": None}
    print("\nTell me what you want to find out, in your own words.")
    text = input("> ").strip()
    for _ in range(MAX_ROUNDS):
        confirm = False; msgs = []; reply = None
        if state["target"] is None and not text:
            text = input("> ").strip(); continue
        if state["target"] is not None and _norm(text) in CONFIRM_PHRASES:
            confirm = True
        else:
            data = _interpret(text, df, state)
            if data is None:
                ops, reply = [], "I could not understand that."
            else:
                ops, reply = data["actions"], data["reply"]
            confirm = _apply(ops, df, state, msgs.append)
        for m in msgs:
            print("\n" + m)
        if reply:
            print("\n" + reply)
        if state["target"] is None:
            text = input("> ").strip(); continue
        problem = target_problem(df, state["target"], state["task"])
        _show_summary(state, df)
        if problem:
            print("\nProblem: " + problem)
        elif confirm and not msgs and state["features"]:
            return state["target"], state["task"], state["features"]
        text = input("> ").strip()
    print("\nI still need more help, so let's choose manually.")
    return None


# ============================================================ 13. DEEP LEARNING
def torch_status():
    if not TORCH_AVAILABLE:
        return {"torch": False, "cuda": False, "device_name": None,
                "device_count": 0, "torch_version": None,
                "message": ("PyTorch is not installed. The app will use the "
                            "scikit-learn neural network (CPU only).")}
    cuda = torch.cuda.is_available()
    if cuda:
        name = torch.cuda.get_device_name(0); count = torch.cuda.device_count()
        return {"torch": True, "cuda": True, "device_name": name,
                "device_count": count, "torch_version": torch.__version__,
                "message": f"GPU detected: {name} ({count} device(s))."}
    return {"torch": True, "cuda": False, "device_name": None,
            "device_count": 0, "torch_version": torch.__version__,
            "message": ("PyTorch installed, no CUDA GPU. Training runs on CPU.")}


class TorchNN(BaseEstimator):
    def __init__(self, task="regression", hidden_layers=(64, 32),
                 activation="relu", dropout=0.0, optimizer="adam",
                 learning_rate=1e-3, weight_decay=1e-4, batch_size=32,
                 max_epochs=200, early_stopping=True, patience=20,
                 use_gpu=True, random_state=42, verbose=0):
        self.task = task; self.hidden_layers = tuple(hidden_layers)
        self.activation = activation; self.dropout = dropout
        self.optimizer = optimizer; self.learning_rate = learning_rate
        self.weight_decay = weight_decay; self.batch_size = batch_size
        self.max_epochs = max_epochs; self.early_stopping = early_stopping
        self.patience = patience; self.use_gpu = use_gpu
        self.random_state = random_state; self.verbose = verbose

    def _device(self):
        if TORCH_AVAILABLE and self.use_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build(self, n_in, n_out):
        acts = {"relu": tnn.ReLU, "tanh": tnn.Tanh,
                "logistic": tnn.Sigmoid, "identity": tnn.Identity}
        act_cls = acts.get(self.activation, tnn.ReLU)
        layers = []; prev = n_in
        for h in self.hidden_layers:
            layers += [tnn.Linear(prev, h), act_cls()]
            if self.dropout > 0: layers.append(tnn.Dropout(self.dropout))
            prev = h
        layers.append(tnn.Linear(prev, n_out))
        return tnn.Sequential(*layers)

    def _to_dense(self, X):
        if hasattr(X, "toarray"): X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        torch.manual_seed(self.random_state)
        X = self._to_dense(X); self.device_ = self._device()
        if self.task == "classification":
            self.classes_ = np.array(sorted(pd.Series(y).unique()))
            self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
            y_idx = np.array([self.class_to_idx_[c] for c in y], dtype=np.int64)
            n_out = len(self.classes_)
            counts = np.bincount(y_idx, minlength=n_out).astype(np.float32)
            weights = counts.sum() / (n_out * np.maximum(counts, 1))
            self.loss_fn_ = tnn.CrossEntropyLoss(
                weight=torch.tensor(weights, device=self.device_))
        else:
            y_idx = np.asarray(y, dtype=np.float32).reshape(-1, 1)
            n_out = 1; self.loss_fn_ = tnn.MSELoss()
        n = len(X); rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        val_size = max(1, int(0.15 * n)) if self.early_stopping and n >= 20 else 0
        val_idx, tr_idx = idx[:val_size], idx[val_size:]
        X_tr = torch.tensor(X[tr_idx], device=self.device_)
        y_tr = torch.tensor(y_idx[tr_idx], device=self.device_)
        if val_size:
            X_val = torch.tensor(X[val_idx], device=self.device_)
            y_val = torch.tensor(y_idx[val_idx], device=self.device_)
        else:
            X_val = y_val = None
        self.net_ = self._build(X.shape[1], n_out).to(self.device_)
        opt_map = {"adam": topt.Adam, "sgd": topt.SGD,
                   "adamw": topt.AdamW, "rmsprop": topt.RMSprop}
        opt_cls = opt_map.get(self.optimizer, topt.Adam)
        kwargs = {"lr": self.learning_rate, "weight_decay": self.weight_decay}
        if self.optimizer == "sgd": kwargs["momentum"] = 0.9
        opt = opt_cls(self.net_.parameters(), **kwargs)
        loader = DataLoader(TensorDataset(X_tr, y_tr),
                            batch_size=min(self.batch_size, len(X_tr)), shuffle=True)
        best_val = float("inf"); best_state = None
        patience_counter = 0; last_epoch = 0
        for epoch in range(self.max_epochs):
            last_epoch = epoch + 1
            self.net_.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss = self.loss_fn_(self.net_(xb), yb)
                loss.backward(); opt.step()
            if X_val is not None:
                self.net_.eval()
                with torch.no_grad():
                    v = float(self.loss_fn_(self.net_(X_val), y_val).item())
                if v < best_val - 1e-5:
                    best_val = v
                    best_state = {k: t.detach().clone() for k, t in self.net_.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience: break
        if best_state is not None: self.net_.load_state_dict(best_state)
        self.epochs_trained_ = last_epoch
        self.device_used_ = str(self.device_)
        return self

    def predict(self, X):
        X = self._to_dense(X); self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.tensor(X, device=self.device_))
            if self.task == "regression":
                return out.cpu().numpy().reshape(-1)
            return self.classes_[out.argmax(dim=1).cpu().numpy()]

    def predict_proba(self, X):
        if self.task != "classification":
            raise RuntimeError("predict_proba only for classification")
        X = self._to_dense(X); self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.tensor(X, device=self.device_))
            return torch.softmax(out, dim=1).cpu().numpy()

    def get_params(self, deep=True):
        return {"task": self.task, "hidden_layers": self.hidden_layers,
                "activation": self.activation, "dropout": self.dropout,
                "optimizer": self.optimizer, "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay, "batch_size": self.batch_size,
                "max_epochs": self.max_epochs, "early_stopping": self.early_stopping,
                "patience": self.patience, "use_gpu": self.use_gpu,
                "random_state": self.random_state, "verbose": self.verbose}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


DEFAULT_NN_CONFIG = {
    "framework": "sklearn", "hidden_layers": [64, 32], "activation": "relu",
    "dropout": 0.0, "optimizer": "adam", "learning_rate": 0.001,
    "weight_decay": 0.0001, "batch_size": 32, "max_epochs": 100,
    "early_stopping": True, "patience": 20, "use_gpu": True,
}


# ============================================================ 14. AGENTIC
_TARGET_HINTS = ("target", "label", "class", "price", "score", "outcome", "survived",
                 "churn", "rating", "salary", "income", "sold", "default", "fraud",
                 "diagnosis", "result", "y")


def heuristic_target(df):
    scored = []
    for c in df.columns:
        kind = column_kind(df[c], c)
        if kind in ("id", "empty"):
            continue
        miss = float(df[c].isna().mean())
        if miss > 0.60:
            continue
        guess, conf, _ = task_evidence(df[c])
        k = int(df[c].nunique(dropna=True))
        score = 0.0
        name = str(c).lower()
        if any(h in name for h in _TARGET_HINTS):
            score += 6
        if guess == "classification" and 2 <= k <= 20:
            score += 3
        if guess == "regression" and kind in ("num", "numunit") and k > 15:
            score += 3
        if conf == "high":
            score += 1
        if kind == "date":
            score -= 4
        if kind == "text":
            score -= 2
        if miss < 0.05:
            score += 1
        if c == df.columns[-1]:
            score += 0.5
        scored.append((score, c, guess))
    if not scored:
        return None
    scored.sort(reverse=True)
    _, target, task = scored[0]
    return {"target": target, "task": task,
            "reason": ("Heuristic pick: name, type, completeness and number of "
                       f"distinct values all point to '{target}' as the outcome.")}


def recommend_target(df):
    fallback = heuristic_target(df)
    candidates = {}
    for c in df.columns:
        kind = column_kind(df[c], c)
        if kind in ("id", "empty"):
            continue
        entry = {"kind": kind, "unique": int(df[c].nunique()),
                 "missing_pct": round(float(df[c].isna().mean() * 100), 1)}
        if kind in ("num", "numunit"):
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v):
                entry["min"] = float(v.min()); entry["max"] = float(v.max())
                entry["mean"] = round(float(v.mean()), 2)
        else:
            entry["samples"] = df[c].dropna().astype(str).head(4).tolist()
        candidates[c] = entry
    if len(candidates) < 2:
        return fallback
    prompt = (
        "You are a data scientist. Given a dataset's column profile, recommend the "
        "single best column to use as the target.\n\n"
        "Rules:\n"
        "- Avoid ID-like columns, dates, and columns that are >60% empty.\n"
        "- Prefer a column where the other columns plausibly correlate with it.\n"
        "- Classification targets should have at most ~30 classes.\n"
        "- Prefer columns whose names look like outcomes (price, label, class, "
        "survived, churn, score).\n\n"
        f"Rows: {len(df)}, columns: {len(df.columns)}\n\n"
        "Candidates (JSON):\n" + json.dumps(candidates, indent=2, default=str)[:4000] + "\n\n"
        "Return ONLY JSON:\n"
        '{"target": "<column name>", "task": "regression" | "classification", '
        '"reason": "<one short sentence>"}'
    )
    try:
        reply = ask(prompt, temperature=0.2) or ""
        start, end = reply.find("{"), reply.rfind("}")
        data = json.loads(reply[start:end + 1])
        target = data.get("target")
        if target not in df.columns:
            return fallback
        guess, conf, _ = task_evidence(df[target])
        if conf == "high":
            data["task"] = guess
        return {"target": target, "task": data.get("task", guess),
                "reason": data.get("reason", "")}
    except Exception:
        return fallback


def agent_pick_models(task, n_rows, n_features, available, profile_summary, fallback=None):
    available = [k for k in available if k in REASON]
    if not available:
        return [], "no models available"
    descriptions = "\n".join(f"- {k}: {REASON[k]}" for k in available)
    prompt = (
        "You are a machine learning engineer. Pick 3 or 4 models to train on this "
        "dataset, ordered with the one you expect to win first.\n\n"
        f"Task: {task}\nRows: {n_rows}\nPrepared features: {n_features}\n"
        f"Profile summary: {profile_summary}\n\n"
        f"Available models:\n{descriptions}\n\n"
        "Rules:\n"
        "- Prefer diverse picks: one simple, one strong, one that handles "
        "non-linearity or imbalance.\n"
        "- For very small data (<500 rows), avoid models that overfit easily.\n"
        "- If classes are unbalanced, prefer tree ensembles or regularized linear "
        "models over KNN/SVM.\n"
        "- If the profile mentions prior runs, favour models that did well before.\n"
        "Return ONLY JSON:\n"
        '{"keys": ["k1", "k2", "k3"], "reason": "<one short sentence>"}'
    )
    try:
        reply = ask(prompt, temperature=0.2) or ""
        start, end = reply.find("{"), reply.rfind("}")
        data = json.loads(reply[start:end + 1])
        keys = [k for k in data.get("keys", []) if k in available][:4]
        if len(keys) < 2:
            return fallback or [], "LLM picked too few; using rule-based fallback"
        return keys, data.get("reason", "the LLM picked these based on the profile")
    except Exception:
        return fallback or [], "LLM unavailable; using rule-based fallback"


def agent_diagnose(results, task, y_tr, X_tr, pipes, use_nlp=False):
    rule_findings, rule_fixes = diagnose(results, task, y_tr, X_tr, pipes)
    pcol = "Test R2" if task == "regression" else "F1"
    best = results.iloc[0]
    try:
        train_str = f"{_metric(task, y_tr, pipes[best['Model']].predict(X_tr)):.3f}"
    except Exception:
        train_str = "unknown"
    balance_note = "N/A"
    if task == "classification":
        vc = y_tr.value_counts()
        if len(vc) >= 2:
            balance_note = (f"largest class {int(vc.iloc[0])}, smallest {int(vc.iloc[-1])}, "
                            f"ratio {vc.iloc[0] / vc.iloc[-1]:.1f}x")
    leaderboard = results.to_string(index=False)
    prompt = (
        "You are a machine learning reviewer. Look at the training results and "
        "pick the SINGLE best next action.\n\n"
        f"Task: {task}\nLeaderboard:\n{leaderboard}\n\n"
        f"Best model: {best['Model']}\nTrain score: {train_str}\n"
        f"Test score: {best[pcol]:.3f}\nClass balance: {balance_note}\n\n"
        "Possible actions:\n"
        "- \"balance\": apply class weights (classification only)\n"
        "- \"simplify\": reduce model complexity\n"
        "- \"none\": accept the result\n\n"
        "Return ONLY JSON:\n"
        '{"findings": ["short finding 1", "short finding 2"], '
        '"action": "balance" | "simplify" | "none", "reason": "<short>"}'
    )
    try:
        reply = ask(prompt, temperature=0.2) or ""
        start, end = reply.find("{"), reply.rfind("}")
        data = json.loads(reply[start:end + 1])
        action = data.get("action", "none")
        if action not in ("balance", "simplify", "none"):
            action = "none"
        return {"findings": data.get("findings", rule_findings),
                "action": action, "reason": data.get("reason", "LLM decision")}
    except Exception:
        action = "none"
        if "balance" in rule_fixes: action = "balance"
        elif "simplify" in rule_fixes: action = "simplify"
        return {"findings": rule_findings, "action": action,
                "reason": "LLM unavailable; used rule-based diagnosis"}


def sample_predictions(pipeline, X_test, y_test, n_correct=6, n_wrong=6):
    try:
        preds = pipeline.predict(X_test)
    except Exception:
        return [], []

    y_arr = np.asarray(y_test)
    correct_pool, wrong_pool = [], []

    rng = np.random.RandomState(42)
    idx_order = rng.permutation(len(X_test))
    feat_cols = list(X_test.columns)[:4]

    for i in idx_order:
        row = X_test.iloc[i]
        pred = preds[i]; actual = y_arr[i]
        is_correct = bool(pred == actual)
        snippet = " · ".join(f"{c}: {str(row[c])[:40]}" for c in feat_cols)
        entry = {"row_index": int(i), "actual": str(actual),
                 "predicted": str(pred), "snippet": snippet}
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(X_test.iloc[[i]])[0]
                entry["confidence"] = round(float(proba.max()), 3)
            except Exception:
                pass
        if is_correct and len(correct_pool) < n_correct:
            correct_pool.append(entry)
        elif not is_correct and len(wrong_pool) < n_wrong:
            wrong_pool.append(entry)
        if len(correct_pool) >= n_correct and len(wrong_pool) >= n_wrong:
            break

    return correct_pool, wrong_pool


# ============================================================ 15. AUTO MODALITY & TUNING
def auto_detect_modality(df):
    """Automatically decide between 'tabular' (ML/DL) and 'nlp' based on column content."""
    from nlp import is_text_column
    text_cols = [c for c in df.columns if is_text_column(df[c], c)]
    if len(text_cols) > 0:
        return "nlp"
    return "tabular"

def agent_tune_hyperparams(model_key, task, current_params):
    """Ask the LLM to suggest slightly different hyperparameters for a given model."""
    prompt = (
        "You are a machine learning engineer tuning a model. "
        f"Model: {model_key}\nTask: {task}\n"
        f"Current hyperparameters: {current_params}\n\n"
        "Suggest ONE variation of these hyperparameters that might improve performance. "
        "Return ONLY a JSON object with the exact parameter names and the new suggested values. "
        "Example: {\"n_estimators\": 200, \"max_depth\": 10}"
    )
    try:
        reply = ask(prompt, temperature=0.3) or ""
        start, end = reply.find("{"), reply.rfind("}")
        return json.loads(reply[start:end+1])
    except Exception:
        return current_params
# ============================================================ XAI HELPERS
def get_pipeline_model(pipeline):
    """Return the innermost sklearn model from a full pipeline."""
    if hasattr(pipeline, "named_steps"):
        return pipeline.named_steps.get("model", pipeline)
    if hasattr(pipeline, "pipe_"):
        return pipeline.pipe_.named_steps.get("model", pipeline)
    return pipeline


def get_feature_names(pipeline, fallback_features):
    """Best-effort extraction of the transformed feature names."""
    try:
        if hasattr(pipeline, "named_steps"):
            prep = pipeline.named_steps.get("prep")
            if prep is not None and hasattr(prep, "get_feature_names_out"):
                return list(prep.get_feature_names_out())
        elif hasattr(pipeline, "pipe_"):
            prep = pipeline.pipe_.named_steps.get("prep")
            if prep is not None and hasattr(prep, "get_feature_names_out"):
                return list(prep.get_feature_names_out())
    except Exception:
        pass
    return list(fallback_features)