"""
NLP engine — text classification, document tools, topic search.

Sections:
  1. Text classification pipeline (TF-IDF + linear classifiers)
  2. Document cleaning + tokenization (Unicode-aware)
  3. Summarization, translation, grammar
  4. Topic clustering and similarity search
"""
import json
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from llm import ask
from ml import column_kind


# ============================================================ 1. CLASSIFICATION
def is_text_column(series, name=""):
    return column_kind(series, name) == "text"


def detect_text_columns(df, exclude=()):
    return [c for c in df.columns if c not in exclude and is_text_column(df[c], c)]


class TextPipeline(BaseEstimator):
    """TF-IDF + linear classifier, with optional numeric/categorical columns."""

    def __init__(self, model):
        self.model = model

    def _split_columns(self, X):
        texts, nums, cats = [], [], []
        for c in X.columns:
            if is_text_column(X[c], c):
                texts.append(c)
            elif (pd.api.types.is_numeric_dtype(X[c])
                  and not pd.api.types.is_bool_dtype(X[c])):
                nums.append(c)
            else:
                cats.append(c)
        return texts, nums, cats

    def _prepare(self, X):
        X = pd.DataFrame(X).copy()
        texts, nums, cats = self._split_columns(X)
        if len(texts) > 1:
            X["__text__"] = (X[texts].fillna("").astype(str).agg(" ".join, axis=1))
            texts = ["__text__"]
        return X, texts, nums, cats

    def fit(self, X, y):
        X, texts, nums, cats = self._prepare(X)
        self.text_columns_ = texts
        transformers = []
        if texts:
            transformers.append(("text", TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=5000,
                sublinear_tf=True, strip_accents="unicode", lowercase=True,
            ), texts[0]))
        if nums:
            transformers.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), nums))
        if cats:
            transformers.append(("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cats))
        if not transformers:
            raise ValueError("No usable columns for the NLP pipeline")
        self.preprocessor_ = ColumnTransformer(transformers)
        self.pipe_ = Pipeline([("prep", self.preprocessor_), ("model", self.model)])
        self.pipe_.fit(X, y)
        return self

    def predict(self, X):
        X, _, _, _ = self._prepare(X)
        return self.pipe_.predict(X)

    def predict_proba(self, X):
        X, _, _, _ = self._prepare(X)
        if hasattr(self.pipe_, "predict_proba"):
            return self.pipe_.predict_proba(X)
        raise AttributeError("underlying model has no predict_proba")

    def get_params(self, deep=True):
        return {"model": self.model}

    def set_params(self, **params):
        if "model" in params:
            self.model = params["model"]
        return self


def nlp_catalog(n_rows):
    c = {}
    c["logistic"] = {
        "name": "Logistic Regression",
        "make": lambda: LogisticRegression(max_iter=2000, C=1.0),
        "blurb": "a linear boundary on the TF-IDF features; standard strong baseline for text",
    }
    c["linear_svc"] = {
        "name": "Linear SVM",
        "make": lambda: CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000), cv=3),
        "blurb": "a linear boundary that maximizes the margin; often the strongest on TF-IDF",
    }
    c["complement_nb"] = {
        "name": "Complement Naive Bayes",
        "make": lambda: ComplementNB(alpha=0.3),
        "blurb": "a fast probabilistic classifier that handles imbalanced text well",
    }
    if n_rows < 2000:
        c["logistic_l1"] = {
            "name": "Logistic Regression (L1)",
            "make": lambda: LogisticRegression(max_iter=2000, C=1.0,
                                                penalty="l1", solver="liblinear"),
            "blurb": "a sparse variant that keeps only the most informative words per class",
        }
    return c


def _unwrap_model(model):
    """If the model is wrapped in a calibrator, drill into the base estimator."""
    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        try:
            inner = model.calibrated_classifiers_[0]
            if hasattr(inner, "estimator"):
                return inner.estimator
            if hasattr(inner, "base_estimator"):
                return inner.base_estimator
        except Exception:
            pass
    if hasattr(model, "estimator"):
        return model.estimator
    return model


def top_words_per_class(pipeline, n=8):
    try:
        prep = pipeline.pipe_.named_steps["prep"]
        model = pipeline.pipe_.named_steps["model"]
        tfidf = prep.named_transformers_.get("text")
        if tfidf is None:
            return None
        vocab = np.array(tfidf.get_feature_names_out())
        base = _unwrap_model(model)
        if not hasattr(base, "coef_"):
            return None
        coef = base.coef_
        classes = getattr(model, "classes_", getattr(base, "classes_", None))
        if classes is None:
            return None
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        out = {}
        for i, cls in enumerate(classes):
            row = coef[i] if i < len(coef) else coef[0]
            row = row[: len(vocab)]
            top_idx = np.argsort(row)[-n:][::-1]
            out[str(cls)] = [str(vocab[j]) for j in top_idx if row[j] > 0]
        return out
    except Exception:
        return None


def top_bigrams_per_class(pipeline, n=8):
    """Same idea as top_words_per_class, but only 2-grams (phrases)."""
    try:
        prep = pipeline.pipe_.named_steps["prep"]
        model = pipeline.pipe_.named_steps["model"]
        tfidf = prep.named_transformers_.get("text")
        if tfidf is None:
            return None
        vocab = np.array(tfidf.get_feature_names_out())
        bigram_mask = np.array([" " in v for v in vocab])
        if not bigram_mask.any():
            return None
        base = _unwrap_model(model)
        if not hasattr(base, "coef_"):
            return None
        coef = base.coef_
        classes = getattr(model, "classes_", getattr(base, "classes_", None))
        if classes is None:
            return None
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        out = {}
        for i, cls in enumerate(classes):
            row = coef[i] if i < len(coef) else coef[0]
            row = row[: len(vocab)]
            masked = np.where(bigram_mask, row, -np.inf)
            top_idx = np.argsort(masked)[-n:][::-1]
            out[str(cls)] = [str(vocab[j]) for j in top_idx if masked[j] > 0]
        return out
    except Exception:
        return None


# ============================================================ 2. CLEAN + TOKENIZE
HTML_TAG = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
WS_RE = re.compile(r"\s+")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
DIGIT_RE = re.compile(r"\d+")
# Unicode-aware word tokenization: matches any letter sequence (Latin, Arabic,
# Cyrillic, Hindi, Chinese, ...) optionally followed by an apostrophe-word, or
# a bare number. Also handles Arabic Tatweel and diacritics implicitly, since
# Python's `re.UNICODE` treats them as combining characters.
WORD_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?|\d+", re.UNICODE)

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in",
    "on", "at", "by", "for", "with", "about", "as", "into", "like", "through",
    "after", "before", "between", "up", "down", "out", "off", "over", "under",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "shall", "should", "may",
    "might", "must", "can", "could", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "this", "that", "these", "those", "not", "no", "yes",
}


def clean_text(text,
               lowercase=True,
               strip_html=True,
               strip_urls=True,
               strip_emails=True,
               strip_punctuation=False,
               strip_digits=False,
               collapse_whitespace=True,
               remove_stopwords=False):
    """Full cleaning pipeline with everything toggleable.

    Returns (cleaned_text, stats). Every operation has a counter so the
    UI can show exactly what was removed.
    """
    original = text or ""
    stats = {"original_chars": len(original)}

    out = original
    if strip_html:
        n = len(HTML_TAG.findall(out)); stats["html_tags_removed"] = n
        out = HTML_TAG.sub(" ", out)
    if strip_urls:
        n = len(URL_RE.findall(out)); stats["urls_removed"] = n
        out = URL_RE.sub(" ", out)
    if strip_emails:
        n = len(EMAIL_RE.findall(out)); stats["emails_removed"] = n
        out = EMAIL_RE.sub(" ", out)

    n = len(CTRL_RE.findall(out)); stats["control_chars_removed"] = n
    out = CTRL_RE.sub(" ", out)

    if lowercase:
        before = out
        out = out.lower()
        stats["lowercased"] = (before != out)

    if strip_punctuation:
        n = len(PUNCT_RE.findall(out)); stats["punctuation_removed"] = n
        out = PUNCT_RE.sub(" ", out)

    if strip_digits:
        n = len(DIGIT_RE.findall(out)); stats["digit_runs_removed"] = n
        out = DIGIT_RE.sub(" ", out)

    if collapse_whitespace:
        out = WS_RE.sub(" ", out).strip()

    if remove_stopwords:
        tokens = out.split()
        kept = [w for w in tokens if w.lower() not in STOP_WORDS]
        stats["stopwords_removed"] = len(tokens) - len(kept)
        out = " ".join(kept)

    stats["cleaned_chars"] = len(out)
    stats["reduction_pct"] = round(
        (1 - len(out) / max(stats["original_chars"], 1)) * 100, 1)
    return out, stats


def tokenize(text, keep_stopwords=True, keep_punctuation=False):
    """Split text into tokens. Returns a list of strings.

    Unicode-aware: works on Arabic, Cyrillic, Greek, Devanagari, etc. —
    not just Latin. Numbers are kept as-is; punctuation is dropped unless
    explicitly requested.
    """
    if text is None:
        return []
    t = str(text)
    tokens = [m.group(0).lower() for m in WORD_TOKEN_RE.finditer(t)]
    if not keep_stopwords:
        tokens = [tok for tok in tokens if tok not in STOP_WORDS]
    return tokens


def key_phrases(text, n=10, from_tokens=None):
    """Unigrams plus bigrams, ranked by count (stop-words and short tokens dropped)."""
    if from_tokens is not None:
        tokens = [t for t in from_tokens if len(t) > 2 and t not in STOP_WORDS
                  and not t.isdigit()]
    else:
        tokens = [t for t in tokenize(text) if len(t) > 2 and t not in STOP_WORDS
                  and not t.isdigit()]
    counts = Counter(tokens)
    bigrams = Counter(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    merged = counts + bigrams
    return [{"phrase": w, "count": c} for w, c in merged.most_common(n)]


# ============================================================ 3. SUMMARIZE / TRANSLATE
def summarize_extractive(text, n=3):
    text = (text or "").strip()
    if not text:
        return {"summary": "", "sentences": [], "note": "empty input"}
    raw = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw if len(s.strip()) >= 20]
    if len(sentences) <= n:
        return {"summary": " ".join(sentences), "sentences": sentences,
                "note": "shorter than the requested size; returned as-is"}
    words = [w.lower().strip(".,;:!?\"'()[]{}") for s in sentences for w in s.split()]
    words = [w for w in words if len(w) > 3 and w not in STOP_WORDS]
    freq = Counter(words)
    if not freq:
        return {"summary": sentences[0], "sentences": sentences[:n],
                "note": "no usable words after stop-word removal"}
    max_freq = max(freq.values())
    norm = {w: c / max_freq for w, c in freq.items()}
    scored = []
    for i, s in enumerate(sentences):
        s_words = [w.lower().strip(".,;:!?\"'()[]{}") for w in s.split()]
        score = sum(norm.get(w, 0) for w in s_words) / max(len(s_words), 1)
        scored.append((i, s, score))
    top = sorted(scored, key=lambda x: x[2], reverse=True)[:n]
    top = sorted(top, key=lambda x: x[0])
    summary = " ".join(s for _, s, _ in top)
    return {"summary": summary, "sentences": [s for _, s, _ in top],
            "note": f"picked {len(top)} of {len(sentences)} sentences by word-frequency"}


def summarize_abstractive(text, target_words=80):
    text = (text or "").strip()
    if not text:
        return {"summary": "", "note": "empty input"}
    prompt = (
        f"Summarize the following text in about {target_words} words. "
        "Use plain English. Preserve the key facts and drop anything that is "
        "example or filler. Output ONLY the summary — no preamble, no quotes.\n\n"
        f"--- text ---\n{text[:8000]}\n--- end ---"
    )
    try:
        out = (ask(prompt, temperature=0.3) or "").strip()
    except Exception as e:
        return {"summary": f"(Summarizer unavailable: {type(e).__name__})",
                "note": "LLM error"}
    return {"summary": out, "note": f"~{len(out.split())} words"}


def translate(text, target_language="Spanish"):
    text = (text or "").strip()
    if not text:
        return {"translation": "", "note": "empty input"}
    prompt = (
        f"Translate the following text into {target_language}.\n"
        f"Output ONLY the {target_language} translation. "
        f"Do NOT repeat the source text. Do not add any preface, explanation, "
        f"or quotation marks.\n\n"
        f"--- SOURCE TEXT (do not output this) ---\n{text[:6000]}\n"
        f"--- END SOURCE ---\n\n"
        f"Now output only the {target_language} translation:"
    )
    try:
        out = (ask(prompt, temperature=0.0) or "").strip()
        # if the model echoed the input, flag it
        if out[:40].lower() == text[:40].lower():
            return {"translation": out,
                    "note": f"⚠ The model echoed the input instead of translating "
                            f"to {target_language}. Try again, or use a shorter text."}
    except Exception as e:
        return {"translation": f"(Translator unavailable: {type(e).__name__})",
                "note": "LLM error"}
    return {"translation": out, "note": f"translated to {target_language}"}


def fix_grammar(text):
    text = (text or "").strip()
    if not text:
        return {"fixed": "", "note": "empty input"}
    prompt = (
        "Fix the grammar, spelling, and punctuation of the text below. "
        "Do not change the meaning, do not add new content, do not shorten it. "
        "Output ONLY the corrected text.\n\n"
        f"--- text ---\n{text[:6000]}\n--- end ---"
    )
    try:
        out = (ask(prompt, temperature=0.1) or "").strip()
    except Exception as e:
        return {"fixed": f"(Grammar tool unavailable: {type(e).__name__})",
                "note": "LLM error"}
    return {"fixed": out, "note": "grammar-only correction"}


# ============================================================ 4. TOPIC SEARCH
def _safe_texts(texts):
    return [str(t) if t is not None else "" for t in texts]


def cluster_documents(texts, n_clusters=None, max_features=2000):
    texts = _safe_texts(texts)
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return None
    vec = TfidfVectorizer(
        max_features=max_features, min_df=1, stop_words="english",
        ngram_range=(1, 2), sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    if n_clusters is None:
        n_clusters = min(6, max(2, len(texts) // 5))
    n_clusters = max(2, min(n_clusters, len(texts) - 1))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    terms = np.array(vec.get_feature_names_out())
    top_terms = {}
    for i in range(n_clusters):
        center = km.cluster_centers_[i]
        top_idx = center.argsort()[-8:][::-1]
        top_terms[i] = [str(terms[j]) for j in top_idx]
    sizes = {i: int((labels == i).sum()) for i in range(n_clusters)}
    try:
        sil = float(silhouette_score(X, labels)) if n_clusters > 1 else 0.0
    except Exception:
        sil = 0.0
    return {"labels": [int(x) for x in labels],
            "top_terms": top_terms, "sizes": sizes,
            "n_clusters": n_clusters, "silhouette": round(sil, 3),
            "vectorizer": vec, "matrix": X, "kept_texts": texts}


def search_documents(query, texts, top_k=5, vectorizer=None, matrix=None):
    texts = _safe_texts(texts)
    texts = [t for t in texts if t.strip()]
    if not texts or not query.strip():
        return []
    if vectorizer is None or matrix is None:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                      sublinear_tf=True)
        matrix = vectorizer.fit_transform(texts)
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        if sims[i] <= 0:
            continue
        snippet = texts[i]
        if len(snippet) > 320:
            snippet = snippet[:320] + "…"
        results.append({"index": int(i), "score": round(float(sims[i]), 3),
                        "text": snippet})
    return results