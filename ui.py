"""
Shared UI helpers: palette, cards, hero, stat boxes, deployment lock,
watermark, Sisyphus quote, glossary.
"""
import os
import streamlit as st


THEME = """
<style>
    .stApp {
        background:
            radial-gradient(1400px 900px at 15% -10%, rgba(34,211,238,0.09), transparent 60%),
            radial-gradient(1200px 800px at 105% 5%, rgba(232,121,249,0.08), transparent 55%),
            #0a0d1a;
    }
    .block-container { padding-top: 7rem; padding-bottom: 5rem; max-width: 1400px; }
    h1, h2, h3, h4 { letter-spacing: -0.02em; color: #e8eef7; }

    /* ---------- hero ---------- */
    .hero {
        padding: 3.4rem 2.8rem 3.6rem 2.8rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(34,211,238,0.10), rgba(232,121,249,0.10));
        border: 1px solid rgba(34,211,238,0.22);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(34,211,238,0.06);
        min-height: 220px;
    }
    .hero::before {
        content: ""; position: absolute; top: -60%; right: -15%;
        width: 620px; height: 620px;
        background: radial-gradient(circle, rgba(34,211,238,0.22), transparent 70%);
        border-radius: 50%; pointer-events: none;
    }
    .hero::after {
        content: ""; position: absolute; bottom: -70%; left: -15%;
        width: 540px; height: 540px;
        background: radial-gradient(circle, rgba(232,121,249,0.16), transparent 70%);
        border-radius: 50%; pointer-events: none;
    }
    .hero h1 {
        font-size: 3rem;
        margin: 0;
        padding-bottom: 6px;
        background: linear-gradient(135deg, #67e8f9 0%, #22d3ee 40%, #e879f9 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        line-height: 1.15;
        position: relative;
        z-index: 1;
        display: inline-block;
    }
    .hero p {
        font-size: 1.12rem; color: #b6c2d4;
        margin: 1rem 0 1.2rem 0; max-width: 760px;
        position: relative; z-index: 1;
    }

    .prose { max-width: 780px; color: #cfd8e3; font-size: 1.05rem; line-height: 1.7; }
    .prose h3 { margin-top: 1.8rem; margin-bottom: 0.4rem; color: #67e8f9;
                font-size: 1.15rem; font-weight: 700; }
    .prose p { margin: 0 0 1rem 0; }
    .prose strong { color: #e8eef7; }

    .card {
        background: linear-gradient(180deg, #131a2e 0%, #10172a 100%);
        border: 1px solid #2a3350; border-radius: 16px;
        padding: 1.3rem 1.5rem; margin-bottom: 1rem;
        transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    }
    .card:hover {
        border-color: rgba(34,211,238,0.6);
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(34,211,238,0.14), 0 0 0 1px rgba(34,211,238,0.15);
    }
    .card h4 { margin: 0 0 0.5rem 0; color: #67e8f9; font-size: 1rem; font-weight: 600; }
    .card p { margin: 0; color: #b6c2d4; font-size: 0.93rem; line-height: 1.5; }
    .card .icon { font-size: 1.7rem; margin-bottom: 0.5rem; display: block; }

    .pill {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        background: rgba(34,211,238,0.14); color: #67e8f9;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.05em;
        margin-right: 6px; margin-bottom: 6px; text-transform: uppercase;
    }
    .pill.purple { background: rgba(232,121,249,0.16); color: #f0a6fc; }
    .pill.amber  { background: rgba(251,191,36,0.16);  color: #fcd34d; }
    .pill.red    { background: rgba(244,63,94,0.16);   color: #fb7185; }

    .stat {
        padding: 1.1rem 1.2rem; border-radius: 14px;
        background: linear-gradient(135deg, rgba(34,211,238,0.07), rgba(232,121,249,0.07));
        border: 1px solid #2a3350; text-align: center;
    }
    .stat .num { font-size: 1.9rem; font-weight: 800; color: #67e8f9; margin: 0; line-height: 1.1; }
    .stat .lbl {
        font-size: 0.72rem; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.09em;
        margin-top: 0.35rem; font-weight: 600;
    }

    .trace {
        font-family: Consolas, Menlo, monospace; font-size: 0.82rem;
        color: #94a3b8; padding: 3px 0;
        border-left: 2px solid rgba(34,211,238,0.35);
        padding-left: 10px; margin-bottom: 3px;
    }
    .trace::before { content: "› "; color: #22d3ee; }

    .deploy-locked { position: relative; overflow: hidden; border-radius: 16px; }
    .deploy-locked .deploy-inner {
        background: linear-gradient(180deg, #131a2e 0%, #10172a 100%);
        border: 1px solid #2a3350; border-radius: 16px;
        padding: 1.5rem 1.7rem;
    }
    .deploy-locked .deploy-inner h4 { color: #fcd34d; margin: 0 0 0.5rem 0; }
    .deploy-locked .deploy-inner p { color: #94a3b8; margin: 0; font-size: 0.9rem; }
    .deploy-locked .lock-overlay {
        position: absolute; inset: 0;
        display: flex; align-items: center; justify-content: center;
        flex-direction: column; gap: 0.4rem;
        background: rgba(10, 13, 26, 0.92);
        opacity: 0; transition: opacity 0.28s ease;
        border-radius: 16px; backdrop-filter: blur(4px);
        pointer-events: none; z-index: 10;
    }
    .deploy-locked:hover .lock-overlay { opacity: 1; }
    .deploy-locked .lock-overlay .lock-icon {
        font-size: 2.4rem; filter: drop-shadow(0 0 12px rgba(251,191,36,0.55));
    }
    .deploy-locked .lock-overlay .lock-text {
        color: #fcd34d; font-weight: 700; font-size: 1rem; letter-spacing: 0.03em;
    }
    .deploy-locked .lock-overlay .lock-sub {
        color: #94a3b8; font-size: 0.82rem; max-width: 380px;
        text-align: center; line-height: 1.4;
    }

    section[data-testid="stSidebar"] { background: #0c1224; border-right: 1px solid #1a2238; }
    section[data-testid="stSidebar"] .stButton button {
        text-align: left; justify-content: flex-start; width: 100%;
        background: transparent; border: 1px solid transparent;
        color: #cfd8e3; font-weight: 500;
        transition: all 0.14s ease; padding: 0.55rem 0.9rem;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: rgba(34,211,238,0.10);
        border-color: rgba(34,211,238,0.35); color: #67e8f9;
    }

    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
        color: #1a1206; font-weight: 700; border: none;
        box-shadow: 0 4px 14px rgba(251,191,36,0.25);
    }
    .stButton button[kind="primary"]:hover {
        box-shadow: 0 8px 26px rgba(251,191,36,0.45);
        transform: translateY(-1px);
    }
    .stButton button:not([kind="primary"]) {
        background: #1c2540; color: #cfd8e3; border: 1px solid #2a3350;
    }
    .stButton button:not([kind="primary"]):hover {
        border-color: rgba(34,211,238,0.55); color: #67e8f9;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #22d3ee, #e879f9);
    }

    hr { border-color: #1a2238; }

    /* ---------- watermark ---------- */
    .wm {
        position: fixed; bottom: 0; left: 0; right: 0;
        text-align: center; padding: 0.55rem 1rem;
        background: linear-gradient(180deg, transparent, rgba(10,13,26,0.94) 40%);
        color: #5c6a85;
        font-size: 0.76rem; letter-spacing: 0.02em;
        z-index: 9999; pointer-events: none;
        backdrop-filter: blur(6px);
    }
    .wm b { color: #94a3b8; font-weight: 600; }
    .wm .brand { color: #67e8f9; font-weight: 700; }
    .wm .sep { opacity: 0.4; margin: 0 0.5rem; }

    /* ---------- Sisyphus quote ---------- */
    .sisyphus {
        margin-top: 3rem;
        margin-bottom: 1rem;
        text-align: right;
        color: #5c6a85;
        font-style: italic;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        opacity: 0.75;
    }
    .sisyphus em {
        color: #7f8b99;
        font-style: italic;
    }
</style>
"""


def inject():
    st.markdown(THEME, unsafe_allow_html=True)


def watermark():
    st.markdown(
        '<div class="wm">'
        '<b>Engr. Anas Yousef Albakoush</b>'
        '<span class="sep">·</span>'
        '<b>Engr. Muad Salah Himeedah</b>'
        '<span class="sep">·</span>'
        '<span class="brand">ML in a Box</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def sisyphus():
    st.markdown(
        '<div class="sisyphus">— <em>One must imagine Sisyphus happy</em></div>',
        unsafe_allow_html=True)


def hero(title, subtitle, pills=None):
    pills_html = ""
    if pills:
        pills_html = " ".join(f'<span class="pill">{p}</span>' for p in pills)
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{pills_html}</div>',
        unsafe_allow_html=True)


def card(icon, title, body, pill=None):
    pill_html = f'<span class="pill">{pill}</span>' if pill else ""
    st.markdown(
        f'<div class="card"><span class="icon">{icon}</span>'
        f'<h4>{title} {pill_html}</h4><p>{body}</p></div>',
        unsafe_allow_html=True)


def stat(num, label):
    st.markdown(
        f'<div class="stat"><p class="num">{num}</p>'
        f'<div class="lbl">{label}</div></div>',
        unsafe_allow_html=True)


def trace_line(text):
    st.markdown(f'<div class="trace">{text}</div>', unsafe_allow_html=True)


def prose(html: str):
    st.markdown(f'<div class="prose">{html}</div>', unsafe_allow_html=True)


def deployment_locked(title="Deploy to production",
                      body="Package the model as a REST API, add authentication, "
                           "and serve it behind a load balancer.",
                      sub="This build is a research prototype. Deployment is "
                          "intentionally disabled so the model stays local."):
    st.markdown(
        f'<div class="deploy-locked">'
        f'<div class="deploy-inner">'
        f'<h4>🚀 {title}</h4>'
        f'<p>{body}</p>'
        f'</div>'
        f'<div class="lock-overlay">'
        f'<div class="lock-icon">🔒</div>'
        f'<div class="lock-text">Deployment is locked</div>'
        f'<div class="lock-sub">{sub}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True)


def profile_photo(path, fallback_icon="👤"):
    """Show a team photo if the file exists, otherwise show the emoji."""
    if path and os.path.isfile(path):
        st.image(path, use_container_width=True)
    else:
        st.markdown(
            f'<div style="text-align:center;font-size:4rem;padding:1rem;">'
            f'{fallback_icon}</div>',
            unsafe_allow_html=True)


GLOSSARY_CATEGORIES = {
    "Regression metrics": [
        ("R² (R-squared)",
         "How much of the target's variation the model explains.",
         "1.0 is perfect. 0 is useless. R² = 0.71 means the model explains 71% "
         "of the target's ups and downs.",
         "Above 0.7 is strong. Below 0.2 means the columns don't predict the target."),
        ("MAE (Mean Absolute Error)",
         "Average size of the prediction error, in the target's own units.",
         "Measure the gap for each prediction. Take absolute values. Average.",
         "Lower is better. Compare against the baseline's MAE."),
        ("RMSE (Root Mean Squared Error)",
         "Like MAE, but punishes big mistakes more.",
         "Square each error first, then average, then take the square root.",
         "Lower is better. RMSE >> MAE means a few predictions are wildly off."),
    ],
    "Classification metrics": [
        ("Accuracy",
         "Fraction of predictions that were exactly right.",
         "Count matches, divide by total. Misleading when one class dominates.",
         "Compare against the baseline. If the baseline is 90%, accuracy lies."),
        ("Precision",
         "When the model says 'class X', how often is it right?",
         "Of everything labeled class X, what fraction truly belongs.",
         "Higher is better. Use it when false alarms are expensive."),
        ("Recall",
         "Of all real cases of class X, how many did the model find?",
         "Of everything that truly is class X, what share the model caught.",
         "Higher is better. Use it when missing a case is expensive."),
        ("F1 score",
         "Balance between precision and recall. One number that considers both.",
         "Harmonic mean of the two.",
         "0.9+ is excellent. Below 0.3 means the model barely helps."),
        ("Confusion matrix",
         "A table showing what the model predicted vs what actually happened.",
         "Rows are reality, columns are predictions. The diagonal is correct.",
         "Look at where the off-diagonal mass is. Those are the mistakes."),
    ],
    "ML models": [
        ("Linear Regression",
         "Fits a straight-line relationship between inputs and a numeric target.",
         "Predicts a weighted sum of the inputs, plus an intercept.",
         "Fast, explainable. Struggles with curved or non-linear patterns."),
        ("Logistic Regression",
         "Linear classifier. Predicts the probability of each class.",
         "Fits a linear boundary, then applies a logistic curve.",
         "Simple, interpretable, often very competitive on text."),
        ("Ridge / Lasso",
         "Linear models with regularization.",
         "Ridge shrinks coefficients; Lasso drops weak ones to zero.",
         "Ridge when inputs are correlated. Lasso for automatic feature selection."),
        ("Decision Tree",
         "Learns a series of yes/no questions that split the data.",
         "Each internal node tests one feature; each leaf predicts.",
         "Very interpretable, but tends to overfit. Base for ensembles."),
        ("Random Forest",
         "Many decision trees voting together.",
         "Each tree trains on a random subset of rows and columns.",
         "Robust, rarely needs tuning. A very common strong default."),
        ("Gradient Boosting",
         "Trees that fix each other's mistakes in sequence.",
         "Each new tree is trained on the errors of the previous ensemble.",
         "Often the strongest performer on tabular data."),
        ("Extra Trees",
         "Like Random Forest but with even more randomness in the splits.",
         "Splits are chosen at random instead of optimised.",
         "Faster to train. Worth trying alongside a forest."),
        ("AdaBoost",
         "Boosting with shallow trees ('stumps') as the base learner.",
         "Later stumps focus on the rows earlier stumps got wrong.",
         "Strong when the signal is simple."),
        ("Bagging",
         "Many instances of one model, each on a random subset of rows.",
         "Predictions are averaged (regression) or voted (classification).",
         "A cheap variance reduction on any base model."),
        ("K-Nearest Neighbours (KNN)",
         "Predicts from the most similar past examples.",
         "Find the K closest rows, then average or vote among them.",
         "Great on small data with few columns. Sensitive to scaling."),
        ("SVM (Support Vector Machine)",
         "Finds the widest possible boundary between classes.",
         "Maximises the margin between the closest points of each class.",
         "Strong on small, clean data. Doesn't scale to huge datasets."),
        ("Naive Bayes",
         "A probabilistic classifier that assumes features are independent.",
         "Each feature votes for a class, weighted by how often it appears.",
         "Extremely fast. Surprisingly strong on text."),
        ("Neural Network (MLP)",
         "Layers of simple units that learn non-linear patterns.",
         "Each layer transforms the previous one. Weights learned by backprop.",
         "Flexible, but needs more data and tuning than tree ensembles."),
        ("Deep Learning",
         "Neural networks with many layers. Same idea as an MLP, scaled up.",
         "More layers, more data, more compute.",
         "Often overkill for tabular data. The right choice for images/text."),
    ],
    "Training concepts": [
        ("Baseline",
         "The dumbest possible model. Shows what 'doing nothing' looks like.",
         "Regression: predicts the average. Classification: most common class.",
         "If your best model barely beats the baseline, the data doesn't predict."),
        ("Overfitting",
         "The model memorised the training rows instead of learning patterns.",
         "Great on training rows, bad on new rows.",
         "The agent checks for it and tries to fix it automatically."),
        ("Underfitting",
         "The model is too simple to capture the pattern.",
         "Bad on training rows and bad on new rows.",
         "Fix by using a more flexible model, or better inputs."),
        ("Class imbalance",
         "One class has far more examples than others.",
         "If 95% of rows are class A, accuracy lies.",
         "The agent warns when the biggest class is 5x the smallest."),
        ("Cross-validation (CV)",
         "The score during training, averaged over several splits.",
         "Data is split into folds; each fold is used as the test set once.",
         "Compare with the test score. A gap means overfitting."),
        ("Train / test split",
         "The data is divided into two parts: learn on one, check on the other.",
         "The model never sees the test rows during training.",
         "80/20 is the common default."),
        ("Feature importance",
         "How much each input column changes the model's predictions.",
         "Shuffle one column at a time, watch the score drop.",
         "Bigger drop = more influential column."),
        ("Regularization",
         "A penalty for complexity that keeps the model from overfitting.",
         "Adds a term to the loss that punishes large weights.",
         "Higher regularization = smoother model."),
        ("Hyperparameter",
         "A setting you choose before training.",
         "Examples: number of trees, max depth, learning rate, dropout rate.",
         "The setup panel lets you set these per model."),
        ("Feature engineering",
         "Creating new inputs from existing ones to help the model learn.",
         "Examples: ratios, products, date parts, keyword flags.",
         "Usually the highest-leverage thing you can do."),
        ("Interaction feature",
         "A new column created by combining two existing columns.",
         "Examples: price / area, weight * height, date_a - date_b.",
         "Captures combined effects that neither column shows alone."),
        ("Imputation",
         "Filling in missing values so a model can use the column.",
         "Common strategies: median, mean, mode, or dropping the row.",
         "The agent's cleaning plan proposes these per column."),
        ("Standardization",
         "Rescaling features so none dominates just because of its units.",
         "Subtract the mean, divide by the standard deviation.",
         "Essential for models that use distances or gradient descent."),
        ("One-hot encoding",
         "Turning a categorical column into many binary columns.",
         "Each category becomes its own 0/1 column.",
         "Standard for text, countries, product types."),
    ],
    "Agentic AI": [
        ("Agentic AI",
         "An LLM in a loop with tools and a way to check its own work.",
         "The agent decides the plan, code executes, a human approves, a review node reacts.",
         "Four decisions: the target, interactions, models, and the repair."),
        ("Human-in-the-loop",
         "The agent pauses for a person to approve before doing something irreversible.",
         "Before any cleaning or training, the graph stops and shows the plan.",
         "The key difference between an automated script and an agent."),
        ("Guardrail",
         "A code check that constrains what the LLM is allowed to do.",
         "The LLM returns JSON. Code validates each field against the real data.",
         "Rejected fields get logged as warnings."),
        ("Tool calling",
         "When the LLM asks the code to run a specific function.",
         "The model outputs a structured request; code executes it.",
         "Prevents the LLM from executing arbitrary code."),
        ("State (in an agent)",
         "Everything the agent needs to remember between steps.",
         "Held in LangGraph's state dict; large objects live in a side cache.",
         "The checkpointer saves state to disk so the graph can pause."),
        ("Checkpoint",
         "A save point in the graph that the agent can pause at and resume from.",
         "LangGraph's `interrupt()` creates one at the human approval step.",
         "The graph truly pauses, not just waits for input()."),
        ("Self-check",
         "A node that inspects its own output and decides whether to try again.",
         "Compares train score to test score, checks class balance.",
         "The result is 'accepted' or a specific repair attempt."),
        ("Run history",
         "A log of every past run, kept on disk.",
         "The agent reads it before picking models.",
         "This is how the agent 'learns' — by reading its own notes."),
    ],
    "NLP": [
        ("Tokenization",
         "Splitting text into individual units (words, punctuation).",
         "The first step of every NLP pipeline.",
         "Tokens are what get counted, weighted, and fed to models."),
        ("Stop-words",
         "Very common words ('the', 'and', 'of') that carry little meaning.",
         "Often removed to reduce noise and shrink the vocabulary.",
         "Useful for frequency analysis. Sometimes harmful for sentiment."),
        ("TF-IDF",
         "How often a word appears in a document, weighted by how rare it is.",
         "Common words get low weight. Distinctive words get high weight.",
         "The result is a numeric matrix classifiers can learn from."),
        ("Bag of words",
         "A count of each word, order ignored.",
         "Every document becomes a vector of word counts.",
         "Fast and effective. Loses word order and grammar."),
        ("N-gram",
         "A sequence of N consecutive words or characters.",
         "Bigram = two words. Trigram = three words.",
         "Captures short phrases. TF-IDF here uses bigrams."),
        ("Cleaning / preprocessing",
         "Stripping HTML, URLs, punctuation, and case before analysis.",
         "Every step is a toggle in this project.",
         "Always look at the stats to see what actually changed."),
        ("Extractive summarization",
         "Picking the most informative sentences, verbatim.",
         "Offline, deterministic, fully explainable.",
         "Fast and safe. No LLM involved."),
        ("Abstractive summarization",
         "Rewriting the text as a shorter version, in the model's own words.",
         "Uses the LLM. Compact and readable, but slower.",
         "Best for long sources that need paraphrasing."),
        ("Topic clustering",
         "Grouping documents into themes with KMeans over TF-IDF vectors.",
         "Documents that share distinctive words end up in the same cluster.",
         "Silhouette above 0.2 means cleanly separated clusters."),
        ("Silhouette score",
         "A measure of how cleanly separated the clusters are.",
         "Ranges from -1 to 1. Above 0.2 is decent, above 0.5 is strong.",
         "Lower means clusters overlap — try a different topic count."),
        ("Cosine similarity",
         "A measure of how similar two documents are, based on their word vectors.",
         "1.0 = identical direction, 0 = unrelated.",
         "Used by the similarity search panel."),
        ("Sentiment analysis",
         "Classifying text as positive, negative, or neutral.",
         "The LLM scores a sample; the dashboard shows the distribution.",
         "Fast for a summary. For production, train a proper classifier."),
        ("Word cloud",
         "A picture where more frequent words appear larger.",
         "Built from the whole corpus with stop-words removed.",
         "A quick way to see the topics at a glance."),
    ],
}


GLOSSARY = {
    term: (short, long_, how)
    for entries in GLOSSARY_CATEGORIES.values()
    for term, short, long_, how in entries
}