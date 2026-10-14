"""Bilingual strings for ML in a Box.

Rounds covered:
  1  Sidebar, home hero, Tutorial
  2  About
  3  Glossary
  4  Ask / Chat
  5  Login screen
  6  Home body + Data Profiler
  7  Run Pipeline — setup
  8  Run Pipeline — plan + train + results + predict
  9  Tuning
 10  Computer Vision
 11  NLP Tools + NLP Dashboard
"""

from ui import GLOSSARY_CATEGORIES as _EN_GLOSSARY


LANGS = {
    "en": {
        # ================================================== SIDEBAR
        "language": "Language / اللغة",
        "home": "Home", "about": "About", "tutorial": "Tutorial",
        "profile": "Data Profiler", "run": "Run Pipeline",
        "cv": "Computer Vision", "tuning": "Tuning",
        "nlp_tools": "NLP Tools", "nlp_dash": "NLP Dashboard",
        "chat": "Ask", "glossary": "Glossary",
        "reset": "Reset", "sign_out": "Sign out", "signed_in": "Signed in",
        "agent_trace": "Agent trace", "loaded_card": "📄 Loaded",
        "sidebar_ask_title": "💬 Ask anything",
        "sidebar_ask_hint": "Uses the current session facts.",
        "sidebar_full_chat": "Full chat",
        "sidebar_clear": "Clear",

        # ================================================== HOME HERO
        "hero_title": "ML in a Box",
        "hero_sub": ("The power of AI, all in one screen — with a click, "
                     "you can be an AI engineer."),
        "hero_quote": "One must imagine Sisyphus happy",
        "tutorial_btn": "Tutorial",
        "profile_btn": "Profile a dataset",
        "tune_btn": "Tune a past run",

        # ================================================== HOME BODY
        "home_intro": "You bring a dataset. The agent handles the rest.",
        "home_decisions_title": "Where the intelligence lives",
        "home_decisions_sub": "Four decisions come from the language model:",
        "home_decisions": [
            ("🎯", "Target", "Which column to predict"),
            ("🔗", "Interactions", "Which columns to combine into new features"),
            ("📊", "Models", "Which 3–4 algorithms to train"),
            ("🔧", "Repair", "How to fix weak results"),
        ],
        "home_fallback": ("Every decision has a rule-based fallback. If the LLM goes "
                          "down, the pipeline still runs — it just explains less."),
        "home_code_title": "Where the code does the work",
        "home_code_body": ("The LLM never touches your rows. It returns JSON. Code "
                           "validates every field against the real data, rejects "
                           "what's wrong, then executes. **The LLM proposes. Code disposes.**"),
        "home_output_title": "What you get out",
        "home_output_1": ("📦", "Model file", "Ready to load and predict"),
        "home_output_2": ("📊", "Leaderboard", "Every model's score on test data"),
        "home_output_3": ("📄", "Reports", "Text, PDF, and a model card"),

        "home_gpu": "⚡ GPU ready · {name}",
        "home_cpu": "🧠 PyTorch installed (CPU). DL runs on CPU — correct, just slower.",
        "home_no_torch": "🧠 Using scikit-learn's neural network. `pip install torch` unlocks DL.",
        "home_signed_in": "Signed in as {name}",
        "home_modules": "Modules enabled for this account: <b>{mods}</b>. "
                        "Anything locked appears in the sidebar with a 🔒.",

        # ================================================== TUTORIAL
        "tut_hero_title": "Tutorial",
        "tut_sub": "Every stage of the pipeline, in the order it happens.",
        "tut_pill_1": "12 steps",
        "tut_pill_2": "plain words",
        "tut_pill_3": "no maths required",
        "tut_video_title": "🎬 Walkthrough video",
        "tut_video_body": ("A short walkthrough will be embedded here. "
                           "(placeholder — drop a YouTube URL into page_info.py "
                           "to enable)"),
        "tut_step_label": "Step",
        "tut_go_profile": "🔎  Go to the Data Profiler →",
        "tut_go_glossary": "📖  Open the Glossary",
        "tutorial_steps": [
            ("1", "Upload a file",
             "Data Profiler page, drop a CSV or Excel. The agent reads it, "
             "guesses each column's type, and describes what it likely represents."),
            ("2", "Let the agent pick a target (or override)",
             "On the Run Pipeline setup screen, the agent can look at the "
             "profile and recommend which column is worth predicting. You can "
             "always override."),
            ("3", "Choose inputs and metric preference",
             "Pick which columns the model may use. For classification, also "
             "tell the agent which mistake hurts more — false alarms "
             "(precision) or missed cases (recall). This changes the metric "
             "the agent optimises."),
            ("4", "Ask the agent to propose interactions",
             "Two columns together sometimes carry more signal than either "
             "alone. The agent proposes up to three combinations (ratio, "
             "product, difference) with a reason each. You approve or uncheck."),
            ("5", "Review the cleaning plan",
             "The LLM proposes a cleaning plan. The validator checks every "
             "field against the real data and drops anything wrong. What you "
             "see is the safe version."),
            ("6", "Approve or reject",
             "Nothing trains until you approve. This is the human-in-the-loop "
             "checkpoint — the piece that makes the system agentic rather than "
             "just automated."),
            ("7", "Watch it train",
             "The agent trains several models plus a dumb baseline, "
             "cross-validates each one, and scores them on a held-out test set."),
            ("8", "Review the self-check",
             "If the best model barely beats the baseline, or if train score "
             "is much higher than test score, the agent notices and tries a "
             "specific fix. It keeps the change only if cross-validation improves."),
            ("9", "Test the model",
             "On the results page, use the Predict panel to enter one row "
             "manually or upload a CSV. You get predictions with confidence."),
            ("10", "Tune it further",
             "Open the Tuning page to rerun a past configuration with tweaks. "
             "Every attempt is recorded, and the agent reads that history on "
             "future runs."),
            ("11", "NLP Tools",
             "For raw text documents (not classification), open the NLP Tools "
             "page. Clean, tokenize, summarize, cluster, and save."),
            ("12", "Advanced Neural Network",
             "On the setup screen under DL mode, configure layer sizes, "
             "activation, optimizer, learning rate, dropout, and see whether "
             "a GPU is available."),
        ],

        # ================================================== ABOUT
        "about_hero_title": "About",
        "about_hero_sub": "What this project is, what it isn't, and who built it.",
        "about_pill_1": "student project",
        "about_pill_2": "agentic pipeline",
        "about_pill_3": "two engineers",
        "about_prose": """
            <h3>What it is</h3>
            <p>ML in a Box is a student project built for the AI and Machine
            Learning track. It wraps a language model around a standard machine
            learning pipeline and asks a simple question: <strong>what would it
            look like if the decisions came from an LLM, while the execution
            stayed in code you can read?</strong></p>

            <p>The pipeline is honest about its limits. It proposes plans, but
            code validates them before anything runs. It picks models, but every
            pick is logged with a reason. It checks its own work, and when the
            result is weak it says so — and suggests specific changes rather than
            pretending the score is good.</p>

            <h3>What it isn't</h3>
            <p>Not a production system. Not a fine-tuned model. Not a replacement
            for a data scientist. It's a working pipeline with an agentic layer on
            top, built to show the ideas clearly enough to defend under
            questioning.</p>

            <h3>The stack</h3>
            <p>LangGraph holds the state machine and the human checkpoint.
            scikit-learn and PyTorch train the models. Streamlit is the interface.
            An OpenAI-compatible LLM handles the four decision nodes. Nothing
            exotic, and nothing hidden behind a framework you can't explain.</p>

            <h3>How it decides</h3>
            <p>Four decisions come from the language model: the target column,
            the interaction features, the model set, and the repair action after
            training. Every one of them has a rule-based fallback, so the pipeline
            still runs when the LLM is unavailable. The interesting part is the
            gap between proposal and execution — the LLM proposes, code
            disposes.</p>
        """,
        "about_us_title": "About us",
        "about_us_caption": "A short video introducing the project and the two "
                            "engineers behind it.",
        "about_video_title": "🎬 About us video",
        "about_video_body": ("A short introduction will be embedded here. "
                             "(placeholder — drop a YouTube URL into page_info.py "
                             "to enable)"),
        "about_team_title": "The team",
        "about_role_anas": "Gamer, unwilling translator and professional big back",
        "about_role_muad": "Professinal Anime watcher and phenomenal strategist.",
        "about_btn_tutorial": "📚  Tutorial",
        "about_btn_glossary": "📖  Glossary",
        "about_btn_start": "🔎  Start with a dataset",

        # ================================================== GLOSSARY
        "gloss_hero_title": "Glossary",
        "gloss_hero_sub": ("Every metric and term used in this project, "
                           "organised as a map. Click a category to zoom in, "
                           "then click a term. The card below updates "
                           "automatically."),
        "gloss_pill_categories": "{n} categories",
        "gloss_pill_click": "click to explore",
        "gloss_pill_plain": "plain words",
        "gloss_explore": "Explore",
        "gloss_old_streamlit": ("Streamlit is older than 1.35. Click-to-select "
                                "on the map is off — use the two dropdowns "
                                "below instead."),
        "gloss_tip": ("Tip: click a block to zoom. Use the breadcrumb at the "
                      "top of the map to zoom back out."),
        "gloss_full_explanation": "Full explanation",
        "gloss_category": "Category",
        "gloss_term": "Term",
        "gloss_how_to_read": "How to read it:",
        "gloss_further_reading": "Further reading:",
        "gloss_wip": "WIP — link coming soon",

        # ================================================== CHAT
        "chat_hero_title": "Ask",
        "chat_hero_sub": ("Type anything. The assistant answers using only the "
                          "real numbers from the current session."),
        "chat_pill_1": "no clicking required",
        "chat_pill_2": "uses real data",
        "chat_pill_3": "won't invent numbers",
        "chat_no_data_title": "📄 No dataset loaded",
        "chat_no_data_body": ("The assistant answers using the current session. "
                              "Right now there is nothing loaded. Drop a CSV or "
                              "Excel file below to give it context."),
        "chat_upload_label": "Upload a CSV or Excel file",
        "chat_try_one": "**Try one:**",
        "chat_you": "You",
        "chat_assistant": "Assistant",
        "chat_input_placeholder": "e.g. why did Random Forest lose?",
        "chat_input_label": "Ask anything",
        "chat_send": "Send",
        "chat_clear_history": "Clear history",
        "chat_thinking": "Thinking…",
        "chat_ctx_rows": "{r} rows × {c} cols",
        "chat_ctx_target": "target: `{t}`",
        "chat_ctx_results": "results available",
        "chat_suggested": [
            "Why did the best model win?",
            "What does R² mean?",
            "Which column mattered the most?",
            "Is the model overfitting?",
            "What columns should I add next?",
            "Explain the leaderboard in plain words.",
            "What is cross-validation?",
            "Did the self-check find anything?",
        ],

        # ================================================== LOGIN
        "login_brand": "🤖 ML in a Box",
        "login_sub": "Sign in to continue.",
        "login_username": "Username",
        "login_password": "Password",
        "login_signin": "Sign in",
        "login_wrong": "Wrong username or password.",
        "login_success": "Signed in.",
        "login_demo_title": "Demo accounts",
        "login_demo_admin": "admin / <code>admin123</code> &nbsp;·&nbsp; full access (ML, DL, NLP, CV)",
        "login_demo_doctor": "doctor / <code>doctor123</code> &nbsp;·&nbsp; ML, DL, NLP",
        "login_demo_jabo": "jabo / <code>jabo123</code> &nbsp;·&nbsp; ML, DL",
        "login_demo_nlp": "nlp_user / <code>nlp123</code> &nbsp;·&nbsp; NLP only",
        "login_demo_cv": "sambatica / <code>sambatica123</code> &nbsp;·&nbsp; CV",

        # ================================================== PROFILE
        "prof_hero_title": "Data Profiler",
        "prof_hero_sub": ("Upload a file to see what every column is, what it "
                          "likely means, and how the pipeline will treat it."),
        "prof_pill_1": "CSV", "prof_pill_2": "Excel",
        "prof_pill_3": "auto-detected types",
        "prof_upload_label": "Drop a CSV or Excel file",
        "prof_try_hint": "Try: netflix_titles.csv · laptopData.csv · houses.csv · books.csv",
        "prof_stat_rows": "rows",
        "prof_stat_cols": "columns",
        "prof_stat_missing": "missing cells",
        "prof_stat_dups": "duplicate rows",
        "prof_columns_title": "Columns",
        "prof_preview": "Preview raw data",
        "prof_describe_spinner": "Asking the language model to describe the columns…",
        "prof_feature_eng_title": "Feature engineering plan",
        "prof_no_conversions": "No conversions needed — every usable column is already numeric.",
        "prof_likely_meaning": "Likely meaning:",
        "prof_auto_conversion": "Auto-conversion:",
        "prof_missing_none": "no missing values",
        "prof_missing_some": "{p:.1f}% missing",
        "prof_dtype": "dtype: {t}",
        "prof_distinct": "{n} distinct",
        "prof_kind_num": "number",
        "prof_kind_numunit": "number with a unit",
        "prof_kind_cat": "category",
        "prof_kind_topcat": "high-cardinality category",
        "prof_kind_bool": "yes/no",
        "prof_kind_date": "date",
        "prof_kind_list": "comma-separated list",
        "prof_kind_text": "free text",
        "prof_kind_id": "ID / code (deleted)",
        "prof_kind_empty": "empty (deleted)",
        "prof_continue_btn": "⚡  Continue to Run Pipeline →",
        "prof_reupload_btn": "↺  Upload a different file",
        "prof_chat_title": "Ask about this dataset",
        "prof_chat_caption": ("Free-form questions about the columns, the shape "
                              "of the data, or what any of it means. The answer "
                              "uses only the profile above."),
        "prof_chat_suggested": [
            "What are the most important columns here?",
            "Which column looks like a target I could predict?",
            "Are there any columns I should remove?",
            "What does the range of values tell me about the data?",
            "Any signs this data will be hard to model?",
        ],
        "prof_chat_input": "Ask anything about this dataset",
        "prof_chat_placeholder": "e.g. which columns have missing values?",
        "prof_chat_send": "Send",
        "prof_chat_clear": "Clear chat",
        "prof_you": "You",
        "prof_assistant": "Assistant",

        # ================================================== SETUP
        "setup_hero_title": "What kind of data are you working with?",
        "setup_hero_sub": ("The pipeline changes based on data type. Pick the "
                           "modality and the agent will load the right tools."),
        "setup_hero_pill_1": "ML / DL",
        "setup_hero_pill_2": "NLP",
        "setup_hero_pill_3": "CV",
        "setup_ml_title": "ML / DL",
        "setup_ml_pill": "tabular",
        "setup_ml_body": ("Structured CSVs. Numbers, categories, dates. Choose "
                          "classical ML or a configurable neural network."),
        "setup_ml_example": "Example: predict house price from size and age.",
        "setup_ml_choose": "Choose ML / DL",
        "setup_ml_locked": "Your account does not have access to tabular ML / DL.",
        "setup_nlp_title": "NLP",
        "setup_nlp_pill": "text",
        "setup_nlp_body": ("CSVs with text columns. Classify reviews by "
                           "sentiment, topic, or spam."),
        "setup_nlp_example": "Example: reviews as positive or negative.",
        "setup_nlp_choose": "Choose NLP",
        "setup_nlp_locked": "Your account does not have access to NLP.",
        "setup_cv_title": "CV",
        "setup_cv_pill": "images",
        "setup_cv_body": ("Image classification with pretrained backbones or a "
                          "custom CNN. Detection and segmentation are "
                          "catalog-only."),
        "setup_cv_example": "Example: classify cats vs dogs from a ZIP.",
        "setup_cv_choose": "Choose CV",
        "setup_cv_locked": "Your account does not have access to Computer Vision.",
        "setup_locked_btn": "Locked",
        "setup_modality": "Modality:",
        "setup_change_modality": "↺ Change modality",
        "setup_nlp_step1": "### Step 1 — NLP setup",
        "setup_ml_step1": "### Step 1 — Learning mode",
        "setup_lm_pick": "Which family of models should this run use?",
        "setup_lm_ml": "📊 Classical ML — trees, ensembles, linear models",
        "setup_lm_dl": "🧠 Deep Learning — neural networks",
        "setup_lm_help": ("Classical ML: faster, more interpretable, best for "
                          "tabular data. Deep Learning: configurable neural "
                          "networks with GPU support."),
        "setup_lm_only_one": "Only one mode is enabled for your account.",
        "setup_ml_mode_title": "📊 Classical ML mode",
        "setup_dl_mode_title": "🧠 Deep Learning mode",
        "setup_ml_mode_body": "Classical models only. The neural network is hidden.",
        "setup_dl_mode_body": ("The neural network will be trained with your "
                               "configuration below."),
        "setup_nn_expander": "⚙️ Neural network configuration",
        "setup_step2": "### Step 2 — What should the agent predict?",
        "setup_agent_card_title": "🧠 The agent can pick for you",
        "setup_agent_card_body": ("Ask the agent to look at the profile and "
                                  "recommend a target column and a task type. "
                                  "You can override below."),
        "setup_agent_btn": "🤖 Ask the agent to recommend a target",
        "setup_agent_reading": "Reading the profile…",
        "setup_agent_fail": "The agent couldn't decide. Pick a column manually below.",
        "setup_agent_suggests": "**Agent suggests: `{t}`** as {task} — {reason}",
        "setup_preview": "Preview: {name} ({rows} rows × {cols} cols)",
        "setup_target_label": "Target column (what to predict)",
        "setup_reading": "Agent reads this as **{task}** (confidence: {conf}) — {reason}",
        "setup_features_label": "Input columns",
        "setup_feature_warn": "'{c}' cannot be used: {why}",
        "setup_nlp_no_text": "NLP mode selected but no text column included.",
        "setup_nlp_switch": ("You included a free-text column. Switch to NLP "
                             "modality for a stronger text pipeline."),
        "setup_step3": "### Step 3 — When this model is wrong, which mistake hurts more?",
        "setup_metric_label": "Metric preference",
        "setup_metric_balanced": "balanced — care equally about false alarms and missed cases",
        "setup_metric_precision": "precision — avoid false alarms (wrong positives)",
        "setup_metric_recall": "recall — avoid missed cases (wrong negatives)",
        "setup_metric_help": ("This changes which cross-validation metric the "
                              "agent optimises and reports. 'Precision' "
                              "penalises false positives. 'Recall' penalises "
                              "false negatives. 'Balanced' uses F1."),
        "setup_step4": "### Step 4 — Combine columns into new ones?",
        "setup_step4_caption": ("Sometimes two columns together carry more "
                                "signal than either alone. The agent can propose "
                                "a few combinations (ratio, product, difference) "
                                "from your numeric columns."),
        "setup_inter_btn": "🤖 Ask the agent to propose interactions",
        "setup_inter_need_two": "Need at least 2 numeric columns to propose interactions.",
        "setup_inter_thinking": "Thinking about interactions…",
        "setup_inter_none": "The agent didn't propose any useful interactions here.",
        "setup_inter_proposals": "**Agent proposals** — uncheck any you don't want:",
        "setup_step5": "### Step 5 — Who picks the models?",
        "setup_mode_label": "Model selection",
        "setup_mode_auto": "Auto — the agent picks and explains",
        "setup_mode_manual": "Manual — I choose",
        "setup_manual_label": "Models to train",
        "setup_pick_hint": "Pick a target and at least one input column.",
        "setup_continue_btn": "Continue → propose a cleaning plan",

        # ================================================== PLAN
        "plan_title": "### Cleaning plan",
        "plan_caption": ("Here is what the agent wants to do to the data before "
                         "training. Nothing has changed yet — you'll approve it "
                         "in a moment."),
        "plan_spinner": "Asking the LLM to draft a plan…",
        "plan_will_happen": "**What will happen**",
        "plan_drop_dupes": "• Delete duplicate rows: **{v}**",
        "plan_yes": "yes", "plan_no": "no",
        "plan_fill_empty": "• Fill empty cells:",
        "plan_fill_one": "    – `{c}`: use the **{m}**",
        "plan_fill_none": "• Fill empty cells: **nothing to do**",
        "plan_trim": "• Trim extreme values in: **{cols}**",
        "plan_nothing": "nothing to do",
        "plan_drop_empty": "• Drop mostly-empty columns: **{cols}**",
        "plan_validator": "**What the validator thinks**",
        "plan_no_warnings": "No warnings — this plan is safe to run.",
        "plan_show_raw": "Show what the LLM originally proposed (before validation)",
        "plan_info": ("The agent's raw plan gets checked against the real data "
                      "before it can touch anything. Anything invalid — a column "
                      "that doesn't exist, a wrong method for a column's type — "
                      "is dropped with a warning."),
        "plan_approve": "✓ Approve and start training",
        "plan_reject": "✗ Reject and choose new inputs",

        # ================================================== TRAIN
        "train_title": "### Training",
        "train_spinner": ("Fitting models, cross-validating, running the "
                          "agentic self-check…"),
        "train_starting": "Starting…",
        "train_fitting": "Fitting models…",
        "train_importance": "Computing feature importance…",
        "train_done": "Done.",
        "train_stopped": "Training stopped: {err}",
        "train_back": "← Back",
        "train_retry_spinner": "Retrying with the {n} most informative features: {feats}",

        # ================================================== RESULTS
        "results_title": "### Results",
        "results_best_model": "best model",
        "results_cv": "cross-val score",
        "results_test": "test score",
        "results_what_numbers_title": "What these numbers mean",
        "results_metric_pref": "Metric preference: <b>{m}</b>. Change this on the setup screen if the wrong kind of mistake is being punished.",
        "results_leaderboard": "### Leaderboard",
        "results_feat_eng": "🧪 Feature engineering that ran before training",
        "results_no_conversions": "No conversions needed.",
        "results_why_title": "### Why each model was chosen",
        "results_why_caption": ("These reasons come from the LLM, written "
                                "against this dataset — row count, feature "
                                "count, and task type."),
        "results_baseline": "Baseline <span class=\"pill amber\">control</span>",
        "results_baseline_body": ("A dumb model that always guesses the same "
                                  "thing. It shows what \"doing nothing\" looks "
                                  "like."),
        "results_tag_user": "your pick",
        "results_tag_agent": "agent's pick",
        "results_user_prefix": "You selected this model on purpose. ",
        "results_per_class": "### Per-class results",
        "results_confusion": "### Confusion matrix",
        "results_roc_title": "### ROC and Precision-Recall curves",
        "results_roc_caption": ("These show how well the classifier separates "
                                "classes across every possible confidence "
                                "threshold. AUC = 1.0 is perfect; 0.5 is a coin "
                                "flip."),
        "results_top_words": "### Top words per class",
        "results_top_bigrams": "### Top phrases per class (bigrams)",
        "results_class_prefix": "**Class: `{c}`**",
        "results_feat_importance": "### Feature importance",
        "results_adv_diag": "🧪 Advanced diagnostics",
        "results_resid_title": "#### Residual diagnostics",
        "results_resid_caption": ("Left: predicted vs actual — points should "
                                  "hug the dashed line. Right: residuals should "
                                  "look like random noise around zero. A "
                                  "pattern (curve, fan shape) means the model "
                                  "is missing structure."),
        "results_mean_resid": "mean residual",
        "results_std_resid": "std of residuals",
        "results_mae_resid": "mean abs error",
        "results_sample_preds": "### Sample predictions",
        "results_sample_caption": "Randomly sampled from the {n} held-out test rows.",
        "results_correct": "**✅ Correct — {n} shown**",
        "results_wrong": "**❌ Wrong — {n} shown**",
        "results_no_wrong": ("No wrong predictions sampled. The model got "
                             "everything we checked right."),
        "results_self_check": "### Agent self-check",
        "results_plain_words": "### In plain words",
        "results_pred_title": "### Try the model on new data",
        "results_pred_caption": ("Uses the trained **{m}** model to predict "
                                 "`{target}`. Nothing here changes the model or "
                                 "the training data."),
        "results_pred_method": "Input method",
        "results_pred_manual": "✏️ Enter one row manually",
        "results_pred_upload": "📤 Upload a CSV",
        "results_pipeline_notes": "Pipeline notes",
        "results_take_further": "### Take this further",
        "results_tune_btn": "⚙️  Tune this run further",
        "results_tune_caption": "Opens the Tuning page with this run's config preloaded.",
        "results_save_export": "### Save and export",
        "results_save_btn": "💾 Save model, report, and PDF",
        "results_csv_btn": "📥 Leaderboard (CSV)",
        "results_card_btn": "📇 Model card (PDF)",
        "results_card_help": "Save first to generate the model card.",
        "results_pdf_btn": "📄 Download PDF report",
        "results_saved_to": "Saved to {folder}",
        "results_save_fail": "Save failed: {err}",
        "results_try_another": "← Try another target",
        "results_nn_title": "🧠 Neural network details",
        "results_inter_title": "🔗 Interaction features used",
        "results_nlp_title": "📝 NLP pipeline used",
        "results_nlp_text_cols": "Text columns: <b>{cols}</b>.",
        "results_metric_help": ("**R² = {r2:.3f}** means the model explains "
                                "about {pct:.0f}% of the target's variation. "
                                "The baseline's R² is {base:.3f}."),
        "results_metric_help_clf": ("**F1 = {f1:.3f}** balances precision and "
                                    "recall. The baseline's F1 is {base:.3f}."),

        # ================================================== PREDICT PANEL
        "predict_title": "#### Predict on a new image",
        "predict_manual_title": "**Fill in the inputs** — interaction columns are computed automatically.",
        "predict_btn": "🔮  Predict",
        "predict_result": "Prediction",
        "predict_confidence": "confidence: {c:.1%}",
        "predict_upload_hint": ("Upload a CSV with the same base feature "
                                "columns"),
        "predict_upload_caption": ("Missing columns will be auto-computed if "
                                   "they're interaction features."),
        "predict_missing": "Missing required columns: {cols}",
        "predict_predicted_n": "Predicted {n} rows.",
        "predict_download": "📥  Download predictions (CSV)",
        "predict_missing_feature": "Missing required feature column: {e}",
        "predict_failed": "Prediction failed: {err}",

        # ================================================== TUNING
        "tune_hero_title": "Tuning",
        "tune_hero_sub": ("Take a past run, change one or two things, and see "
                          "whether the score improves. Every attempt is "
                          "recorded."),
        "tune_pill_1": "history-aware",
        "tune_pill_2": "compare", "tune_pill_3": "log",
        "tune_load_first": "Load a dataset first, then come back to Tuning.",
        "tune_go_profile": "🔎  Go to Data Profiler",
        "tune_no_runs": "No runs recorded yet for **{source}**. Train a model first, then come back to tune it.",
        "tune_go_run": "⚡  Go to Run Pipeline",
        "tune_history_title": "### History for this dataset",
        "tune_best_ever": "Best ever for this dataset",
        "tune_best_ever_body": ("<b>{model}</b> scored <b>{score:.3f}</b> as "
                                "<code>{task}</code> predicting "
                                "<code>{target}</code>."),
        "tune_last_run": "### Tune the last run",
        "tune_last_caption": ("Target: `{target}` · task: `{task}` · {n} "
                              "features · original score: {score:.3f}"),
        "tune_target": "Target",
        "tune_inputs": "Input columns",
        "tune_metric": "Metric preference",
        "tune_family": "Model family",
        "tune_selection": "Model selection",
        "tune_models": "Models",
        "tune_retrain": "🚀  Retrain with these settings",
        "tune_spinner": "Retraining…",
        "tune_result_title": "Result",
        "tune_result_body": ("New best: <b>{model}</b> scored <b>{score:.3f}</b>. "
                             "This <b>{verb}</b> the previous run by {delta:+.3f}."),
        "tune_open_results": "→ Open full results",
        "tune_fail": "Retrain failed: {err}",

        # ================================================== CV
        "cv_hero_title": "Computer Vision",
        "cv_hero_sub": ("Train image classifiers with pretrained backbones, or "
                        "build a small custom CNN from scratch. Detection and "
                        "segmentation are on the roadmap — currently locked."),
        "cv_pill_1": "classification",
        "cv_pill_2": "detection (WIP)",
        "cv_pill_3": "segmentation (WIP)",
        "cv_pill_4": "transfer learning",
        "cv_tab_class": "Classification",
        "cv_tab_detect": "🔒 Detection",
        "cv_tab_seg": "🔒 Segmentation",
        "cv_sec_1": "### 1. Dataset",
        "cv_sec_2": "### 2. Architecture",
        "cv_sec_3": "### 3. Training",
        "cv_sec_4": "### 4. Train",
        "cv_sec_5": "### 5. Results",
        "cv_zip_title": "What kind of ZIP?",
        "cv_zip_body": ("Upload a ZIP whose top level contains one folder per "
                        "class."),
        "cv_zip_min": ("Each class folder needs at least a handful of images. "
                       "20+ per class is a decent minimum. More is better."),
        "cv_zip_upload": "Upload dataset ZIP",
        "cv_zip_extract": "Extracting…",
        "cv_zip_fail": ("Could not find class folders in that ZIP. Make sure "
                        "the top level has one subfolder per class."),
        "cv_stat_classes": "classes",
        "cv_stat_train": "training images",
        "cv_stat_val": "validation images",
        "cv_classes_prefix": "Classes: ",
        "cv_diff_dataset": "↺ Upload a different dataset",
        "cv_backbone": "Backbone",
        "cv_custom_caption": ("Build a small CNN from scratch. The conv stack "
                              "ends with a global pool, then runs your dense "
                              "chain to the class count."),
        "cv_suggested_backbone": "Suggested backbone: **{name}** — {reason}",
        "cv_epochs": "Epochs",
        "cv_lr": "Learning rate",
        "cv_bs": "Batch size",
        "cv_freeze": "Freeze backbone (transfer learning)",
        "cv_freeze_help": "Recommended. Trains only the final layer.",
        "cv_start_btn": "🚀 Start training",
        "cv_training": "Training…",
        "cv_evaluating": "Evaluating…",
        "cv_progress_epoch": "Epoch {e}/{n}",
        "cv_progress_line": "epoch {e}: train_loss={tl:.3f} · val_loss={vl:.3f} · val_acc={va:.3f}",
        "cv_val_acc": "validation accuracy",
        "cv_architecture": "architecture",
        "cv_confusion": "#### Confusion matrix",
        "cv_per_class": "#### Per-class results",
        "cv_sample_preds": "#### Sample predictions",
        "cv_predict_title": "#### Predict on a new image",
        "cv_predict_using": "Using: {m}",
        "cv_predict_loaded": " (loaded from a saved slot)",
        "cv_predict_label": "Prediction",
        "cv_predict_conf": "confidence {c:.1f}%",
        "cv_predict_fail": "Prediction failed: {err}",
        "cv_saved_title": "### 💾 Saved models",
        "cv_saved_caption": ("Keep up to {n} trained models. They are stored on "
                             "disk, so they survive restarts — load one any "
                             "time to predict without retraining."),
        "cv_saved_already": "The model you just trained is already saved.",
        "cv_saved_full": ("All {n} slots are full. Delete one below to save "
                          "this model."),
        "cv_saved_train_first": "Train a model above, then save it here.",
        "cv_saved_name_placeholder": "my model",
        "cv_saved_name_label": "Name (optional)",
        "cv_saved_save_btn": "💾 Save ({cur}/{max})",
        "cv_saved_confirm_flash": "Model saved.",
        "cv_saved_load": "Load",
        "cv_saved_loading": "Loading model…",
        "cv_saved_load_flash": "Loaded “{m}”. Use it in the predict box above.",
        "cv_saved_delete": "Delete",
        "cv_saved_confirm": "Confirm",
        "cv_saved_cancel": "Cancel",
        "cv_saved_confirm_warn": "Delete this model permanently? This cannot be undone.",
        "cv_saved_in_use": "in use",
        "cv_saved_saved_at": "saved {when}",
        "cv_saved_size": "{mb} MB",
        "cv_saved_val_acc": "{p:.1f}% val acc",
        "cv_detect_wip_title": "🔒 Detection — Work in Progress",
        "cv_detect_wip_body": ("Detection finds <em>and locates</em> objects. "
                               "Every training image needs a bounding box per "
                               "object, in COCO JSON format (image path, class "
                               "label, x/y/width/height)."),
        "cv_detect_wip_body2": ("This tab will let you upload a COCO-annotated "
                                "ZIP, fine-tune a detector, and run inference — "
                                "but the training loop is not built yet."),
        "cv_detect_wip_foot": ("The architecture list below is a read-only "
                               "roadmap. No upload, no training, no inference "
                               "is available here."),
        "cv_detect_arch_title": "### Architectures planned (5)",
        "cv_detect_locked": "This feature is locked until the training loop is implemented.",
        "cv_seg_wip_title": "🔒 Segmentation — Work in Progress",
        "cv_seg_wip_body": ("Segmentation labels <em>every pixel</em>. Each "
                            "training image needs a companion mask PNG where "
                            "each pixel value is the class index."),
        "cv_seg_wip_body2": ("This tab will let you upload image+mask pairs, "
                             "fine-tune a segmenter, and run inference — but "
                             "the training loop is not built yet."),
        "cv_seg_wip_foot": ("The architecture list below is a read-only roadmap. "
                            "No upload, no training, no inference is available "
                            "here."),
        "cv_seg_arch_title": "### Architectures planned (5)",
        "cv_seg_locked": "This feature is locked until the training loop is implemented.",

        # ================================================== NLP TOOLS
        "nlp_tools_hero_title": "NLP Tools",
        "nlp_tools_hero_sub": ("Clean a document, tokenize it, summarize it, "
                               "cluster a corpus into themes, and save the "
                               "useful parts."),
        "nlp_tools_pill_1": "clean", "nlp_tools_pill_2": "tokenize",
        "nlp_tools_pill_3": "summarize", "nlp_tools_pill_4": "cluster",
        "nlp_tools_pill_5": "export",
        "nlp_1_title": "### 1. Give it text",
        "nlp_paste_tab": "Paste text",
        "nlp_upload_tab": "Upload file",
        "nlp_paste_placeholder": ("Paste an article, review, email thread, or "
                                  "any long text…"),
        "nlp_upload_label": "Upload .txt, .md, .html, .htm, .pdf, .docx, or CSV",
        "nlp_no_input": "Paste text above or upload a file to begin.",
        "nlp_input_caption": "Input: {chars:,} characters · {docs} document(s).",
        "nlp_2_title": "### 2. Clean the text",
        "nlp_2_caption": "Every option is a toggle. Nothing runs until you click the button.",
        "nlp_clean_options": "Cleaning options",
        "nlp_opt_lower": "Lowercase everything",
        "nlp_opt_html": "Strip HTML tags",
        "nlp_opt_urls": "Strip URLs",
        "nlp_opt_emails": "Strip emails",
        "nlp_opt_punct": "Strip punctuation (. , ! ? …)",
        "nlp_opt_digits": "Strip numbers",
        "nlp_opt_ws": "Collapse extra whitespace",
        "nlp_opt_stops": "Remove stop-words (the, and, of)",
        "nlp_opt_changed": ("Options changed since the last cleaning. Click "
                            "**Clean text** to apply them."),
        "nlp_clean_btn": "🧹 Clean text",
        "nlp_cleaned_output": "Cleaned output",
        "nlp_cleaned_title": "View cleaned text",
        "nlp_reduction": "size reduction",
        "nlp_3_title": "### 3. Tokenize",
        "nlp_3_caption": ("Split the cleaned text into individual tokens. This "
                          "is what every NLP model actually sees."),
        "nlp_tok_keep_stops": "Keep stop-words",
        "nlp_tokenize_btn": "🔤 Tokenize",
        "nlp_tokens": "Tokens",
        "nlp_all_tokens": "All tokens",
        "nlp_all_tokens_caption": "Showing first 500 of {n}.",
        "nlp_4_title": "### 4. Key phrases",
        "nlp_phrases_btn": "🔑 Extract key phrases",
        "nlp_5_title": "### 5. Summarize",
        "nlp_extractive_title": "**Extractive** — most important sentences, verbatim",
        "nlp_sentences": "Sentences",
        "nlp_extract_btn": "Extract summary",
        "nlp_extractive_output": "Extractive summary",
        "nlp_abstractive_title": "**Abstractive** — rewritten in its own words",
        "nlp_approx_words": "Approx. words",
        "nlp_abstract_btn": "Generate summary",
        "nlp_abstractive_output": "Abstractive summary",
        "nlp_summarizing": "Summarizing…",
        "nlp_6_title": "### 6. Topic clustering",
        "nlp_6_caption": ("Group documents by theme. Each theme is named by its "
                          "most distinctive words."),
        "nlp_cluster_split": "Split this text into documents",
        "nlp_cluster_split_label": "Split by",
        "nlp_cluster_split_blank": "blank lines (paragraphs)",
        "nlp_cluster_split_sent": "sentence boundaries",
        "nlp_cluster_split_words": "every N words",
        "nlp_cluster_chunk": "Words per chunk",
        "nlp_cluster_split_ok": "Split into {n} documents.",
        "nlp_cluster_split_no": ("Split into only {n} — need at least 3 to "
                                 "cluster."),
        "nlp_cluster_need_3": ("Need at least 3 documents. Either load a CSV "
                               "with 3+ rows, or use the splitter above."),
        "nlp_topics_count": "Number of topics",
        "nlp_cluster_btn": "🔬 Find topics",
        "nlp_clustering": "Clustering…",
        "nlp_topics_prefix": "**{n} topics** · silhouette **{sil}** (above 0.2 means the clusters are cleanly separated)",
        "nlp_topic": "Topic {i}",
        "nlp_topic_docs": "{n} docs",
        "nlp_7_title": "### 7. Similarity search",
        "nlp_search_placeholder": "e.g. shipping delays, refund request",
        "nlp_search_results": "Results",
        "nlp_search_btn": "🔍 Search",
        "nlp_search_score": "score {s}",
        "nlp_search_doc": "doc #{i}",
        "nlp_search_none": "No matches.",
        "nlp_8_title": "### 8. Translate",
        "nlp_lang_label": "Target language",
        "nlp_translate_btn": "Translate",
        "nlp_translating": "Translating to {lang}…",
        "nlp_translation_output": "Translation",
        "nlp_9_title": "### 9. Fix grammar and spelling",
        "nlp_grammar_btn": "Fix grammar",
        "nlp_correcting": "Correcting…",
        "nlp_original": "**Original**",
        "nlp_corrected": "**Corrected**",
        "nlp_10_title": "### 10. Save to archive",
        "nlp_10_caption": ("Download the cleaned text and summaries as Excel, "
                           "SQL, or a text bundle."),
        "nlp_export_fmt": "Format",
        "nlp_export_excel": "Excel (.xlsx)",
        "nlp_export_sql": "SQL INSERT statements",
        "nlp_export_txt": "Plain text bundle",
        "nlp_download_xlsx": "Download .xlsx",
        "nlp_download_csv": "Download .csv",
        "nlp_download_sql": "Download .sql",
        "nlp_download_txt": "Download .txt bundle",
        "nlp_openpyxl_missing": "openpyxl not installed — falling back to CSV.",

        # ================================================== NLP DASHBOARD
        "nlp_dash_hero_title": "NLP Dashboard",
        "nlp_dash_hero_sub": ("This page is for **corpora** — a whole set of "
                              "documents looked at together. If you want to "
                              "process a single document, use NLP Tools "
                              "instead. Here you get: a word cloud of what the "
                              "corpus is about, sentiment distribution across "
                              "documents, text length distribution, the most "
                              "common two-word phrases, and how the vocabulary "
                              "grows as you read more."),
        "nlp_dash_pill_1": "corpus view",
        "nlp_dash_pill_2": "CSV / PDF / DOCX / TXT",
        "nlp_dash_pill_3": "many documents",
        "nlp_dash_upload": ("Upload a corpus — CSV with a text column, or a "
                            "single document (.pdf, .docx, .txt, .md, .html)"),
        "nlp_dash_use_loaded": "You have documents loaded from NLP Tools.",
        "nlp_dash_use_loaded_btn": "Use loaded documents",
        "nlp_dash_csv_pick": "Which column holds the text?",
        "nlp_dash_split_label": ("This is one document. How should I split it "
                                 "into multiple for the corpus view?"),
        "nlp_dash_split_1": "paragraphs (blank lines)",
        "nlp_dash_split_2": "sentences",
        "nlp_dash_split_3": "fixed-size chunks",
        "nlp_dash_chunk": "Words per chunk",
        "nlp_dash_split_ok": "Split into {n} pseudo-documents.",
        "nlp_dash_stat_docs": "documents",
        "nlp_dash_stat_words": "total words",
        "nlp_dash_stat_avg": "avg words/doc",
        "nlp_dash_stat_vocab": "unique words",
        "nlp_dash_wordcloud": "### Word cloud",
        "nlp_dash_wc_missing": "Install `wordcloud` to see the cloud: `pip install wordcloud`.",
        "nlp_dash_sentiment": "### Sentiment distribution",
        "nlp_dash_sentiment_caption": ("The LLM scores a sample of up to 15 "
                                       "documents as negative, neutral, or "
                                       "positive."),
        "nlp_dash_scoring": "Scoring sentiment…",
        "nlp_dash_llm_unavail": "LLM unavailable; sentiment skipped.",
        "nlp_dash_sent_method": "Method: {m} · sample of {n}",
        "nlp_dash_length": "### Text length distribution",
        "nlp_dash_length_x": "words per document",
        "nlp_dash_length_y": "documents",
        "nlp_dash_bigrams": "### Top phrases (bigrams)",
        "nlp_dash_vocab_growth": "### Vocabulary growth",
        "nlp_dash_vocab_x": "documents read",
        "nlp_dash_vocab_y": "unique words seen",
        "nlp_dash_vocab_caption": ("A flat tail means the corpus vocabulary has "
                                   "stabilised — you've seen most of what it "
                                   "has to say. A rising curve means there's "
                                   "still more variety to discover."),
    },

    "ar": {
        # ================================================== SIDEBAR
        "language": "اللغة / Language",
        "home": "الرئيسية", "about": "حول", "tutorial": "الشرح",
        "profile": "محلّل البيانات", "run": "تشغيل المسار",
        "cv": "الرؤية الحاسوبية", "tuning": "الضبط",
        "nlp_tools": "أدوات اللغة", "nlp_dash": "لوحة اللغة",
        "chat": "اسأل", "glossary": "المعجم",
        "reset": "إعادة", "sign_out": "خروج", "signed_in": "مسجّل الدخول",
        "agent_trace": "أثر الوكيل", "loaded_card": "📄 محمّل",
        "sidebar_ask_title": "💬 اسأل أي شيء",
        "sidebar_ask_hint": "يستخدم حقائق الجلسة الحالية.",
        "sidebar_full_chat": "المحادثة كاملة",
        "sidebar_clear": "مسح",

        # ================================================== HOME HERO
        "hero_title": "التعلّم الآلي في صندوق",
        "hero_sub": ("قوة الذكاء الاصطناعي في شاشة واحدة — "
                     "بضغطة زر تصبح مهندس ذكاء اصطناعي."),
        "hero_quote": "يجب أن نتخيّل سيزيف سعيداً",
        "tutorial_btn": "الشرح",
        "profile_btn": "حلّل بياناتك",
        "tune_btn": "اضبط تجربة سابقة",

        # ================================================== HOME BODY
        "home_intro": "أنت تُحضر مجموعة بيانات، والوكيل يتولّى الباقي.",
        "home_decisions_title": "أين يعيش الذكاء",
        "home_decisions_sub": "أربعة قرارات من النموذج اللغوي:",
        "home_decisions": [
            ("🎯", "الهدف", "أيّ عمود نتنبّأ به"),
            ("🔗", "التفاعلات", "أيّ أعمدة ندمجها لميزات جديدة"),
            ("📊", "النماذج", "٣-٤ خوارزميات للتدريب"),
            ("🔧", "الإصلاح", "كيف نصلح النتائج الضعيفة"),
        ],
        "home_fallback": ("كل قرار له بديل قائم على القواعد. لو تعذّر الوصول إلى "
                          "النموذج اللغوي، يستمرّ المسار بالعمل — فقط يشرح أقل."),
        "home_code_title": "أين يعمل الكود",
        "home_code_body": ("النموذج اللغوي لا يلمس صفوفك أبداً. يُرجع JSON. الكود "
                           "يتحقق من كل حقل مقابل البيانات الحقيقية، يرفض الخاطئ، "
                           "ثم ينفّذ. **النموذج يقترح، الكود يُنفّذ.**"),
        "home_output_title": "ما تحصل عليه",
        "home_output_1": ("📦", "ملف النموذج", "جاهز للتحميل والتنبؤ"),
        "home_output_2": ("📊", "لوحة الصدارة", "نتيجة كل نموذج على بيانات الاختبار"),
        "home_output_3": ("📄", "التقارير", "نص وPDF وبطاقة نموذج"),

        "home_gpu": "⚡ GPU جاهز · {name}",
        "home_cpu": "🧠 PyTorch مثبّت (CPU). التعلّم العميق يعمل على CPU — صحيح، لكن أبطأ.",
        "home_no_torch": "🧠 نستخدم شبكة scikit-learn العصبية. `pip install torch` يفتح التعلّم العميق.",
        "home_signed_in": "مسجّل الدخول باسم {name}",
        "home_modules": "الوحدات المفعّلة لهذا الحساب: <b>{mods}</b>. "
                        "كل ما هو مقفل يظهر في الشريط الجانبي مع 🔒.",

        # ================================================== TUTORIAL
        "tut_hero_title": "الشرح",
        "tut_sub": "كل مرحلة من المسار، بالترتيب الذي تحدث فيه.",
        "tut_pill_1": "١٢ خطوة",
        "tut_pill_2": "بكلمات بسيطة",
        "tut_pill_3": "بدون رياضيات",
        "tut_video_title": "🎬 فيديو الشرح",
        "tut_video_body": ("سيُضمَّن شرح قصير هنا. "
                           "(عنصر نائب — أضف رابط يوتيوب في page_info.py للتفعيل)"),
        "tut_step_label": "الخطوة",
        "tut_go_profile": "🔎  اذهب إلى محلّل البيانات →",
        "tut_go_glossary": "📖  افتح المعجم",
        "tutorial_steps": [
            ("١", "ارفع ملفاً",
             "في صفحة محلّل البيانات، أفلت ملف CSV أو Excel. يقرأه الوكيل، "
             "ويخمّن نوع كل عمود، ويصف ما يمثّله على الأرجح."),
            ("٢", "دع الوكيل يختار الهدف (أو تجاوزه)",
             "في شاشة إعداد تشغيل المسار، ينظر الوكيل إلى الملف ويوصي "
             "بالعمود الذي يستحق التنبؤ. يمكنك دائماً تجاوز اقتراحه."),
            ("٣", "اختر المُدخلات وتفضيل المقياس",
             "اختر الأعمدة التي يجوز للنموذج استخدامها. في التصنيف، أخبر "
             "الوكيل أيضاً أيّ خطأ أكثر إيلاماً — الإنذارات الكاذبة (الضبط) "
             "أم الحالات المفقودة (الاسترجاع). يغيّر هذا المقياس الذي "
             "يحسّنه الوكيل."),
            ("٤", "اطلب من الوكيل اقتراح تفاعلات",
             "أحياناً يحمل عمودان معاً إشارة أقوى من أيّ منهما منفرداً. "
             "يقترح الوكيل حتى ثلاث تركيبات (نسبة، حاصل ضرب، فرق) مع سبب "
             "لكل واحدة. توافق أو تلغي التحديد."),
            ("٥", "راجع خطة التنظيف",
             "يقترح النموذج اللغوي خطة تنظيف. يتحقق المدقّق من كل حقل مقابل "
             "البيانات الفعلية ويُسقط أي شيء خاطئ. ما تراه هو النسخة الآمنة."),
            ("٦", "وافق أو ارفض",
             "لا يتدرّب شيء حتى توافق. هذه هي نقطة التفتيش البشرية — "
             "الجزء الذي يجعل النظام وكيلياً وليس مجرّد أتمتة."),
            ("٧", "شاهده يتدرّب",
             "يدرّب الوكيل عدّة نماذج بالإضافة إلى خطّ أساس بسيط، ويجري "
             "التحقق المتقاطع لكل منها، ويسجّل نتيجتها على مجموعة اختبار "
             "محجوزة."),
            ("٨", "راجع الفحص الذاتي",
             "إذا كان أفضل نموذج بالكاد يتفوّق على الخطّ الأساس، أو كانت "
             "نتيجة التدريب أعلى بكثير من نتيجة الاختبار، يلاحظ الوكيل ذلك "
             "ويجرّب إصلاحاً محدّداً. لا يبقي التغيير إلا إذا تحسّن التحقق "
             "المتقاطع."),
            ("٩", "اختبر النموذج",
             "في صفحة النتائج، استخدم لوحة التنبؤ لإدخال صفّ يدوياً أو رفع "
             "ملف CSV. تحصل على تنبؤات مع درجة الثقة."),
            ("١٠", "اضبطه أكثر",
             "افتح صفحة الضبط لإعادة تشغيل إعداد سابق بتعديلات. تُسجّل كل "
             "محاولة، ويقرأ الوكيل هذا السجل في المرات التالية."),
            ("١١", "أدوات اللغة",
             "للنصوص الخام (ليس التصنيف)، افتح صفحة أدوات اللغة. تنظيف، "
             "تقطيع، تلخيص، تجميع، وحفظ."),
            ("١٢", "الشبكة العصبية المتقدّمة",
             "في شاشة الإعداد تحت وضع التعلّم العميق، اضبط أحجام الطبقات، "
             "دالة التنشيط، المُحسّن، معدّل التعلّم، والإسقاط، وتحقق من "
             "توفّر GPU."),
        ],

        # ================================================== ABOUT
        "about_hero_title": "حول",
        "about_hero_sub": "ما هو هذا المشروع، وما ليس كذلك، ومن بناه.",
        "about_pill_1": "مشروع طلابي",
        "about_pill_2": "مسار وكيلي",
        "about_pill_3": "مهندسان",
        "about_prose": """
            <h3>ما هو</h3>
            <p>التعلّم الآلي في صندوق مشروع طلابي بُني لمسار الذكاء الاصطناعي
            وتعلّم الآلة. يلفّ نموذجاً لغوياً حول مسار تعلّم آلي قياسي ويسأل
            سؤالاً بسيطاً: <strong>كيف سيبدو الأمر لو جاءت القرارات من نموذج
            لغوي، بينما بقي التنفيذ في كود يمكنك قراءته؟</strong></p>

            <p>المسار صادق حول حدوده. يقترح خططاً، لكن الكود يتحقق منها قبل أن
            يُشغّل أي شيء. يختار نماذج، لكن كل اختيار يُسجَّل مع سبب. يفحص
            عمله بنفسه، وعندما تكون النتيجة ضعيفة يقول ذلك — ويقترح تغييرات
            محدّدة بدلاً من التظاهر بأن النتيجة جيدة.</p>

            <h3>ما ليس كذلك</h3>
            <p>ليس نظام إنتاج. ليس نموذجاً مضبوطاً. ليس بديلاً عن عالِم بيانات.
            هو مسار عامل مع طبقة وكيلية فوقه، بُني ليعرض الأفكار بوضوح كافٍ
            للدفاع عنها أمام الأسئلة.</p>

            <h3>الحزمة التقنية</h3>
            <p>LangGraph يحمل آلة الحالة ونقطة التفتيش البشرية.
            scikit-learn و PyTorch يدرّبان النماذج. Streamlit هي الواجهة.
            نموذج لغوي متوافق مع OpenAI يتولّى عُقد القرار الأربع. لا شيء
            غريب، ولا شيء مخفيّ خلف إطار لا تستطيع تفسيره.</p>

            <h3>كيف يقرّر</h3>
            <p>أربعة قرارات تأتي من النموذج اللغوي: عمود الهدف، وميزات
            التفاعل، ومجموعة النماذج، وإجراء الإصلاح بعد التدريب. كل واحد
            منها له بديل قائم على قواعد، لذا يستمرّ المسار بالعمل حتى لو
            تعذّر الوصول إلى النموذج اللغوي. الجزء المثير هو الفجوة بين
            الاقتراح والتنفيذ — النموذج يقترح، والكود يُنفّذ.</p>
        """,
        "about_us_title": "من نحن",
        "about_us_caption": "فيديو قصير يعرّف بالمشروع والمهندسين اللذين بنياه.",
        "about_video_title": "🎬 فيديو من نحن",
        "about_video_body": ("سيُضمَّن تعريف قصير هنا. "
                             "(عنصر نائب — أضف رابط يوتيوب في page_info.py للتفعيل)"),
        "about_team_title": "الفريق",
        "about_role_anas": "لاعب، مترجم غير راغب، وأكل كثير محترف",
        "about_role_muad": "مشاهد أنمي محترف، استراتيجي بارع.",
        "about_btn_tutorial": "📚  الشرح",
        "about_btn_glossary": "📖  المعجم",
        "about_btn_start": "🔎  ابدأ بمجموعة بيانات",

        # ================================================== GLOSSARY
        "gloss_hero_title": "المعجم",
        "gloss_hero_sub": ("كل مقياس ومصطلح مستخدَم في هذا المشروع، منظّماً "
                           "كخريطة. انقر فئة للتكبير، ثم انقر مصطلحاً. "
                           "البطاقة أدناه تتحدّث تلقائياً."),
        "gloss_pill_categories": "{n} فئات",
        "gloss_pill_click": "انقر للاستكشاف",
        "gloss_pill_plain": "بكلمات بسيطة",
        "gloss_explore": "استكشف",
        "gloss_old_streamlit": ("إصدار Streamlit أقدم من 1.35. النقر للاختيار "
                                "على الخريطة معطّل — استخدم القائمتين أدناه."),
        "gloss_tip": ("نصيحة: انقر كتلة للتكبير. استخدم شريط التنقّل في أعلى "
                      "الخريطة للتصغير."),
        "gloss_full_explanation": "الشرح الكامل",
        "gloss_category": "الفئة",
        "gloss_term": "المصطلح",
        "gloss_how_to_read": "كيف تقرأه:",
        "gloss_further_reading": "قراءة إضافية:",
        "gloss_wip": "قيد العمل — الرابط قادم قريباً",

        # ================================================== CHAT
        "chat_hero_title": "اسأل",
        "chat_hero_sub": ("اكتب أيّ شيء. يجيب المساعد بالاعتماد فقط على "
                          "الأرقام الحقيقية من الجلسة الحالية."),
        "chat_pill_1": "بدون نقر",
        "chat_pill_2": "يستخدم بيانات حقيقية",
        "chat_pill_3": "لن يختلق أرقاماً",
        "chat_no_data_title": "📄 لا توجد مجموعة بيانات محمّلة",
        "chat_no_data_body": ("يجيب المساعد باستخدام الجلسة الحالية. حالياً "
                              "لا يوجد شيء محمّل. أفلت ملف CSV أو Excel أدناه "
                              "لإعطائه سياقاً."),
        "chat_upload_label": "ارفع ملف CSV أو Excel",
        "chat_try_one": "**جرّب واحداً:**",
        "chat_you": "أنت",
        "chat_assistant": "المساعد",
        "chat_input_placeholder": "مثلاً: لماذا خسر Random Forest؟",
        "chat_input_label": "اسأل أي شيء",
        "chat_send": "إرسال",
        "chat_clear_history": "مسح المحادثة",
        "chat_thinking": "أفكّر…",
        "chat_ctx_rows": "{r} صفاً × {c} عموداً",
        "chat_ctx_target": "الهدف: `{t}`",
        "chat_ctx_results": "النتائج متاحة",
        "chat_suggested": [
            "لماذا فاز أفضل نموذج؟",
            "ماذا يعني R²؟",
            "أيّ عمود كان الأكثر تأثيراً؟",
            "هل النموذج يعاني من الإفراط في التخصيص؟",
            "ما الأعمدة التي أضيفها بعد ذلك؟",
            "اشرح لوحة الصدارة بكلمات بسيطة.",
            "ما هو التحقق المتقاطع؟",
            "هل وجد الفحص الذاتي شيئاً؟",
        ],

        # ================================================== LOGIN
        "login_brand": "🤖 التعلّم الآلي في صندوق",
        "login_sub": "سجّل الدخول للمتابعة.",
        "login_username": "اسم المستخدم",
        "login_password": "كلمة المرور",
        "login_signin": "تسجيل الدخول",
        "login_wrong": "اسم المستخدم أو كلمة المرور خاطئة.",
        "login_success": "تم تسجيل الدخول.",
        "login_demo_title": "حسابات تجريبية",
        "login_demo_admin": "admin / <code>admin123</code> &nbsp;·&nbsp; وصول كامل (ML، DL، NLP، CV)",
        "login_demo_doctor": "doctor / <code>doctor123</code> &nbsp;·&nbsp; ML، DL، NLP",
        "login_demo_jabo": "jabo / <code>jabo123</code> &nbsp;·&nbsp; ML، DL",
        "login_demo_nlp": "nlp_user / <code>nlp123</code> &nbsp;·&nbsp; NLP فقط",
        "login_demo_cv": "sambatica / <code>sambatica123</code> &nbsp;·&nbsp; CV",

        # ================================================== PROFILE
        "prof_hero_title": "محلّل البيانات",
        "prof_hero_sub": ("ارفع ملفاً لترى ما هو كل عمود، وما يعنيه على "
                          "الأرجح، وكيف سيتعامل معه المسار."),
        "prof_pill_1": "CSV", "prof_pill_2": "Excel",
        "prof_pill_3": "أنواع تُكتشف تلقائياً",
        "prof_upload_label": "أفلت ملف CSV أو Excel",
        "prof_try_hint": "جرّب: netflix_titles.csv · laptopData.csv · houses.csv · books.csv",
        "prof_stat_rows": "صفوف",
        "prof_stat_cols": "أعمدة",
        "prof_stat_missing": "خلايا مفقودة",
        "prof_stat_dups": "صفوف مكرّرة",
        "prof_columns_title": "الأعمدة",
        "prof_preview": "معاينة البيانات الخام",
        "prof_describe_spinner": "نطلب من النموذج اللغوي وصف الأعمدة…",
        "prof_feature_eng_title": "خطة هندسة الميزات",
        "prof_no_conversions": "لا حاجة لتحويلات — كل عمود قابل للاستخدام رقمي بالفعل.",
        "prof_likely_meaning": "المعنى المحتمل:",
        "prof_auto_conversion": "التحويل التلقائي:",
        "prof_missing_none": "لا قيم مفقودة",
        "prof_missing_some": "{p:.1f}% مفقود",
        "prof_dtype": "النوع: {t}",
        "prof_distinct": "{n} قيمة مميّزة",
        "prof_kind_num": "رقم",
        "prof_kind_numunit": "رقم مع وحدة",
        "prof_kind_cat": "فئة",
        "prof_kind_topcat": "فئة عالية التنوّع",
        "prof_kind_bool": "نعم/لا",
        "prof_kind_date": "تاريخ",
        "prof_kind_list": "قائمة مفصولة بفواصل",
        "prof_kind_text": "نص حر",
        "prof_kind_id": "معرّف (محذوف)",
        "prof_kind_empty": "فارغ (محذوف)",
        "prof_continue_btn": "⚡  متابعة إلى تشغيل المسار →",
        "prof_reupload_btn": "↺  ارفع ملفاً آخر",
        "prof_chat_title": "اسأل عن هذه المجموعة",
        "prof_chat_caption": ("أسئلة حرّة عن الأعمدة، شكل البيانات، أو معنى "
                              "أي منها. تستخدم الإجابة الملف أعلاه فقط."),
        "prof_chat_suggested": [
            "ما أهم الأعمدة هنا؟",
            "أيّ عمود يبدو هدفاً قابلاً للتنبؤ؟",
            "هل هناك أعمدة يجب حذفها؟",
            "ماذا يخبرني مدى القيم عن البيانات؟",
            "هل هناك إشارات على صعوبة النمذجة؟",
        ],
        "prof_chat_input": "اسأل أي شيء عن هذه المجموعة",
        "prof_chat_placeholder": "مثلاً: أيّ أعمدة فيها قيم مفقودة؟",
        "prof_chat_send": "إرسال",
        "prof_chat_clear": "مسح المحادثة",
        "prof_you": "أنت",
        "prof_assistant": "المساعد",

        # ================================================== SETUP
        "setup_hero_title": "ما نوع البيانات التي تعمل عليها؟",
        "setup_hero_sub": ("يتغيّر المسار حسب نوع البيانات. اختر النمط وسيحمل "
                           "الوكيل الأدوات المناسبة."),
        "setup_hero_pill_1": "ML / DL",
        "setup_hero_pill_2": "NLP",
        "setup_hero_pill_3": "CV",
        "setup_ml_title": "ML / DL",
        "setup_ml_pill": "جدولي",
        "setup_ml_body": ("ملفات CSV منظّمة. أرقام، فئات، تواريخ. اختر "
                          "التعلّم الآلي الكلاسيكي أو شبكة عصبية قابلة للضبط."),
        "setup_ml_example": "مثال: التنبؤ بسعر منزل من المساحة والعمر.",
        "setup_ml_choose": "اختر ML / DL",
        "setup_ml_locked": "حسابك لا يملك وصولاً إلى ML / DL الجدولي.",
        "setup_nlp_title": "NLP",
        "setup_nlp_pill": "نص",
        "setup_nlp_body": ("ملفات CSV بأعمدة نصية. صنّف المراجعات حسب "
                           "المشاعر أو الموضوع أو الرسائل المزعجة."),
        "setup_nlp_example": "مثال: مراجعات إيجابية أو سلبية.",
        "setup_nlp_choose": "اختر NLP",
        "setup_nlp_locked": "حسابك لا يملك وصولاً إلى NLP.",
        "setup_cv_title": "CV",
        "setup_cv_pill": "صور",
        "setup_cv_body": ("تصنيف صور بشبكات مُدرَّبة مسبقاً أو CNN مخصّص. "
                          "الكشف والتقسيم للعرض فقط."),
        "setup_cv_example": "مثال: تصنيف قطط مقابل كلاب من ملف ZIP.",
        "setup_cv_choose": "اختر CV",
        "setup_cv_locked": "حسابك لا يملك وصولاً إلى الرؤية الحاسوبية.",
        "setup_locked_btn": "مقفل",
        "setup_modality": "النمط:",
        "setup_change_modality": "↺ تغيير النمط",
        "setup_nlp_step1": "### الخطوة 1 — إعداد NLP",
        "setup_ml_step1": "### الخطوة 1 — نمط التعلّم",
        "setup_lm_pick": "أيّ عائلة من النماذج يستخدم هذا التشغيل؟",
        "setup_lm_ml": "📊 تعلّم آلي كلاسيكي — أشجار، تجمّعات، نماذج خطية",
        "setup_lm_dl": "🧠 تعلّم عميق — شبكات عصبية",
        "setup_lm_help": ("تعلّم آلي كلاسيكي: أسرع، أكثر قابلية للتفسير، "
                          "الأفضل للبيانات الجدولية. تعلّم عميق: شبكات عصبية "
                          "قابلة للضبط مع دعم GPU."),
        "setup_lm_only_one": "نمط واحد فقط مفعّل لحسابك.",
        "setup_ml_mode_title": "📊 نمط التعلّم الآلي الكلاسيكي",
        "setup_dl_mode_title": "🧠 نمط التعلّم العميق",
        "setup_ml_mode_body": "نماذج كلاسيكية فقط. الشبكة العصبية مخفية.",
        "setup_dl_mode_body": "ستُدرَّب الشبكة العصبية بإعدادك أدناه.",
        "setup_nn_expander": "⚙️ إعداد الشبكة العصبية",
        "setup_step2": "### الخطوة 2 — ما الذي يجب أن يتنبّأ به الوكيل؟",
        "setup_agent_card_title": "🧠 يمكن للوكيل أن يختار عنك",
        "setup_agent_card_body": ("اطلب من الوكيل النظر إلى الملف واقتراح عمود "
                                  "هدف ونوع مهمة. يمكنك تجاوز ذلك أدناه."),
        "setup_agent_btn": "🤖 اطلب من الوكيل اقتراح هدف",
        "setup_agent_reading": "نقرأ الملف…",
        "setup_agent_fail": "لم يتمكّن الوكيل من القرار. اختر عموداً يدوياً أدناه.",
        "setup_agent_suggests": "**الوكيل يقترح: `{t}`** كـ{task} — {reason}",
        "setup_preview": "معاينة: {name} ({rows} صفاً × {cols} عموداً)",
        "setup_target_label": "عمود الهدف (ما نتنبّأ به)",
        "setup_reading": "يقرأ الوكيل هذا كـ**{task}** (الثقة: {conf}) — {reason}",
        "setup_features_label": "أعمدة المُدخلات",
        "setup_feature_warn": "لا يمكن استخدام '{c}': {why}",
        "setup_nlp_no_text": "تم اختيار نمط NLP لكن لا يوجد عمود نصي ضمن المُدخلات.",
        "setup_nlp_switch": ("أضفت عموداً نصياً حرّاً. بدّل إلى نمط NLP لمسار "
                             "نصوص أقوى."),
        "setup_step3": "### الخطوة 3 — عندما يخطئ النموذج، أيّ خطأ أكثر إيلاماً؟",
        "setup_metric_label": "تفضيل المقياس",
        "setup_metric_balanced": "متوازن — نهتم بالتساوي بالإنذارات الكاذبة والحالات المفقودة",
        "setup_metric_precision": "الضبط — تجنّب الإنذارات الكاذبة (إيجابيات خاطئة)",
        "setup_metric_recall": "الاسترجاع — تجنّب الحالات المفقودة (سلبيات خاطئة)",
        "setup_metric_help": ("يغيّر هذا المقياس الذي يحسّنه الوكيل ويقدمه "
                              "خلال التحقق المتقاطع. 'الضبط' يعاقب الإيجابيات "
                              "الكاذبة. 'الاسترجاع' يعاقب السلبيات الكاذبة. "
                              "'متوازن' يستخدم F1."),
        "setup_step4": "### الخطوة 4 — دمج الأعمدة في أعمدة جديدة؟",
        "setup_step4_caption": ("أحياناً يحمل عمودان معاً إشارة أقوى من أي "
                                "منهما منفرداً. يستطيع الوكيل اقتراح تركيبات "
                                "(نسبة، حاصل ضرب، فرق) من أعمدة رقمية."),
        "setup_inter_btn": "🤖 اطلب من الوكيل اقتراح تفاعلات",
        "setup_inter_need_two": "نحتاج عمودين رقميين على الأقل لاقتراح تفاعلات.",
        "setup_inter_thinking": "نفكّر في التفاعلات…",
        "setup_inter_none": "لم يقترح الوكيل أيّ تفاعلات مفيدة هنا.",
        "setup_inter_proposals": "**اقتراحات الوكيل** — ألغِ ما لا تريده:",
        "setup_step5": "### الخطوة 5 — من يختار النماذج؟",
        "setup_mode_label": "اختيار النموذج",
        "setup_mode_auto": "تلقائي — الوكيل يختار ويشرح",
        "setup_mode_manual": "يدوي — أنا أختار",
        "setup_manual_label": "النماذج للتدريب",
        "setup_pick_hint": "اختر هدفاً وعمود مُدخل واحداً على الأقل.",
        "setup_continue_btn": "متابعة → اقتراح خطة تنظيف",

        # ================================================== PLAN
        "plan_title": "### خطة التنظيف",
        "plan_caption": ("هذا ما يريد الوكيل فعله بالبيانات قبل التدريب. لم "
                         "يتغيّر شيء بعد — ستوافق عليه بعد قليل."),
        "plan_spinner": "نطلب من النموذج اللغوي صياغة خطة…",
        "plan_will_happen": "**ما سيحدث**",
        "plan_drop_dupes": "• حذف الصفوف المكرّرة: **{v}**",
        "plan_yes": "نعم", "plan_no": "لا",
        "plan_fill_empty": "• ملء الخلايا الفارغة:",
        "plan_fill_one": "    – `{c}`: استخدام **{m}**",
        "plan_fill_none": "• ملء الخلايا الفارغة: **لا شيء**",
        "plan_trim": "• تقليص القيم الشاذّة في: **{cols}**",
        "plan_nothing": "لا شيء",
        "plan_drop_empty": "• حذف أعمدة شبه فارغة: **{cols}**",
        "plan_validator": "**رأي المدقّق**",
        "plan_no_warnings": "لا تحذيرات — هذه الخطة آمنة للتنفيذ.",
        "plan_show_raw": "اعرض ما اقترحه النموذج اللغوي أصلاً (قبل التحقق)",
        "plan_info": ("يُفحص اقتراح الوكيل مقابل البيانات الحقيقية قبل أن يلمس "
                      "أيّ شيء. أي شيء غير صالح — عمود غير موجود، طريقة خاطئة "
                      "لنوع العمود — يُسقَط مع تحذير."),
        "plan_approve": "✓ موافق وابدأ التدريب",
        "plan_reject": "✗ ارفض واختر مُدخلات جديدة",

        # ================================================== TRAIN
        "train_title": "### التدريب",
        "train_spinner": ("ندرّب النماذج، ونجري التحقق المتقاطع، وننفّذ الفحص "
                          "الذاتي الوكيلي…"),
        "train_starting": "نبدأ…",
        "train_fitting": "ندرّب النماذج…",
        "train_importance": "نحسب أهمية الميزات…",
        "train_done": "تمّ.",
        "train_stopped": "توقّف التدريب: {err}",
        "train_back": "← رجوع",
        "train_retry_spinner": "إعادة المحاولة بأهم {n} ميزات: {feats}",

        # ================================================== RESULTS
        "results_title": "### النتائج",
        "results_best_model": "أفضل نموذج",
        "results_cv": "نتيجة التحقق المتقاطع",
        "results_test": "نتيجة الاختبار",
        "results_what_numbers_title": "ماذا تعني هذه الأرقام",
        "results_metric_pref": "تفضيل المقياس: <b>{m}</b>. غيّره في شاشة الإعداد إذا كان النوع الخطأ من الأخطاء يُعاقَب.",
        "results_leaderboard": "### لوحة الصدارة",
        "results_feat_eng": "🧪 هندسة الميزات التي جَرَت قبل التدريب",
        "results_no_conversions": "لا حاجة لتحويلات.",
        "results_why_title": "### لماذا اختير كل نموذج",
        "results_why_caption": ("هذه الأسباب من النموذج اللغوي، مصوغة لهذه "
                                "المجموعة — عدد الصفوف، عدد الميزات، ونوع المهمة."),
        "results_baseline": "خط الأساس <span class=\"pill amber\">مرجع</span>",
        "results_baseline_body": ("نموذج غبي يخمّن الشيء نفسه دائماً. يُظهر "
                                  "كيف يبدو 'عدم فعل أي شيء'."),
        "results_tag_user": "اختيارك",
        "results_tag_agent": "اختيار الوكيل",
        "results_user_prefix": "اخترت هذا النموذج بنفسك. ",
        "results_per_class": "### النتائج لكل فئة",
        "results_confusion": "### مصفوفة الالتباس",
        "results_roc_title": "### منحنيات ROC و Precision-Recall",
        "results_roc_caption": ("تُظهر هذه المنحنيات مدى جودة فصل المصنّف بين "
                                "الفئات عند كل عتبة ثقة ممكنة. AUC = 1.0 مثالي؛ "
                                "0.5 يعادل رمي العملة."),
        "results_top_words": "### أهم الكلمات لكل فئة",
        "results_top_bigrams": "### أهم العبارات لكل فئة (bigrams)",
        "results_class_prefix": "**الفئة: `{c}`**",
        "results_feat_importance": "### أهمية الميزات",
        "results_adv_diag": "🧪 تشخيصات متقدّمة",
        "results_resid_title": "#### تشخيصات البواقي",
        "results_resid_caption": ("يسار: المتوقّع مقابل الفعلي — يجب أن تلتصق "
                                  "النقاط بالخط المتقطّع. يمين: يجب أن تبدو "
                                  "البواقي كضجيج عشوائي حول الصفر. أي نمط "
                                  "(منحنى، شكل مروحي) يعني أن النموذج يفوّت "
                                  "بنية ما."),
        "results_mean_resid": "متوسط البواقي",
        "results_std_resid": "الانحراف المعياري للبواقي",
        "results_mae_resid": "متوسط الخطأ المطلق",
        "results_sample_preds": "### عيّنة من التنبؤات",
        "results_sample_caption": "مأخوذة عشوائياً من {n} صفّ اختبار محجوز.",
        "results_correct": "**✅ الصحيحة — {n} معروضة**",
        "results_wrong": "**❌ الخاطئة — {n} معروضة**",
        "results_no_wrong": ("لم تُعرض تنبؤات خاطئة. النموذج أصاب في كل ما "
                             "فحصناه."),
        "results_self_check": "### الفحص الذاتي للوكيل",
        "results_plain_words": "### بكلمات بسيطة",
        "results_pred_title": "### جرّب النموذج على بيانات جديدة",
        "results_pred_caption": ("يستخدم النموذج المدرَّب **{m}** للتنبؤ بـ"
                                 "`{target}`. لا شيء هنا يغيّر النموذج أو "
                                 "بيانات التدريب."),
        "results_pred_method": "طريقة الإدخال",
        "results_pred_manual": "✏️ أدخل صفاً واحداً يدوياً",
        "results_pred_upload": "📤 ارفع ملف CSV",
        "results_pipeline_notes": "ملاحظات المسار",
        "results_take_further": "### خُذه أبعد",
        "results_tune_btn": "⚙️  اضبط هذا التشغيل أكثر",
        "results_tune_caption": "يفتح صفحة الضبط مع إعداد هذا التشغيل محمّلاً مسبقاً.",
        "results_save_export": "### الحفظ والتصدير",
        "results_save_btn": "💾 احفظ النموذج، التقرير، وPDF",
        "results_csv_btn": "📥 لوحة الصدارة (CSV)",
        "results_card_btn": "📇 بطاقة النموذج (PDF)",
        "results_card_help": "احفظ أولاً لتوليد بطاقة النموذج.",
        "results_pdf_btn": "📄 حمّل تقرير PDF",
        "results_saved_to": "تم الحفظ في {folder}",
        "results_save_fail": "فشل الحفظ: {err}",
        "results_try_another": "← جرّب هدفاً آخر",
        "results_nn_title": "🧠 تفاصيل الشبكة العصبية",
        "results_inter_title": "🔗 ميزات التفاعل المستخدمة",
        "results_nlp_title": "📝 مسار NLP المستخدم",
        "results_nlp_text_cols": "الأعمدة النصية: <b>{cols}</b>.",
        "results_metric_help": ("**R² = {r2:.3f}** يعني أن النموذج يفسّر "
                                "حوالي {pct:.0f}% من تغيّر الهدف. "
                                "R² لخط الأساس = {base:.3f}."),
        "results_metric_help_clf": ("**F1 = {f1:.3f}** يوازن بين الضبط "
                                    "والاسترجاع. F1 لخط الأساس = {base:.3f}."),

        # ================================================== PREDICT PANEL
        "predict_title": "#### توقّع صورة جديدة",
        "predict_manual_title": "**املأ المُدخلات** — أعمدة التفاعل تُحسب تلقائياً.",
        "predict_btn": "🔮  توقّع",
        "predict_result": "التنبؤ",
        "predict_confidence": "الثقة: {c:.1%}",
        "predict_upload_hint": "ارفع ملف CSV بنفس أعمدة المُدخلات الأساسية",
        "predict_upload_caption": ("الأعمدة المفقودة ستُحسب تلقائياً إذا كانت "
                                   "ميزات تفاعل."),
        "predict_missing": "أعمدة مطلوبة مفقودة: {cols}",
        "predict_predicted_n": "تم التنبؤ بـ {n} صفّاً.",
        "predict_download": "📥  حمّل التنبؤات (CSV)",
        "predict_missing_feature": "عمود ميزة مطلوب مفقود: {e}",
        "predict_failed": "فشل التنبؤ: {err}",

        # ================================================== TUNING
        "tune_hero_title": "الضبط",
        "tune_hero_sub": ("خُذ تشغيلاً سابقاً، غيّر شيئاً أو شيئين، وراقب هل "
                          "تتحسّن النتيجة. كل محاولة تُسجَّل."),
        "tune_pill_1": "واعٍ بالسجل",
        "tune_pill_2": "مقارنة", "tune_pill_3": "سجل",
        "tune_load_first": "حمّل مجموعة بيانات أولاً، ثم ارجع إلى الضبط.",
        "tune_go_profile": "🔎  اذهب إلى محلّل البيانات",
        "tune_no_runs": "لا تشغيلات مسجّلة بعد لـ **{source}**. درّب نموذجاً أولاً، ثم ارجع لضبطه.",
        "tune_go_run": "⚡  اذهب إلى تشغيل المسار",
        "tune_history_title": "### سجل هذه المجموعة",
        "tune_best_ever": "الأفضل على الإطلاق لهذه المجموعة",
        "tune_best_ever_body": ("<b>{model}</b> حقّق <b>{score:.3f}</b> كـ"
                                "<code>{task}</code> يتنبّأ بـ"
                                "<code>{target}</code>."),
        "tune_last_run": "### اضبط آخر تشغيل",
        "tune_last_caption": ("الهدف: `{target}` · المهمة: `{task}` · {n} "
                              "ميزات · النتيجة الأصلية: {score:.3f}"),
        "tune_target": "الهدف",
        "tune_inputs": "أعمدة المُدخلات",
        "tune_metric": "تفضيل المقياس",
        "tune_family": "عائلة النماذج",
        "tune_selection": "اختيار النموذج",
        "tune_models": "النماذج",
        "tune_retrain": "🚀  أعد التدريب بهذه الإعدادات",
        "tune_spinner": "نعيد التدريب…",
        "tune_result_title": "النتيجة",
        "tune_result_body": ("الأفضل الجديد: <b>{model}</b> حقّق <b>{score:.3f}</b>. "
                             "هذا <b>{verb}</b> التشغيل السابق بـ {delta:+.3f}."),
        "tune_open_results": "→ افتح النتائج كاملة",
        "tune_fail": "فشل إعادة التدريب: {err}",

        # ================================================== CV
        "cv_hero_title": "الرؤية الحاسوبية",
        "cv_hero_sub": ("درّب مصنّفات صور بشبكات مُدرَّبة مسبقاً، أو ابنِ CNN "
                        "صغيراً من الصفر. الكشف والتقسيم على خارطة الطريق — "
                        "مقفلان حالياً."),
        "cv_pill_1": "تصنيف",
        "cv_pill_2": "كشف (WIP)",
        "cv_pill_3": "تقسيم (WIP)",
        "cv_pill_4": "تعلّم بالنقل",
        "cv_tab_class": "التصنيف",
        "cv_tab_detect": "🔒 الكشف",
        "cv_tab_seg": "🔒 التقسيم",
        "cv_sec_1": "### 1. المجموعة",
        "cv_sec_2": "### 2. البنية",
        "cv_sec_3": "### 3. التدريب",
        "cv_sec_4": "### 4. ابدأ التدريب",
        "cv_sec_5": "### 5. النتائج",
        "cv_zip_title": "أيّ نوع من ZIP؟",
        "cv_zip_body": "ارفع ملف ZIP يحتوي في مستواه الأعلى على مجلد لكل فئة.",
        "cv_zip_min": ("كل مجلد فئة يحتاج صوراً كافية على الأقل. 20+ لكل فئة "
                       "حدّ أدنى معقول. الأكثر أفضل."),
        "cv_zip_upload": "ارفع ملف ZIP للمجموعة",
        "cv_zip_extract": "نستخرج…",
        "cv_zip_fail": ("لم نجد مجلدات الفئات في هذا الملف. تأكّد من وجود "
                        "مجلد فرعي لكل فئة في المستوى الأعلى."),
        "cv_stat_classes": "فئات",
        "cv_stat_train": "صور تدريب",
        "cv_stat_val": "صور تحقق",
        "cv_classes_prefix": "الفئات: ",
        "cv_diff_dataset": "↺ ارفع مجموعة أخرى",
        "cv_backbone": "الشبكة الأساسية",
        "cv_custom_caption": ("ابنِ CNN صغيراً من الصفر. تنتهي كومة الالتفاف "
                              "بمجمع عام، ثم تشغّل سلسلة الكثيفة حتى عدد الفئات."),
        "cv_suggested_backbone": "الشبكة المقترحة: **{name}** — {reason}",
        "cv_epochs": "الحقب",
        "cv_lr": "معدّل التعلّم",
        "cv_bs": "حجم الدفعة",
        "cv_freeze": "جمّد الشبكة الأساسية (تعلّم بالنقل)",
        "cv_freeze_help": "موصى به. يدرّب الطبقة الأخيرة فقط.",
        "cv_start_btn": "🚀 ابدأ التدريب",
        "cv_training": "نتدرّب…",
        "cv_evaluating": "نقيّم…",
        "cv_progress_epoch": "الحقبة {e}/{n}",
        "cv_progress_line": "الحقبة {e}: خسارة_تدريب={tl:.3f} · خسارة_تحقق={vl:.3f} · دقة_تحقق={va:.3f}",
        "cv_val_acc": "دقة التحقق",
        "cv_architecture": "البنية",
        "cv_confusion": "#### مصفوفة الالتباس",
        "cv_per_class": "#### النتائج لكل فئة",
        "cv_sample_preds": "#### عيّنة من التنبؤات",
        "cv_predict_title": "#### توقّع صورة جديدة",
        "cv_predict_using": "نستخدم: {m}",
        "cv_predict_loaded": " (محمّل من فتحة محفوظة)",
        "cv_predict_label": "التنبؤ",
        "cv_predict_conf": "الثقة {c:.1f}%",
        "cv_predict_fail": "فشل التنبؤ: {err}",
        "cv_saved_title": "### 💾 النماذج المحفوظة",
        "cv_saved_caption": ("احتفظ بما يصل إلى {n} نماذج مدرَّبة. مخزّنة على "
                             "القرص، لذا تصمد بعد إعادة التشغيل — حمّل أحدها "
                             "في أي وقت للتنبؤ دون إعادة تدريب."),
        "cv_saved_already": "النموذج الذي درّبته للتوّ محفوظ بالفعل.",
        "cv_saved_full": ("كل الفتحات الـ{n} ممتلئة. احذف واحدة أدناه لحفظ "
                          "هذا النموذج."),
        "cv_saved_train_first": "درّب نموذجاً أعلاه، ثم احفظه هنا.",
        "cv_saved_name_placeholder": "نموذجي",
        "cv_saved_name_label": "الاسم (اختياري)",
        "cv_saved_save_btn": "💾 احفظ ({cur}/{max})",
        "cv_saved_confirm_flash": "تم حفظ النموذج.",
        "cv_saved_load": "تحميل",
        "cv_saved_loading": "نحمّل النموذج…",
        "cv_saved_load_flash": "تم تحميل «{m}». استخدمه في صندوق التنبؤ أعلاه.",
        "cv_saved_delete": "حذف",
        "cv_saved_confirm": "تأكيد",
        "cv_saved_cancel": "إلغاء",
        "cv_saved_confirm_warn": "حذف هذا النموذج نهائياً؟ لا يمكن التراجع.",
        "cv_saved_in_use": "قيد الاستخدام",
        "cv_saved_saved_at": "حُفظ {when}",
        "cv_saved_size": "{mb} م.ب",
        "cv_saved_val_acc": "{p:.1f}% دقة تحقق",
        "cv_detect_wip_title": "🔒 الكشف — قيد العمل",
        "cv_detect_wip_body": ("الكشف يجد <em>ويحدّد موقع</em> الأجسام. كل صورة "
                               "تدريب تحتاج صندوق إحاطة لكل جسم، بصيغة COCO "
                               "JSON (مسار الصورة، تسمية الفئة، x/y/العرض/الارتفاع)."),
        "cv_detect_wip_body2": ("ستتيح هذه التبويبة رفع ملف ZIP بصيغة COCO، "
                                "وضبط كاشف، وتشغيل الاستدلال — لكن حلقة "
                                "التدريب لم تُبنَ بعد."),
        "cv_detect_wip_foot": ("قائمة البنى أدناه خارطة طريق للقراءة فقط. لا "
                               "رفع، لا تدريب، لا استدلال متاح هنا."),
        "cv_detect_arch_title": "### البنى المخطّطة (5)",
        "cv_detect_locked": "هذه الميزة مقفلة حتى تُنفَّذ حلقة التدريب.",
        "cv_seg_wip_title": "🔒 التقسيم — قيد العمل",
        "cv_seg_wip_body": ("التقسيم يسم كل بكسل. كل صورة تدريب تحتاج قناع PNG "
                            "مرافقاً حيث كل بكسل يحمل رقم الفئة."),
        "cv_seg_wip_body2": ("ستتيح هذه التبويبة رفع أزواج صورة+قناع، وضبط "
                             "مقسّم، وتشغيل الاستدلال — لكن حلقة التدريب لم "
                             "تُبنَ بعد."),
        "cv_seg_wip_foot": ("قائمة البنى أدناه خارطة طريق للقراءة فقط. لا رفع، "
                            "لا تدريب، لا استدلال متاح هنا."),
        "cv_seg_arch_title": "### البنى المخطّطة (5)",
        "cv_seg_locked": "هذه الميزة مقفلة حتى تُنفَّذ حلقة التدريب.",

        # ================================================== NLP TOOLS
        "nlp_tools_hero_title": "أدوات اللغة",
        "nlp_tools_hero_sub": ("نظّف مستنداً، قطّعه، لخّصه، جمّع مجموعة نصوص "
                               "في موضوعات، واحفظ الأجزاء المفيدة."),
        "nlp_tools_pill_1": "تنظيف", "nlp_tools_pill_2": "تقطيع",
        "nlp_tools_pill_3": "تلخيص", "nlp_tools_pill_4": "تجميع",
        "nlp_tools_pill_5": "تصدير",
        "nlp_1_title": "### 1. أعطِه نصاً",
        "nlp_paste_tab": "الصق نصاً",
        "nlp_upload_tab": "ارفع ملفاً",
        "nlp_paste_placeholder": ("الصق مقالاً، مراجعة، سلسلة بريد إلكتروني، "
                                  "أو أيّ نص طويل…"),
        "nlp_upload_label": "ارفع .txt أو .md أو .html أو .htm أو .pdf أو .docx أو CSV",
        "nlp_no_input": "الصق نصاً أعلاه أو ارفع ملفاً للبدء.",
        "nlp_input_caption": "الإدخال: {chars:,} حرفاً · {docs} مستند(ات).",
        "nlp_2_title": "### 2. نظّف النص",
        "nlp_2_caption": "كل خيار مفتاح تبديل. لا شيء يعمل حتى تنقر الزر.",
        "nlp_clean_options": "خيارات التنظيف",
        "nlp_opt_lower": "تحويل الكل إلى أحرف صغيرة",
        "nlp_opt_html": "إزالة وسوم HTML",
        "nlp_opt_urls": "إزالة الروابط",
        "nlp_opt_emails": "إزالة البريد الإلكتروني",
        "nlp_opt_punct": "إزالة الترقيم (. , ! ? …)",
        "nlp_opt_digits": "إزالة الأرقام",
        "nlp_opt_ws": "دمج المسافات الزائدة",
        "nlp_opt_stops": "إزالة كلمات التوقف (في، من، على)",
        "nlp_opt_changed": ("تغيّرت الخيارات منذ آخر تنظيف. انقر **نظّف النص** "
                            "لتطبيقها."),
        "nlp_clean_btn": "🧹 نظّف النص",
        "nlp_cleaned_output": "النص المُنظَّف",
        "nlp_cleaned_title": "اعرض النص المنظَّف",
        "nlp_reduction": "تقليص الحجم",
        "nlp_3_title": "### 3. التقطيع",
        "nlp_3_caption": ("قسّم النص المنظَّف إلى وحدات فردية. هذا ما يراه كل "
                          "نموذج NLP فعلاً."),
        "nlp_tok_keep_stops": "احتفظ بكلمات التوقف",
        "nlp_tokenize_btn": "🔤 قطّع",
        "nlp_tokens": "الوحدات",
        "nlp_all_tokens": "كل الوحدات",
        "nlp_all_tokens_caption": "يُعرض أول 500 من {n}.",
        "nlp_4_title": "### 4. العبارات المفتاحية",
        "nlp_phrases_btn": "🔑 استخرج العبارات المفتاحية",
        "nlp_5_title": "### 5. التلخيص",
        "nlp_extractive_title": "**استخلاصي** — أهم الجمل، حرفياً",
        "nlp_sentences": "الجمل",
        "nlp_extract_btn": "استخرج الملخّص",
        "nlp_extractive_output": "الملخّص الاستخلاصي",
        "nlp_abstractive_title": "**توليدي** — إعادة صياغة بكلماته",
        "nlp_approx_words": "عدد الكلمات التقريبي",
        "nlp_abstract_btn": "ولّد ملخّصاً",
        "nlp_abstractive_output": "الملخّص التوليدي",
        "nlp_summarizing": "نلخّص…",
        "nlp_6_title": "### 6. تجميع المواضيع",
        "nlp_6_caption": ("جمّع المستندات حسب الموضوع. كل موضوع يُسمّى بأكثر "
                          "كلماته تمييزاً."),
        "nlp_cluster_split": "قسّم هذا النص إلى مستندات",
        "nlp_cluster_split_label": "قسّم حسب",
        "nlp_cluster_split_blank": "أسطر فارغة (فقرات)",
        "nlp_cluster_split_sent": "حدود الجمل",
        "nlp_cluster_split_words": "كل N كلمة",
        "nlp_cluster_chunk": "كلمات لكل قطعة",
        "nlp_cluster_split_ok": "قُسِّم إلى {n} مستنداً.",
        "nlp_cluster_split_no": "قُسِّم إلى {n} فقط — نحتاج 3 على الأقل للتجميع.",
        "nlp_cluster_need_3": ("نحتاج 3 مستندات على الأقل. إمّا حمّل ملف CSV "
                               "بـ 3+ صفوف، أو استخدم المُقسِّم أعلاه."),
        "nlp_topics_count": "عدد المواضيع",
        "nlp_cluster_btn": "🔬 ابحث عن مواضيع",
        "nlp_clustering": "نجمّع…",
        "nlp_topics_prefix": "**{n} مواضيع** · silhouette **{sil}** (فوق 0.2 يعني مجموعات منفصلة بوضوح)",
        "nlp_topic": "الموضوع {i}",
        "nlp_topic_docs": "{n} مستند",
        "nlp_7_title": "### 7. البحث بالتشابه",
        "nlp_search_placeholder": "مثلاً: تأخير الشحن، طلب استرجاع",
        "nlp_search_results": "النتائج",
        "nlp_search_btn": "🔍 ابحث",
        "nlp_search_score": "النتيجة {s}",
        "nlp_search_doc": "مستند #{i}",
        "nlp_search_none": "لا نتائج.",
        "nlp_8_title": "### 8. الترجمة",
        "nlp_lang_label": "اللغة الهدف",
        "nlp_translate_btn": "ترجم",
        "nlp_translating": "نترجم إلى {lang}…",
        "nlp_translation_output": "الترجمة",
        "nlp_9_title": "### 9. تصحيح النحو والإملاء",
        "nlp_grammar_btn": "صحّح النحو",
        "nlp_correcting": "نصحّح…",
        "nlp_original": "**الأصلي**",
        "nlp_corrected": "**المصحَّح**",
        "nlp_10_title": "### 10. الحفظ في الأرشيف",
        "nlp_10_caption": ("حمّل النص المنظَّف والملخّصات بصيغة Excel أو SQL "
                           "أو حزمة نصية."),
        "nlp_export_fmt": "الصيغة",
        "nlp_export_excel": "Excel (.xlsx)",
        "nlp_export_sql": "عبارات SQL INSERT",
        "nlp_export_txt": "حزمة نصية",
        "nlp_download_xlsx": "حمّل .xlsx",
        "nlp_download_csv": "حمّل .csv",
        "nlp_download_sql": "حمّل .sql",
        "nlp_download_txt": "حمّل حزمة .txt",
        "nlp_openpyxl_missing": "openpyxl غير مثبّت — نستخدم CSV بدلاً منه.",

        # ================================================== NLP DASHBOARD
        "nlp_dash_hero_title": "لوحة اللغة",
        "nlp_dash_hero_sub": ("هذه الصفحة لمجموعات النصوص — مجموعة كاملة من "
                              "المستندات تُنظر معاً. لمعالجة مستند واحد استخدم "
                              "أدوات اللغة. هنا تحصل على: سحابة كلمات تبيّن "
                              "موضوع المجموعة، توزيع المشاعر عبر المستندات، "
                              "توزيع أطوال النصوص، أكثر العبارات المكوّنة من "
                              "كلمتين، وكيف تنمو المفردات مع القراءة."),
        "nlp_dash_pill_1": "عرض مجموعة",
        "nlp_dash_pill_2": "CSV / PDF / DOCX / TXT",
        "nlp_dash_pill_3": "مستندات كثيرة",
        "nlp_dash_upload": ("ارفع مجموعة — CSV بعمود نصي، أو مستند واحد "
                            "(.pdf، .docx، .txt، .md، .html)"),
        "nlp_dash_use_loaded": "لديك مستندات محمّلة من أدوات اللغة.",
        "nlp_dash_use_loaded_btn": "استخدم المستندات المحمّلة",
        "nlp_dash_csv_pick": "أيّ عمود يحمل النص؟",
        "nlp_dash_split_label": ("هذا مستند واحد. كيف أقسمه إلى عدّة أجزاء "
                                 "لعرض المجموعة؟"),
        "nlp_dash_split_1": "فقرات (أسطر فارغة)",
        "nlp_dash_split_2": "جمل",
        "nlp_dash_split_3": "قطع بحجم ثابت",
        "nlp_dash_chunk": "كلمات لكل قطعة",
        "nlp_dash_split_ok": "قُسِّم إلى {n} مستنداً زائفاً.",
        "nlp_dash_stat_docs": "مستندات",
        "nlp_dash_stat_words": "إجمالي الكلمات",
        "nlp_dash_stat_avg": "متوسط الكلمات/مستند",
        "nlp_dash_stat_vocab": "كلمات مميّزة",
        "nlp_dash_wordcloud": "### سحابة الكلمات",
        "nlp_dash_wc_missing": "ثبّت `wordcloud` لرؤية السحابة: `pip install wordcloud`.",
        "nlp_dash_sentiment": "### توزيع المشاعر",
        "nlp_dash_sentiment_caption": ("يقيّم النموذج اللغوي عيّنة تصل إلى 15 "
                                       "مستنداً كسلبي أو محايد أو إيجابي."),
        "nlp_dash_scoring": "نقيّم المشاعر…",
        "nlp_dash_llm_unavail": "النموذج اللغوي غير متاح؛ تخطّينا المشاعر.",
        "nlp_dash_sent_method": "الطريقة: {m} · عيّنة من {n}",
        "nlp_dash_length": "### توزيع أطوال النصوص",
        "nlp_dash_length_x": "كلمات لكل مستند",
        "nlp_dash_length_y": "مستندات",
        "nlp_dash_bigrams": "### أهم العبارات (bigrams)",
        "nlp_dash_vocab_growth": "### نمو المفردات",
        "nlp_dash_vocab_x": "المستندات المقروءة",
        "nlp_dash_vocab_y": "الكلمات المميّزة المرئية",
        "nlp_dash_vocab_caption": ("ذيل مسطّح يعني أن مفردات المجموعة استقرّت "
                                   "— رأيت معظم ما لديها. منحنى صاعد يعني أن "
                                   "هناك تنوّعاً أكثر للاستكشاف."),
    },
}


# ============================================================ ARABIC GLOSSARY
GLOSSARY_AR = {
    "مقاييس الانحدار": [
        ("R² (R-squared)",
         "مقدار تباين الهدف الذي يفسّره النموذج.",
         "1.0 مثالي. 0 عديم الفائدة. R² = 0.71 يعني أن النموذج يفسّر 71% من "
         "صعود وهبوط الهدف.",
         "أعلى من 0.7 نتيجة قوية. أقل من 0.2 يعني أن الأعمدة لا تتنبّأ بالهدف."),
        ("MAE (متوسط الخطأ المطلق)",
         "متوسط حجم خطأ التنبؤ، بوحدات الهدف نفسها.",
         "قس الفجوة لكل تنبؤ، خذ القيم المطلقة، ثم احسب المتوسط.",
         "أقل = أفضل. قارنه بـ MAE لخط الأساس."),
        ("RMSE (الجذر التربيعي لمتوسط مربع الخطأ)",
         "مثل MAE لكن يعاقب الأخطاء الكبيرة أكثر.",
         "ربّع كل خطأ أولاً، ثم احسب المتوسط، ثم خذ الجذر التربيعي.",
         "أقل = أفضل. RMSE >> MAE يعني أن بعض التنبؤات بعيدة جداً."),
    ],
    "مقاييس التصنيف": [
        ("الدقة (Accuracy)",
         "نسبة التنبؤات التي كانت صحيحة تماماً.",
         "اعتبر المطابقات، اقسمها على الإجمالي. مضلّلة عندما تهيمن فئة واحدة.",
         "قارنها بخط الأساس. لو كان خط الأساس 90%، فالدقة تخدع."),
        ("الضبط (Precision)",
         "عندما يقول النموذج 'الفئة X'، ما مدى صحته؟",
         "من كل ما صُنِّف الفئة X، ما النسبة التي تنتمي فعلاً إليها.",
         "أعلى = أفضل. استخدمه عندما تكون الإنذارات الكاذبة مكلفة."),
        ("الاسترجاع (Recall)",
         "من كل الحالات الحقيقية للفئة X، كم حالة التقطها النموذج؟",
         "من كل ما هو فعلاً الفئة X، ما النسبة التي التقطها النموذج.",
         "أعلى = أفضل. استخدمه عندما تكون تفويت حالة مكلفاً."),
        ("نتيجة F1",
         "توازن بين الضبط والاسترجاع. رقم واحد يأخذ الاثنين في الحساب.",
         "المتوسط التوافقي للاثنين.",
         "0.9+ ممتاز. أقل من 0.3 يعني أن النموذج بالكاد يساعد."),
        ("مصفوفة الالتباس (Confusion matrix)",
         "جدول يبيّن ما تنبّأ به النموذج مقابل ما حدث فعلاً.",
         "الصفوف = الواقع، الأعمدة = التنبؤات. القُطر هو الصحيح.",
         "انظر أين تتركّز الكتلة خارج القُطر. تلك هي الأخطاء."),
    ],
    "نماذج التعلّم الآلي": [
        ("الانحدار الخطي (Linear Regression)",
         "يلائم علاقة خطّية بين المُدخلات وهدف رقمي.",
         "يتنبّأ بمجموع موزون للمُدخلات، زائد حدّ ثابت.",
         "سريع وقابل للتفسير. يتعثّر مع الأنماط المنحنية أو غير الخطية."),
        ("الانحدار اللوجستي (Logistic Regression)",
         "مصنّف خطي. يتنبّأ باحتمال كل فئة.",
         "يلائم حدّاً خطياً، ثم يطبّق منحنى لوجستياً.",
         "بسيط، قابل للتفسير، وغالباً منافس جداً في النصوص."),
        ("Ridge / Lasso",
         "نماذج خطية مع تنظيم.",
         "Ridge يقلّص المعاملات؛ Lasso يُسقط الضعيف منها إلى الصفر.",
         "Ridge عندما تكون المُدخلات مترابطة. Lasso لاختيار الميزات آلياً."),
        ("شجرة القرار (Decision Tree)",
         "تتعلّم سلسلة أسئلة نعم/لا تقسّم البيانات.",
         "كل عقدة داخلية تختبر ميزة؛ كل ورقة تتنبّأ.",
         "قابلة للتفسير جداً، لكنها تميل إلى الإفراط في التخصيص. أساس "
         "لتجمّعات النماذج."),
        ("الغابة العشوائية (Random Forest)",
         "أشجار قرار كثيرة تصوّت معاً.",
         "كل شجرة تدرّبت على مجموعة جزئية عشوائية من الصفوف والأعمدة.",
         "متينة، ونادراً ما تحتاج ضبطاً. افتراضي قوي شائع."),
        ("تعزيز التدرّج (Gradient Boosting)",
         "أشجار تُصلح أخطاء بعضها بالترتيب.",
         "كل شجرة جديدة تدرّب على أخطاء المجموعة السابقة.",
         "غالباً الأقوى على البيانات الجدولية."),
        ("Extra Trees",
         "مثل الغابة العشوائية لكن مع عشوائية أكبر في التقسيمات.",
         "تُختار التقسيمات عشوائياً بدلاً من تحسينها.",
         "أسرع في التدريب. يستحق التجربة إلى جانب الغابة."),
        ("AdaBoost",
         "تعزيز بأشجار سطحية (جذوع) كمُتعلّم أساسي.",
         "الجذوع المتأخّرة تركّز على الصفوف التي أخطأتها الجذوع السابقة.",
         "قوي عندما تكون الإشارة بسيطة."),
        ("Bagging",
         "عدّة نسخ من نموذج واحد، كل على مجموعة صفوف عشوائية.",
         "التنبؤات تُتوسّط (انحدار) أو تُصوَّت (تصنيف).",
         "تقليل رخيص للتباين على أي نموذج أساسي."),
        ("K-Nearest Neighbours (KNN)",
         "يتنبّأ من الأمثلة السابقة الأكثر تشابهاً.",
         "أوجد K صفوف أقرب، ثم وسّط أو صوّت بينها.",
         "رائع على بيانات صغيرة بأعمدة قليلة. حسّاس للقياس."),
        ("SVM (Support Vector Machine)",
         "يجد أوسع حدّ ممكن بين الفئات.",
         "يعظّم الهامش بين أقرب نقاط من كل فئة.",
         "قوي على البيانات الصغيرة النظيفة. لا يتوسّع لبيانات ضخمة."),
        ("Naive Bayes",
         "مصنّف احتمالي يفترض استقلال الميزات.",
         "كل ميزة تصوّت لفئة، موزونة بمعدّل ظهورها.",
         "سريع للغاية. قوي بشكل مفاجئ على النصوص."),
        ("الشبكة العصبية (MLP)",
         "طبقات من وحدات بسيطة تتعلّم أنماطاً غير خطية.",
         "كل طبقة تحوّل السابقة. الأوزان تُتعلَّم بالانتشار العكسي.",
         "مرنة، لكن تحتاج بيانات وضبطاً أكثر من تجمّعات الأشجار."),
        ("التعلّم العميق (Deep Learning)",
         "شبكات عصبية بطبقات كثيرة. نفس فكرة MLP بمقياس أكبر.",
         "طبقات أكثر، بيانات أكثر، حوسبة أكثر.",
         "غالباً مبالَغ فيه للبيانات الجدولية. الاختيار الصحيح للصور والنصوص."),
    ],
    "مفاهيم التدريب": [
        ("خط الأساس (Baseline)",
         "أبسط نموذج ممكن. يُظهر كيف يبدو 'عدم فعل أي شيء'.",
         "انحدار: يتنبّأ بالمتوسط. تصنيف: الفئة الأكثر شيوعاً.",
         "لو بالكاد تجاوز أفضل نموذج خط الأساس، فالبيانات لا تتنبّأ."),
        ("الإفراط في التخصيص (Overfitting)",
         "النموذج حفظ صفوف التدريب بدلاً من تعلّم الأنماط.",
         "رائع على صفوف التدريب، سيّئ على صفوف جديدة.",
         "الوكيل يفحصه ويحاول إصلاحه تلقائياً."),
        ("نقص التخصيص (Underfitting)",
         "النموذج بسيط أكثر من اللازم لالتقاط النمط.",
         "سيّئ على صفوف التدريب وسيّئ على صفوف جديدة.",
         "الإصلاح بنموذج أكثر مرونة، أو مُدخلات أفضل."),
        ("اختلال الفئات (Class imbalance)",
         "فئة واحدة لها أمثلة أكثر بكثير من غيرها.",
         "لو 95% من الصفوف الفئة A، فالدقة تخدع.",
         "الوكيل يحذّر عندما تكون أكبر فئة 5 أضعاف الأصغر."),
        ("التحقق المتقاطع (Cross-validation)",
         "النتيجة خلال التدريب، متوسّطة على عدّة تقسيمات.",
         "البيانات تُقسَّم إلى طيّات؛ كل طيّة تُستخدم كاختبار مرّة واحدة.",
         "قارنه بنتيجة الاختبار. الفجوة تعني إفراطاً في التخصيص."),
        ("تقسيم التدريب/الاختبار",
         "البيانات تُقسَم إلى جزأين: تعلّم على أحدهما، تحقّق على الآخر.",
         "النموذج لا يرى صفوف الاختبار خلال التدريب.",
         "80/20 هو الافتراضي الشائع."),
        ("أهمية الميزة (Feature importance)",
         "كم يُغيّر كل عمود مُدخل تنبؤات النموذج.",
         "بدّل عموداً واحداً في كل مرّة، وراقب هبوط النتيجة.",
         "هبوط أكبر = عمود أكثر تأثيراً."),
        ("التنظيم (Regularization)",
         "عقوبة على التعقيد تمنع الإفراط في التخصيص.",
         "تُضاف عبارة إلى دالة الخطأ تعاقب الأوزان الكبيرة.",
         "تنظيم أعلى = نموذج أنعم."),
        ("المُعامل الفائق (Hyperparameter)",
         "إعداد تختاره قبل التدريب.",
         "أمثلة: عدد الأشجار، أقصى عمق، معدّل التعلّم، نسبة الإسقاط.",
         "لوحة الإعداد تتيح لك تعيينها لكل نموذج."),
        ("هندسة الميزات (Feature engineering)",
         "إنشاء مُدخلات جديدة من الموجودة لمساعدة النموذج على التعلّم.",
         "أمثلة: نِسَب، حواصل ضرب، أجزاء التاريخ، أعلام كلمات مفتاحية.",
         "غالباً الشيء الأعلى مردوداً الذي يمكنك فعله."),
        ("ميزة التفاعل (Interaction feature)",
         "عمود جديد يُنشأ بدمج عمودين موجودين.",
         "أمثلة: السعر / المساحة، الوزن * الطول، تاريخ_أ - تاريخ_ب.",
         "يلتقط تأثيرات مركّبة لا يُظهرها أي عمود منفرداً."),
        ("التعبئة (Imputation)",
         "ملء القيم المفقودة ليتمكّن النموذج من استخدام العمود.",
         "استراتيجيات شائعة: الوسيط، المتوسط، المنوال، أو حذف الصف.",
         "خطة التنظيف التي يقترحها الوكيل تقترح هذه لكل عمود."),
        ("التقييس (Standardization)",
         "إعادة قياس الميزات حتى لا يهيمن أي منها بسبب وحداته فقط.",
         "اطرح المتوسط، اقسم على الانحراف المعياري.",
         "ضروري للنماذج التي تستخدم المسافات أو الانحدار التدريجي."),
        ("الترميز الأحادي (One-hot encoding)",
         "تحويل عمود فئوي إلى أعمدة ثنائية كثيرة.",
         "كل فئة تصبح عمودها 0/1 الخاص.",
         "قياسي للنصوص والبلدان وأنواع المنتجات."),
    ],
    "الذكاء الوكيلي": [
        ("الذكاء الوكيلي (Agentic AI)",
         "نموذج لغوي في حلقة مع أدوات وطريقة لفحص عمله.",
         "الوكيل يقرّر الخطة، الكود يُنفّذ، إنسان يوافق، عقدة مراجعة تتفاعل.",
         "أربعة قرارات: الهدف، التفاعلات، النماذج، والإصلاح."),
        ("الإنسان في الحلقة (Human-in-the-loop)",
         "الوكيل يتوقف لينال موافقة إنسان قبل فعل لا رجعة فيه.",
         "قبل أي تنظيف أو تدريب، تتوقف الحالة وتعرض الخطة.",
         "الفرق الجوهري بين سكربت آلي ووكيل."),
        ("الحاجز (Guardrail)",
         "فحص برمجي يقيّد ما يُسمح للنموذج اللغوي بفعله.",
         "النموذج يُرجع JSON. الكود يتحقق من كل حقل مقابل البيانات الفعلية.",
         "الحقول المرفوضة تُسجَّل كتحذيرات."),
        ("استدعاء الأدوات (Tool calling)",
         "عندما يطلب النموذج اللغوي من الكود تشغيل دالة محدّدة.",
         "النموذج يُخرج طلباً منظّماً؛ الكود ينفّذه.",
         "يمنع النموذج من تنفيذ كود اعتباطي."),
        ("الحالة (State)",
         "كل ما يحتاج الوكيل تذكّره بين الخطوات.",
         "محفوظة في قاموس حالة LangGraph؛ الكائنات الكبيرة في ذاكرة جانبية.",
         "المُدقّق يحفظ الحالة على القرص حتى تستطيع الحالة التوقف."),
        ("نقطة التفتيش (Checkpoint)",
         "نقطة حفظ في الحالة يمكن للوكيل التوقف عندها والاستئناف منها.",
         "interrupt() في LangGraph ينشئ واحدة عند خطوة الموافقة البشرية.",
         "الحالة تتوقف فعلاً، لا تنتظر input() فقط."),
        ("الفحص الذاتي (Self-check)",
         "عقدة تفحص مُخرجها وتقرّر إن كانت ستعيد المحاولة.",
         "تقارن نتيجة التدريب بنتيجة الاختبار، وتفحص توازن الفئات.",
         "النتيجة 'مقبول' أو محاولة إصلاح محدّدة."),
        ("سجل التشغيل (Run history)",
         "سجل بكل تشغيل سابق، محفوظ على القرص.",
         "الوكيل يقرأه قبل اختيار النماذج.",
         "هكذا 'يتعلّم' الوكيل — بقراءة ملاحظاته."),
    ],
    "معالجة اللغة الطبيعية": [
        ("التقطيع (Tokenization)",
         "تقسيم النص إلى وحدات فردية (كلمات، ترقيم).",
         "أول خطوة في كل مسار معالجة لغة.",
         "الوحدات هي ما يُعدّ ويُوزَن ويُغذّى للنماذج."),
        ("كلمات التوقف (Stop-words)",
         "كلمات شائعة جداً ('في'، 'من'، 'على') تحمل معنى ضئيلاً.",
         "غالباً تُحذف لتقليل الضجيج وتصغير المفردات.",
         "مفيدة لتحليل التكرار. أحياناً ضارّة لتحليل المشاعر."),
        ("TF-IDF",
         "كم مرّة تظهر كلمة في مستند، موزوناً بندرتها.",
         "الكلمات الشائعة تأخذ وزناً منخفضاً. الكلمات المميّزة وزناً عالياً.",
         "الناتج مصفوفة رقمية يستطيع المصنّف التعلّم منها."),
        ("حقيبة الكلمات (Bag of words)",
         "عدّ لكل كلمة، مع تجاهل الترتيب.",
         "كل مستند يصبح متجهاً من عدّات الكلمات.",
         "سريع وفعّال. يفقد ترتيب الكلمات والقواعد."),
        ("N-gram",
         "تسلسل من N كلمة أو حرف متتالٍ.",
         "Bigram = كلمتان. Trigram = ثلاث كلمات.",
         "يلتقط العبارات القصيرة. TF-IDF هنا يستخدم bigrams."),
        ("التنظيف / المعالجة المسبقة",
         "إزالة HTML وURLs والترقيم وحالة الأحرف قبل التحليل.",
         "كل خطوة هي مفتاح تبديل في هذا المشروع.",
         "انظر دائماً إلى الإحصاءات لترى ما تغيّر فعلاً."),
        ("التلخيص الاستخلاصي",
         "اختيار الجمل الأكثر إفادة، حرفياً.",
         "غير متصل، حتمي، قابل للتفسير بالكامل.",
         "سريع وآمن. لا نموذج لغوي متورّط."),
        ("التلخيص التوليدي",
         "إعادة صياغة النص في نسخة أقصر، بكلمات النموذج.",
         "يستخدم النموذج اللغوي. مكثّف ومقروء، لكن أبطأ.",
         "الأفضل للمصادر الطويلة التي تحتاج إعادة صياغة."),
        ("تجميع المواضيع (Topic clustering)",
         "تجميع المستندات في موضوعات باستخدام KMeans على متجهات TF-IDF.",
         "المستندات التي تشارك كلمات مميّزة تنتهي في نفس المجموعة.",
         "معامل Silhouette أعلى من 0.2 يعني مجموعات منفصلة بوضوح."),
        ("معامل Silhouette",
         "مقياس لمدى انفصال المجموعات بوضوح.",
         "يتراوح من -1 إلى 1. فوق 0.2 جيد، فوق 0.5 قوي.",
         "أقل يعني تداخل المجموعات — جرّب عدداً مختلفاً من المواضيع."),
        ("تشابه الجيب (Cosine similarity)",
         "مقياس لتشابه مستندين، بناءً على متجهات كلماتهما.",
         "1.0 = نفس الاتجاه، 0 = غير مترابطين.",
         "يستخدمه لوح البحث بالتشابه."),
        ("تحليل المشاعر (Sentiment analysis)",
         "تصنيف النص إلى إيجابي أو سلبي أو محايد.",
         "النموذج اللغوي يقيّم عيّنة؛ اللوحة تُظهر التوزيع.",
         "سريع لملخّص. للإنتاج، درّب مصنّفاً حقيقياً."),
        ("سحابة الكلمات (Word cloud)",
         "صورة تظهر فيها الكلمات الأكثر تكراراً بحجم أكبر.",
         "مبنية من المجموعة كاملة مع إزالة كلمات التوقف.",
         "طريقة سريعة لرؤية المواضيع بنظرة واحدة."),
    ],
}


def t(key):
    """Return the localized string (or list) for `key`. Falls back to English."""
    import streamlit as st
    lang = st.session_state.get("lang", "en")
    table = LANGS.get(lang, LANGS["en"])
    if key in table:
        return table[key]
    return LANGS["en"].get(key, key)


def get_glossary():
    """Return the glossary dict for the current language."""
    import streamlit as st
    lang = st.session_state.get("lang", "en")
    if lang == "ar":
        return GLOSSARY_AR
    return _EN_GLOSSARY