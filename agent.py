"""
LangGraph orchestration for ML-in-a-Box.

Why this is agentic:
  - The LLM proposes a cleaning plan; code validates it (guardrail).
  - The LLM picks the model strategy from the profile.
  - The LLM reviews the leaderboard and picks the repair action.
  - The graph has a loop: prune features and retrain.
  - A human checkpoint pauses the graph before training.

State is small; DataFrames and pipelines live in a side cache keyed by run_id.
"""
import operator
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from ml import (load_table_with_notes, get_plan, validate_plan, restrict_plan,
                run as train_run, task_evidence, select_features,
                ask_user_models, export_all, interview, suggest_features,
                DEFAULT_NN_CONFIG, prune_features)


_CTX = {}


def _ctx(state):
    return _CTX.setdefault(state.get("run_id", "default"), {})


class State(TypedDict):
    source_path: str
    run_id: str
    max_iterations: int
    load_notes: list
    target: str
    task: str
    features: list
    mode: str
    model_keys: list
    proposed_plan: dict
    clean_plan: dict
    warnings: list
    iteration: int
    last_decision: str
    output_folder: str
    log: Annotated[list, operator.add]


def node_ingest(state):
    df, notes = load_table_with_notes(state["source_path"])
    _ctx(state)["df"] = df
    return {"load_notes": notes, "iteration": 0,
            "log": [f"Ingested {len(df)} rows x {len(df.columns)} cols "
                    f"from {state['source_path']}"]}


def node_interview(state):
    df = _ctx(state)["df"]
    print("\nColumns:", list(df.columns))
    try:
        answer = interview(df)
    except Exception as e:
        print(f"(interview failed: {type(e).__name__}; falling back to manual)")
        answer = None
    if answer is None:
        target = input("Target column: ").strip().strip('"').strip("'")
        if target not in df.columns:
            raise SystemExit(f"'{target}' is not a column.")
        guess, conf, reason = task_evidence(df[target])
        print(f"Reading: {guess} ({conf}) — {reason}")
        task = guess
        features = list(select_features(df, target)[0])
    else:
        target, task, features = answer
        guess, conf, reason = task_evidence(df[target])
        if guess != task and conf == "high":
            print(f"Note: data says {guess} ({reason}); using that.")
            task = guess
    return {"target": target, "task": task, "features": features,
            "log": [f"Task={task}  target='{target}'  features={len(features)}"]}


def node_choose_models(state):
    print("\nHow should I pick models?  1=Auto (agent)  2=Manual  [1]")
    mode = input("> ").strip() or "1"
    keys = []
    if mode == "2":
        keys = ask_user_models(state["task"], int(len(_ctx(state)["df"]) * 0.8)) or []
    return {"mode": "auto" if mode == "1" else "manual", "model_keys": keys,
            "log": [f"Model selection: {'agent' if mode == '1' else 'manual'}"]}


def node_plan(state):
    plan = get_plan(_ctx(state)["df"])
    return {"proposed_plan": plan, "log": ["LLM proposed a cleaning plan"]}


def node_validate(state):
    clean, warnings = validate_plan(state["proposed_plan"], _ctx(state)["df"])
    features = state.get("features") or list(_ctx(state)["df"].columns)
    restricted = restrict_plan(clean, features)
    return {"clean_plan": restricted, "warnings": warnings,
            "log": [f"Validated plan ({len(warnings)} warning(s))"] +
                   [f"  ! {w}" for w in warnings]}


def node_human_checkpoint(state):
    decision = interrupt({"question": "Approve this cleaning plan?",
                          "plan": state["clean_plan"],
                          "warnings": state["warnings"]})
    return {"log": [f"Human decision: {decision}"]}


def node_train(state):
    df = _ctx(state)["df"]
    features = state["features"]
    target = state["target"]
    df_run = df[features + [target]] if features else df
    it = state.get("iteration", 0) + 1
    print(f"\nTraining round {it}, please wait...")
    try:
        results, notes, extras = train_run(df_run, target, state["clean_plan"],
                                            state["task"], state["model_keys"] or None,
                                            self_check=True,
                                            nn_config=DEFAULT_NN_CONFIG)
    except ValueError as e:
        return {"iteration": it, "last_decision": "finish",
                "log": [f"Training round {it} failed: {e}"]}
    cvcol = results.columns[1] if not results.empty else None
    prev = _ctx(state).get("results")
    if prev is not None and not prev.empty and cvcol and not results.empty:
        prev_best = float(prev.iloc[0][cvcol])
        new_best = float(results.iloc[0][cvcol])
        if new_best <= prev_best:
            return {"iteration": it,
                    "log": [f"Training round {it}: {new_best:.3f} not better "
                            f"than round {it - 1} ({prev_best:.3f}); keeping earlier."]}
    _ctx(state)["results"] = results
    _ctx(state)["notes"] = notes
    _ctx(state)["extras"] = extras
    return {"iteration": it, "log": [f"Training round {it}: best = {extras['best_name']}"]}


def node_review(state):
    results = _ctx(state).get("results")
    if results is None or results.empty:
        return {"last_decision": "finish", "log": ["No results; finishing"]}
    best = results.iloc[0]
    col = "Test R2" if state["task"] == "regression" else "F1"
    score = float(best[col])
    is_baseline = best["Model"].startswith("Baseline")
    if state["iteration"] >= state["max_iterations"]:
        return {"last_decision": "finish",
                "log": [f"Iteration limit ({state['iteration']}); finishing"]}
    extras = _ctx(state).get("extras") or {}
    selfcheck = " ".join(extras.get("selfcheck") or []).lower()
    weak = is_baseline or score < 0.3
    overfitting = "memorizing" in selfcheck or "overfit" in selfcheck
    if weak or overfitting:
        return {"last_decision": "retry",
                "log": [f"Score {score:.2f} is weak or overfit; pruning low-signal features"]}
    return {"last_decision": "finish", "log": [f"Score {score:.2f} accepted"]}


def node_prune(state):
    feats = state.get("features") or []
    if len(feats) <= 3:
        return {"last_decision": "finish",
                "log": ["Too few features to prune; stopping."]}
    df = _ctx(state)["df"]
    keep, ranked = prune_features(df, feats, state["target"], state["task"])
    if set(keep) == set(feats):
        return {"last_decision": "finish",
                "log": ["Prune would keep the same columns; stopping."]}
    return {"features": keep, "last_decision": "retry",
            "log": [f"Kept top-{len(keep)} by mutual information: {keep} "
                    f"(dropped {ranked[len(keep):]})" ]}


def node_advise(state):
    results = _ctx(state).get("results")
    if results is None or results.empty:
        _ctx(state)["advice"] = ["No results; no advice available."]
        return {"log": ["No results; skipped feature advice"]}
    df = _ctx(state)["df"]
    used = state.get("features") or []
    unused = [c for c in df.columns if c not in used and c != state["target"]]
    if not unused:
        _ctx(state)["advice"] = ["All columns in use."]
        return {"log": ["All columns in use"]}
    best = results.iloc[0]
    col = "Test R2" if state["task"] == "regression" else "F1"
    score = float(best[col])
    header = f"Unused columns available: {', '.join(unused)}"
    if score >= 0.5:
        lines = [header, f"Score {score:.2f} is acceptable."]
    else:
        summary = (f"best={best['Model']} {col}={score:.3f}, "
                   f"baseline={float(results.iloc[-1][col]):.3f}")
        suggestions = suggest_features(df, state["target"], used, state["task"], summary)
        lines = [header, "The model is weak. Feature advice:"] + \
                ["  + " + s for s in suggestions]
    _ctx(state)["advice"] = lines
    return {"log": ["Computed feature advice"]}


def node_report(state):
    results = _ctx(state).get("results")
    if results is None or results.empty:
        print("\nTraining did not complete; nothing to report.")
        return {"log": ["No results"]}
    extras = _ctx(state)["extras"]
    notes = _ctx(state)["notes"]
    advice = _ctx(state).get("advice") or []
    print("\nPLAN:", state["clean_plan"])
    print("WARNINGS:", state["warnings"] or "none")
    print("\nPIPELINE NOTES:")
    for n in notes: print(" -", n)
    print("\nLEADERBOARD:")
    print(results.to_string(index=False))
    if extras.get("per_class") is not None:
        print("\nPER-CLASS RESULTS:")
        print(extras["per_class"].to_string())
    if extras.get("confusion") is not None:
        print("\nCONFUSION MATRIX:")
        print(extras["confusion"].to_string())
    print("\nSELF-CHECK:")
    for line in extras.get("selfcheck", []): print(" -", line)
    print("\nIN PLAIN WORDS:")
    for line in extras.get("explanations", []): print(" -", line)
    if advice:
        print("\nFEATURE ADVICE:")
        for line in advice: print(" -", line)
    return {"log": ["Printed the full report"]}


def node_export(state):
    results = _ctx(state).get("results")
    if results is None or results.empty:
        return {"log": ["Nothing to export."]}
    try:
        folder = export_all(state["source_path"], state["target"], state["task"],
                             state["clean_plan"], state["warnings"], state["load_notes"],
                             results, _ctx(state)["notes"], _ctx(state)["extras"])
    except Exception as e:
        return {"log": [f"Export failed: {type(e).__name__}: {e}"]}
    _CTX.pop(state.get("run_id", "default"), None)
    return {"output_folder": folder, "log": [f"Saved to {folder}"]}


def build_graph():
    # Clear the global cache so a second CLI run in the same process
    # doesn't pick up the previous run's dataset.
    _CTX.clear()
    
    g = StateGraph(State)
    for name, fn in [("ingest", node_ingest), ("interview", node_interview),
                      ("choose_models", node_choose_models), ("plan", node_plan),
                      ("validate", node_validate), ("human_checkpoint", node_human_checkpoint),
                      ("train", node_train), ("review", node_review),
                      ("prune", node_prune), ("advise", node_advise),
                      ("report", node_report), ("export", node_export)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "interview")
    g.add_edge("interview", "choose_models")
    g.add_edge("choose_models", "plan")
    g.add_edge("plan", "validate")
    g.add_edge("validate", "human_checkpoint")
    g.add_edge("human_checkpoint", "train")
    g.add_edge("train", "review")
    g.add_conditional_edges("review", lambda s: s["last_decision"],
                             {"retry": "prune", "finish": "advise"})
    g.add_conditional_edges("prune", lambda s: s.get("last_decision", "retry"),
                             {"retry": "train", "finish": "advise"})
    g.add_edge("advise", "report")
    g.add_edge("report", "export")
    g.add_edge("export", END)
    return g.compile(checkpointer=MemorySaver())