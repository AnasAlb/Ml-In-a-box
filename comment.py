"""
KIND OF AGENTIC
---------------
Human-in-the-loop. LLM proposes, code disposes. Nothing runs until code
validates it and a human approves. Every decision has a rule-based fallback,
so the pipeline works even if the LLM is down.

WHAT THE AGENT DECIDES
----------------------
It walks a fixed pipeline, but stops at four points to decide:

  1. Target        — picks the column to predict from the profile.
  2. Interactions  — proposes pairs of columns to combine.
  3. Model set     — chooses 3–4 algorithms suited to the data.
  4. Repair        — reviews the leaderboard, accepts or suggests a fix.

EVERYTHING ELSE IS CODE
-----------------------
  - Cleaning plan: LLM drafts, code validates, human approves, then it runs.
  - Training: fixed scikit-learn pipeline. No LLM.
  - Weak score: graph prunes features and retrains (a loop, max 2 rounds).
  - History: every run is logged to outputs/_run_history.json; the next run
    reads it before picking models. The agent "learns" by reading its notes.

WHERE IT LIVES
--------------
  ml.py      recommend_target           pick target column
  ml.py      agent_propose_interactions propose feature combinations
  ml.py      agent_pick_models          pick model set
  ml.py      agent_diagnose             pick repair action
  ml.py      validate_plan              guardrail on the LLM's cleaning plan
  ml.py      _self_check                keep LLM's fix only if CV improves
  ml.py      load_history               feed past runs into the next prompt
  agent.py   build_graph                the state machine (the loop itself)
  agent.py   node_human_checkpoint      the pause before training
  llm.py     ask                        single entry point for every LLM call

WHAT IS NOT AGENTIC
-------------------
Training, feature engineering, the model catalog, the CV page, the NLP tools
page, cleaning and tokenizing. All deterministic. The LLM only touches the
four decisions above.

SHORT VERSION
-------------
Four LLM calls in ml.py make the decisions. agent.py loops them with a human
checkpoint. Guardrails in ml.py let code overrule the LLM at any point.


The agents:

recommend_target — a planner. Looks at the data, says which column to predict.
agent_propose_interactions — a feature engineer. Says which columns to combine.
agent_pick_models — a selector. Picks 3–4 models suited to the data.
agent_diagnose — a critic. Reviews the score, says accept or fix.
That's it. Four small agents, one graph, one human approval.

Human-in-the-loop + LangGraph workflow agent.
Human-in-the-loop = it pauses for approval before training.
LangGraph = the state machine (agent.py) that wires the steps together as nodes and edges
"""