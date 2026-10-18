"""Driver for the LangGraph agent. Run this instead of run_all.py.

This file is intentionally self-contained: it does NOT import run_all,
because run_all has module-level code that runs the whole pipeline on import.
"""
import os
import sys
import uuid

from langgraph.types import Command

from agent import build_graph


def simple_choose_file():
    """Minimal file chooser that never imports run_all."""
    text = input("\nFile name or full path (or press Enter to browse): "
                 ).strip().strip('"').strip("'")

    if text:
        text = os.path.expanduser(text)
        if os.path.isfile(text):
            return text
        # search a few common places
        for base in (os.getcwd(),
                     os.path.expanduser("~/Downloads"),
                     os.path.expanduser("~/Desktop"),
                     os.path.expanduser("~/Documents")):
            cand = os.path.join(base, text)
            if os.path.isfile(cand):
                return cand
        # try to match by stem inside those folders
        stem = os.path.splitext(text)[0].lower()
        for base in (os.getcwd(),
                     os.path.expanduser("~/Downloads"),
                     os.path.expanduser("~/Desktop"),
                     os.path.expanduser("~/Documents")):
            if not os.path.isdir(base):
                continue
            for name in os.listdir(base):
                if os.path.splitext(name)[0].lower() == stem:
                    return os.path.join(base, name)

    # browse
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose your data file",
            filetypes=[("Data files", "*.csv *.xlsx *.xls"), ("All files", "*.*")])
        root.destroy()
        if path:
            return path
    except Exception:
        pass

    sys.exit("No file chosen.")


def print_checkpoint(payload):
    """Plain-English display of the cleaning plan."""
    plan = payload.get("plan") or {}
    print("\n--- HUMAN CHECKPOINT ---")
    print("The agent wants to clean the data like this:")
    print(f"  Drop duplicate rows: {'yes' if plan.get('drop_duplicates') else 'no'}")

    imp = plan.get("impute") or {}
    if imp:
        print("  Fill missing values:")
        for col, method in imp.items():
            print(f"    - {col}: use the {method}")
    else:
        print("  Fill missing values: none")

    out = plan.get("outlier_columns") or []
    if out:
        print("  Cap extreme values in: " + ", ".join(out))
    else:
        print("  Cap extreme values: none")

    if payload.get("warnings"):
        print("  Warnings from the validator:")
        for w in payload["warnings"]:
            print("    ! " + w)


def ask_approval():
    """Loop until the user gives a clear yes or no."""
    while True:
        try:
            ans = input("Approve? (yes/no) > ").strip().lower()
        except KeyboardInterrupt:
            print("\nCancelled by the user.")
            sys.exit(1)
            
        if ans in ("", "y", "yes", "ok", "okay", "approve", "go"):
            return True
        if ans in ("n", "no", "cancel", "abort", "stop"):
            return False
        print("Please type 'yes' or 'no'.")


def main():
    try:
        path = simple_choose_file()
        app = build_graph()

        # Unique per run so state from a previous session cannot leak in.
        run_id = str(uuid.uuid4())[:8]
        config = {"configurable": {"thread_id": run_id}}
        init = {
            "source_path": path,
            "run_id": run_id,
            "max_iterations": 2,
            "log": [],
        }

        # First pass — runs until it finishes or pauses at the human checkpoint.
        app.invoke(init, config=config)

        # Handle human checkpoints (LangGraph interrupt / resume).
        while True:
            snap = app.get_state(config)
            if not snap.next:
                break

            try:
                payload = snap.tasks[0].interrupts[0].value
                print_checkpoint(payload)
            except Exception:
                print("\n--- HUMAN CHECKPOINT (no payload) ---")

            if ask_approval():
                app.invoke(Command(resume="approved"), config=config)
            else:
                print("\nCancelled by the user. Nothing was trained or saved.")
                return 1

        final = app.get_state(config).values

        print("\n--- AGENT TRACE ---")
        for line in final.get("log", []):
            print(" -", line)

        if final.get("output_folder"):
            print("\nOutput folder:", final["output_folder"])
        return 0

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 1


if __name__ == "__main__":
    sys.exit(main())