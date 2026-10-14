import datetime
import os
import re
import joblib


def _safe(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "run"


def export_all(source_path, target, task, plan, warnings_list, load_notes,
               results, notes, extras):
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
    """Build a formatted PDF report into `folder`. Returns the PDF path,
    or None if reportlab is not installed."""
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
        author="Anas Albakoush, Muad Himeedah",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=20,
                        textColor=colors.HexColor("#0a2540"), spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                        textColor=colors.HexColor("#0a2540"),
                        spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10,
                          leading=14)
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


def export_model_card(folder, target, task, results, extras, load_notes):
    """Write a one-page model card as model_card.pdf into `folder`.
    Returns the path, or None if reportlab is missing."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle)
    except ImportError:
        return None

    path = os.path.join(folder, "model_card.pdf")
    doc = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=f"Model Card — {target}",
        author="ML in a Box")

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=20,
                        textColor=colors.HexColor("#0a2540"), spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                        textColor=colors.HexColor("#0a2540"),
                        spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["BodyText"],
                          fontSize=10, leading=14)
    warn = ParagraphStyle("warn", parent=body,
                          textColor=colors.HexColor("#7a1f1f"),
                          backColor=colors.HexColor("#fdeaea"),
                          borderPadding=6)

    best = results.iloc[0]
    task_col = "Test R2" if task == "regression" else "F1"
    score = float(best[task_col])

    story = []
    story.append(Paragraph("Model Card", h1))
    story.append(Paragraph(
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M}", body))
    story.append(Spacer(1, 10))

    story.append(Paragraph("What this model does", h2))
    story.append(Paragraph(
        f"Predicts <b>{target}</b> as a <b>{task}</b> task, "
        f"using {len(extras.get('features', []))} input columns "
        f"and the <b>{best['Model']}</b> algorithm.", body))

    story.append(Paragraph("Training data", h2))
    rows = [
        ["Training rows", str(extras.get("n_train", "?"))],
        ["Test rows", str(extras.get("n_test", "?"))],
        ["Input columns", ", ".join(extras.get("features", []))],
        ["Target column", str(target)],
        ["Task type", str(task)],
        ["Load notes", ", ".join(load_notes) if load_notes else "none"],
    ]
    t = Table(rows, colWidths=[4.2 * cm, 12.3 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f8")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5d1de")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t)

    story.append(Paragraph("Performance", h2))
    if task == "regression":
        story.append(Paragraph(
            f"Best model: <b>{best['Model']}</b>. "
            f"R² = {best['Test R2']}, MAE = {best['Test MAE']}, "
            f"RMSE = {best['Test RMSE']}.", body))
    else:
        story.append(Paragraph(
            f"Best model: <b>{best['Model']}</b>. "
            f"Accuracy = {best['Accuracy']}, F1 = {best['F1']}, "
            f"Precision = {best['Precision']}, Recall = {best['Recall']}.", body))

    story.append(Paragraph("Limitations and warnings", h2))
    lines = [
        "• This model was evaluated only on the data you supplied. "
        "It will not generalise to very different populations.",
        "• The feature set is fixed. Any input missing these columns "
        "will fail or produce meaningless output.",
    ]
    if task == "classification":
        base = results.iloc[-1]
        if float(base.get("Accuracy", 0)) >= 0.7:
            lines.append("• The dataset is imbalanced. Accuracy is misleading "
                         "here; check per-class F1 before trusting the model.")
    if score < 0.5:
        lines.append("• Overall performance is weak. Do not deploy this model "
                     "for real decisions without more data or better features.")
    lines.append("• The pipeline is a research prototype. It has not been "
                 "audited for fairness, privacy, or adversarial robustness.")
    for l in lines:
        story.append(Paragraph(l, warn if "Do not deploy" in l else body))

    story.append(Paragraph("Intended use", h2))
    story.append(Paragraph(
        "Educational demonstrations and internal exploration only. "
        "Not intended for decisions affecting people, health, finance, "
        "safety, or any regulated domain.", body))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Engr. Anas Yousef Albakoush &nbsp;·&nbsp; Engr. Muad Salah Himeedah "
        "&nbsp;·&nbsp; ML in a Box", body))

    doc.build(story)
    return path