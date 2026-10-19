"""
Computer Vision page.

Classification: full pipeline — upload a ZIP, pick a backbone (or build a
custom CNN), train, evaluate, predict.

Detection: YOLOv8. Two modes:
  - Zero-shot: pre-trained COCO weights, no training required.
  - Fine-tune: train on your own YOLO-format dataset (images/ + labels/).

Segmentation: LOCKED (Work in Progress).
"""
import os
import copy
import shutil
import uuid
from html import escape as esc

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

import ui
import auth
from i18n import t
from cv import (torch_status, CLASSIFICATION_BACKBONES, DETECTION_BACKBONES,
                 SEGMENTATION_BACKBONES, YOLO_BACKBONES,
                 default_layer_spec, extract_zip_to_folder,
                 build_dataloaders, build_classifier, count_trainable,
                 count_total, train_classifier, evaluate, denormalize,
                 predict_single_image, list_saved_models, save_model_slot,
                 load_model_slot, delete_model_slot, recommend_backbone,
                 predict_yolov8, prepare_yolo_dataset, train_yolov8)


MAX_SAVED = 3   # per user

_LAYER_TYPES = ["conv2d", "maxpool", "batchnorm", "dropout", "dense"]


def _init_state():
    defaults = {
        "cv_task": "classification",
        "cv_extracted_root": None,
        "cv_workdir": None,
        "cv_class_names": None,
        "cv_n_train": 0,
        "cv_n_val": 0,
        "cv_model": None,
        "cv_model_label": None,
        "cv_model_classes": None,
        "cv_input_size": 224,
        "cv_backbone_key": None,
        "cv_custom_layers": None,
        "cv_model_saved": False,
        "cv_loaded_slot": None,
        "cv_history": None,
        "cv_eval": None,
        "cv_layer_spec": default_layer_spec(),
        "cv_upl_n": 0,
        "cv_save_n": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _username():
    u = auth.current_user()
    return (u or {}).get("username", "")


def _clean_layers(spec):
    return [{k: v for k, v in layer.items() if not k.startswith("_")}
            for layer in (spec or [])]


def _idx(options, value, default=0):
    try:
        return options.index(value)
    except ValueError:
        return default


def _backbone_label(key):
    if key == "custom":
        return "Custom CNN"
    return CLASSIFICATION_BACKBONES.get(key, {}).get("name", str(key))


# ============================================================ CLASSIFICATION
def _class_tab():
    status = torch_status()
    if not status["torch"]:
        st.error(status["message"])
        st.code("pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cpu", language=None)
        return

    st.caption(status["message"])

    _train_flow()

    if st.session_state.get("cv_eval") is not None:
        _render_results()
    if st.session_state.get("cv_model") is not None:
        _render_predict()
    _render_saved_models()


def _reset_dataset():
    wd = st.session_state.get("cv_workdir")
    if wd and os.path.isdir(wd):
        shutil.rmtree(wd, ignore_errors=True)
    for k in ("cv_extracted_root", "cv_workdir", "cv_class_names"):
        st.session_state[k] = None
    st.session_state["cv_n_train"] = 0
    st.session_state["cv_n_val"] = 0
    st.session_state["cv_upl_n"] += 1


def _train_flow():
    # ---- 1. dataset
    st.markdown(t("cv_sec_1"))
    st.markdown(
        f'<div class="card">'
        f'<h4>{t("cv_zip_title")}</h4>'
        f'<p>{t("cv_zip_body")}</p>'
        f'<pre style="color:#b6c2d4;font-size:0.85rem;">'
        f'dataset.zip\n'
        f'  ├── cats/\n'
        f'  │     img1.jpg  img2.jpg  …\n'
        f'  ├── dogs/\n'
        f'  │     img1.jpg  img2.jpg  …\n'
        f'  └── birds/\n'
        f'        img1.jpg  …</pre>'
        f'<p>{t("cv_zip_min")}</p>'
        f'</div>', unsafe_allow_html=True)

    if not st.session_state.get("cv_extracted_root"):
        up = st.file_uploader(
            t("cv_zip_upload"), type=["zip"],
            key=f"cv_zip_{st.session_state['cv_upl_n']}",
            label_visibility="collapsed")
        if up is not None:
            workdir = os.path.join("outputs", f"cv_{uuid.uuid4().hex[:6]}")
            try:
                with st.spinner(t("cv_zip_extract")):
                    root = extract_zip_to_folder(up.getvalue(), workdir)
                if root is None:
                    st.error(t("cv_zip_fail"))
                    shutil.rmtree(workdir, ignore_errors=True)
                else:
                    _, _, classes, n_tr, n_val = build_dataloaders(
                        root, batch_size=4)
                    st.session_state["cv_extracted_root"] = root
                    st.session_state["cv_workdir"] = workdir
                    st.session_state["cv_class_names"] = classes
                    st.session_state["cv_n_train"] = n_tr
                    st.session_state["cv_n_val"] = n_val
                    st.rerun()
            except Exception as e:
                shutil.rmtree(workdir, ignore_errors=True)
                st.error(f"Could not use that ZIP: {type(e).__name__}: {e}")
    else:
        classes = st.session_state["cv_class_names"]
        c1, c2, c3 = st.columns(3)
        with c1: ui.stat(len(classes), t("cv_stat_classes"))
        with c2: ui.stat(st.session_state["cv_n_train"], t("cv_stat_train"))
        with c3: ui.stat(st.session_state["cv_n_val"], t("cv_stat_val"))
        st.caption(t("cv_classes_prefix") + ", ".join(classes))
        if st.button(t("cv_diff_dataset")):
            _reset_dataset()
            st.rerun()

    root = st.session_state.get("cv_extracted_root")
    if not root:
        return

    # ---- 2. architecture
    st.divider()
    st.markdown(t("cv_sec_2"))
    arch_options = list(CLASSIFICATION_BACKBONES.keys()) + ["custom"]
    labels = {k: CLASSIFICATION_BACKBONES[k]["name"]
              for k in CLASSIFICATION_BACKBONES}
    labels["custom"] = "🛠 Custom CNN (build your own)"

    rec = recommend_backbone(st.session_state.get("cv_n_train") or 0,
                             len(st.session_state.get("cv_class_names") or []))
    st.info(t("cv_suggested_backbone").format(name=rec["name"],
                                                reason=rec["reason"]))
    default_idx = _idx(arch_options, rec["key"], 0)
    pick = st.radio(t("cv_backbone"), arch_options,
                    format_func=lambda k: labels[k],
                    index=default_idx,
                    key="cv_arch_pick")

    if pick != "custom":
        info = CLASSIFICATION_BACKBONES[pick]
        st.markdown(
            f'<div class="card" style="border-left:3px solid #22d3ee;">'
            f'<h4>{esc(info["name"])} '
            f'<span class="pill">{esc(info["params"])} params</span></h4>'
            f'<p>{esc(info["blurb"])}</p></div>',
            unsafe_allow_html=True)

    custom_layers = None
    if pick == "custom":
        st.caption(t("cv_custom_caption"))
        custom_layers = _layer_builder()

    # ---- 3. training config
    st.divider()
    st.markdown(t("cv_sec_3"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        epochs = st.slider(t("cv_epochs"), 1, 30, 5, key="cv_epochs")
    with c2:
        lr = st.select_slider(t("cv_lr"),
                              options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
                              value=1e-3, key="cv_lr")
    with c3:
        bs = st.slider(t("cv_bs"), 4, 64, 16, key="cv_bs")
    with c4:
        freeze = st.checkbox(t("cv_freeze"),
                             value=True, key="cv_freeze",
                             help=t("cv_freeze_help"))

    # ---- 4. build model + train
    st.divider()
    st.markdown(t("cv_sec_4"))
    if st.button(t("cv_start_btn"), type="primary"):
        _run_training(pick, custom_layers, epochs, lr, bs, freeze, root)


def _layer_builder():
    if not st.session_state.get("cv_layer_spec"):
        st.session_state["cv_layer_spec"] = default_layer_spec()
    spec = st.session_state["cv_layer_spec"]
    for layer in spec:
        layer.setdefault("_id", uuid.uuid4().hex[:6])

    st.markdown("**Layers (top of list = first applied to the image)**")
    to_remove = None
    for i, layer in enumerate(spec):
        lid = layer["_id"]
        cols = st.columns([3, 5, 1])
        with cols[0]:
            kind = st.selectbox(
                f"Layer {i+1} type", _LAYER_TYPES,
                index=_idx(_LAYER_TYPES, layer.get("type"), 0),
                key=f"ly_type_{lid}", label_visibility="collapsed")
            layer["type"] = kind
        with cols[1]:
            if kind == "conv2d":
                layer["out_channels"] = int(st.number_input(
                    "Out channels", 4, 512,
                    int(layer.get("out_channels", 32)), step=4,
                    key=f"ly_oc_{lid}", label_visibility="collapsed"))
                ks = [1, 3, 5, 7]
                layer["kernel"] = st.selectbox(
                    "Kernel", ks,
                    index=_idx(ks, int(layer.get("kernel", 3)), 1),
                    key=f"ly_k_{lid}", label_visibility="collapsed")
            elif kind == "maxpool":
                ks = [2, 3, 4]
                layer["kernel"] = st.selectbox(
                    "Pool kernel", ks,
                    index=_idx(ks, int(layer.get("kernel", 2)), 0),
                    key=f"ly_pk_{lid}", label_visibility="collapsed")
            elif kind == "dropout":
                layer["p"] = st.slider("Drop probability", 0.0, 0.7,
                                        float(layer.get("p", 0.3)), 0.05,
                                        key=f"ly_p_{lid}",
                                        label_visibility="collapsed")
            elif kind == "dense":
                layer["units"] = int(st.number_input(
                    "Units", 4, 2048,
                    int(layer.get("units", 128)), step=4,
                    key=f"ly_u_{lid}", label_visibility="collapsed"))
            else:
                st.caption("BatchNorm — no parameters")
        with cols[2]:
            if st.button("✕", key=f"ly_rm_{lid}"):
                to_remove = i
    if to_remove is not None:
        spec.pop(to_remove)
        st.session_state["cv_layer_spec"] = spec
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("＋ Add layer"):
            spec.append({"type": "conv2d", "out_channels": 64, "kernel": 3,
                         "_id": uuid.uuid4().hex[:6]})
            st.session_state["cv_layer_spec"] = spec
            st.rerun()
    with c2:
        if st.button("↺ Reset to default"):
            st.session_state["cv_layer_spec"] = default_layer_spec()
            st.rerun()

    n_conv = sum(1 for l in spec if l["type"] == "conv2d")
    if n_conv == 0:
        st.warning("No conv2d layer — the network will only see the average "
                   "colour of each image. Add at least one conv layer.")
    st.caption(f"{len(spec)} layers. The final classifier layer is added "
               f"automatically for your number of classes.")
    return _clean_layers(spec)


def _run_training(backbone_key, custom_layers, epochs, lr, bs, freeze, root):
    classes = st.session_state["cv_class_names"]
    n_classes = len(classes)
    device = _device()

    with st.spinner("Preparing data…"):
        try:
            tr_loader, va_loader, class_names, n_tr, n_val = \
                build_dataloaders(root, batch_size=bs,
                                   input_size=CLASSIFICATION_BACKBONES
                                   .get(backbone_key, {}).get("input_size", 224))
        except Exception as e:
            st.error(f"Could not load data: {type(e).__name__}: {e}")
            return

    try:
        model, input_size, label = build_classifier(
            backbone_key, n_classes, freeze_backbone=freeze,
            custom_layers=custom_layers)
    except Exception as e:
        st.error(f"Could not build model: {type(e).__name__}: {e}")
        return

    trainable = count_trainable(model)
    total = count_total(model)
    st.info(f"{label} · {total:,} parameters, {trainable:,} trainable "
            f"(backbone {'frozen' if freeze and backbone_key != 'custom' else 'active'})")

    progress = st.progress(0, text=t("cv_training"))
    log_box = st.empty()
    chart_slot = st.empty()

    live = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}

    def _plot_live():
        if not live["epoch"]:
            return
        df = pd.DataFrame(live)
        fig = px.line(df, x="epoch", y=["train_loss", "val_loss"],
                      color_discrete_sequence=["#22d3ee", "#e879f9"])
        fig.update_layout(height=280, margin=dict(t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#b6c2d4"),
                          legend_title_text="",
                          xaxis_title="epoch", yaxis_title="loss")
        chart_slot.plotly_chart(fig, use_container_width=True,
                                 key=f"live_chart_{len(live['epoch'])}")

    def _cb(epoch, total_epochs, train_loss, val_loss, val_acc):
        progress.progress(int(100 * epoch / total_epochs),
                          text=t("cv_progress_epoch").format(e=epoch, n=total_epochs))
        log_box.caption(t("cv_progress_line").format(
            e=epoch, tl=train_loss, vl=val_loss, va=val_acc))
        live["epoch"].append(epoch)
        live["train_loss"].append(float(train_loss))
        live["val_loss"].append(float(val_loss))
        live["val_acc"].append(float(val_acc))
        _plot_live()

    try:
        history = train_classifier(model, tr_loader, va_loader, device,
                                    epochs=epochs, lr=lr, progress_cb=_cb)
    except Exception as e:
        st.error(f"Training failed: {type(e).__name__}: {e}")
        return

    progress.progress(100, text=t("cv_evaluating"))
    try:
        acc, cm, per_class, samples = evaluate(model, va_loader, device,
                                                class_names)
    except Exception as e:
        st.error(f"Evaluation failed: {type(e).__name__}: {e}")
        return

    ss = st.session_state
    ss["cv_model"] = model
    ss["cv_model_label"] = label
    ss["cv_model_classes"] = list(class_names)
    ss["cv_input_size"] = input_size
    ss["cv_backbone_key"] = backbone_key
    ss["cv_custom_layers"] = copy.deepcopy(custom_layers) \
        if backbone_key == "custom" else None
    ss["cv_model_saved"] = False
    ss["cv_loaded_slot"] = None
    ss["cv_history"] = history
    ss["cv_eval"] = {
        "acc": acc, "cm": cm, "per_class": per_class,
        "samples": samples, "n_classes": n_classes,
        "class_names": class_names,
    }
    st.rerun()


def _plot_layout(fig, height=300):
    fig.update_layout(height=height, margin=dict(t=10, b=10),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#b6c2d4"))
    return fig


def _render_results():
    ev = st.session_state["cv_eval"]
    history = st.session_state["cv_history"]
    label = st.session_state["cv_model_label"]

    st.divider()
    st.markdown(t("cv_sec_5"))
    c1, c2, c3 = st.columns(3)
    with c1: ui.stat(f"{ev['acc']*100:.1f}%", t("cv_val_acc"))
    with c2: ui.stat(label, t("cv_architecture"))
    with c3: ui.stat(ev["n_classes"], t("cv_stat_classes"))

    if history:
        df = pd.DataFrame({
            "epoch": list(range(1, len(history["train_loss"]) + 1)),
            "train loss": history["train_loss"],
            "val loss": history["val_loss"],
            "val accuracy": history["val_acc"],
        })
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(df, x="epoch", y=["train loss", "val loss"],
                          color_discrete_sequence=["#22d3ee", "#e879f9"])
            _plot_layout(fig).update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(df, x="epoch", y="val accuracy",
                          color_discrete_sequence=["#22d3ee"])
            st.plotly_chart(_plot_layout(fig), use_container_width=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(t("cv_confusion"))
        cm = ev["cm"]
        fig = px.imshow(cm.values,
                        x=[c.replace("Predicted ", "", 1) for c in cm.columns],
                        y=[r.replace("Actual ", "", 1) for r in cm.index],
                        color_continuous_scale=[[0, "#131a2e"], [1, "#22d3ee"]],
                        aspect="auto", text_auto=True)
        _plot_layout(fig, height=max(320, 32 * len(cm)))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(t("cv_per_class"))
        st.dataframe(ev["per_class"], use_container_width=True)

    st.markdown(t("cv_sample_preds"))
    samples = ev["samples"]
    class_names = ev["class_names"]
    if samples:
        correct = [s for s in samples if s[1] == s[2]][:4]
        wrong = [s for s in samples if s[1] != s[2]][:4]
        pool = correct + wrong
        if not pool:
            pool = samples[:min(8, len(samples))]
        cols = st.columns(min(4, len(pool)))
        for i, (tensor, true_i, pred_i, conf) in enumerate(pool):
            with cols[i % len(cols)]:
                st.image(denormalize(tensor), use_container_width=True)
                ok = "✅" if true_i == pred_i else "❌"
                st.markdown(
                    f'<div style="text-align:center;font-size:0.82rem;'
                    f'color:#b6c2d4;">{ok} true: <b>{esc(class_names[true_i])}</b><br>'
                    f'pred: <b>{esc(class_names[pred_i])}</b> · {conf*100:.0f}%</div>',
                    unsafe_allow_html=True)


def _render_predict():
    classes = (st.session_state.get("cv_model_classes")
               or st.session_state.get("cv_class_names") or [])
    st.divider()
    st.markdown(t("cv_predict_title"))
    caption = t("cv_predict_using").format(
        m=st.session_state.get('cv_model_label') or 'current model')
    if st.session_state.get("cv_loaded_slot"):
        caption += t("cv_predict_loaded")
    st.caption(caption)

    input_mode = st.radio("Input Method",
                          ["📤 Upload Image", "📸 Live Webcam"], horizontal=True)

    pil = None
    if input_mode == "📤 Upload Image":
        up = st.file_uploader("Upload a single image",
                              type=["png", "jpg", "jpeg", "webp", "bmp"],
                              key="cv_single")
        if up:
            try:
                pil = Image.open(up).convert("RGB")
            except Exception as e:
                st.error(f"Could not open image: {type(e).__name__}: {e}")
    else:
        camera_img = st.camera_input("Take a picture")
        if camera_img:
            try:
                pil = Image.open(camera_img).convert("RGB")
            except Exception as e:
                st.error(f"Could not open camera image: {type(e).__name__}: {e}")

    if pil is None:
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(pil, use_container_width=True)
    with c2:
        try:
            label, conf, probs = predict_single_image(
                st.session_state["cv_model"], pil, classes, _device(),
                input_size=st.session_state.get("cv_input_size", 224))
        except Exception as e:
            st.error(t("cv_predict_fail").format(
                err=f"{type(e).__name__}: {e}"))
            return
        st.markdown(
            f'<div class="card" style="border-left:3px solid #fbbf24;">'
            f'<h4>{t("cv_predict_label")}</h4>'
            f'<p style="font-size:1.5rem;font-weight:700;color:#e8eef7;'
            f'margin:0.3rem 0;">{esc(label)}</p>'
            f'<p style="color:#94a3b8;margin:0;">'
            f'{t("cv_predict_conf").format(c=conf*100)}</p></div>',
            unsafe_allow_html=True)
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
        for cls, p in top:
            st.progress(min(max(p, 0.0), 1.0), text=f"{cls}: {p*100:.1f}%")


# ------------------------------------------------------------ saved models
def _render_saved_models():
    ss = st.session_state
    user = _username()
    slots = list_saved_models(owner=user)
    full = len(slots) >= MAX_SAVED

    st.divider()
    st.markdown(t("cv_saved_title"))
    st.caption(t("cv_saved_caption").format(n=MAX_SAVED))

    flash = ss.pop("cv_flash", None)
    if flash:
        st.success(flash)

    can_save = (ss.get("cv_model") is not None
                and ss.get("cv_eval") is not None
                and ss.get("cv_backbone_key"))
    if can_save:
        if ss.get("cv_model_saved"):
            st.info(t("cv_saved_already"))
        elif full:
            st.warning(t("cv_saved_full").format(n=MAX_SAVED))
        else:
            c1, c2 = st.columns([3, 1])
            with c1:
                name = st.text_input(
                    t("cv_saved_name_label"),
                    key=f"cv_save_name_{ss['cv_save_n']}",
                    placeholder=ss.get("cv_model_label") or t("cv_saved_name_placeholder"),
                    max_chars=40)
            with c2:
                st.markdown("<div style='height:1.7rem'></div>",
                            unsafe_allow_html=True)
                if st.button(t("cv_saved_save_btn").format(
                        cur=len(slots), max=MAX_SAVED),
                             type="primary", use_container_width=True):
                    ev = ss["cv_eval"]
                    try:
                        save_model_slot(
                            ss["cv_model"],
                            (name or "").strip() or ss["cv_model_label"],
                            ss["cv_backbone_key"],
                            _clean_layers(ss.get("cv_custom_layers")),
                            ev["class_names"],
                            ss.get("cv_input_size", 224),
                            ev["acc"], ev["n_classes"], owner=user)
                    except Exception as e:
                        st.error(f"Could not save: {type(e).__name__}: {e}")
                    else:
                        ss["cv_model_saved"] = True
                        ss["cv_save_n"] += 1
                        ss["cv_flash"] = t("cv_saved_confirm_flash")
                        st.rerun()
    elif not slots:
        st.caption(t("cv_saved_train_first"))

    if not slots:
        return

    for s in slots:
        sid = s["id"]
        in_use = ss.get("cv_loaded_slot") == sid
        classes = s.get("class_names") or []
        shown = ", ".join(classes[:6]) + (" …" if len(classes) > 6 else "")
        saved_at = str(s.get("saved_at", "")).replace("T", " ")
        c1, c2, c3 = st.columns([6, 1.2, 1.2])
        with c1:
            border = "#22d3ee" if in_use else "#2a3350"
            badge = f' <span class="pill">{t("cv_saved_in_use")}</span>' if in_use else ""
            st.markdown(
                f'<div class="card" style="padding:0.8rem 1rem;'
                f'border-left:3px solid {border};">'
                f'<b>{esc(str(s.get("label", sid)))}</b>{badge}<br>'
                f'<span style="color:#94a3b8;font-size:0.85rem;">'
                f'{esc(_backbone_label(s.get("backbone_key")))} · '
                f'{t("cv_saved_val_acc").format(p=float(s.get("accuracy", 0))*100)} · '
                f'{len(classes)} · ({esc(shown)})<br>'
                f'{t("cv_saved_saved_at").format(when=esc(saved_at))} · '
                f'{t("cv_saved_size").format(mb=s.get("size_mb", 0))}</span>'
                f'</div>', unsafe_allow_html=True)
        with c2:
            if st.button(t("cv_saved_load"), key=f"cv_load_{sid}",
                         disabled=in_use, use_container_width=True):
                try:
                    with st.spinner(t("cv_saved_loading")):
                        model, meta = load_model_slot(sid)
                except Exception as e:
                    st.error(f"Could not load: {type(e).__name__}: {e}")
                else:
                    ss["cv_model"] = model
                    ss["cv_model_label"] = meta.get("label", sid)
                    ss["cv_model_classes"] = list(meta.get("class_names", []))
                    ss["cv_input_size"] = int(meta.get("input_size", 224))
                    ss["cv_backbone_key"] = meta.get("backbone_key")
                    ss["cv_custom_layers"] = meta.get("custom_layers")
                    ss["cv_model_saved"] = True
                    ss["cv_loaded_slot"] = sid
                    ss["cv_eval"] = None
                    ss["cv_history"] = None
                    ss["cv_flash"] = t("cv_saved_load_flash").format(
                        m=meta.get("label", sid))
                    st.rerun()
        with c3:
            if ss.get("cv_confirm_del") == sid:
                if st.button(t("cv_saved_confirm"), key=f"cv_cdel_{sid}",
                             type="primary", use_container_width=True):
                    delete_model_slot(sid)
                    ss.pop("cv_confirm_del", None)
                    if ss.get("cv_loaded_slot") == sid:
                        ss["cv_loaded_slot"] = None
                        ss["cv_model_saved"] = False
                    st.rerun()
            else:
                if st.button(t("cv_saved_delete"), key=f"cv_del_{sid}",
                             use_container_width=True):
                    ss["cv_confirm_del"] = sid
                    st.rerun()
        if ss.get("cv_confirm_del") == sid:
            st.warning(t("cv_saved_confirm_warn"))
            if st.button(t("cv_saved_cancel"), key=f"cv_cancel_{sid}"):
                ss.pop("cv_confirm_del", None)
                st.rerun()


# ============================================================ DETECTION (YOLOV8)
def _detection_tab():
    st.markdown("### 🎯 Object Detection (YOLOv8)")
    
    tab_zs, tab_ft = st.tabs(["⚡ Zero-Shot Detection", "🎓 Fine-Tune on Custom Dataset"])
    
    with tab_zs:
        _detection_zero_shot()
    with tab_ft:
        _detection_finetune()


def _detection_zero_shot():
    st.caption(
        "Zero-shot mode uses pre-trained COCO weights (80 common classes). "
        "No training required."
    )
    up = st.file_uploader(
        "Upload an image for detection",
        type=["png", "jpg", "jpeg", "webp"], key="det_zs_img")
    if up is None:
        return
    try:
        pil = Image.open(up).convert("RGB")
    except Exception as e:
        st.error(f"Could not open image: {type(e).__name__}: {e}")
        return

    try:
        with st.spinner("Running YOLOv8 inference..."):
            annotated_img, df = predict_yolov8(pil)
    except ImportError as e:
        st.error(f"Missing dependency: {e}")
        return
    except Exception as e:
        st.error(f"Detection failed: {type(e).__name__}: {e}")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.image(pil, caption="Original", use_container_width=True)
    with c2:
        st.image(annotated_img, caption="YOLOv8 detections", use_container_width=True)

    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No objects detected in this image.")


def _detection_finetune():
    ss = st.session_state

    st.markdown("**Step 1 — Upload a YOLO-format dataset (.zip)**")
    st.caption(
        "Expect `images/train/`, `images/val/`, `labels/train/`, "
        "`labels/val/` inside the zip. Missing `val/` falls back to train. "
        "Labels must be YOLO `.txt` (class cx cy w h). See "
        "https://docs.ultralytics.com/datasets/detect/ for the exact format.")

    up = st.file_uploader(
        "YOLO dataset (.zip)", type=["zip"], key="det_ft_zip")

    if up is not None:
        sig = (up.name, getattr(up, "size", 0))
        if ss.get("_yolo_zip_sig") != sig:
            workdir = os.path.join("outputs", f"yolo_ds_{uuid.uuid4().hex[:6]}")
            try:
                with st.spinner("Extracting and inspecting dataset..."):
                    yaml_path, names = prepare_yolo_dataset(up.getvalue(), workdir)
            except Exception as e:
                shutil.rmtree(workdir, ignore_errors=True)
                st.error(f"Could not prepare dataset: {type(e).__name__}: {e}")
                return
            ss["_yolo_zip_sig"] = sig
            ss["_yolo_workdir"] = workdir
            ss["_yolo_yaml"] = yaml_path
            ss["_yolo_names"] = names
            ss["_yolo_weights"] = None
            st.success(
                f"Dataset ready · {len(names)} classes: "
                f"{', '.join(names.values())}")

    yaml_path = ss.get("_yolo_yaml")
    if not yaml_path:
        return

    st.markdown("**Step 2 — Training configuration**")
    c1, c2 = st.columns(2)
    with c1:
        base = st.selectbox(
            "Base model",
            list(YOLO_BACKBONES.keys()),
            format_func=lambda k: (
                f"{YOLO_BACKBONES[k]['name']} — "
                f"{YOLO_BACKBONES[k]['params']} · {YOLO_BACKBONES[k]['speed']}"),
            key="det_ft_base")
    with c2:
        epochs = st.slider("Epochs", 5, 200, 10, 5, key="det_ft_epochs")

    c1, c2 = st.columns(2)
    with c1:
        imgsz = st.select_slider(
            "Image size",
            options=[320, 416, 512, 640, 768], value=640,
            key="det_ft_imgsz")
    with c2:
        batch = st.slider("Batch size", 4, 64, 16, 4, key="det_ft_batch")

    st.warning(
        "⚠️ **CPU Training Warning:** YOLOv8 training on Streamlit Cloud (CPU only) is very slow. "
        "Start with a small number of epochs (5-10) to make sure it works. "
        "If it crashes with out-of-memory, reduce the batch size."
    )

    st.markdown("**Step 3 — Train**")
    if st.button("🚀 Fine-tune YOLOv8", type="primary", key="det_ft_train_btn"):
        run_name = f"yolo_run_{uuid.uuid4().hex[:6]}"
        project_dir = os.path.join("outputs", "yolo_runs")

        status = st.status("Training YOLOv8...", expanded=True)
        progress = st.progress(0, text="Starting...")
        epoch_log = st.empty()

        def _cb(epoch, epochs, metrics):
            frac = min(1.0, epoch / max(epochs, 1))
            progress.progress(frac, text=f"Epoch {epoch}/{epochs}")
            try:
                epoch_log.caption(f"Epoch {epoch}/{epochs} — {metrics}")
            except Exception:
                pass

        try:
            best_weights, _ = train_yolov8(
                data_yaml=yaml_path,
                base_model=YOLO_BACKBONES[base]["weights"],
                epochs=int(epochs),
                imgsz=int(imgsz),
                batch=int(batch),
                project_dir=project_dir,
                run_name=run_name,
                workers=0,          # safest on Windows
                progress_cb=_cb,
            )
            status.update(label="Training complete ✅", state="complete")
            ss["_yolo_weights"] = best_weights
            st.success(f"Model trained → {best_weights}")
        except Exception as e:
            status.update(label="Training failed ❌", state="error")
            st.error(f"Training failed: {type(e).__name__}: {e}")
            return

    weights = ss.get("_yolo_weights")
    if not weights:
        return
    st.markdown("**Step 4 — Test the fine-tuned model**")
    st.caption(f"Using weights: `{weights}`")
    test_up = st.file_uploader(
        "Upload a test image",
        type=["png", "jpg", "jpeg", "webp"], key="det_ft_test_img")
    if test_up is not None:
        try:
            pil = Image.open(test_up).convert("RGB")
        except Exception as e:
            st.error(f"Could not open image: {type(e).__name__}: {e}")
            return
        try:
            with st.spinner("Running inference..."):
                annotated, df = predict_yolov8(pil, weights_path=weights)
        except Exception as e:
            st.error(f"Inference failed: {type(e).__name__}: {e}")
            return
        c1, c2 = st.columns(2)
        with c1:
            st.image(pil, caption="Original", use_container_width=True)
        with c2:
            st.image(annotated, caption="Fine-tuned detections",
                     use_container_width=True)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No objects detected. Try a different image, or train "
                    "for more epochs.")


# ============================================================ SEGMENTATION (LOCKED)
def _segmentation_tab():
    st.markdown(
        f'<div class="card" style="border-left:3px solid #fbbf24;'
        f'background:linear-gradient(180deg,rgba(251,191,36,0.08),'
        f'rgba(251,191,36,0.02));">'
        f'<h4 style="color:#fcd34d;">{t("cv_seg_wip_title")}</h4>'
        f'<p>{t("cv_seg_wip_body")}</p>'
        f'<p>{t("cv_seg_wip_body2")}</p>'
        f'<p style="color:#94a3b8;font-size:0.85rem;margin-top:0.6rem;">'
        f'{t("cv_seg_wip_foot")}</p>'
        f'</div>', unsafe_allow_html=True)

    st.markdown(t("cv_seg_arch_title"))
    for key, info in SEGMENTATION_BACKBONES.items():
        st.markdown(
            f'<div class="card" style="opacity:0.55;">'
            f'<h4>🔒 {esc(info["name"])}</h4>'
            f'<p>{esc(info["blurb"])}</p></div>',
            unsafe_allow_html=True)

    st.info(t("cv_seg_locked"))


# ============================================================ MAIN
def render_cv():
    _init_state()

    if not auth.is_logged_in():
        st.error("Sign in required. The CV page needs an authenticated session.")
        return
    if not auth.can_use("cv"):
        st.error("Your account does not have access to Computer Vision.")
        return

    tab1, tab2, tab3 = st.tabs([t("cv_tab_class"), t("cv_tab_detect"),
                                 t("cv_tab_seg")])
    with tab1:
        _class_tab()
    with tab2:
        _detection_tab()
    with tab3:
        _segmentation_tab()