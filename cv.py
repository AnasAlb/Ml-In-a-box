"""
Computer Vision engine.

Provides:
  - torch_status()
  - CLASSIFICATION_BACKBONES / DETECTION_BACKBONES / SEGMENTATION_BACKBONES
  - default_layer_spec()
  - extract_zip_to_folder()
  - build_dataloaders()
  - build_classifier() / count_trainable() / count_total()
  - train_classifier() / evaluate()
  - denormalize() / predict_single_image()
  - Persistent save slots:
      list_saved_models() / save_model_slot() /
      load_model_slot() / delete_model_slot()

Detection and segmentation are catalog-only (no training loop) — that
matches the page's stated scope.
"""
import io
import os
import re
import json
import zipfile
import datetime

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, random_split
    import torchvision.transforms as T
    import torchvision.models as tvm
    TORCH = True
except ImportError:
    TORCH = False


# ============================================================ 1. STATUS
def torch_status():
    if not TORCH:
        return {
            "torch": False, "cuda": False, "device_name": None,
            "device_count": 0, "torch_version": None,
            "message": ("PyTorch is not installed. Install it with "
                        "`pip install torch torchvision`."),
        }
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return {
            "torch": True, "cuda": True, "device_name": name,
            "device_count": torch.cuda.device_count(),
            "torch_version": torch.__version__,
            "message": f"GPU detected: {name} "
                       f"({torch.cuda.device_count()} device(s)).",
        }
    return {
        "torch": True, "cuda": False, "device_name": None,
        "device_count": 0, "torch_version": torch.__version__,
        "message": ("PyTorch installed, no CUDA GPU. Training runs on CPU."),
    }


# ============================================================ 2. CATALOGS
CLASSIFICATION_BACKBONES = {
    "resnet18": {
        "name": "ResNet-18", "params": "11.7M", "input_size": 224,
        "blurb": "Compact residual network. Strong accuracy, fast to train — "
                 "a great default.",
        "ctor": lambda: tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT),
    },
    "resnet50": {
        "name": "ResNet-50", "params": "25.6M", "input_size": 224,
        "blurb": "Deeper ResNet. More accurate than ResNet-18, roughly 3x slower.",
        "ctor": lambda: tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT),
    },
    "mobilenet": {
        "name": "MobileNet-V3 Small", "params": "2.5M", "input_size": 224,
        "blurb": "Very small and fast. Good when data or compute are limited.",
        "ctor": lambda: tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.DEFAULT),
    },
    "efficientnet": {
        "name": "EfficientNet-B0", "params": "5.3M", "input_size": 224,
        "blurb": "Balanced accuracy per FLOP. Often the best small backbone.",
        "ctor": lambda: tvm.efficientnet_b0(
            weights=tvm.EfficientNet_B0_Weights.DEFAULT),
    },
    "vgg16": {
        "name": "VGG-16", "params": "138M", "input_size": 224,
        "blurb": "Classic deep CNN. Heavy and slow — use only if you need it.",
        "ctor": lambda: tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT),
    },
}

DETECTION_BACKBONES = {
    "fasterrcnn": {"name": "Faster R-CNN",
                   "blurb": "Two-stage detector. Accurate, slower at inference."},
    "retinanet":  {"name": "RetinaNet",
                   "blurb": "One-stage detector with focal loss for class imbalance."},
    "ssd":        {"name": "SSD",
                   "blurb": "Single-shot multi-scale detector. Fast, lighter accuracy."},
    "yolo":       {"name": "YOLO family",
                   "blurb": "Real-time one-stage detector. Excellent speed/accuracy."},
    "detr":       {"name": "DETR",
                   "blurb": "Transformer detector. No anchors, no NMS."},
}

SEGMENTATION_BACKBONES = {
    "unet":      {"name": "U-Net",
                  "blurb": "Encoder-decoder with skip connections. Strong on small masks."},
    "deeplabv3": {"name": "DeepLabV3",
                  "blurb": "Atrous convolution for dense prediction. Classic baseline."},
    "maskrcnn":  {"name": "Mask R-CNN",
                  "blurb": "Instance segmentation built on Faster R-CNN."},
    "segformer": {"name": "SegFormer",
                  "blurb": "Transformer-based segmentation. Efficient and accurate."},
    "sam":       {"name": "SAM",
                  "blurb": "Promptable Segment-Anything. Zero-shot, needs prompts."},
}


def default_layer_spec():
    """Reasonable starter stack for the custom CNN builder."""
    return [
        {"type": "conv2d",  "out_channels": 32, "kernel": 3},
        {"type": "maxpool", "kernel": 2},
        {"type": "conv2d",  "out_channels": 64, "kernel": 3},
        {"type": "maxpool", "kernel": 2},
        {"type": "dropout", "p": 0.3},
        {"type": "dense",   "units": 128},
    ]


# ============================================================ 3. DATASET
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")


class _ImageFolderDataset(Dataset):
    """Flat directory-of-classes dataset. Paths are resolved at __init__."""

    def __init__(self, root, classes, transform):
        self.transform = transform
        self.samples = []
        for idx, cls in enumerate(classes):
            d = os.path.join(root, cls)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith(_IMG_EXTS):
                    self.samples.append((os.path.join(d, fn), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def extract_zip_to_folder(zip_bytes, dest):
    """Extract a dataset zip and return the folder that holds one
    subdirectory per class. Returns None if no such folder is found."""
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(dest)

    cur = dest
    for _ in range(6):  # descend a few levels if the zip is nested
        children = [c for c in os.listdir(cur)
                    if os.path.isdir(os.path.join(cur, c))
                    and not c.startswith("__")]
        if len(children) >= 2:
            return cur
        if len(children) == 1:
            cur = os.path.join(cur, children[0])
        else:
            return None
    return None


def _build_transforms(input_size):
    train_tf = T.Compose([
        T.Resize((int(input_size * 1.15), int(input_size * 1.15))),
        T.RandomCrop(input_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def recommend_backbone(n_train, n_classes):
    """Pick a classification backbone from dataset size — no LLM required."""
    if n_train < 80:
        key = "mobilenet"
        reason = (f"{n_train} training images is small, so MobileNet-V3 is less "
                  "likely to memorize the set.")
    elif n_train < 400:
        key = "resnet18"
        reason = (f"{n_train} images is a typical transfer-learning size; ResNet-18 "
                  "is a strong default.")
    elif n_classes >= 8:
        key = "efficientnet"
        reason = (f"{n_classes} classes with {n_train} images — EfficientNet-B0 "
                  "usually gives more accuracy per parameter.")
    else:
        key = "resnet50"
        reason = (f"{n_train} images is enough for ResNet-50 if the backbone stays "
                  "frozen at first.")
    info = CLASSIFICATION_BACKBONES[key]
    return {"key": key, "name": info["name"], "reason": reason}


def build_dataloaders(root, batch_size=16, input_size=224, val_split=0.2):
    if not TORCH:
        raise RuntimeError("PyTorch is not installed.")
    classes = sorted([c for c in os.listdir(root)
                      if os.path.isdir(os.path.join(root, c))
                      and not c.startswith("__")])
    if len(classes) < 2:
        raise ValueError("Need at least 2 class folders inside the dataset.")

    train_tf, eval_tf = _build_transforms(input_size)

    train_ds = _ImageFolderDataset(root, classes, train_tf)
    eval_ds  = _ImageFolderDataset(root, classes, eval_tf)

    n_total = len(train_ds)
    if n_total < 2:
        raise ValueError("Found fewer than 2 images total in the dataset.")
    n_val = max(1, int(n_total * val_split))
    n_val = min(n_val, n_total - 1)
    n_tr = n_total - n_val

    gen = torch.Generator().manual_seed(42)
    idx = torch.randperm(n_total, generator=gen).tolist()
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]

    from torch.utils.data import Subset
    tr_subset = Subset(train_ds, tr_idx)
    va_subset = Subset(eval_ds, va_idx)

    tr_loader = DataLoader(tr_subset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=torch.cuda.is_available())
    va_loader = DataLoader(va_subset, batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=torch.cuda.is_available())
    return tr_loader, va_loader, classes, n_tr, n_val


# ============================================================ 4. MODELS
def _build_custom_cnn(layers, n_classes):
    """Build a small CNN from the layer spec.

    Conv / pool / batchnorm / dropout run on the spatial tensor.
    The first `dense` inserts a global average pool + flatten so we do not
    blow up parameter counts from a huge flatten. Final head -> n_classes.
    """
    modules = []
    c_in = 3
    in_dense = False
    dense_in = None

    for layer in layers:
        t = layer.get("type")
        if t == "conv2d":
            k = int(layer.get("kernel", 3))
            modules.append(nn.Conv2d(c_in, int(layer["out_channels"]),
                                     kernel_size=k, padding=k // 2))
            modules.append(nn.ReLU(inplace=True))
            c_in = int(layer["out_channels"])
        elif t == "maxpool":
            modules.append(nn.MaxPool2d(int(layer.get("kernel", 2))))
        elif t == "batchnorm":
            modules.append(nn.BatchNorm2d(c_in))
        elif t == "dropout":
            modules.append(nn.Dropout(float(layer.get("p", 0.3))))
        elif t == "dense":
            if not in_dense:
                modules.append(nn.AdaptiveAvgPool2d(1))
                modules.append(nn.Flatten())
                dense_in = c_in
                in_dense = True
            modules.append(nn.Linear(dense_in, int(layer["units"])))
            modules.append(nn.ReLU(inplace=True))
            dense_in = int(layer["units"])

    if not in_dense:
        modules.append(nn.AdaptiveAvgPool2d(1))
        modules.append(nn.Flatten())
        dense_in = c_in

    modules.append(nn.Linear(dense_in, n_classes))
    return nn.Sequential(*modules)


def build_classifier(backbone_key, n_classes, freeze_backbone=True,
                     custom_layers=None):
    """Return (model, input_size, human_label)."""
    if not TORCH:
        raise RuntimeError("PyTorch is not installed.")

    if backbone_key == "custom":
        layers = custom_layers or default_layer_spec()
        model = _build_custom_cnn(layers, n_classes)
        return model, 224, "Custom CNN"

    if backbone_key not in CLASSIFICATION_BACKBONES:
        raise ValueError(f"Unknown backbone: {backbone_key}")

    info = CLASSIFICATION_BACKBONES[backbone_key]
    model = info["ctor"]()

    # Swap the classifier head for our class count.
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif hasattr(model, "classifier"):
        clf = model.classifier
        if isinstance(clf, nn.Sequential):
            last = clf[-1]
            if isinstance(last, nn.Linear):
                clf[-1] = nn.Linear(last.in_features, n_classes)
        elif isinstance(clf, nn.Linear):
            model.classifier = nn.Linear(clf.in_features, n_classes)
    else:
        raise ValueError(f"Don't know how to attach a head to {backbone_key}.")

    if freeze_backbone:
        head_prefixes = ("fc.", "classifier.")
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(head_prefixes)

    return model, info["input_size"], info["name"]


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================ 5. TRAIN / EVAL
def _device_of(model):
    return next(model.parameters()).device


def _class_weights(loader, n_classes, device):
    counts = torch.zeros(n_classes, dtype=torch.float32)
    for _, yb in loader:
        for v in yb.view(-1).tolist():
            if 0 <= int(v) < n_classes:
                counts[int(v)] += 1
    counts = counts.clamp(min=1)
    weights = counts.sum() / (n_classes * counts)
    return weights.to(device)


def train_classifier(model, tr_loader, va_loader, device,
                     epochs=5, lr=1e-3, progress_cb=None, patience=4):
    """Train and return a history dict:
       {"train_loss": [...], "val_loss": [...], "val_acc": [...]}"""
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters — is the backbone frozen "
                           "without a head?")
    n_classes = None
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        n_classes = int(last_linear.out_features)
    if n_classes and n_classes >= 2:
        loss_fn = nn.CrossEntropyLoss(
            weight=_class_weights(tr_loader, n_classes, device))
    else:
        loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(1, patience // 2))

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val = float("inf")
    best_state = None
    stall = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
            seen += xb.size(0)
        tr_loss = running / max(seen, 1)

        vl_loss, vl_acc = _evaluate_loss_acc(model, va_loader, device, loss_fn)
        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(vl_loss))
        history["val_acc"].append(float(vl_acc))
        sched.step(vl_loss)

        if vl_loss < best_val - 1e-4:
            best_val = vl_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1

        if progress_cb:
            try:
                progress_cb(epoch, epochs, tr_loss, vl_loss, vl_acc)
            except Exception:
                pass
        if stall >= patience and epoch >= 3:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return history


@torch.no_grad()
def _evaluate_loss_acc(model, loader, device, loss_fn):
    model.eval()
    total, seen, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total += loss_fn(out, yb).item() * xb.size(0)
        seen += xb.size(0)
        correct += (out.argmax(dim=1) == yb).sum().item()
    return total / max(seen, 1), correct / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    """Return (accuracy, confusion_df, per_class_df, samples).

    samples: list of (cpu_tensor_image, true_idx, pred_idx, confidence)
    """
    model.eval()
    all_true, all_pred = [], []
    samples = []

    for xb, yb in loader:
        xb_d = xb.to(device)
        logits = model(xb_d)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        all_true.extend(yb.tolist())
        all_pred.extend(pred.cpu().tolist())

        for i in range(xb.size(0)):
            samples.append((xb[i].cpu(), int(yb[i]),
                            int(pred[i]), float(conf[i])))

    n = max(len(all_true), 1)
    acc = sum(int(t == p) for t, p in zip(all_true, all_pred)) / n

    k = len(class_names)
    cm_values = np.zeros((k, k), dtype=int)
    for t, p in zip(all_true, all_pred):
        if 0 <= t < k and 0 <= p < k:
            cm_values[t, p] += 1
    cm = pd.DataFrame(
        cm_values,
        index=[f"Actual {c}" for c in class_names],
        columns=[f"Predicted {c}" for c in class_names],
    )

    rows = []
    for i, c in enumerate(class_names):
        tp = int(sum(1 for t, p in zip(all_true, all_pred) if t == i and p == i))
        fp = int(sum(1 for t, p in zip(all_true, all_pred) if t != i and p == i))
        fn = int(sum(1 for t, p in zip(all_true, all_pred) if t == i and p != i))
        support = int(sum(1 for t in all_true if t == i))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({
            "class": c,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1-score": round(f1, 3),
            "support": support,
        })
    per_class = pd.DataFrame(rows)

    return acc, cm, per_class, samples


# ============================================================ 6. INFERENCE
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize(tensor):
    """CHW tensor -> HWC float image in [0, 1] for st.image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * _STD + _MEAN
    return np.clip(img, 0.0, 1.0)


def predict_single_image(model, pil_image, class_names, device, input_size=224):
    """Return (label, confidence, {class_name: prob})."""
    if not TORCH:
        raise RuntimeError("PyTorch is not installed.")

    tf = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(pil_image.convert("RGB")).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    idx = int(probs.argmax())
    label = class_names[idx] if idx < len(class_names) else str(idx)
    confidence = float(probs[idx])
    by_class = {c: float(probs[i]) for i, c in enumerate(class_names)}
    return label, confidence, by_class


# ============================================================ 7. PERSISTENT SAVE SLOTS
CV_SAVE_DIR = os.path.join("outputs", "_cv_saved_models")
os.makedirs(CV_SAVE_DIR, exist_ok=True)


def _slug(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "model"


def list_saved_models(owner=None):
    """Return metadata for every saved slot on disk, newest first.

    If `owner` is provided, only slots saved by that user are returned.
    Pass owner="" to mean "the anonymous user".
    """
    if not os.path.isdir(CV_SAVE_DIR):
        return []
    slots = []
    for fn in os.listdir(CV_SAVE_DIR):
        if not fn.endswith(".json"):
            continue
        meta_path = os.path.join(CV_SAVE_DIR, fn)
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        if owner is not None and meta.get("owner", "") != owner:
            continue
        slot_id = os.path.splitext(fn)[0]
        meta["id"] = slot_id
        meta["_weights_file"] = os.path.join(CV_SAVE_DIR, f"{slot_id}.pt")
        slots.append(meta)
    # newest first — sort by saved_at if present, else by id
    slots.sort(key=lambda m: m.get("saved_at", m.get("id", "")), reverse=True)
    return slots


def save_model_slot(model, label, backbone_key, custom_layers, class_names,
                    input_size, accuracy, n_classes, owner=""):
    """Persist one trained model to disk. Returns its metadata dict.

    `owner` is stored inside the metadata and (as a slug) inside the slot id,
    so different users can save models with the same label at the same
    second without collision.
    """
    if not TORCH:
        raise RuntimeError("PyTorch is not installed.")
    os.makedirs(CV_SAVE_DIR, exist_ok=True)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    owner_slug = _slug(owner) if owner else "anon"
    slot_id = f"{owner_slug}__{_slug(label)}__{stamp}"

    meta = {
        "id": slot_id,
        "owner": str(owner),
        "label": str(label),
        "backbone_key": str(backbone_key),
        "custom_layers": custom_layers,
        "class_names": list(class_names),
        "input_size": int(input_size),
        "n_classes": int(n_classes),
        "accuracy": float(accuracy),
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    # Always save CPU state so files are portable between machines.
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    weights_path = os.path.join(CV_SAVE_DIR, f"{slot_id}.pt")
    torch.save(cpu_state, weights_path)

    # Record file size for the UI.
    try:
        meta["size_mb"] = round(os.path.getsize(weights_path) / (1024 * 1024), 1)
    except Exception:
        meta["size_mb"] = 0

    with open(os.path.join(CV_SAVE_DIR, f"{slot_id}.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


def load_model_slot(slot_id):
    """Rebuild a saved model on the current device.

    Returns (model, meta). The model is in eval() mode.
    """
    if not TORCH:
        raise RuntimeError("PyTorch is not installed.")

    meta_path = os.path.join(CV_SAVE_DIR, f"{slot_id}.json")
    weights_path = os.path.join(CV_SAVE_DIR, f"{slot_id}.pt")

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"No metadata for slot '{slot_id}'.")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"No weights for slot '{slot_id}'.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    model, _input_size, _label = build_classifier(
        meta["backbone_key"], meta["n_classes"],
        freeze_backbone=False,
        custom_layers=meta.get("custom_layers"),
    )

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)

    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, meta


def delete_model_slot(slot_id):
    """Remove both .pt and .json for the given slot. Silent if missing."""
    removed = []
    for ext in (".pt", ".json"):
        p = os.path.join(CV_SAVE_DIR, f"{slot_id}{ext}")
        if os.path.isfile(p):
            try:
                os.remove(p)
                removed.append(p)
            except Exception:
                pass
    return removed


# ============================================================ 8. PRETRAINED DETECT / SEGMENT
def predict_detection(pil_image, score_thresh=0.5):
    """Zero-shot COCO Faster R-CNN on one image. Returns (annotated PIL, rows)."""
    if not TORCH:
        raise RuntimeError("PyTorch is not installed.")
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    img = pil_image.convert("RGB")
    x = weights.transforms()(img).to(device)
    with torch.no_grad():
        out = model([x])[0]
    cats = weights.meta.get("categories") or []
    rows, draw_boxes = [], []
    for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
        if float(score) < score_thresh:
            continue
        lab = cats[int(label)] if int(label) < len(cats) else str(int(label))
        xyxy = [float(v) for v in box.cpu().tolist()]
        rows.append({"label": lab, "confidence": round(float(score), 3),
                     "x1": round(xyxy[0], 1), "y1": round(xyxy[1], 1),
                     "x2": round(xyxy[2], 1), "y2": round(xyxy[3], 1)})
        draw_boxes.append((xyxy, lab, float(score)))
    annotated = img.copy()
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(annotated)
    for xyxy, lab, score in draw_boxes:
        draw.rectangle(xyxy, outline="#22d3ee", width=3)
        draw.text((xyxy[0] + 4, xyxy[1] + 4), f"{lab} {score:.0%}", fill="#22d3ee")
    return annotated, rows


def predict_segmentation(pil_image):
    """Zero-shot DeepLabV3 (VOC/COCO classes) on one image. Returns (overlay, legend)."""
    if not TORCH:
        raise RuntimeError("PyTorch is not installed.")
    from torchvision.models.segmentation import (
        deeplabv3_resnet50, DeepLabV3_ResNet50_Weights)

    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    img = pil_image.convert("RGB")
    x = weights.transforms()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)["out"].argmax(1)[0].cpu().numpy()
    cats = weights.meta.get("categories") or [str(i) for i in range(int(pred.max()) + 1)]
    used = sorted(int(v) for v in np.unique(pred) if v < len(cats))
    palette = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    rng = np.random.RandomState(0)
    colors = {0: (0, 0, 0)}
    legend = []
    for i in used:
        if i == 0:
            legend.append({"class": cats[i] if i < len(cats) else str(i),
                           "pixels": int((pred == i).sum())})
            continue
        colors[i] = tuple(int(v) for v in rng.randint(40, 240, size=3))
        legend.append({"class": cats[i] if i < len(cats) else str(i),
                       "pixels": int((pred == i).sum())})
        palette[pred == i] = colors[i]
    overlay = Image.fromarray(palette).resize(img.size, Image.NEAREST)
    blended = Image.blend(img, overlay.convert("RGB"), 0.45)
    return blended, legend