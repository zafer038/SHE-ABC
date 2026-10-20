"""U-Net (ImageNet-pretrained ResNet18 encoder) fine-tuning and evaluation on GPU."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import IGNORE

NUM_CLASSES = 6


def set_determinism():
    """Deterministic GPU settings. Some U-Net backward kernels (e.g. nearest upsampling) have no
    deterministic CUDA implementation; they only warn, and the residual run-to-run noise is
    measured by hpo.benchmark and hpo.final_test (re-evaluation of the stored best settings)."""
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def library_versions():
    import importlib.metadata as md
    out = {}
    for name in ("torch", "segmentation-models-pytorch", "timm", "numpy", "tifffile"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)*255
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)*255


class GPUData:
    """Whole dataset kept on the GPU as uint8 (Potsdam patches: ~0.9 GB)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, device: str = "cuda"):
        self.device = torch.device(device)
        self.x = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2))).to(self.device)
        self.y = torch.from_numpy(np.ascontiguousarray(labels)).to(self.device)
        self.mean, self.std = _MEAN.to(self.device), _STD.to(self.device)

    def normalise(self, xb_uint8):
        return ((xb_uint8.float() - self.mean)/self.std).contiguous(memory_format=torch.channels_last)


def build_model(device):
    import segmentation_models_pytorch as smp
    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=NUM_CLASSES)
    return model.to(device).to(memory_format=torch.channels_last)


def _augment(x, y, p, gen):
    b = x.shape[0]
    for op in ("h", "v", "r"):
        mask = torch.rand(b, generator=gen, device=x.device) < p
        if not mask.any():
            continue
        if op == "h":
            x[mask], y[mask] = x[mask].flip(-1), y[mask].flip(-1)
        elif op == "v":
            x[mask], y[mask] = x[mask].flip(-2), y[mask].flip(-2)
        else:
            x[mask], y[mask] = x[mask].rot90(1, (-2, -1)), y[mask].rot90(1, (-2, -1))
    return x, y


def _dice_loss(logits, y):
    valid = (y != IGNORE)
    probs = logits.float().softmax(1)*valid.unsqueeze(1)
    onehot = F.one_hot(torch.where(valid, y, torch.zeros_like(y)), NUM_CLASSES).permute(0, 3, 1, 2).float()
    onehot = onehot*valid.unsqueeze(1)
    inter = (probs*onehot).sum((0, 2, 3))
    denom = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    return 1 - ((2*inter + 1)/(denom + 1)).mean()


def confusion(pred, y):
    valid = y != IGNORE
    idx = y[valid].long()*NUM_CLASSES + pred[valid].long()
    return torch.bincount(idx, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def scores(cm):
    cm = cm.double()
    tp = cm.diag()
    iou = tp/(cm.sum(0) + cm.sum(1) - tp).clamp_min(1)
    return {"miou": float(iou.mean()), "oa": float(tp.sum()/cm.sum().clamp_min(1)),
            "per_class_iou": iou.tolist()}


@torch.no_grad()
def predict_logits(model, data: GPUData, idx, flips=("id",), batch=16):
    """Logits for full patches (padded to a multiple of 32, then cropped back)."""
    model.eval()
    outs = {f: [] for f in flips}
    h, w = data.x.shape[-2:]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    for s in range(0, len(idx), batch):
        xb = data.normalise(data.x[idx[s:s + batch]])
        xb = F.pad(xb, (0, pw, 0, ph), mode="reflect")
        for f in flips:
            xin = xb if f == "id" else xb.flip(-1) if f == "h" else xb.flip(-2)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(xin)
            out = out if f == "id" else out.flip(-1) if f == "h" else out.flip(-2)
            outs[f].append(out[..., :h, :w].half())
    return {f: torch.cat(v) for f, v in outs.items()}


def evaluate_model(model, data: GPUData, idx):
    logits = predict_logits(model, data, idx)["id"]
    cm = confusion(logits.argmax(1), data.y[idx])
    return scores(cm)


def train(hp: dict, data: GPUData, train_idx, epochs: int, seed: int, crop: int = 256):
    """Fine-tune a fresh pretrained U-Net with hyperparameters hp. Returns the model."""
    import random
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen_cpu = torch.Generator().manual_seed(seed)
    gen_dev = torch.Generator(device=data.device).manual_seed(seed)
    model = build_model(data.device)
    enc = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc, "lr": hp["lr"]*hp["encoder_lr_scale"]},
                             {"params": rest, "lr": hp["lr"]}], weight_decay=hp["weight_decay"])
    bs = int(hp["batch_size"])
    steps = max(1, math.ceil(len(train_idx)/bs))
    total = epochs*steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5*(1 + math.cos(math.pi*min(t, total)/total)))
    scaler = torch.amp.GradScaler("cuda")
    train_idx_t = torch.as_tensor(np.asarray(train_idx), device=data.device)
    h, w = data.x.shape[-2:]
    w_dice = float(hp["dice_weight"])
    model.train()
    diverged = False
    for _ in range(epochs):
        order = train_idx_t[torch.randperm(len(train_idx_t), generator=gen_cpu).to(data.device)]
        for s in range(0, len(order), bs):
            bidx = order[s:s + bs]
            i = int(torch.randint(0, h - crop + 1, (1,), generator=gen_cpu))
            j = int(torch.randint(0, w - crop + 1, (1,), generator=gen_cpu))
            xb = data.x[bidx][..., i:i + crop, j:j + crop].clone()
            yb = data.y[bidx][..., i:i + crop, j:j + crop].clone()
            xb, yb = _augment(xb, yb, float(hp["aug_prob"]), gen_dev)
            xb, yb = data.normalise(xb), yb.long()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                ce = F.cross_entropy(logits, yb, ignore_index=IGNORE)
            loss = (1 - w_dice)*ce + (w_dice*_dice_loss(logits, yb) if w_dice > 0 else 0.0)
            if not torch.isfinite(loss):
                diverged = True
                break
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        if diverged:
            break
    model.diverged = diverged
    return model


class FinetuneObjective:
    """x -> 1 - validation mIoU after a short fine-tuning run (fixed training seed per HPO run)."""

    def __init__(self, data: GPUData, train_idx, val_idx, epochs: int, decode, seed: int):
        self.data, self.train_idx, self.val_idx = data, train_idx, val_idx
        self.epochs, self.decode, self.seed = epochs, decode, seed

    def __call__(self, x):
        hp = self.decode(x)
        started = time.perf_counter()
        model = train(hp, self.data, self.train_idx, self.epochs, self.seed)
        if model.diverged:
            info = {"miou": 0.0, "oa": 0.0, "diverged": True}
        else:
            info = {**evaluate_model(model, self.data, self.val_idx), "diverged": False}
        info.update({"train_seconds": time.perf_counter() - started, "hp": hp})
        del model
        torch.cuda.empty_cache()
        return 1.0 - info["miou"], info


class PostprocObjective:
    """x -> 1 - validation mIoU of a fixed model after class-bias + flip-TTA post-processing."""

    def __init__(self, logits: dict, labels, decode, chunk: int = 48):
        self.logits, self.labels, self.decode, self.chunk = logits, labels, decode, chunk

    def predict_cm(self, p, logits=None, labels=None):
        logits = logits or self.logits
        labels = self.labels if labels is None else labels
        bias = torch.tensor(p["class_bias"], device=labels.device, dtype=torch.float32).view(1, -1, 1, 1)
        wh, wv = p["w_hflip"], p["w_vflip"]
        cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long, device=labels.device)
        n = labels.shape[0]
        for s in range(0, n, self.chunk):
            comb = (logits["id"][s:s + self.chunk].float() + wh*logits["h"][s:s + self.chunk].float()
                    + wv*logits["v"][s:s + self.chunk].float())/(1 + wh + wv) + bias
            cm += confusion(comb.argmax(1), labels[s:s + self.chunk])
        return cm

    def __call__(self, x):
        p = self.decode(x)
        info = scores(self.predict_cm(p))
        info["params"] = p
        return 1.0 - info["miou"], info


def load_or_train_base(cfg, data: GPUData, idx, root):
    """Post-processing experiment: one base model, trained once on the training set and stored
    with a sidecar describing how it was made. Only validation scores are computed here."""
    import json
    path = root / cfg["base_model"]
    meta_path = path.with_suffix(".json")
    spec = {"base_hp": cfg["base_hp"], "base_epochs": cfg["base_epochs"], "base_seed": cfg["base_seed"],
            "train_patches": len(idx["train"])}
    if path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("spec") != spec:
            raise SystemExit(f"{path} was trained with different settings; delete it or change base_model")
        model = build_model(data.device)
        model.load_state_dict(torch.load(path, map_location=data.device))
    else:
        started = time.perf_counter()
        model = train(cfg["base_hp"], data, idx["train"], cfg["base_epochs"], cfg["base_seed"])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        meta = {"spec": spec, "train_seconds": time.perf_counter() - started}
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(f"base model trained in {meta['train_seconds']:.0f}s -> {path}", flush=True)
    return model, {"spec": spec, "val": evaluate_model(model, data, idx["val"])}
