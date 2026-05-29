# -*- coding: utf-8 -*-

import os
import sys  
import csv
import math
import time
import json
import random
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pickle

# ===== toggles =====
AUTO_POS_WEIGHT = True     # batch-wise (1-p)/p; set False to disable
WARMUP_BATCHES  = 3        # forward-only dry-run batches to materialize lazy modules
SEED            = 0

# ===== your modules =====
from model_refine6 import TriComplexPredictor
from dataset_esm_dcmap_fp import PROTACBagDataset, protac_bag_collate

# ---------------- Utils ----------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def torch_to_numpy_fallback(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    try:
        return t.numpy()
    except RuntimeError as e:
        if "Numpy is not available" in str(e):
            return np.asarray(t.tolist(), dtype=np.float32)
        raise

def to_float_label(y: torch.Tensor, device: torch.device, target_shape: Tuple[int, int]) -> torch.Tensor:
    if isinstance(y, torch.Tensor):
        yf = y.to(device).float()
    else:
        yf = torch.tensor(y, dtype=torch.float32, device=device)
    if yf.dim() == 1:
        yf = yf.unsqueeze(-1)
    if yf.shape[1] != target_shape[1]:
        yf = yf.expand(-1, target_shape[1])
    return yf

def masked_bce_sum(
    logits: torch.Tensor,
    targets: torch.Tensor,
    auto_pos_weight: bool = AUTO_POS_WEIGHT,
    given_pos_weight: Optional[float] = None,
    label_smoothing: float = 0.0,
):
    """
    Returns: (sum_loss, n_effective, acc@0.5, pos_rate, pred_pos_rate@0.5)
    Masks out NaN; optionally uses pos_weight=(1-p)/p computed on-current-batch.
    Accuracy uses rounded targets (before smoothing) to avoid metric bias.
    """
    mask = ~torch.isnan(targets)
    if not mask.any():
        z = torch.tensor(0.0, device=logits.device)
        return z, 0, float('nan'), float('nan'), float('nan')

    l = logits[mask]
    t = targets[mask].float()

    # keep hard labels for acc metric
    t_hard = (t >= 0.5).float()

    if label_smoothing > 0:
        eps = float(label_smoothing)
        t = t * (1.0 - eps) + 0.5 * eps

    pos_weight_to_use = None
    if given_pos_weight is not None:
        pos_weight_to_use = given_pos_weight
    elif auto_pos_weight:
        p = t.mean().item()
        if 0.0 < p < 1.0:
            pos_weight_to_use = (1.0 - p) / p

    if pos_weight_to_use is not None and pos_weight_to_use > 0:
        pw = torch.full((l.size(-1),), float(pos_weight_to_use), device=l.device, dtype=l.dtype)
        lsum = F.binary_cross_entropy_with_logits(l, t, pos_weight=pw, reduction='sum')
    else:
        lsum = F.binary_cross_entropy_with_logits(l, t, reduction='sum')

    probs = torch.sigmoid(l)
    preds05 = (probs >= 0.5).float()
    acc = (preds05 == t_hard).float().mean().item()
    pos_rate = t_hard.mean().item()
    pred_pos_rate = preds05.mean().item()
    return lsum, int(mask.sum().item()), acc, pos_rate, pred_pos_rate

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray):
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(np.unique(y_true)) < 2:
            return float('nan'), float('nan')
        return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))
    except Exception:
        return float('nan'), float('nan')

def sweep_threshold(probs: np.ndarray, y_true: np.ndarray, steps: int = 101):
    if probs.size == 0 or len(np.unique(y_true)) < 2:
        return {"best_thr": 0.5, "acc_best": float('nan'), "f1_best": float('nan')}
    best_acc, best_thr_acc = -1.0, 0.5
    best_f1, best_thr_f1 = -1.0, 0.5
    for thr in np.linspace(0.0, 1.0, steps):
        pred = (probs >= thr).astype(np.float32)
        acc = (pred == y_true).mean()
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        fn = float(((pred == 0) & (y_true == 1)).sum())
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-12)
        if acc > best_acc:
            best_acc, best_thr_acc = acc, thr
        if f1 > best_f1:
            best_f1, best_thr_f1 = f1, thr
    return {"best_thr": float(best_thr_f1), "acc_best": float(best_acc), "f1_best": float(best_f1)}

@torch.no_grad()
def count_label1_distribution(loader: DataLoader) -> dict:
    pos = neg = total = missing = 0
    for batch in loader:
        y = batch.get("label1", None)
        if y is None:
            continue
        if isinstance(y, torch.Tensor):
            arr = torch_to_numpy_fallback(y).reshape(-1)
        else:
            arr = np.asarray(y).reshape(-1)
        mask = ~np.isnan(arr)
        missing += int((~mask).sum())
        arr = arr[mask]
        total += int(arr.size)
        if arr.size:
            pos += int((arr >= 0.5).sum())
            neg += int((arr < 0.5).sum())
    pos_rate = float(pos) / max(1, total)
    return {"pos": pos, "neg": neg, "total": total, "missing": missing, "pos_rate": pos_rate}

def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)

def save_run_config(config_path: str, config: Dict[str, Any]):
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

# ---------------- Train / Eval ----------------
def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scaler: GradScaler,
                    device: torch.device,
                    use_amp: bool = True,
                    max_grad_norm: float = 1.0,
                    given_pos_weight: Optional[float] = None,
                    label_smoothing: float = 0.0) -> Dict[str, float]:
    model.train()
    tot_sum, tot_cnt = 0.0, 0
    tot_sum1, cnt1, accs1, pos1, ppos1 = 0.0, 0, [], [], []

    pbar = tqdm(loader, desc="Train", ncols=100)
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            out = model(batch)                      # model moves graphs to device internally
            logits1 = out["logits1"]                # [B, 1] (single-task)

            loss_sum, denom = 0.0, 0
            y1 = batch.get("label1", None)
            if y1 is not None:
                y1f = to_float_label(y1, device, logits1.shape)
                l1, n1, a1, pr1, ppr1 = masked_bce_sum(
                    logits1, y1f, auto_pos_weight=AUTO_POS_WEIGHT,
                    given_pos_weight=given_pos_weight, label_smoothing=label_smoothing
                )
                loss_sum += l1; denom += n1
                tot_sum1 += float(l1.item()); cnt1 += n1
                if not math.isnan(a1): accs1.append(a1)
                if not math.isnan(pr1): pos1.append(pr1)
                if not math.isnan(ppr1): ppos1.append(ppr1)

            loss = (loss_sum / max(denom, 1))

        scaler.scale(loss).backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        tot_sum += float(loss_sum.item()); tot_cnt += denom
        pbar.set_postfix({
            "loss": f"{(tot_sum / max(tot_cnt,1)):.4f}",
            "acc1": f"{(np.mean(accs1) if accs1 else float('nan')):.3f}",
        })

    return {
        "loss": tot_sum / max(tot_cnt, 1),
        "loss1": (tot_sum1 / max(cnt1, 1)) if cnt1 else float('nan'),
        "acc1": float(np.mean(accs1)) if accs1 else float('nan'),
        "n1": cnt1,
        "pos_rate1": float(np.mean(pos1)) if pos1 else float('nan'),
        "pred_pos_rate1": float(np.mean(ppos1)) if ppos1 else float('nan'),
    }

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             use_amp: bool = True) -> Dict[str, float]:
    model.eval()
    tot_sum, tot_cnt = 0.0, 0
    tot_sum1, cnt1, accs1_05, pos1, ppos1 = 0.0, 0, [], [], []

    probs1_all, y1_all = [], []

    pbar = tqdm(loader, desc="Valid", ncols=100)
    for batch in pbar:
        with autocast(enabled=use_amp):
            out = model(batch)
            logits1 = out["logits1"]

            loss_sum, denom = 0.0, 0
            y1 = batch.get("label1", None)
            if y1 is not None:
                y1f = to_float_label(y1, device, logits1.shape)
                l1, n1, a1, pr1, ppr1 = masked_bce_sum(
                    logits1, y1f, auto_pos_weight=False, given_pos_weight=None, label_smoothing=0.0
                )
                loss_sum += l1; denom += n1
                tot_sum1 += float(l1.item()); cnt1 += n1
                if not math.isnan(a1): accs1_05.append(a1)
                if not math.isnan(pr1): pos1.append(pr1)
                if not math.isnan(ppr1): ppos1.append(ppr1)

                mask = ~torch.isnan(y1f)
                if mask.any():
                    probs1_all.append(torch.sigmoid(logits1[mask]).detach().cpu().view(-1))
                    y1_all.append(y1f[mask].detach().cpu().view(-1))

            loss = (loss_sum / max(denom, 1))
        tot_sum += float(loss_sum.item()); tot_cnt += denom
        pbar.set_postfix({"loss": f"{(tot_sum / max(tot_cnt,1)):.4f}"})

    out: Dict[str, Any] = {
        "loss": tot_sum / max(tot_cnt, 1),
        "loss1": (tot_sum1 / max(cnt1, 1)) if cnt1 else float('nan'),
        "acc1@0.5": float(np.mean(accs1_05)) if accs1_05 else float('nan'),
        "n1": cnt1,
        "pos_rate1": float(np.mean(pos1)) if pos1 else float('nan'),
        "pred_pos_rate1@0.5": float(np.mean(ppos1)) if ppos1 else float('nan'),
    }

    if probs1_all:
        p1 = torch_to_numpy_fallback(torch.cat(probs1_all))
        t1 = torch_to_numpy_fallback(torch.cat(y1_all))
        sweep1 = sweep_threshold(p1, t1, steps=101)
        auroc1, aupr1 = _safe_auc(t1, p1)
        out.update({
            "best_thr1": sweep1["best_thr"],
            "acc1@best": sweep1["acc_best"],
            "f1_1": sweep1["f1_best"],
            "auroc1": auroc1,
            "auprc1": aupr1,
        })

    return out

# ---------------- Warmup ----------------
@torch.no_grad()
def parameter_warmup_dry_run(model: nn.Module, loader: DataLoader, warm_batches: int = WARMUP_BATCHES):
    if warm_batches <= 0:
        return
    model.eval()
    it = iter(loader)
    for _ in range(warm_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        _ = model(batch)

# ---------------- EarlyStopping ----------------
class EarlyStopper:
    def __init__(self, patience: int = 50, min_delta: float = 0.0):
        self.best = float('inf')
        self.bad = 0
        self.patience = patience
        self.min_delta = min_delta
    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value; self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience

# ---------------- Main ----------------
def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None or s.strip() == "":
        return None
    return [int(x) for x in s.split(",")]

def main():
    parser = argparse.ArgumentParser(description="Train TriComplexPredictor (model_refine5_TransGraph) on label1 only")
    parser.add_argument('--data', type=str, default=r"G:\TriComplex\database\protac0901_3358_esm_DCMAP_fp.pkl")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1)
    parser.add_argument('--num_workers', type=int, default=0)   
    parser.add_argument('--no_amp', action='store_true', help='Disable AMP even if CUDA available')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--ckpt_dir', type=str, default='./log_TransPos')
    parser.add_argument('--logs_dir', type=str, default='./log_TransPos')

    # model capacity
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_layers_gnn', type=int, default=3)

    # protac graph
    parser.add_argument('--num_node_embeddings_protac', type=int, default=10)
    parser.add_argument('--num_edge_embeddings_protac', type=int, default=6)

    # protein graph specifics
    parser.add_argument('--protein_node_fields', type=str, default="23, 7, 24",
                        help='Comma-separated caps for protein node multi-fields, e.g., "64,8,40"; '
                             'leave empty for dynamic enlargement')
    parser.add_argument('--num_edge_embeddings_protein', type=int, default=6)

    parser.add_argument('--gnn_heads', type=int, default=4)

    # PROTAC fingerprints dims (for logging / sanity; 模型本身也会 lazy 创建)
    parser.add_argument('--maccs_dim', type=int, default=167)
    parser.add_argument('--ecfp_dim', type=int, default=2048)

    # --- new regularization/optim args ---
    parser.add_argument('--label_smoothing', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=100)

    parser.add_argument('--seq_dropout', type=float, default=0.1)
    parser.add_argument('--seq_tokendrop', type=float, default=0.1)
    parser.add_argument('--gnn_dropout', type=float, default=0.1)
    parser.add_argument('--gnn_dropedge', type=float, default=0.1)
    parser.add_argument('--cm_drop2d', type=float, default=0.1)
    parser.add_argument('--fuse_dropout', type=float, default=0.0)

    parser.add_argument('--esm_num_layers', type=int, default=5)
    parser.add_argument('--esm_attn_heads', type=int, default=8)
    parser.add_argument('--esm_dropout', type=float, default=0.0)

    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    # === dataset ===
    with open(args.data, 'rb') as f:
        packed = pickle.load(f)

    ds = PROTACBagDataset(
        packed,
        filter_empty=True,
        shuffle_within_name_each_epoch=True
    )

    # split train/val
    n_total = len(ds)
    n_val = int(round(n_total * args.val_ratio))
    n_train = n_total - n_val
    train_set, val_set = random_split(ds, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(SEED))

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=protac_bag_collate, num_workers=args.num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=protac_bag_collate, num_workers=args.num_workers, pin_memory=pin_mem)

    # label1 distribution (train only) -> fixed pos_weight
    dist = count_label1_distribution(train_loader)
    print(f"[Label1 distribution | train] pos={dist['pos']}  neg={dist['neg']}  total={dist['total']}  "
          f"missing(NaN)={dist['missing']}  pos_rate={dist['pos_rate']:.3f}")
    pos_rate = max(1e-6, min(1 - 1e-6, dist['pos_rate']))
    fixed_pos_weight = (1.0 - pos_rate) / pos_rate

    # === model ===
    vocab_size = packed.get("vocab_size", 128)
    protein_node_field_sizes = parse_int_list(args.protein_node_fields)

    model = TriComplexPredictor(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers_gnn=args.num_layers_gnn,
        num_classes1=1,
        num_classes2=1,   
        num_node_embeddings_protac=args.num_node_embeddings_protac,
        num_edge_embeddings_protac=args.num_edge_embeddings_protac,
        protein_node_field_sizes=protein_node_field_sizes,
        num_edge_embeddings_protein=args.num_edge_embeddings_protein,

        seq_dropout=args.seq_dropout,
        seq_tokendrop=args.seq_tokendrop,
        gnn_dropout=args.gnn_dropout,
        gnn_dropedge=args.gnn_dropedge,
        cm_drop2d=args.cm_drop2d,
        fuse_dropout=args.fuse_dropout,

        esm_num_layers=args.esm_num_layers,
        esm_attn_heads=args.esm_attn_heads,
        esm_dropout=args.esm_dropout,
        gnn_heads=args.gnn_heads,
        maccs_dim=args.maccs_dim,
        ecfp_dim=args.ecfp_dim,
    ).to(device)

    # warmup to materialize lazy/dynamic modules in encoders
    parameter_warmup_dry_run(model, train_loader, warm_batches=WARMUP_BATCHES)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / max(1, args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, (args.epochs - args.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    total_params, trainable_params = count_parameters(model)
    run_config = {
        "run_id": stamp,
        "argv": " ".join(sys.argv),
        "timestamp": stamp,
        "raw_args": vars(args),
        "data": {
            "path": args.data,
            "n_total": n_total,
            "n_train": n_train,
            "n_val": n_val,
            "label1_distribution_train": dist,          
            "fixed_pos_weight": float(fixed_pos_weight)
        },
        "model": {
            "class": "TriComplexPredictor",
            "vocab_size": int(vocab_size),
            "hidden_dim": int(args.hidden_dim),
            "num_layers_gnn": int(args.num_layers_gnn),
            "protein_node_field_sizes": protein_node_field_sizes,
            "num_node_embeddings_protac": int(args.num_node_embeddings_protac),
            "num_edge_embeddings_protac": int(args.num_edge_embeddings_protac),
            "num_edge_embeddings_protein": int(args.num_edge_embeddings_protein),
            "gnn_heads": int(args.gnn_heads),
            "fp_dims": {
                "maccs_dim": int(args.maccs_dim),
                "ecfp_dim": int(args.ecfp_dim),
            },
            "dropouts": {
                "seq_dropout": float(args.seq_dropout),
                "seq_tokendrop": float(args.seq_tokendrop),
                "gnn_dropout": float(args.gnn_dropout),
                "gnn_dropedge": float(args.gnn_dropedge),
                "cm_drop2d": float(args.cm_drop2d),
                "fuse_dropout": float(args.fuse_dropout),
            },
            "parameters": {
                "total": total_params,
                "trainable": trainable_params
            }
        },
        "train": {
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "warmup_epochs": int(args.warmup_epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "max_grad_norm": float(args.max_grad_norm),
            "val_ratio": float(args.val_ratio),
            "AUTO_POS_WEIGHT": bool(AUTO_POS_WEIGHT),
            "given_pos_weight": float(fixed_pos_weight),
            "label_smoothing": float(args.label_smoothing),
            "patience": int(args.patience),
            "seed": int(SEED),
            "use_amp": bool(use_amp)
        },
        "env": {
            "torch": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "paths": {
            "ckpt_dir": os.path.abspath(args.ckpt_dir),
            "logs_dir": os.path.abspath(args.logs_dir),
        }
    }
    config_path = os.path.join(args.logs_dir, f"config_{stamp}.json")
    save_run_config(config_path, run_config)
    print(f"Run config saved -> {config_path}")

    # CSV logger
    log_path = os.path.join(args.logs_dir, f'{args.model_name}_{stamp}.csv')
    log_fields = [
        'run_id',  # NEW
        'epoch','lr',
        'train_loss','train_loss1','train_acc1','train_pred_pos_rate1',
        'val_loss','val_loss1','val_acc1@0.5','val_best_thr1','val_acc1@best','val_f1_1','val_auroc1','val_auprc1',
        'val_pos_rate1','val_pred_pos_rate1@0.5','is_best'
    ]
    with open(log_path, 'w', newline='') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=log_fields)
        writer.writeheader()

    print(f"CSV log -> {log_path}")

    best_val = float('inf')
    stopper = EarlyStopper(patience=args.patience)

    for epoch in range(1, args.epochs + 1):
        print(f"==== Epoch {epoch}/{args.epochs} ====")
        if hasattr(ds, "on_epoch_start"):
            try:
                ds.on_epoch_start()
            except Exception:
                pass

        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device,
                                      use_amp=use_amp, max_grad_norm=args.max_grad_norm,
                                      given_pos_weight=fixed_pos_weight, label_smoothing=args.label_smoothing)
        val_stats = evaluate(model, val_loader, device, use_amp=use_amp)

        scheduler.step()

        cur_lr = optimizer.param_groups[0]['lr']
        print(
            "Train: loss={loss:.4f} | loss1={loss1:.4f} n1={n1} "
            "acc1={acc1:.3f} pos1={pos1:.3f} pred1@0.5={pp1:.3f}".format(
                loss=train_stats['loss'], loss1=train_stats['loss1'], n1=train_stats['n1'],
                acc1=train_stats['acc1'], pos1=train_stats['pos_rate1'], pp1=train_stats['pred_pos_rate1'],
            )
        )
        print(
            "Valid: loss={loss:.4f} | loss1={loss1:.4f} n1={n1} "
            "acc1@0.5={a105:.3f} best_thr1={bt1:.2f} acc1@best={ab1:.3f} "
            "F1_1={f1_1:.3f} AUROC1={au1:.3f} AUPRC1={ap1:.3f} "
            "pos1={pos1:.3f} pred1@0.5={pp1:.3f}".format(
                loss=val_stats['loss'], loss1=val_stats['loss1'], n1=val_stats['n1'],
                a105=val_stats.get('acc1@0.5', float('nan')),
                bt1=val_stats.get('best_thr1', float('nan')), ab1=val_stats.get('acc1@best', float('nan')),
                f1_1=val_stats.get('f1_1', float('nan')), au1=val_stats.get('auroc1', float('nan')),
                ap1=val_stats.get('auprc1', float('nan')),
                pos1=val_stats['pos_rate1'],
                pp1=val_stats['pred_pos_rate1@0.5'],
            )
        )

        # save best
        is_best = False
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            is_best = True
            ckpt_path = os.path.join(
                args.ckpt_dir,
                f"{args.model_name}_{stamp}_best_epoch{epoch:03d}_val{best_val:.4f}.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "val_loss": best_val,
                "run_config": run_config,   # NEW: 让 ckpt 自描述
            }, ckpt_path)
            print(f"=> Saved best checkpoint to {ckpt_path}")

        # write CSV row
        row = {
            'run_id': stamp,  
            'epoch': epoch,
            'lr': cur_lr,
            'train_loss': train_stats['loss'],
            'train_loss1': train_stats['loss1'],
            'train_acc1': train_stats['acc1'],
            'train_pred_pos_rate1': train_stats['pred_pos_rate1'],
            'val_loss': val_stats['loss'],
            'val_loss1': val_stats['loss1'],
            'val_acc1@0.5': val_stats.get('acc1@0.5', float('nan')),
            'val_best_thr1': val_stats.get('best_thr1', float('nan')),
            'val_acc1@best': val_stats.get('acc1@best', float('nan')),
            'val_f1_1': val_stats.get('f1_1', float('nan')),
            'val_auroc1': val_stats.get('auroc1', float('nan')),
            'val_auprc1': val_stats.get('auprc1', float('nan')),
            'val_pos_rate1': val_stats.get('pos_rate1', float('nan')),
            'val_pred_pos_rate1@0.5': val_stats.get('pred_pos_rate1@0.5', float('nan')),
            'is_best': int(is_best),
        }
        with open(log_path, 'a', newline='') as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=log_fields)
            writer.writerow(row)

        # early stopping
        if stopper.step(val_stats['loss']):
            print(f"Early stopping triggered. Best val loss: {stopper.best:.4f}")
            break

    print("Training finished.")
    print(f"Logs saved to: {log_path}")
    print(f"Config saved to: {config_path}")

if __name__ == "__main__":
    main()
