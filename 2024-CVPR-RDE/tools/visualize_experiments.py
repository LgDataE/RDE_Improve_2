import argparse
import copy
import csv
import json
import logging
import math
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataloader
from datasets.bases import ImageTextDataset
from datasets.build import build_transforms, change_path, collate
from datasets.cuhkpedes import CUHKPEDES
from datasets.icfgpedes import ICFGPEDES
from datasets.rstpreid import RSTPReid
from model import build_model
from utils.checkpoint import Checkpointer
from utils.iotools import load_train_configs
from utils.metrics import Evaluator, rank


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--root_dir", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--rstpreid_anno_path", type=str, default="")
    parser.add_argument("--val_dataset", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cmc_max_rank", type=int, default=10)
    parser.add_argument("--score_sample_size", type=int, default=50000)
    parser.add_argument("--embedding_num_ids", type=int, default=8)
    parser.add_argument("--embedding_max_queries_per_id", type=int, default=3)
    parser.add_argument("--embedding_max_images_per_id", type=int, default=2)
    parser.add_argument("--num_retrieval", type=int, default=8)
    parser.add_argument("--retrieval_topk", type=int, default=5)
    parser.add_argument("--retrieval_branch", type=str, default="GE+RFE")
    parser.add_argument("--heatmap_num_images", type=int, default=6)
    parser.add_argument("--heatmap_alpha", type=float, default=0.45)
    parser.add_argument("--heatmap_highlight_topk", type=int, default=8)
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--plot_ccd", action="store_true")
    parser.add_argument("--ccd_batch_size", type=int, default=64)
    parser.add_argument("--ccd_max_batches", type=int, default=0)
    return parser.parse_args()


def setup_logger(output_dir: Path):
    logger = logging.getLogger("visualize_experiments")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(output_dir / "visualize_log.txt", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def maybe_resolve_repo_path(path_str: str):
    path_str = str(path_str or "").strip()
    if not path_str:
        return path_str
    path = Path(path_str)
    if path.is_absolute() or path.exists():
        return str(path)
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return str(repo_path)
    return path_str


def load_runtime_args(cli_args):
    args = load_train_configs(cli_args.config_file)
    args.training = False
    args.distributed = False
    args.local_rank = 0
    args.val_dataset = cli_args.val_dataset
    args.test_batch_size = int(cli_args.test_batch_size)
    args.num_workers = int(cli_args.num_workers)
    if cli_args.root_dir:
        args.root_dir = cli_args.root_dir
    if cli_args.dataset_name:
        args.dataset_name = cli_args.dataset_name
    if cli_args.rstpreid_anno_path:
        args.rstpreid_anno_path = cli_args.rstpreid_anno_path
    if hasattr(args, "noisy_file"):
        args.noisy_file = maybe_resolve_repo_path(args.noisy_file)
    return args


def build_dataset(args):
    if args.dataset_name == "CUHK-PEDES":
        return CUHKPEDES(root=args.root_dir, verbose=False)
    if args.dataset_name == "ICFG-PEDES":
        return ICFGPEDES(root=args.root_dir, verbose=False)
    if args.dataset_name == "RSTPReid":
        return RSTPReid(root=args.root_dir, verbose=False, anno_path=getattr(args, "rstpreid_anno_path", ""))
    raise ValueError(f"Unsupported dataset: {args.dataset_name}")


def dataset_stats(dataset):
    return {
        "train": {
            "ids": len(dataset.train_id_container),
            "images": len(dataset.train_annos),
            "captions": len(dataset.train),
        },
        "test": {
            "ids": len(dataset.test_id_container),
            "images": len(dataset.test_annos),
            "captions": len(dataset.test["captions"]),
        },
        "val": {
            "ids": len(dataset.val_id_container),
            "images": len(dataset.val_annos),
            "captions": len(dataset.val["captions"]),
        },
    }


def plot_dataset_overview(stats_dict, output_dir: Path):
    with open(output_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)
    splits = ["train", "val", "test"]
    metrics = ["ids", "images", "captions"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, metrics):
        vals = [stats_dict[s][metric] for s in splits]
        ax.bar(splits, vals, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_title(metric.upper())
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_model(args, ckpt_path: str, num_classes: int, device: torch.device):
    model = build_model(args, num_classes)
    Checkpointer(model).load(f=ckpt_path)
    model = model.to(device)
    if device.type == "cpu":
        model = model.float()
    model.eval()
    return model


def normalize_feats(qfeats, gfeats):
    return F.normalize(qfeats.float(), p=2, dim=1), F.normalize(gfeats.float(), p=2, dim=1)


def normalize_branch_name(branch: str):
    branch = str(branch or "").strip().upper().replace(" ", "")
    mapping = {
        "BGE": "GE",
        "TSE": "RFE",
        "BGE+TSE": "GE+RFE",
        "GE": "GE",
        "RFE": "RFE",
        "GE+RFE": "GE+RFE",
    }
    if branch not in mapping:
        raise ValueError(f"Unsupported branch name: {branch}")
    return mapping[branch]


def compute_similarity(branch, q_bge, g_bge, q_tse, g_tse):
    branch = normalize_branch_name(branch)
    if branch == "GE":
        return q_bge @ g_bge.t()
    if branch == "RFE":
        return q_tse @ g_tse.t()
    return 0.5 * ((q_bge @ g_bge.t()) + (q_tse @ g_tse.t()))


def compute_metric_row(branch, sims, qids, gids, max_rank):
    branch = normalize_branch_name(branch)
    cmc, mAP, mINP, _ = rank(sims, q_pids=qids, g_pids=gids, max_rank=max_rank, get_mAP=True)
    cmc_np = cmc.detach().cpu().numpy()
    row = {
        "task": branch,
        "R1": float(cmc_np[0]),
        "R5": float(cmc_np[min(4, len(cmc_np) - 1)]),
        "R10": float(cmc_np[min(9, len(cmc_np) - 1)]),
        "mAP": to_float(mAP),
        "mINP": to_float(mINP),
    }
    row["rSum"] = row["R1"] + row["R5"] + row["R10"]
    return row, cmc_np


def sample_scores(sims, qids, gids, max_samples, seed):
    mask = qids.view(-1, 1).eq(gids.view(1, -1))
    pos = sims[mask].detach().cpu().numpy()
    neg = sims[~mask].detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    if len(pos) > max_samples:
        pos = rng.choice(pos, size=max_samples, replace=False)
    if len(neg) > max_samples:
        neg = rng.choice(neg, size=max_samples, replace=False)
    return pos, neg


def save_metrics(rows, cmc_dict, output_dir: Path):
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(output_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])
        writer.writeheader()
        writer.writerows(rows)
    np.savez(output_dir / "cmc_curves.npz", **cmc_dict)
    metrics = ["R1", "R5", "R10", "mAP", "mINP", "rSum"]
    labels = [r["task"] for r in rows]
    x = np.arange(len(metrics))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, row in enumerate(rows):
        ax.bar(x + (idx - 1) * width, [row[m] for m in metrics], width=width, label=row["task"])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "metric_bar_chart.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cmc(cmc_dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, cmc in cmc_dict.items():
        ax.plot(np.arange(1, len(cmc) + 1), cmc, marker="o", label=name)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CMC Curves")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cmc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(score_dict, output_dir: Path):
    fig, axes = plt.subplots(len(score_dict), 1, figsize=(9, 4 * len(score_dict)))
    if len(score_dict) == 1:
        axes = [axes]
    for ax, (name, (pos, neg)) in zip(axes, score_dict.items()):
        ax.hist(neg, bins=50, alpha=0.55, density=True, label="negative", color="#E45756")
        ax.hist(pos, bins=50, alpha=0.55, density=True, label="positive", color="#4C78A8")
        ax.set_title(f"Similarity Distribution - {name}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "score_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def pca_project(features: np.ndarray, n_components: int = 2):
    centered = features.astype(np.float32) - features.astype(np.float32).mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:n_components].T


def compute_zoom_limits(points: np.ndarray, pad_ratio: float = 0.45, min_pad: float = 0.08):
    x_min = float(points[:, 0].min())
    x_max = float(points[:, 0].max())
    y_min = float(points[:, 1].min())
    y_max = float(points[:, 1].max())
    x_span = max(x_max - x_min, min_pad)
    y_span = max(y_max - y_min, min_pad)
    x_pad = max(x_span * pad_ratio, min_pad)
    y_pad = max(y_span * pad_ratio, min_pad)
    return [x_min - x_pad, x_max + x_pad], [y_min - y_pad, y_max + y_pad]


def find_best_projection_zoom(q_proj: np.ndarray, g_proj: np.ndarray, q_labels: np.ndarray, g_labels: np.ndarray, selected_ids):
    best = None
    for pid in selected_ids:
        pid = int(pid)
        q_points = q_proj[q_labels == pid]
        g_points = g_proj[g_labels == pid]
        if len(q_points) == 0 or len(g_points) == 0:
            continue
        combined = np.concatenate([q_points, g_points], axis=0)
        pairwise = np.linalg.norm(q_points[:, None, :] - g_points[None, :, :], axis=-1)
        centroid = combined.mean(axis=0)
        compactness = np.linalg.norm(combined - centroid, axis=1).mean()
        score = float(pairwise.mean() + 0.35 * compactness)
        xlim, ylim = compute_zoom_limits(combined)
        candidate = {
            "pid": pid,
            "score": score,
            "xlim": xlim,
            "ylim": ylim,
            "center": [float(centroid[0]), float(centroid[1])],
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate
    return best


def add_projection_zoom_inset(ax, q_proj: np.ndarray, g_proj: np.ndarray, q_labels: np.ndarray, g_labels: np.ndarray, selected_ids, pid_to_color):
    zoom_info = find_best_projection_zoom(q_proj, g_proj, q_labels, g_labels, selected_ids)
    if zoom_info is None:
        return None
    axins = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=1.0)
    for pid in selected_ids:
        pid = int(pid)
        color = pid_to_color[pid]
        q_mask = q_labels == pid
        g_mask = g_labels == pid
        axins.scatter(q_proj[q_mask, 0], q_proj[q_mask, 1], marker="o", s=26, color=color, alpha=0.8)
        axins.scatter(g_proj[g_mask, 0], g_proj[g_mask, 1], marker="^", s=34, color=color, alpha=0.85)
    focus_pid = int(zoom_info["pid"])
    focus_color = pid_to_color[focus_pid]
    q_mask = q_labels == focus_pid
    g_mask = g_labels == focus_pid
    axins.scatter(q_proj[q_mask, 0], q_proj[q_mask, 1], marker="o", s=42, color=focus_color, edgecolors="black", linewidths=0.8)
    axins.scatter(g_proj[g_mask, 0], g_proj[g_mask, 1], marker="^", s=52, color=focus_color, edgecolors="black", linewidths=0.8)
    axins.set_xlim(*zoom_info["xlim"])
    axins.set_ylim(*zoom_info["ylim"])
    axins.set_title(f"Zoom-in PID {focus_pid}", fontsize=8)
    axins.grid(alpha=0.22)
    axins.tick_params(axis="both", labelsize=6)
    for spine in axins.spines.values():
        spine.set_edgecolor(focus_color)
        spine.set_linewidth(1.2)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec=focus_color, lw=1.0)
    return zoom_info


def plot_embedding_projection(q_bge, g_bge, q_tse, g_tse, qids, gids, output_dir: Path, seed: int, num_ids: int, max_queries_per_id: int, max_images_per_id: int):
    qids_np = qids.detach().cpu().numpy()
    gids_np = gids.detach().cpu().numpy()
    common_ids = np.array(sorted(set(qids_np.tolist()) & set(gids_np.tolist())))
    if len(common_ids) == 0:
        return
    rng = np.random.default_rng(seed)
    if len(common_ids) > num_ids:
        selected_ids = sorted(rng.choice(common_ids, size=num_ids, replace=False).tolist())
    else:
        selected_ids = common_ids.tolist()

    selected_q = []
    selected_g = []
    q_labels = []
    g_labels = []
    for pid in selected_ids:
        q_idx = np.where(qids_np == pid)[0]
        g_idx = np.where(gids_np == pid)[0]
        q_pick = rng.choice(q_idx, size=min(len(q_idx), max_queries_per_id), replace=False)
        g_pick = rng.choice(g_idx, size=min(len(g_idx), max_images_per_id), replace=False)
        selected_q.extend(q_pick.tolist())
        selected_g.extend(g_pick.tolist())
        q_labels.extend([pid] * len(q_pick))
        g_labels.extend([pid] * len(g_pick))
    if not selected_q or not selected_g:
        return

    selected_q = np.array(selected_q, dtype=np.int64)
    selected_g = np.array(selected_g, dtype=np.int64)
    q_labels = np.array(q_labels)
    g_labels = np.array(g_labels)

    branch_feats = {
        "GE": (q_bge.detach().cpu().numpy(), g_bge.detach().cpu().numpy()),
        "RFE": (q_tse.detach().cpu().numpy(), g_tse.detach().cpu().numpy()),
    }
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(selected_ids))))
    pid_to_color = {int(pid): tuple(color) for color, pid in zip(colors, selected_ids)}
    fig, axes = plt.subplots(1, len(branch_feats), figsize=(7 * len(branch_feats), 6))
    if len(branch_feats) == 1:
        axes = [axes]
    zoom_summaries = {}

    for ax, (branch, (q_feat, g_feat)) in zip(axes, branch_feats.items()):
        merged = np.concatenate([q_feat[selected_q], g_feat[selected_g]], axis=0)
        proj = pca_project(merged, n_components=2)
        q_proj = proj[:len(selected_q)]
        g_proj = proj[len(selected_q):]
        for color, pid in zip(colors, selected_ids):
            q_mask = q_labels == pid
            g_mask = g_labels == pid
            ax.scatter(q_proj[q_mask, 0], q_proj[q_mask, 1], marker="o", s=45, color=color, alpha=0.85)
            ax.scatter(g_proj[g_mask, 0], g_proj[g_mask, 1], marker="^", s=60, color=color, alpha=0.9)
        zoom_info = add_projection_zoom_inset(ax, q_proj, g_proj, q_labels, g_labels, selected_ids, pid_to_color)
        if zoom_info is not None:
            zoom_summaries[branch] = zoom_info
        ax.set_title(f"Embedding Projection - {branch}")
        ax.grid(alpha=0.3)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    pid_handles = [
        Line2D([0], [0], marker="o", color="w", label=f"PID {pid}", markerfacecolor=color, markersize=8)
        for color, pid in zip(colors, selected_ids)
    ]
    modality_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", label="Text", markersize=7),
        Line2D([0], [0], marker="^", color="black", linestyle="None", label="Image", markersize=8),
    ]
    axes[0].legend(handles=pid_handles, loc="upper left", fontsize=8)
    axes[-1].legend(handles=modality_handles, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "embedding_projection.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "selected_ids": selected_ids,
        "num_query_points": int(len(selected_q)),
        "num_gallery_points": int(len(selected_g)),
        "zoom_regions": zoom_summaries,
    }
    with open(output_dir / "embedding_projection.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def resolve_grid_size(model, img_size):
    visual = getattr(getattr(model, "base_model", None), "visual", None)
    if visual is None:
        return None
    num_y = getattr(visual, "num_y", None)
    num_x = getattr(visual, "num_x", None)
    if num_y is not None and num_x is not None:
        return int(num_y), int(num_x)
    resolution = getattr(visual, "input_resolution", img_size)
    if isinstance(resolution, (list, tuple)):
        img_h, img_w = int(resolution[0]), int(resolution[1])
    else:
        img_h = int(resolution)
        img_w = int(resolution)
    conv1 = getattr(visual, "conv1", None)
    if conv1 is None:
        return None
    kernel = conv1.kernel_size if isinstance(conv1.kernel_size, tuple) else (conv1.kernel_size, conv1.kernel_size)
    stride = conv1.stride if isinstance(conv1.stride, tuple) else (conv1.stride, conv1.stride)
    num_y = (img_h - int(kernel[0])) // int(stride[0]) + 1
    num_x = (img_w - int(kernel[1])) // int(stride[1]) + 1
    if num_y <= 0 or num_x <= 0:
        return None
    return int(num_y), int(num_x)


def compute_rfe_patch_data(model, images, grid_size):
    vis_layer = getattr(model, "visul_emb_layer", None)
    base_model = getattr(model, "base_model", None)
    if vis_layer is None or base_model is None:
        return None
    num_y, num_x = grid_size
    with torch.no_grad():
        if hasattr(model, "_update_reliability_blend"):
            model._update_reliability_blend()
        base_features, atten = base_model.encode_image(images)
        atten = atten.detach().clone()
        base_features = base_features.detach()
        bs = base_features.size(0)
        k = max(1, int((atten.size(1) - 1) * vis_layer.ratio))
        atten[torch.arange(bs, device=atten.device), :, 0] = -1
        atten_topk_raw = atten[:, 0].topk(dim=-1, k=k)[1]
        atten_topk = atten_topk_raw.unsqueeze(-1).expand(bs, k, base_features.size(2))
        selected_features = torch.gather(input=base_features, dim=1, index=atten_topk)
        selected_features = F.normalize(selected_features.float(), p=2, dim=-1).to(dtype=vis_layer.fc.weight.dtype)
        features = vis_layer.fc(selected_features)
        features = vis_layer.mlp(selected_features) + features
        logits = vis_layer.reliability_head(features).squeeze(-1).float()
        selected_atten = atten[:, 0].gather(dim=1, index=atten_topk_raw).detach().float()
        logits = logits + 0.1 * selected_atten
        num_keep = max(1, int(k * vis_layer.topk_reliable_ratio))
        topk_reliable_idx = logits.topk(num_keep, dim=1)[1]
        topk_reliable_mask = torch.zeros_like(logits, dtype=torch.bool)
        topk_reliable_mask.scatter_(1, topk_reliable_idx, True)
        logits = logits.masked_fill(~topk_reliable_mask, -1e4)
        weights = torch.softmax(logits, dim=1)
        full_map = torch.zeros(bs, atten.size(1), device=weights.device, dtype=weights.dtype)
        full_map.scatter_(1, atten_topk_raw, weights)
        patch_map = full_map[:, 1:]
        if patch_map.size(1) != num_y * num_x:
            return None
        return {
            "patch_maps": patch_map.view(bs, num_y, num_x).detach().cpu().numpy(),
            "selected_patch_tokens": atten_topk_raw.detach().cpu().numpy(),
            "selected_patch_weights": weights.detach().cpu().numpy(),
        }


def select_query_indices(topk_indices, qids, gids, num_queries, seed):
    top1_correct = gids[topk_indices[:, 0]].eq(qids).cpu().numpy()
    ok = np.where(top1_correct)[0]
    bad = np.where(~top1_correct)[0]
    rng = np.random.default_rng(seed)
    chosen = []
    if len(bad) > 0:
        for item in rng.choice(bad, size=min(len(bad), max(1, num_queries // 2)), replace=False).tolist():
            if item not in chosen:
                chosen.append(int(item))
    remain = num_queries - len(chosen)
    if remain > 0 and len(ok) > 0:
        for item in rng.choice(ok, size=min(len(ok), remain), replace=False).tolist():
            if item not in chosen:
                chosen.append(int(item))
    if len(chosen) < num_queries:
        all_idx = np.arange(len(qids))
        for item in rng.choice(all_idx, size=min(len(all_idx), num_queries - len(chosen)), replace=False).tolist():
            if item not in chosen:
                chosen.append(int(item))
    return chosen[:num_queries]


def plot_retrieval_examples(ds, args, topk_indices, topk_scores, qids, gids, output_dir: Path, topk: int, num_queries: int, seed: int):
    img_paths = list(ds["img_paths"])
    if getattr(args, "test_dt_type", 1) == 0:
        img_paths = [change_path(args.dataset_name, p) for p in img_paths]
    chosen = select_query_indices(topk_indices, qids, gids, num_queries, seed)
    retrieval_dir = output_dir / "retrieval_examples"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for out_idx, q_idx in enumerate(chosen):
        caption = str(ds["captions"][q_idx]).strip()
        wrap_width = 30 if len(caption) <= 130 else 27
        wrapped_caption = textwrap.fill(caption, width=wrap_width)
        num_caption_lines = wrapped_caption.count("\n") + 1
        fig_height = max(5.3, 4.8 + 0.24 * max(0, num_caption_lines - 4))
        width_ratios = [2.35] + [1.0] * topk
        fig, axes = plt.subplots(
            1,
            topk + 1,
            figsize=(5.0 + 3.2 * topk, fig_height),
            gridspec_kw={"width_ratios": width_ratios},
        )
        axes[0].axis("off")
        pid = int(qids[q_idx].item())
        caption_fontsize = 11.8 if num_caption_lines <= 6 else 11.0
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        axes[0].text(
            0.03,
            0.97,
            f"Query {q_idx}\nPID={pid}",
            va="top",
            ha="left",
            fontsize=16,
            fontweight="bold",
            transform=axes[0].transAxes,
        )
        axes[0].text(
            0.03,
            0.80,
            wrapped_caption,
            va="top",
            ha="left",
            fontsize=caption_fontsize,
            linespacing=1.45,
            transform=axes[0].transAxes,
            bbox={"facecolor": "#F7F7F7", "edgecolor": "#D0D0D0", "boxstyle": "round,pad=0.55"},
        )
        recs = []
        for rank_idx in range(topk):
            gallery_idx = int(topk_indices[q_idx, rank_idx].item())
            score = float(topk_scores[q_idx, rank_idx].item())
            correct = int(gids[gallery_idx].item()) == pid
            img = Image.open(img_paths[gallery_idx]).convert("RGB")
            axes[rank_idx + 1].imshow(img)
            axes[rank_idx + 1].set_xticks([])
            axes[rank_idx + 1].set_yticks([])
            axes[rank_idx + 1].set_title(f"R{rank_idx + 1} | {score:.3f}", color=("green" if correct else "red"), fontsize=10)
            for spine in axes[rank_idx + 1].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3)
                spine.set_edgecolor("green" if correct else "red")
            recs.append({"rank": rank_idx + 1, "gallery_idx": gallery_idx, "pid": int(gids[gallery_idx].item()), "score": score, "correct": bool(correct), "img_path": img_paths[gallery_idx]})
        fig.tight_layout(pad=1.2, w_pad=1.35)
        out_path = retrieval_dir / f"query_{out_idx:02d}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        summary.append({"query_index": int(q_idx), "query_pid": pid, "caption": caption, "output_image": str(out_path), "retrievals": recs})
    with open(output_dir / "retrieval_examples.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def select_gallery_indices(gids, num_images, seed):
    gids_np = gids.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    chosen = []
    unique_ids = np.array(sorted(np.unique(gids_np)))
    if len(unique_ids) > 0:
        sampled_ids = rng.choice(unique_ids, size=min(len(unique_ids), num_images), replace=False)
        for pid in sampled_ids.tolist():
            candidates = np.where(gids_np == pid)[0]
            if len(candidates) == 0:
                continue
            pick = int(rng.choice(candidates))
            if pick not in chosen:
                chosen.append(pick)
    if len(chosen) < num_images:
        for item in rng.choice(np.arange(len(gids_np)), size=min(len(gids_np), num_images - len(chosen)), replace=False).tolist():
            if int(item) not in chosen:
                chosen.append(int(item))
    return chosen[:num_images]


def patch_token_to_rect(token_idx: int, grid_size, image_size):
    num_y, num_x = int(grid_size[0]), int(grid_size[1])
    width, height = int(image_size[0]), int(image_size[1])
    patch_idx = max(0, int(token_idx))
    row = patch_idx // num_x
    col = patch_idx % num_x
    patch_w = width / max(1, num_x)
    patch_h = height / max(1, num_y)
    x0 = col * patch_w
    y0 = row * patch_h
    return x0, y0, patch_w, patch_h


def draw_patch_grid(ax, grid_size, image_size, color="white", linewidth=0.6, alpha=0.35):
    num_y, num_x = int(grid_size[0]), int(grid_size[1])
    width, height = int(image_size[0]), int(image_size[1])
    for gx in range(1, num_x):
        x = width * gx / max(1, num_x)
        ax.plot([x, x], [0, height], color=color, linewidth=linewidth, alpha=alpha)
    for gy in range(1, num_y):
        y = height * gy / max(1, num_y)
        ax.plot([0, width], [y, y], color=color, linewidth=linewidth, alpha=alpha)


def draw_highlighted_patches(ax, top_tokens, top_weights, grid_size, image_size):
    for rank, (token_idx, weight) in enumerate(zip(top_tokens, top_weights), start=1):
        x0, y0, patch_w, patch_h = patch_token_to_rect(int(token_idx), grid_size, image_size)
        rect = Rectangle((x0, y0), patch_w, patch_h, fill=False, edgecolor="yellow", linewidth=2.0)
        ax.add_patch(rect)
        ax.text(
            x0 + patch_w * 0.08,
            y0 + patch_h * 0.20,
            f"{rank}",
            color="black",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "yellow", "edgecolor": "black", "boxstyle": "round,pad=0.15", "alpha": 0.9},
        )
        ax.text(
            x0 + patch_w * 0.08,
            y0 + patch_h * 0.52,
            f"{float(weight):.2f}",
            color="white",
            fontsize=7,
            fontweight="bold",
            bbox={"facecolor": "black", "edgecolor": "none", "boxstyle": "round,pad=0.12", "alpha": 0.65},
        )


def plot_rfe_patch_heatmaps(ds, args, model, gids, output_dir: Path, seed: int, num_images: int, alpha: float, highlight_topk: int, logger):
    if num_images <= 0:
        return
    grid_size = resolve_grid_size(model, getattr(args, "img_size", 224))
    if grid_size is None:
        logger.info("Skip RFE patch heatmaps: unable to determine visual token grid.")
        return
    img_paths = list(ds["img_paths"])
    if getattr(args, "test_dt_type", 1) == 0:
        img_paths = [change_path(args.dataset_name, p) for p in img_paths]
    chosen = select_gallery_indices(gids, num_images, seed)
    if not chosen:
        return
    transform = build_transforms(img_size=args.img_size, aug=False, is_train=False)
    originals = []
    tensors = []
    valid_indices = []
    for gallery_idx in chosen:
        try:
            img = Image.open(img_paths[gallery_idx]).convert("RGB")
        except Exception as exc:
            logger.info(f"Skip heatmap sample {img_paths[gallery_idx]}: {exc}")
            continue
        originals.append(img.copy())
        tensors.append(transform(img))
        valid_indices.append(gallery_idx)
    if not tensors:
        logger.info("Skip RFE patch heatmaps: no valid images loaded.")
        return
    device = next(model.parameters()).device
    batch = torch.stack(tensors, dim=0).to(device)
    patch_data = compute_rfe_patch_data(model, batch, grid_size)
    if patch_data is None:
        logger.info("Skip RFE patch heatmaps: failed to compute patch weights.")
        return
    out_dir = output_dir / "rfe_patch_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_maps = patch_data["patch_maps"]
    selected_patch_tokens = patch_data["selected_patch_tokens"]
    selected_patch_weights = patch_data["selected_patch_weights"]
    summary = []
    for local_idx, gallery_idx in enumerate(valid_indices):
        pid = int(gids[gallery_idx].item())
        image = originals[local_idx]
        image_np = np.asarray(image)
        heatmap = patch_maps[local_idx]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-12)
        heatmap_img = Image.fromarray(np.uint8(np.clip(heatmap * 255.0, 0, 255)), mode="L").resize(image.size, resample=Image.BICUBIC)
        heatmap_np = np.asarray(heatmap_img).astype(np.float32) / 255.0
        token_ids = [int(v) - 1 for v in selected_patch_tokens[local_idx].tolist() if int(v) > 0]
        token_weights = [float(v) for v in selected_patch_weights[local_idx].tolist()]
        ranked_pairs = sorted(zip(token_ids, token_weights), key=lambda item: item[1], reverse=True)
        ranked_pairs = [item for item in ranked_pairs if item[1] > 0]
        top_pairs = ranked_pairs[:max(1, int(highlight_topk))] if ranked_pairs else []
        top_tokens = [item[0] for item in top_pairs]
        top_weights = [item[1] for item in top_pairs]
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
        axes[0].imshow(image_np)
        axes[0].set_title(f"Original\nPID={pid}")
        im = axes[1].imshow(heatmap_np, cmap="jet")
        axes[1].set_title("RFE Patch Importance")
        axes[2].imshow(image_np)
        draw_patch_grid(axes[2], grid_size, image.size)
        draw_highlighted_patches(axes[2], top_tokens, top_weights, grid_size, image.size)
        axes[2].set_title("Patch Grid + Top Patches")
        axes[3].imshow(image_np)
        axes[3].imshow(heatmap_np, cmap="jet", alpha=alpha)
        draw_patch_grid(axes[3], grid_size, image.size, color="white", linewidth=0.5, alpha=0.28)
        draw_highlighted_patches(axes[3], top_tokens, top_weights, grid_size, image.size)
        axes[3].set_title("Overlay + Highlighted Patches")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        fig.tight_layout()
        out_path = out_dir / f"gallery_{gallery_idx:05d}_pid_{pid}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        summary.append({
            "gallery_index": int(gallery_idx),
            "gallery_pid": pid,
            "img_path": img_paths[gallery_idx],
            "output_image": str(out_path),
            "grid_size": [int(grid_size[0]), int(grid_size[1])],
            "selected_patch_tokens": token_ids,
            "selected_patch_weights": token_weights,
            "highlighted_top_patches": [{"patch_index": int(token_idx), "weight": float(weight), "rank": int(rank + 1)} for rank, (token_idx, weight) in enumerate(top_pairs)],
        })
    with open(output_dir / "rfe_patch_heatmaps.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def plot_learning_curves(run_dir: str, output_dir: Path, logger):
    run_dir = str(run_dir or "").strip()
    if not run_dir:
        return
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception:
        logger.info("Skip learning curves: tensorboard is not available.")
        return
    event_files = sorted(Path(run_dir).rglob("events.out.tfevents*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        logger.info("Skip learning curves: no TensorBoard event files found.")
        return
    acc = event_accumulator.EventAccumulator(str(event_files[-1]))
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    wanted = ["loss", "bge_loss", "tse_loss", "R1", "lr", "temperature"]
    available = [tag for tag in wanted if tag in tags]
    if not available:
        logger.info("Skip learning curves: scalar tags not found.")
        return
    cols = 2
    rows = math.ceil(len(available) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).reshape(-1)
    display_names = {"loss": "Loss", "bge_loss": "GE loss", "tse_loss": "RFE loss", "R1": "R1", "lr": "LR", "temperature": "Temperature"}
    y_labels = {
        "loss": "Loss",
        "bge_loss": "Loss",
        "tse_loss": "Loss",
        "R1": "Accuracy (%)",
        "lr": "Learning Rate",
        "temperature": "Temperature",
    }
    for ax, tag in zip(axes, available):
        events = acc.Scalars(tag)
        ax.plot([e.step for e in events], [e.value for e in events], marker="o", linewidth=1.5)
        ax.set_title(display_names.get(tag, tag))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_labels.get(tag, "Value"))
        ax.grid(alpha=0.3)
    for ax in axes[len(available):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def split_prob(prob, threshold=0.5):
    if prob.min() > threshold:
        threshold = np.sort(prob)[len(prob) // 100]
    return (prob > threshold).astype(np.int32)


def plot_ccd(args, model, device, output_dir: Path, batch_size: int, max_batches: int, logger):
    if float(getattr(args, "noisy_rate", 0.0)) <= 0:
        logger.info("Skip CCD plot: noisy_rate <= 0.")
        return
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        logger.info("Skip CCD plot: scikit-learn is not available.")
        return
    vis_args = copy.deepcopy(args)
    vis_args.img_aug = False
    vis_args.txt_aug = False
    dataset = build_dataset(vis_args)
    transform = build_transforms(img_size=vis_args.img_size, aug=False, is_train=False)
    train_set = ImageTextDataset(dataset.train, vis_args, transform=transform, text_length=vis_args.text_length)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=vis_args.num_workers, collate_fn=collate)
    losses_a = np.zeros(len(train_set), dtype=np.float32)
    losses_b = np.zeros(len(train_set), dtype=np.float32)
    filled = np.zeros(len(train_set), dtype=bool)
    for batch_idx, batch in enumerate(train_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        index = batch["index"].detach().cpu().numpy()
        with torch.no_grad():
            loss_a, loss_b, _, _ = model.compute_per_loss(batch)
        losses_a[index] = loss_a.detach().cpu().numpy()
        losses_b[index] = loss_b.detach().cpu().numpy()
        filled[index] = True
    real = np.asarray(train_set.real_correspondences)[filled]
    losses_a = losses_a[filled]
    losses_b = losses_b[filled]
    if len(losses_a) == 0:
        logger.info("Skip CCD plot: no train samples processed.")
        return
    losses_a = (losses_a - losses_a.min()) / (losses_a.max() - losses_a.min() + 1e-12)
    losses_b = (losses_b - losses_b.min()) / (losses_b.max() - losses_b.min() + 1e-12)
    gmm_a = GaussianMixture(n_components=2, random_state=0).fit(losses_a.reshape(-1, 1))
    gmm_b = GaussianMixture(n_components=2, random_state=0).fit(losses_b.reshape(-1, 1))
    prob_a = gmm_a.predict_proba(losses_a.reshape(-1, 1))[:, gmm_a.means_.argmin()]
    prob_b = gmm_b.predict_proba(losses_b.reshape(-1, 1))[:, gmm_b.means_.argmin()]
    pred_a = split_prob(prob_a)
    pred_b = split_prob(prob_b)
    consensus = (pred_a + pred_b > 1).astype(np.int32)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(losses_a[real == 1], bins=40, alpha=0.6, label="clean", color="#4C78A8")
    axes[0].hist(losses_a[real == 0], bins=40, alpha=0.6, label="noisy", color="#E45756")
    axes[0].set_title("GE Loss Distribution")
    axes[0].legend()
    axes[1].hist(losses_b[real == 1], bins=40, alpha=0.6, label="clean", color="#4C78A8")
    axes[1].hist(losses_b[real == 0], bins=40, alpha=0.6, label="noisy", color="#E45756")
    axes[1].set_title("RFE Loss Distribution")
    axes[1].legend()
    scatter_n = min(4000, len(losses_a))
    pick = np.random.default_rng(0).choice(len(losses_a), size=scatter_n, replace=False)
    axes[2].scatter(losses_a[pick], losses_b[pick], c=consensus[pick], cmap="coolwarm", s=10, alpha=0.7)
    axes[2].set_xlabel("GE loss")
    axes[2].set_ylabel("RFE loss")
    axes[2].set_title("CCD consensus split")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ccd_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    cli_args = parse_args()
    output_dir = Path(cli_args.output_dir) if cli_args.output_dir else Path(cli_args.ckpt).resolve().parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    set_seed(cli_args.seed)
    cli_args.retrieval_branch = normalize_branch_name(cli_args.retrieval_branch)
    cli_args.heatmap_alpha = min(max(float(cli_args.heatmap_alpha), 0.0), 1.0)
    args = load_runtime_args(cli_args)
    dataset = build_dataset(args)
    ds = dataset.val if cli_args.val_dataset == "val" else dataset.test
    plot_dataset_overview(dataset_stats(dataset), output_dir)
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    device = torch.device("cuda" if cli_args.device == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = load_model(args, cli_args.ckpt, num_classes, device)
    evaluator = Evaluator(test_img_loader, test_txt_loader)
    q_bge, g_bge, qids, gids = evaluator._compute_embedding(model)
    q_tse, g_tse, _, _ = evaluator._compute_embedding_tse(model)
    q_bge, g_bge = normalize_feats(q_bge, g_bge)
    q_tse, g_tse = normalize_feats(q_tse, g_tse)
    plot_embedding_projection(
        q_bge,
        g_bge,
        q_tse,
        g_tse,
        qids,
        gids,
        output_dir,
        cli_args.seed,
        cli_args.embedding_num_ids,
        cli_args.embedding_max_queries_per_id,
        cli_args.embedding_max_images_per_id,
    )
    metric_rows = []
    cmc_dict = {}
    score_dict = {}
    retrieval_cache = None
    for idx, branch in enumerate(["GE", "RFE", "GE+RFE"]):
        sims = compute_similarity(branch, q_bge, g_bge, q_tse, g_tse)
        row, cmc = compute_metric_row(branch, sims, qids, gids, cli_args.cmc_max_rank)
        metric_rows.append(row)
        cmc_dict[branch] = cmc
        score_dict[branch] = sample_scores(sims, qids, gids, cli_args.score_sample_size, cli_args.seed + idx)
        if branch == cli_args.retrieval_branch:
            topk_scores, topk_indices = torch.topk(sims, k=cli_args.retrieval_topk, dim=1, largest=True, sorted=True)
            retrieval_cache = (topk_indices.cpu(), topk_scores.cpu())
        del sims
    save_metrics(metric_rows, cmc_dict, output_dir)
    plot_cmc(cmc_dict, output_dir)
    plot_score_distributions(score_dict, output_dir)
    if retrieval_cache is not None:
        plot_retrieval_examples(ds, args, retrieval_cache[0], retrieval_cache[1], qids, gids, output_dir, cli_args.retrieval_topk, cli_args.num_retrieval, cli_args.seed)
    plot_rfe_patch_heatmaps(ds, args, model, gids, output_dir, cli_args.seed, cli_args.heatmap_num_images, cli_args.heatmap_alpha, cli_args.heatmap_highlight_topk, logger)
    plot_learning_curves(cli_args.run_dir or str(Path(cli_args.config_file).resolve().parent), output_dir, logger)
    if cli_args.plot_ccd:
        plot_ccd(args, model, device, output_dir, cli_args.ccd_batch_size, cli_args.ccd_max_batches, logger)
    logger.info(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
