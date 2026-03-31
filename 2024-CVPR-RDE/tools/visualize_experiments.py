import argparse
import copy
import csv
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    parser.add_argument("--retrieval_branch", type=str, default="BGE+TSE", choices=["BGE", "TSE", "BGE+TSE"])
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


def compute_similarity(branch, q_bge, g_bge, q_tse, g_tse):
    if branch == "BGE":
        return q_bge @ g_bge.t()
    if branch == "TSE":
        return q_tse @ g_tse.t()
    return 0.5 * ((q_bge @ g_bge.t()) + (q_tse @ g_tse.t()))


def compute_metric_row(branch, sims, qids, gids, max_rank):
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
        "BGE": (q_bge.detach().cpu().numpy(), g_bge.detach().cpu().numpy()),
        "TSE": (q_tse.detach().cpu().numpy(), g_tse.detach().cpu().numpy()),
    }
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(selected_ids))))
    fig, axes = plt.subplots(1, len(branch_feats), figsize=(7 * len(branch_feats), 6))
    if len(branch_feats) == 1:
        axes = [axes]

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
    axes[0].legend(handles=pid_handles, loc="best", fontsize=8)
    axes[-1].legend(handles=modality_handles, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "embedding_projection.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "selected_ids": selected_ids,
        "num_query_points": int(len(selected_q)),
        "num_gallery_points": int(len(selected_g)),
    }
    with open(output_dir / "embedding_projection.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


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
        fig, axes = plt.subplots(1, topk + 1, figsize=(3.3 * (topk + 1), 4.4))
        axes[0].axis("off")
        caption = ds["captions"][q_idx]
        pid = int(qids[q_idx].item())
        axes[0].text(0.0, 1.0, f"Query {q_idx}\nPID={pid}", va="top", fontsize=11, fontweight="bold")
        axes[0].text(0.0, 0.82, caption, va="top", wrap=True, fontsize=10)
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
        fig.tight_layout()
        out_path = retrieval_dir / f"query_{out_idx:02d}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        summary.append({"query_index": int(q_idx), "query_pid": pid, "caption": caption, "output_image": str(out_path), "retrievals": recs})
    with open(output_dir / "retrieval_examples.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


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
    for ax, tag in zip(axes, available):
        events = acc.Scalars(tag)
        ax.plot([e.step for e in events], [e.value for e in events], marker="o", linewidth=1.5)
        ax.set_title(tag)
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
    axes[0].set_title("BGE Loss Distribution")
    axes[0].legend()
    axes[1].hist(losses_b[real == 1], bins=40, alpha=0.6, label="clean", color="#4C78A8")
    axes[1].hist(losses_b[real == 0], bins=40, alpha=0.6, label="noisy", color="#E45756")
    axes[1].set_title("TSE Loss Distribution")
    axes[1].legend()
    scatter_n = min(4000, len(losses_a))
    pick = np.random.default_rng(0).choice(len(losses_a), size=scatter_n, replace=False)
    axes[2].scatter(losses_a[pick], losses_b[pick], c=consensus[pick], cmap="coolwarm", s=10, alpha=0.7)
    axes[2].set_xlabel("BGE loss")
    axes[2].set_ylabel("TSE loss")
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
    for idx, branch in enumerate(["BGE", "TSE", "BGE+TSE"]):
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
    plot_learning_curves(cli_args.run_dir or str(Path(cli_args.config_file).resolve().parent), output_dir, logger)
    if cli_args.plot_ccd:
        plot_ccd(args, model, device, output_dir, cli_args.ccd_batch_size, cli_args.ccd_max_batches, logger)
    logger.info(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
