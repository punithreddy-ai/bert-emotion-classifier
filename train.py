"""
Fine-tuning BERT for Multi-Class Emotion Detection
Dataset : dair-ai/emotion  (6 classes: sadness, joy, love, anger, fear, surprise)
Author  : Your Name
"""

import os, json, logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────── Config ────────────────────────────────────────

class Config:
    MODEL_NAME   = "bert-base-uncased"
    DATASET_NAME = "dair-ai/emotion"
    OUTPUT_DIR   = "./models/bert-emotion"
    LOGS_DIR     = "./logs"
    MAX_LEN      = 128
    BATCH_SIZE   = 32
    EPOCHS       = 1
    LR           = 2e-5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    DROPOUT      = 0.1
    SEED         = 42
    LABELS       = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    NUM_LABELS   = 6
    ID2LABEL     = {i: l for i, l in enumerate(["sadness","joy","love","anger","fear","surprise"])}
    LABEL2ID     = {l: i for i, l in enumerate(["sadness","joy","love","anger","fear","surprise"])}


cfg = Config()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(cfg.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# ──────────────────────────── Dataset ───────────────────────────────────────

def load_and_tokenize(tokenizer):
    logger.info("Loading dair-ai/emotion …")
    ds = load_dataset(cfg.DATASET_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=cfg.MAX_LEN,
        )

    tokenized = ds.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return tokenized


# ──────────────────────────── Metrics ───────────────────────────────────────

def compute_metrics(preds, labels):
    return {
        "accuracy":     accuracy_score(labels, preds),
        "f1_macro":     f1_score(labels, preds, average="macro"),
        "f1_weighted":  f1_score(labels, preds, average="weighted"),
    }


# ──────────────────────────── Train / eval ──────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, epoch: int):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc=f"Epoch {epoch} [train]"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        out = model(**batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
        all_preds.extend(out.logits.argmax(-1).cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    return total_loss / len(loader), compute_metrics(all_preds, all_labels)


@torch.no_grad()
def evaluate(model, loader, split: str = "val"):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc=f"  [{split}]"):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total_loss += out.loss.item()
        all_preds.extend(out.logits.argmax(-1).cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    return (
        total_loss / len(loader),
        compute_metrics(all_preds, all_labels),
        all_preds,
        all_labels,
    )


# ──────────────────────────── Visualisation ─────────────────────────────────

def plot_curves(history: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BERT Fine-tuning — Emotion Detection", fontsize=14, fontweight="bold")

    axes[0].plot(ep, history["train_loss"], "o-", label="Train", color="#2563EB")
    axes[0].plot(ep, history["val_loss"],   "s--", label="Val",  color="#DC2626")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-Entropy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["train_f1"], "o-", label="Train", color="#2563EB")
    axes[1].plot(ep, history["val_f1"],   "s--", label="Val",  color="#DC2626")
    axes[1].set(title="Macro F1", xlabel="Epoch", ylabel="F1 Score", ylim=[0, 1])
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    logger.info(f"Saved → {path}")
    plt.close()


def plot_confusion_matrix(labels, preds, save_dir: str):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cfg.LABELS, yticklabels=cfg.LABELS, ax=ax)
    ax.set(title="Confusion Matrix — Test Set", xlabel="Predicted", ylabel="True")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    logger.info(f"Saved → {path}")
    plt.close()


# ──────────────────────────── Entry point ───────────────────────────────────

def main():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.LOGS_DIR, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(cfg.MODEL_NAME)
    ds        = load_and_tokenize(tokenizer)

    # num_workers=0 for Windows compatibility (avoids multiprocessing spawn issues)
    # pin_memory only helps with CUDA; disabled here for CPU/Windows safety
    train_loader = DataLoader(ds["train"],      batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(ds["validation"], batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(ds["test"],       batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    logger.info(f"Sizes — Train: {len(ds['train'])} | Val: {len(ds['validation'])} | Test: {len(ds['test'])}")

    model = BertForSequenceClassification.from_pretrained(
        cfg.MODEL_NAME,
        num_labels=cfg.NUM_LABELS,
        id2label=cfg.ID2LABEL,
        label2id=cfg.LABEL2ID,
        hidden_dropout_prob=cfg.DROPOUT,
        attention_probs_dropout_prob=cfg.DROPOUT,
    ).to(device)

    total_steps  = len(train_loader) * cfg.EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    optimizer    = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY, eps=1e-8)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history   = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1   = 0.0
    best_epoch = 0

    for epoch in range(1, cfg.EPOCHS + 1):
        tr_loss, tr_m        = train_epoch(model, train_loader, optimizer, scheduler, epoch)
        vl_loss, vl_m, _, _ = evaluate(model, val_loader)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_f1"].append(tr_m["f1_macro"])
        history["val_f1"].append(vl_m["f1_macro"])

        logger.info(
            f"Epoch {epoch:02d} | "
            f"tr_loss={tr_loss:.4f}  tr_acc={tr_m['accuracy']:.4f}  tr_f1={tr_m['f1_macro']:.4f} | "
            f"vl_loss={vl_loss:.4f}  vl_acc={vl_m['accuracy']:.4f}  vl_f1={vl_m['f1_macro']:.4f}"
        )

        if vl_m["f1_macro"] > best_f1:
            best_f1, best_epoch = vl_m["f1_macro"], epoch
            model.save_pretrained(cfg.OUTPUT_DIR)
            tokenizer.save_pretrained(cfg.OUTPUT_DIR)
            logger.info(f"  ✓ Best model saved (val F1 = {best_f1:.4f})")

    logger.info(f"\nBest epoch = {best_epoch}  |  val F1 = {best_f1:.4f}")

    # ── Test set ──
    best_model = BertForSequenceClassification.from_pretrained(cfg.OUTPUT_DIR).to(device)
    te_loss, te_m, te_preds, te_labels = evaluate(best_model, test_loader, "test")
    logger.info(f"Test | loss={te_loss:.4f}  acc={te_m['accuracy']:.4f}  f1={te_m['f1_macro']:.4f}")
    print("\n" + classification_report(te_labels, te_preds, target_names=cfg.LABELS))

    # ── Save artefacts ──
    plot_curves(history, cfg.LOGS_DIR)
    plot_confusion_matrix(te_labels, te_preds, cfg.LOGS_DIR)

    results = {"best_epoch": best_epoch, **te_m, "history": history}
    with open(os.path.join(cfg.LOGS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Training pipeline complete ✓")


if __name__ == "__main__":
    main()
