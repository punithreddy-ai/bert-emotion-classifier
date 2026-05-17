"""
Inference module — BERT Emotion Classifier
Usage:
    from src.predict import EmotionPredictor
    predictor = EmotionPredictor("./models/bert-emotion")
    result = predictor.predict("I'm so excited about this!")
    results = predictor.predict_batch(["I love this", "I'm terrified"])
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification
from dataclasses import dataclass
from typing import Union


LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOJI_MAP = {
    "sadness":  "😢",
    "joy":      "😄",
    "love":     "❤️",
    "anger":    "😠",
    "fear":     "😨",
    "surprise": "😲",
}


@dataclass
class EmotionResult:
    text: str
    label: str
    emoji: str
    confidence: float
    all_scores: dict[str, float]

    def __str__(self) -> str:
        top = f"{self.emoji} {self.label.upper()} ({self.confidence:.1%})"
        breakdown = "  ".join(f"{l}: {s:.2%}" for l, s in self.all_scores.items())
        return f"{top}\n  {self.text!r}\n  [{breakdown}]"


class EmotionPredictor:
    """
    Production-ready inference class for the fine-tuned BERT emotion model.

    Args:
        model_path : Path to the saved model directory (from BertForSequenceClassification.save_pretrained)
        device     : 'cuda', 'cpu', or 'auto' (default). Auto selects GPU if available.
        max_len    : Maximum tokenisation length (default 128).
    """

    def __init__(
        self,
        model_path: str = "./models/bert-emotion",
        device: str = "auto",
        max_len: int = 128,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device  = torch.device(device)
        self.max_len = max_len

        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label  # {0: "sadness", …}

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, text: str) -> EmotionResult:
        """Predict emotion for a single text string."""
        return self.predict_batch([text])[0]

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> list[EmotionResult]:
        """
        Predict emotions for a list of texts.
        Automatically batches to avoid OOM on long lists.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            results.extend(self._infer_chunk(chunk))
        return results

    def top_k(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """Return the top-k emotions sorted by probability."""
        result = self.predict(text)
        sorted_scores = sorted(result.all_scores.items(), key=lambda x: -x[1])
        return sorted_scores[:k]

    # ── Internal ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer_chunk(self, texts: list[str]) -> list[EmotionResult]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits                        # (B, num_labels)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()        # (B, num_labels)

        results = []
        for text, prob_row in zip(texts, probs):
            idx   = int(np.argmax(prob_row))
            label = self.id2label[idx]
            scores = {self.id2label[j]: float(p) for j, p in enumerate(prob_row)}
            results.append(
                EmotionResult(
                    text=text,
                    label=label,
                    emoji=EMOJI_MAP.get(label, "🤔"),
                    confidence=float(prob_row[idx]),
                    all_scores=scores,
                )
            )
        return results


# ── CLI demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="BERT Emotion Predictor")
    parser.add_argument("--model", default="./models/bert-emotion", help="Model directory")
    parser.add_argument("--text",  nargs="+", help="Text(s) to classify")
    args = parser.parse_args()

    predictor = EmotionPredictor(args.model)

    texts = args.text or [
        "I just got promoted — I'm over the moon!",
        "I can't believe they cancelled the show, I'm devastated.",
        "There's something moving in the dark and I'm terrified.",
        "How dare they treat people like that!",
        "Wait — you're getting married? When?!",
        "You are my sunshine, my only sunshine.",
    ]

    print(f"\n{'─' * 60}")
    for t in texts:
        r = predictor.predict(t)
        print(r)
        print()
