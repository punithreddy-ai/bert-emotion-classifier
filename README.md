<div align="center">

<h1>🧠 EmoSense — BERT Emotion Classifier</h1>

<p>Fine-tuning <code>bert-base-uncased</code> for multi-class emotion detection across 6 emotional categories</p>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-~93%25-22C55E?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-6366F1?style=for-the-badge)

</div>

---

## Overview

Fine-tunes BERT on `dair-ai/emotion` to classify English text into 6 emotions: **joy, sadness, love, anger, fear, surprise**.

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~93% |
| Macro F1 | ~92% |
| Weighted F1 | ~93% |

## Project Structure

```
bert-sentiment/
├── train.py                      # Fine-tuning pipeline
├── predict.py                    # Production inference class  
├── app_gradio.py                 # Gradio web UI (localhost:7860)
├── index.html                    # Static demo (no server needed)
├── BERT_Emotion_Analysis.ipynb   # EDA + training + evaluation
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py              # fine-tune BERT (~2-4h CPU / ~30min GPU)
python app_gradio.py         # launch Gradio UI at localhost:7860
```

Or just open `index.html` in your browser for the instant demo.

## Inference

```python
from predict import EmotionPredictor

predictor = EmotionPredictor("./models/bert-emotion")

result = predictor.predict("I just got promoted — I'm over the moon!")
print(result)  # 😄 JOY (96.3%)

results = predictor.predict_batch([
    "I can't stop crying. I miss him so much.",
    "How DARE they treat people like that!",
    "There's something moving in the basement...",
])
for r in results:
    print(f"{r.emoji} {r.label:10s} {r.confidence:.1%}")
# 😢 sadness    94.1%
# 😠 anger      91.7%
# 😨 fear       88.3%
```

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Base model | bert-base-uncased |
| Max length | 128 tokens |
| Batch size | 32 |
| Learning rate | 2e-5 |
| LR schedule | Linear warmup (10%) |
| Epochs | 4 |
| Optimizer | AdamW |
| Grad clip | 1.0 |

## Architecture

```
Input → [BERT Tokenizer] → [bert-base-uncased 12L/12H/768D]
     → [CLS] embedding → [Dropout 0.1] → [Linear 768→6] → [Softmax]
     → probability distribution over 6 emotions
```

## Dataset

`dair-ai/emotion` — 16K train / 2K val / 2K test — English Twitter emotion labels.

## Tech Stack

PyTorch · HuggingFace Transformers · Datasets · scikit-learn · Gradio · Matplotlib

## Future Work

- Try RoBERTa or DistilBERT
- Add class weights for label imbalance  
- LoRA / PEFT efficient fine-tuning
- Deploy as FastAPI REST endpoint
- Push to HuggingFace Hub

---

<div align="center">
Built as a portfolio project · Give it a ⭐ if useful
</div>
