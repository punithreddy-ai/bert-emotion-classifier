"""
Gradio Demo — BERT Emotion Classifier
Run: python app_gradio.py
Then open: http://localhost:7860
"""

import gradio as gr
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from transformers import BertTokenizerFast, BertForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "./models/bert-emotion"   # path to your fine-tuned model
LABELS     = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOJIS     = {"sadness": "😢", "joy": "😄", "love": "❤️", "anger": "😠", "fear": "😨", "surprise": "😲"}
COLORS     = {"sadness": "#3B82F6", "joy": "#F59E0B", "love": "#EC4899",
              "anger": "#EF4444", "fear": "#8B5CF6", "surprise": "#10B981"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model from {MODEL_PATH} …")
try:
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    model     = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    MODEL_LOADED = True
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"⚠ Could not load fine-tuned model ({e})\n  → Running in demo mode with base BERT")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model     = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=6,
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id={l: i for i, l in enumerate(LABELS)},
    ).to(device)
    model.eval()
    MODEL_LOADED = False


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(text: str):
    if not text or not text.strip():
        return {}, None, "Please enter some text."

    enc = tokenizer(
        text, padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    )
    enc    = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits
    probs  = F.softmax(logits, dim=-1)[0].cpu().numpy()

    scores = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    top    = max(scores, key=scores.get)

    # ── Bar chart ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#13131A")
    ax.set_facecolor("#13131A")

    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    labels_sorted = [l for l, _ in sorted_items]
    values_sorted = [v for _, v in sorted_items]
    bar_colors    = [COLORS[l] for l in labels_sorted]

    bars = ax.barh(labels_sorted, values_sorted, color=bar_colors, height=0.55)

    for bar, val in zip(bars, values_sorted):
        ax.text(
            min(val + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", ha="left",
            color="white", fontsize=10, fontweight="bold"
        )

    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Probability", color="#6B6880", fontsize=10)
    ax.tick_params(colors="white", labelsize=11)
    ax.spines[:].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.invert_yaxis()

    # Emoji labels
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(
        [f"{EMOJIS[l]}  {l.capitalize()}" for l in labels_sorted],
        color="white", fontsize=11
    )

    plt.tight_layout(pad=1.2)

    # ── Summary text ─────────────────────────────────────────────────────────
    conf    = scores[top]
    summary = f"{EMOJIS[top]}  **{top.upper()}**  —  {conf:.1%} confidence"
    if not MODEL_LOADED:
        summary += "\n\n⚠️ *Demo mode: train the model first for accurate results.*"

    return scores, fig, summary


# ── Examples ──────────────────────────────────────────────────────────────────
EXAMPLES = [
    ["I just got promoted — I'm absolutely over the moon!"],
    ["I can't stop crying. I miss him so much."],
    ["How DARE they treat people like that — completely unacceptable!"],
    ["There's something in the basement and it keeps making noises at night."],
    ["Wait — you're getting married?! I had NO idea!"],
    ["You are the most important person in my world, always."],
    ["I feel so empty and hopeless. Nothing seems to matter anymore."],
    ["I can't believe I actually won! This is incredible!"],
]


# ── Gradio UI ─────────────────────────────────────────────────────────────────

css = """
body { background: #0A0A0F; }
.gradio-container { background: #0A0A0F !important; font-family: 'Inter', sans-serif; }
#title { text-align: center; color: white; margin-bottom: 0.5rem; }
#subtitle { text-align: center; color: #6B6880; font-size: 14px; margin-bottom: 1.5rem; }
.gr-button-primary { background: white !important; color: black !important; font-weight: 600 !important; }
"""

with gr.Blocks(css=css, title="EmoSense — BERT Emotion Detector", theme=gr.themes.Base(
    primary_hue="purple",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
)) as demo:

    gr.Markdown("# 🧠 EmoSense — BERT Emotion Detector", elem_id="title")
    gr.Markdown(
        "Fine-tuned `bert-base-uncased` on **16,000** English texts · "
        "6 emotion classes · ~93% test accuracy",
        elem_id="subtitle"
    )

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Type anything… e.g. 'I can't believe we actually made it!'",
                lines=4,
                max_lines=8,
            )
            analyze_btn = gr.Button("Analyze Emotion →", variant="primary", size="lg")

            gr.Markdown("### Try an example")
            gr.Examples(
                examples=EXAMPLES,
                inputs=text_input,
                label="",
            )

            with gr.Accordion("Model info", open=False):
                gr.Markdown(f"""
**Base model:** `bert-base-uncased` (110M params)  
**Dataset:** `dair-ai/emotion` — 16K train / 2K val / 2K test  
**Classes:** sadness, joy, love, anger, fear, surprise  
**Training:** 4 epochs · AdamW lr=2e-5 · linear warmup  
**Test accuracy:** ~93% · Macro F1: ~92%  
**Device:** `{device}`  
**Model loaded:** {"✅ Fine-tuned" if MODEL_LOADED else "⚠️ Base (not fine-tuned)"}
                """)

        with gr.Column(scale=1):
            summary_out = gr.Markdown(label="Result")
            chart_out   = gr.Plot(label="Emotion Probabilities")
            scores_out  = gr.Label(label="All scores", num_top_classes=6)

    analyze_btn.click(
        fn=predict,
        inputs=text_input,
        outputs=[scores_out, chart_out, summary_out],
    )
    text_input.submit(
        fn=predict,
        inputs=text_input,
        outputs=[scores_out, chart_out, summary_out],
    )

    gr.Markdown(
        "---\n"
        "*Built with PyTorch + HuggingFace Transformers · "
        "[dair-ai/emotion dataset](https://huggingface.co/datasets/dair-ai/emotion)*"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,        # set True to get a public URL
        show_error=True,
        inbrowser=True,     # auto-opens browser
    )
