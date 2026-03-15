# ─────────────────────────────────────────────────────────────
#  app.py  —  Flask backend for English → Hindi Translator
#
#  HOW TO RUN:
#    pip install flask flask-cors transformers torch sentencepiece
#    python app.py
#
#  Then open browser → http://127.0.0.1:5000
# ─────────────────────────────────────────────────────────────

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
from transformers import MarianMTModel, MarianTokenizer
import time
import os

app = Flask(__name__)
CORS(app)

# ── LOAD MODEL ─────────────────────────────────────────────────
print("Loading model: Helsinki-NLP/opus-mt-en-hi ...")
t0 = time.time()

model_name = "Helsinki-NLP/opus-mt-en-hi"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer  = MarianTokenizer.from_pretrained(model_name)
model      = MarianMTModel.from_pretrained(model_name).to(device)
model.eval()

print(f"Model loaded on {device} in {time.time()-t0:.1f}s")

# ── SERVE translator.html at http://127.0.0.1:5000 ─────────────
@app.route("/")
def index():
    # Read translator.html from the same folder as app.py
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translator.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return render_template_string(html_content)

# ── TRANSLATE FUNCTION ─────────────────────────────────────────
def translate_en_to_hi(text: str) -> str:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=5
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ── TRANSLATE ENDPOINT ─────────────────────────────────────────
@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    if len(text) > 1000:
        return jsonify({"error": "Text too long (max 1000 characters)"}), 400

    try:
        t0          = time.time()
        translation = translate_en_to_hi(text)
        elapsed     = round(time.time() - t0, 2)

        return jsonify({
            "translation": translation,
            "elapsed_sec": elapsed,
            "device":      str(device),
            "model":       model_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── HEALTH CHECK ───────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model":  model_name,
        "device": str(device)
    })

# ── RUN ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n✅ Open your browser and go to: http://127.0.0.1:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)