from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import time
import math

# Force UTF-8 for Windows
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pipeline import HybridClassifier

app = Flask(__name__, static_folder="dashboard", static_url_path="")
CORS(app)

# Load pipeline once at startup
print("Loading AegisPrompt Hybrid Classifier...")
classifier = HybridClassifier()
print("Classifier ready.")

# ──────────────────────────────────────────────
# Serve dashboard
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("dashboard", "index.html")

# ──────────────────────────────────────────────
# POST /api/scan  — Live Inspector
# Body: { "text": "...", "w_heur": 0.4, "w_model": 0.4, "w_tfidf": 0.2, "threshold": 0.5 }
# ──────────────────────────────────────────────
RULE_DESCRIPTIONS = {
    "1_instruction_override": "Instruction Override — Matches 'ignore/forget/bypass previous instructions'",
    "2_system_prompt_leak":   "System Prompt Leak — Detects attempts to reveal system prompt",
    "3_roleplay_jailbreak":   "Roleplay Jailbreak — Matches DAN, sudo, developer mode persona adoption",
    "4_output_formatting":    "Output Formatting Exploit — base64/hex encode+output patterns",
    "5_virtualization_shell": "Virtualization/Shell — Simulating bash/python terminal",
    "6_reverse_psychology":   "Reverse Psychology — Social engineering via urgency or emotion",
    "7_payload_splitting":    "Payload Splitting — Concat/spell-out obfuscation",
    "8_hypothetical_scenario":"Hypothetical Scenario — Fictional framing to bypass guards",
}

@app.route("/api/scan", methods=["POST"])
def scan():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    w_heur      = float(data.get("w_heur",     0.4))
    w_model     = float(data.get("w_model",    0.4))
    w_tfidf     = float(data.get("w_tfidf",    0.2))
    threshold   = float(data.get("threshold",  0.5))

    t_start = time.perf_counter()
    result  = classifier.predict(
        text,
        w_heur=w_heur, w_model=w_model, w_tfidf=w_tfidf,
        threshold=threshold
    )
    latency_ms = round((time.perf_counter() - t_start) * 1000, 1)

    # Build rules table
    rules_triggered = result["details"]["rules_triggered"]
    rules_table = [
        {
            "id":          r,
            "description": RULE_DESCRIPTIONS.get(r, r),
            "risk":        "High" if r in ("1_instruction_override", "3_roleplay_jailbreak") else "Medium",
        }
        for r in rules_triggered
    ]

    # Entropy (Shannon) as a simple text complexity indicator
    from collections import Counter
    freq = Counter(text.lower())
    total = sum(freq.values())
    entropy = -sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0)
    entropy = round(entropy, 2)

    # Complexity label
    words = len(text.split())
    complexity = "Low" if words < 15 else ("Medium" if words < 40 else "High")

    return jsonify({
        "prediction":   result["prediction"],
        "decision":     result["decision"],
        "score":        round(result["score"], 4),
        "heuristic":    round(result["details"]["heuristic_score"], 4),
        "model_prob":   round(result["details"]["model_prob"] or 0, 4),
        "tfidf":        round(result["details"]["tfidf_score"] or 0, 4),
        "rules":        rules_table,
        "latency_ms":   latency_ms,
        "entropy":      entropy,
        "complexity":   complexity,
        "flagged_substrings": len(rules_triggered),
    })

# ──────────────────────────────────────────────
# POST /api/batch  — Batch Analysis
# Body: { "prompts": ["...", "..."] }
# ──────────────────────────────────────────────
@app.route("/api/batch", methods=["POST"])
def batch():
    data = request.get_json(force=True)
    prompts = data.get("prompts", [])
    if not prompts:
        return jsonify({"error": "No prompts provided"}), 400

    results = []
    for i, text in enumerate(prompts):
        res = classifier.predict(text.strip())
        results.append({
            "id":         i + 1,
            "text":       text[:120],
            "prediction": res["prediction"],
            "decision":   res["decision"],
            "score":      round(res["score"], 4),
            "rules":      res["details"]["rules_triggered"],
            "model_prob": round(res["details"]["model_prob"] or 0, 4),
        })

    total       = len(results)
    injections  = sum(1 for r in results if r["prediction"] == 1)
    benign      = total - injections
    avg_score   = round(sum(r["score"] for r in results) / total, 4) if total else 0

    # Score distribution buckets (0-1 in 10 bins)
    bins = [0] * 10
    for r in results:
        bucket = min(int(r["score"] * 10), 9)
        bins[bucket] += 1

    return jsonify({
        "total":        total,
        "injections":   injections,
        "benign":       benign,
        "avg_score":    avg_score,
        "items":        results,
        "histogram":    bins,
    })

# ──────────────────────────────────────────────
# GET /api/metrics  — Diagnostic Metrics Center
# Loads from results/metrics.json
# ──────────────────────────────────────────────
import json as _json

@app.route("/api/metrics", methods=["GET"])
def metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), "results", "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({"error": "metrics.json not found. Run evaluate_pipeline.py first."}), 404

    with open(metrics_path, "r") as f:
        m = _json.load(f)

    # Derive confusion matrix values from available data (67-sample test set)
    total_test        = 67
    total_benign      = 40
    total_injection   = 27
    tp = round(m["test_recall"] * total_injection)
    fn = total_injection - tp
    fp = m["test_set_fps"]
    tn = total_benign - fp

    return jsonify({
        "accuracy":             round(m["test_accuracy"] * 100, 2),
        "f1":                   round(m["test_f1"] * 100, 2),
        "precision":            round(m["test_precision"] * 100, 2),
        "recall":               round(m["test_recall"] * 100, 2),
        "auc":                  round(m["test_auc"], 4),
        "red_team_total":       m["red_team_total"],
        "red_team_blocked":     m["red_team_blocked"],
        "red_team_block_rate":  round(m["red_team_block_rate"], 1),
        "tricky_fps":           m["tricky_benign_fps"],
        "test_fps":             m["test_set_fps"],
        "confusion_matrix":     {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    })

# ──────────────────────────────────────────────
# GET /api/false_positives  — FP inspector data
# ──────────────────────────────────────────────
@app.route("/api/false_positives", methods=["GET"])
def false_positives():
    fp_path = os.path.join(os.path.dirname(__file__), "results", "false_positives.md")
    if not os.path.exists(fp_path):
        return jsonify({"items": []})

    # Parse the false_positives.md for flagged entries
    items = []
    with open(fp_path, "r", encoding="utf-8") as f:
        content = f.read()

    import re
    # Extract flagged prompt blocks
    blocks = re.findall(
        r'\[FLAGGED\].*?\n- \*\*Text\*\*: `(.+?)`\n- \*\*Combined Score\*\*: `(.+?)`\n- \*\*Heuristics Rules\*\*: `(.+?)`',
        content
    )
    for text, score, rules in blocks:
        rules_list = eval(rules) if rules.startswith("[") else [rules] if rules != "[]" else []
        items.append({
            "text":  text[:100],
            "score": float(score),
            "rules": rules_list,
            "flag_type": rules_list[0].replace("_", " ").title() if rules_list else "Semantic Shadow"
        })

    return jsonify({"items": items})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*60)
    print(f"  AegisPrompt Dashboard running at http://localhost:{port}")
    print("="*60 + "\n")
    app.run(debug=False, port=port, host="0.0.0.0")
