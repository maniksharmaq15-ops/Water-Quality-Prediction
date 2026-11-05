import json, os
import joblib, pandas as pd
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)
ART = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
MODEL, THR, FEATURES = ART["model"], ART["threshold"], ART["features"]
try:
    METRICS = json.load(open(os.path.join(os.path.dirname(__file__), "metrics.json")))
except Exception:
    METRICS = {}

# plausible physical ranges, used for input validation
RANGES = {"ph": (0, 14), "Hardness": (0, 500), "Solids": (0, 100000), "Chloramines": (0, 20),
          "Sulfate": (0, 600), "Conductivity": (0, 1000), "Organic_carbon": (0, 40),
          "Trihalomethanes": (0, 200), "Turbidity": (0, 10)}

def predict(values: dict):
    row, errors = {}, []
    for f in FEATURES:
        raw = values.get(f)
        if raw in (None, ""):
            row[f] = float("nan")  # model imputes missing values
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{f} must be a number"); continue
        lo, hi = RANGES[f]
        if not lo <= v <= hi:
            errors.append(f"{f} must be between {lo} and {hi}")
        row[f] = v
    if errors:
        return None, errors
    prob = float(MODEL.predict_proba(pd.DataFrame([row], columns=FEATURES))[0, 1])
    return {"potable": prob >= THR, "probability_potable": round(prob, 3), "threshold": round(THR, 3)}, None

PAGE = """<!doctype html><title>Water Quality Prediction</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>body{font-family:system-ui;max-width:560px;margin:2rem auto;padding:0 1rem}label{display:block;margin:.6rem 0 .2rem}
input{width:100%;padding:.5rem;box-sizing:border-box}button{margin-top:1rem;padding:.7rem 1.2rem}
.r{margin-top:1rem;padding:1rem;border-radius:8px;background:#eef}.e{background:#fee}</style>
<h1>Water Quality Prediction</h1><p>Leave a field empty if unknown.</p>
<form method=post action="/">{% for f in features %}<label>{{f}} ({{ranges[f][0]}}-{{ranges[f][1]}})
<input name="{{f}}" type=number step=any value="{{vals.get(f,'')}}"></label>{% endfor %}
<button>Predict</button></form>
{% if result %}<div class=r><b>{{'Potable' if result.potable else 'Not potable'}}</b>
(probability {{result.probability_potable}}, threshold {{result.threshold}})</div>{% endif %}
{% for e in errors or [] %}<div class="r e">{{e}}</div>{% endfor %}
<p><small>Educational model. Do not use to decide if real water is safe to drink.</small></p>"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = errors = None
    vals = request.form if request.method == "POST" else {}
    if request.method == "POST":
        result, errors = predict(vals)
    return render_template_string(PAGE, features=FEATURES, ranges=RANGES, vals=vals, result=result, errors=errors)

@app.post("/predict")
def api_predict():
    result, errors = predict(request.get_json(silent=True) or {})
    return (jsonify(errors=errors), 400) if errors else jsonify(result)

@app.get("/health")
def health():
    return jsonify(status="ok", **METRICS)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
