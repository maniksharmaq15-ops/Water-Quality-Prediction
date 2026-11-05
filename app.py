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
 
SAMPLES = {
    "good": {"ph": 7.17, "Hardness": 203.41, "Solids": 20401.1, "Chloramines": 7.68, "Sulfate": 287.09, "Conductivity": 315.55,
             "Organic_carbon": 14.53, "Trihalomethanes": 74.41, "Turbidity": 3.94},
    "bad": {"ph": 10.22, "Hardness": 248.07, "Solids": 28749.72, "Chloramines": 7.51, "Sulfate": 393.66, "Conductivity": 283.65,
            "Organic_carbon": 13.79, "Trihalomethanes": 84.6, "Turbidity": 2.67},
}
 
PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AquaCheck | Water Potability Predictor</title>
<style>
:root{--bg:#f4f8fb;--card:#fff;--ink:#0f2233;--mute:#5b7183;--line:#dce6ee;--brand:#0b7ac7;--brand2:#12b5a6;--ok:#12915a;--bad:#d64545;--okbg:#e6f7ef;--badbg:#fdecec}
@media(prefers-color-scheme:dark){:root{--bg:#0b1620;--card:#122331;--ink:#e8f1f7;--mute:#92a8b8;--line:#223a4c;--okbg:#0f3326;--badbg:#3a1a1d;--ok:#3ddc97;--bad:#ff7a7a}}
*{box-sizing:border-box}body{margin:0;font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{background:linear-gradient(135deg,var(--brand),var(--brand2));color:#fff;padding:2.2rem 1rem 4.5rem;text-align:center}
header h1{margin:0;font-size:clamp(1.6rem,4vw,2.3rem);letter-spacing:-.5px}header p{margin:.4rem auto 0;max-width:560px;opacity:.92}
main{max-width:980px;margin:-3rem auto 2rem;padding:0 1rem;display:grid;gap:1.2rem;grid-template-columns:1.6fr 1fr}
@media(max-width:820px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:1.3rem;box-shadow:0 6px 24px rgba(10,40,70,.08)}
.card h2{margin:0 0 .2rem;font-size:1.1rem}.sub{color:var(--mute);font-size:.88rem;margin:0 0 1rem}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:.8rem}@media(max-width:480px){.grid{grid-template-columns:1fr}}
label{font-size:.82rem;font-weight:600;color:var(--mute);display:block;margin-bottom:.2rem}
.f small{font-weight:400}
input{width:100%;padding:.6rem .7rem;border:1px solid var(--line);border-radius:10px;background:var(--bg);color:var(--ink);font-size:1rem}
input:focus{outline:2px solid var(--brand);border-color:transparent}input.bad{border-color:var(--bad)}
.row{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:1.1rem}
button{cursor:pointer;border:0;border-radius:10px;padding:.7rem 1.1rem;font-size:.95rem;font-weight:600}
.go{background:linear-gradient(135deg,var(--brand),var(--brand2));color:#fff;flex:1;min-width:140px}
.ghost{background:transparent;border:1px solid var(--line);color:var(--ink)}button:disabled{opacity:.6;cursor:wait}
#res{text-align:center}.badge{display:inline-block;padding:.5rem 1rem;border-radius:999px;font-weight:700;font-size:1.15rem;margin:.6rem 0}
.ok{background:var(--okbg);color:var(--ok)}.no{background:var(--badbg);color:var(--bad)}
.bar{position:relative;height:12px;border-radius:99px;background:linear-gradient(90deg,var(--bad),#f2c14e,var(--ok));margin:1.6rem 0 .4rem}
.bar i{position:absolute;top:-5px;width:4px;height:22px;background:var(--ink);border-radius:2px;transform:translateX(-50%);transition:left .5s}
.bar b{position:absolute;top:-20px;font-size:.7rem;color:var(--mute);transform:translateX(-50%);font-weight:600}
.lab{display:flex;justify-content:space-between;font-size:.75rem;color:var(--mute)}
.empty{color:var(--mute);padding:2rem 0}.err{color:var(--bad);font-size:.9rem;text-align:left}
.stat{display:flex;justify-content:space-between;padding:.45rem 0;border-bottom:1px solid var(--line);font-size:.9rem}.stat:last-child{border:0}
.note{font-size:.78rem;color:var(--mute);margin-top:1rem}footer{text-align:center;color:var(--mute);font-size:.8rem;padding:0 1rem 2rem}
</style></head><body>
<header><h1>AquaCheck</h1><p>Enter water-quality measurements and a machine-learning model estimates whether the water is potable.</p></header>
<main>
<section class="card"><h2>Water measurements</h2><p class="sub">Leave a field empty if unknown. Missing values are estimated automatically.</p>
<form id="f" novalidate><div class="grid">
{% for f in features %}<div class="f"><label for="{{f}}">{{labels[f]}} <small>({{ranges[f][0]}}&ndash;{{ranges[f][1]}})</small></label>
<input id="{{f}}" name="{{f}}" type="number" step="any" inputmode="decimal"></div>{% endfor %}
</div><div class="row"><button class="go" id="go" type="submit">Predict potability</button>
<button class="ghost" type="button" data-s="good">Sample: safe</button><button class="ghost" type="button" data-s="bad">Sample: unsafe</button>
<button class="ghost" type="button" id="clr">Clear</button></div><div class="err" id="err"></div></form></section>
<aside><section class="card" id="res"><h2>Result</h2><div class="empty" id="empty">Fill the form and press <b>Predict</b>.</div>
<div id="out" hidden><div class="badge" id="badge"></div><div>Probability potable: <b id="pp"></b></div>
<div class="bar"><i id="mk"></i><b id="tk"></b></div><div class="lab"><span>Not potable</span><span>Potable</span></div>
<p class="note">The marker shows the probability; the tick shows the decision threshold.</p></div></section>
<section class="card" style="margin-top:1.2rem"><h2>About the model</h2><p class="sub">Measured on a held-out test set</p>
<div class="stat"><span>Model</span><b>{{m.get('model','-')}}</b></div><div class="stat"><span>ROC-AUC</span><b>{{m.get('roc_auc','-')}}</b></div>
<div class="stat"><span>Recall (potable)</span><b>{{m.get('recall','-')}}</b></div><div class="stat"><span>F1 score</span><b>{{m.get('f1','-')}}</b></div>
<p class="note">Educational project. Signal in this dataset is weak, so treat results as indicative only. Never use it to decide whether real water is safe to drink.</p></section></aside>
</main><footer>Water Quality Prediction &middot; B.Tech CSE final-year project</footer>
<script>
const S={good:__GOOD__,bad:__BAD__},F=document.getElementById('f'),$=i=>document.getElementById(i);
document.querySelectorAll('[data-s]').forEach(b=>b.onclick=()=>{const s=S[b.dataset.s];for(const k in s)$(k).value=s[k];$('err').textContent=''});
$('clr').onclick=()=>{F.reset();$('out').hidden=true;$('empty').hidden=false;$('err').textContent='';document.querySelectorAll('input').forEach(i=>i.classList.remove('bad'))};
F.onsubmit=async e=>{e.preventDefault();$('err').textContent='';const body={};
document.querySelectorAll('input').forEach(i=>{i.classList.remove('bad');body[i.name]=i.value});
$('go').disabled=true;$('go').textContent='Predicting...';
try{const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();
if(!r.ok){$('err').innerHTML=(d.errors||['Something went wrong']).map(x=>'&bull; '+x).join('<br>');
(d.errors||[]).forEach(x=>{const el=$(x.split(' ')[0]);if(el)el.classList.add('bad')});return}
$('empty').hidden=true;$('out').hidden=false;const b=$('badge');b.textContent=d.potable?'Likely potable':'Likely not potable';b.className='badge '+(d.potable?'ok':'no');
$('pp').textContent=(d.probability_potable*100).toFixed(1)+'%';$('mk').style.left=(d.probability_potable*100)+'%';
$('tk').style.left=(d.threshold*100)+'%';$('tk').textContent='|'}
catch(x){$('err').textContent='Network error. If the site was idle, wait a few seconds and retry.'}
finally{$('go').disabled=false;$('go').textContent='Predict potability'}};
</script></body></html>"""
PAGE = PAGE.replace("__GOOD__", json.dumps(SAMPLES["good"])).replace("__BAD__", json.dumps(SAMPLES["bad"]))
 
LABELS = {"ph": "pH", "Hardness": "Hardness (mg/L)", "Solids": "Total dissolved solids (ppm)", "Chloramines": "Chloramines (ppm)",
          "Sulfate": "Sulfate (mg/L)", "Conductivity": "Conductivity (\u03bcS/cm)", "Organic_carbon": "Organic carbon (ppm)",
          "Trihalomethanes": "Trihalomethanes (\u03bcg/L)", "Turbidity": "Turbidity (NTU)"}
 
@app.get("/")
def index():
    return render_template_string(PAGE, features=FEATURES, ranges=RANGES, labels=LABELS, m=METRICS)
 
@app.post("/predict")
def api_predict():
    result, errors = predict(request.get_json(silent=True) or {})
    return (jsonify(errors=errors), 400) if errors else jsonify(result)
 
@app.get("/health")
def health():
    return jsonify(status="ok", **METRICS)
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
