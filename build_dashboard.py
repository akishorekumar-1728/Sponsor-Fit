"""Build the standalone dashboard:  python build_dashboard.py  ->  dashboard/index.html
Open the file in any browser (or VS Code 'Live Server'). No installation or server needed."""
import json
import pandas as pd
import preprocess
from core import BASE, CRITERIA, LABELS, load_data, ahp_weights, entropy_weights

HINTS = {
    "reach": "How many people follow the sport (published audience data).",
    "audience_fit": "Do the fans match your 18-34 buyers?",
    "engagement": "How passionate and active are the fans?",
    "image_congruence": "Does the sport's image suit a health brand?",
    "cost_efficiency": "Value for money, allowing for sponsor clutter.",
    "growth": "Is interest in the sport rising?",
    "risk_safety": "Stability of the league, stars and rules.",
}


def build():
    _, log = preprocess.build()
    df = load_data()
    w_ahp, cr = ahp_weights()
    w_ent = entropy_weights(df)
    raw = pd.read_csv(BASE / "data" / "raw_indicators.csv").fillna("")
    sports = []
    for _, r in df.iterrows():
        d = {"sport": r.sport, "property": r.property, "rationale": r.rationale, "confidence": r.confidence,
             "audience_base_m": float(r.audience_base_m), "reach_imputed": bool(r.reach_imputed)}
        d.update({k: float(r[k]) for k in CRITERIA})
        sports.append(d)
    data = {
        "scenario": "FitFuel: a mid-size health-drink brand, target 18-34, budget Rs 10 crore",
        "criteria": [{"key": k, "label": LABELS[k], "hint": HINTS[k]} for k in CRITERIA],
        "sports": sports,
        "weights": {"ahp": [float(w_ahp[k]) for k in CRITERIA], "entropy": [float(w_ent[k]) for k in CRITERIA]},
        "ahp_cr": cr,
        "raw": json.loads(raw.to_json(orient="records")),
        "log": log,
        "notes": f"The brand-default weights come from pairwise comparisons (consistency ratio {cr:.2f}, below the 0.10 limit).",
    }
    html = (BASE / "dashboard_template.html").read_text(encoding="utf-8")
    out = BASE / "dashboard"; out.mkdir(exist_ok=True)
    (out / "index.html").write_text(html.replace("/*DATA*/", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    print("Wrote", out / "index.html")


if __name__ == "__main__":
    build()
