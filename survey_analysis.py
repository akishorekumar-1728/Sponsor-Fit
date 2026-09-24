"""Analyse YOUR real Google Forms survey: python survey_analysis.py data/survey_responses.csv
Expected columns (see survey_form_questions.md):
  age_group, gender, city_tier, weekly_watch_hours, health_drink_freq,
  interest_<sport> and intent_<sport> (1-5) for: cricket, kabaddi, football, badminton, hockey, running
Outputs: results/survey_summary.csv, results/survey_segments.csv, charts/08_survey_*.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from core import BASE

SPORTS = ["cricket", "kabaddi", "football", "badminton", "hockey", "running"]


def main(path):
    df = pd.read_csv(path)
    need = [f"{p}_{s}" for p in ("interest", "intent") for s in SPORTS]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"Missing columns in survey file: {miss}")
    df = df.dropna(subset=need).copy()
    n = len(df)
    print(f"Valid responses: {n}")
    if n < 30:
        print("WARNING: fewer than 30 responses - treat results as indicative only.")

    # 1. interest vs purchase intent by sport
    summ = pd.DataFrame({"mean_interest": [df[f"interest_{s}"].mean() for s in SPORTS],
                         "mean_purchase_intent": [df[f"intent_{s}"].mean() for s in SPORTS]},
                        index=[s.title() for s in SPORTS]).round(2)
    summ["intent_per_interest"] = (summ.mean_purchase_intent / summ.mean_interest).round(2)
    summ.to_csv(BASE / "results" / "survey_summary.csv")
    print(summ.sort_values("mean_purchase_intent", ascending=False))

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(SPORTS))
    ax.bar(x - 0.2, summ.mean_interest, 0.4, label="Interest in following", color="#2E86AB")
    ax.bar(x + 0.2, summ.mean_purchase_intent, 0.4, label="Purchase intent if brand sponsors", color="#17BEBB")
    ax.set_xticks(x, summ.index); ax.set_ylim(0, 5); ax.legend(frameon=False)
    ax.set_title(f"Survey (n={n}): interest vs purchase intent (1-5)")
    fig.tight_layout(); fig.savefig(BASE / "charts" / "08_survey_interest_intent.png", dpi=200); plt.close(fig)

    # 2. fan segments (k-means on interest scores)
    X = StandardScaler().fit_transform(df[[f"interest_{s}" for s in SPORTS]])
    best_k, best_s = 2, -1
    for k in range(2, min(6, n // 5 + 1)):
        lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        s = silhouette_score(X, lab)
        if s > best_s:
            best_k, best_s = k, s
    df["segment"] = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit_predict(X)
    print(f"\nChosen k={best_k} (silhouette {best_s:.2f})")
    seg = df.groupby("segment")[[f"interest_{s}" for s in SPORTS] + [f"intent_{s}" for s in SPORTS]
                                + [c for c in ("weekly_watch_hours", "health_drink_freq") if c in df.columns]].mean().round(2)
    seg.insert(0, "size", df.groupby("segment").size())
    seg.to_csv(BASE / "results" / "survey_segments.csv")
    print(seg.T)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else BASE / "data" / "survey_responses.csv")
