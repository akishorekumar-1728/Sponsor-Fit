"""Step 1 of the pipeline: build the modelling dataset from sourced raw indicators + expert ratings.
Run:  python preprocess.py     ->  data/sports_scores.csv  and  results/preprocessing_log.txt

Cleaning / preprocessing steps (for the report):
 1. Select the audience-base indicator (Ormax 2024) as the common yardstick, because TV-only reach is not
    comparable across sports or between TV and OTT.
 2. Resolve source conflicts (kabaddi 280M vs 208M) in favour of the primary source.
 3. Handle missing data (running has no TV/OTT audience base): set to the observed minimum, then flag it.
 4. Min-max scale reach to a 1-10 score, so it is on the same scale as the expert-rated criteria.
 5. Join with expert-rated criteria and save the final modelling table.
"""
import pandas as pd
from core import BASE

RAW, EXP = BASE / "data" / "raw_indicators.csv", BASE / "data" / "expert_scores.csv"
OUT, LOG = BASE / "data" / "sports_scores.csv", BASE / "results" / "preprocessing_log.txt"


def build():
    log = []
    raw = pd.read_csv(RAW)
    exp = pd.read_csv(EXP)
    log.append(f"Raw indicator rows: {len(raw)}; sports: {raw.sport.nunique()}; expert-rated sports: {len(exp)}")

    aud = raw[(raw.indicator == "audience_base") & (raw.used_in_model == 1)].copy()
    aud["value"] = pd.to_numeric(aud["value"], errors="coerce")
    log.append("Step 1: yardstick = Ormax audience_base (million people).")

    alt = raw[raw.indicator == "audience_base_alt"]
    for _, r in alt.iterrows():
        log.append(f"Step 2: conflicting value for {r.sport} ({r.value}M from {r.source}) ignored; "
                   f"primary source value {aud.loc[aud.sport == r.sport, 'value'].iloc[0]:.0f}M used.")

    lo = aud["value"].min()
    n_missing = int(aud["value"].isna().sum())
    for s in aud.loc[aud["value"].isna(), "sport"]:
        log.append(f"Step 3: {s} has no comparable audience base -> imputed with observed minimum ({lo:.0f}M); "
                   f"flagged as imputed because participation, not TV/OTT viewing, is its reach.")
    aud["imputed"] = aud["value"].isna()
    aud["value"] = aud["value"].fillna(lo)

    hi = aud["value"].max()
    aud["reach"] = (1 + 9 * (aud["value"] - lo) / (hi - lo)).round(2)
    log.append(f"Step 4: min-max scaling reach = 1 + 9*(x - {lo:.0f})/({hi:.0f} - {lo:.0f}).")

    df = exp.merge(aud[["sport", "value", "imputed", "reach"]].rename(columns={"value": "audience_base_m"}),
                   on="sport", how="left")
    cols = ["sport", "property", "reach", "audience_fit", "engagement", "image_congruence",
            "cost_efficiency", "growth", "risk_safety", "audience_base_m", "reach_imputed",
            "rationale", "confidence"]
    df = df.rename(columns={"imputed": "reach_imputed"})[cols]
    assert df.isna().sum()[["reach", "audience_fit", "engagement", "image_congruence", "cost_efficiency",
                            "growth", "risk_safety"]].sum() == 0, "missing criteria after preprocessing"
    df.to_csv(OUT, index=False)
    log.append(f"Step 5: saved {len(df)} rows x {len(df.columns)} columns to {OUT.name}")
    (BASE / "results").mkdir(exist_ok=True)
    LOG.write_text("\n".join(log), encoding="utf-8")
    return df, log


if __name__ == "__main__":
    d, lg = build()
    print("\n".join(lg))
    print(d[["sport", "audience_base_m", "reach_imputed", "reach"]].to_string(index=False))
