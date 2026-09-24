"""
SponsorFit - core analysis functions (free tools only: numpy, pandas, scipy, scikit-learn).

Brand scenario (hypothetical): "FitFuel", a mid-size Indian health-drink brand
targeting 18-34 year olds, with a fixed sponsorship budget of Rs 10 crore.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
CRITERIA = ["reach", "audience_fit", "engagement", "image_congruence",
            "cost_efficiency", "growth", "risk_safety"]
LABELS = {
    "reach": "Audience reach",
    "audience_fit": "Audience-brand fit",
    "engagement": "Fan engagement",
    "image_congruence": "Image congruence",
    "cost_efficiency": "Cost efficiency",
    "growth": "Growth trend",
    "risk_safety": "Risk safety",
}

# ---------------------------------------------------------------- data
def load_data(path=None) -> pd.DataFrame:
    path = path or BASE / "data" / "sports_scores.csv"
    if not Path(path).exists():                      # build from raw sourced indicators
        import preprocess
        preprocess.build()
    df = pd.read_csv(path)
    missing = [c for c in CRITERIA if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df[CRITERIA] = df[CRITERIA].apply(pd.to_numeric, errors="raise")
    if df[CRITERIA].isna().any().any():
        raise ValueError("Dataset has missing criterion scores")
    return df


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale each criterion to 0-1 (all criteria are 'higher is better')."""
    x = df[CRITERIA].astype(float)
    rng = (x.max() - x.min()).replace(0, 1)
    return (x - x.min()) / rng

# ---------------------------------------------------------------- AHP
# Pairwise judgements on Saaty's 1-9 scale for a mid-size brand that wants
# sales-linked returns on a limited budget:
# order = reach, fit, engagement, image, cost, growth, risk
_AHP_UPPER = {
    (0, 1): 1/2, (0, 2): 2, (0, 3): 1, (0, 4): 1/2, (0, 5): 2, (0, 6): 3,
    (1, 2): 3,   (1, 3): 1, (1, 4): 1, (1, 5): 3, (1, 6): 4,
    (2, 3): 1/2, (2, 4): 1/3, (2, 5): 1, (2, 6): 2,
    (3, 4): 1,   (3, 5): 3, (3, 6): 3,
    (4, 5): 3,   (4, 6): 4,
    (5, 6): 2,
}
_RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def ahp_matrix() -> np.ndarray:
    n = len(CRITERIA)
    m = np.ones((n, n))
    for (i, j), v in _AHP_UPPER.items():
        m[i, j] = v
        m[j, i] = 1 / v
    return m


def ahp_weights(m: np.ndarray | None = None):
    """Principal-eigenvector weights and consistency ratio (CR < 0.10 is acceptable)."""
    m = ahp_matrix() if m is None else m
    vals, vecs = np.linalg.eig(m)
    k = np.argmax(vals.real)
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    n = m.shape[0]
    ci = (vals[k].real - n) / (n - 1)
    cr = ci / _RI[n] if _RI[n] else 0.0
    return pd.Series(w, index=CRITERIA), float(cr)

# ---------------------------------------------------------------- entropy
def entropy_weights(df: pd.DataFrame) -> pd.Series:
    """Objective weights: criteria that discriminate more between sports get more weight."""
    x = df[CRITERIA].astype(float).clip(lower=1e-9)
    p = x / x.sum()
    k = 1 / np.log(len(x))
    e = -k * (p * np.log(p)).sum()
    d = 1 - e
    return d / d.sum()

# ---------------------------------------------------------------- scoring
def wsm_score(df: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Weighted-sum model on normalised scores, scaled 0-100."""
    return (normalise(df)[CRITERIA] * w[CRITERIA]).sum(axis=1) * 100


def topsis_score(df: pd.DataFrame, w: pd.Series) -> pd.Series:
    """TOPSIS closeness to the ideal solution (0-1)."""
    x = df[CRITERIA].astype(float)
    v = x / np.sqrt((x ** 2).sum()) * w[CRITERIA]
    best, worst = v.max(), v.min()
    d_best = np.sqrt(((v - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((v - worst) ** 2).sum(axis=1))
    return d_worst / (d_best + d_worst)


def rank_table(df: pd.DataFrame, w_ahp: pd.Series, w_ent: pd.Series) -> pd.DataFrame:
    """Primary model = weighted sum with AHP (brand-priority) weights.
    Cross-checks = TOPSIS with the same weights, and the data-driven entropy-weight scenario."""
    out = pd.DataFrame({"sport": df["sport"]})
    out["WSM_AHP"] = wsm_score(df, w_ahp).values
    out["TOPSIS_AHP"] = topsis_score(df, w_ahp).values * 100
    out["WSM_Entropy"] = wsm_score(df, w_ent).values
    for c in ["WSM_AHP", "TOPSIS_AHP", "WSM_Entropy"]:
        out["rank_" + c] = out[c].rank(ascending=False).astype(int)
    return out.sort_values("WSM_AHP", ascending=False).reset_index(drop=True)

# ---------------------------------------------------------------- robustness
def monte_carlo(df: pd.DataFrame, base_w: pd.Series, n=5000, concentration=30, seed=42):
    """
    Randomise the weights around the base weights (Dirichlet) and record how often
    each sport lands in each rank. Lower concentration = more uncertainty about weights.
    Returns (rank_probability DataFrame, mean_rank Series).
    """
    rng = np.random.default_rng(seed)
    alpha = np.clip(base_w[CRITERIA].values * concentration, 0.05, None)
    W = rng.dirichlet(alpha, size=n)
    N = normalise(df)[CRITERIA].values
    scores = N @ W.T                       # sports x n
    ranks = (-scores).argsort(axis=0).argsort(axis=0) + 1
    probs = np.zeros((len(df), len(df)))
    for r in range(1, len(df) + 1):
        probs[:, r - 1] = (ranks == r).mean(axis=1)
    prob_df = pd.DataFrame(probs, index=df["sport"],
                           columns=[f"Rank {r}" for r in range(1, len(df) + 1)])
    return prob_df, pd.Series(ranks.mean(axis=1), index=df["sport"], name="mean_rank")



def score_noise_test(df: pd.DataFrame, w: pd.Series, n=5000, noise=1.5, seed=7):
    """Robustness of the RATINGS: add random error (+-noise points) to every expert-rated criterion
    (all criteria except the data-derived 'reach'), clip to 1-10, and count how often each sport ranks #1."""
    rng = np.random.default_rng(seed)
    expert = [c for c in CRITERIA if c != "reach"]
    wins = pd.Series(0.0, index=df["sport"])
    for _ in range(n):
        d = df.copy()
        d[expert] = (d[expert] + rng.uniform(-noise, noise, size=(len(d), len(expert)))).clip(1, 10)
        sc = wsm_score(d, w)
        wins[df.loc[sc.idxmax(), "sport"]] += 1
    return wins / n


def one_at_a_time(df: pd.DataFrame, base_w: pd.Series, swing=0.5) -> pd.DataFrame:
    """Increase / decrease each weight by +-50% (re-normalised) and report the winner."""
    rows = []
    for c in CRITERIA:
        for label, f in (("-50%", 1 - swing), ("+50%", 1 + swing)):
            w = base_w.copy()
            w[c] *= f
            w = w / w.sum()
            s = wsm_score(df, w)
            rows.append({"criterion": LABELS[c], "change": label,
                         "top_sport": df.loc[s.idxmax(), "sport"],
                         "top_score": round(s.max(), 1)})
    return pd.DataFrame(rows)



def weight_sweep(df: pd.DataFrame, base_w: pd.Series, criterion: str, steps=101) -> pd.DataFrame:
    """Vary one criterion's weight from 0 to 100%, scaling the others in proportion to their
    base weights, and return each sport's Fit Index at every step (a 'decision map')."""
    others = [c for c in CRITERIA if c != criterion]
    rest = base_w[others] / base_w[others].sum()
    rows = []
    for x in np.linspace(0, 1, steps):
        w = pd.Series(0.0, index=CRITERIA)
        w[criterion] = x
        w[others] = rest * (1 - x)
        rows.append(wsm_score(df, w).values)
    out = pd.DataFrame(rows, columns=df["sport"].values)
    out.insert(0, "weight", np.linspace(0, 1, steps))
    return out


def break_even(sweep: pd.DataFrame, sport: str):
    """Smallest weight at which `sport` is ranked #1 (None if never)."""
    sports = [c for c in sweep.columns if c != "weight"]
    lead = sweep[sports].idxmax(axis=1)
    hit = sweep.loc[lead == sport, "weight"]
    return None if hit.empty else float(hit.min())

def allocate_budget(df: pd.DataFrame, scores: pd.Series, budget_cr=10.0, top_k=2,
                    power=2, reserve_pct=20):
    """Split the budget across the top-k sports in proportion to score**power,
    after ring-fencing `reserve_pct` % for digital content, activation and measurement."""
    top = scores.sort_values(ascending=False).head(top_k)
    share = top ** power / (top ** power).sum() * (100 - reserve_pct) / 100
    out = pd.DataFrame({"sport": df.loc[top.index, "sport"].values,
                        "score": top.values.round(1),
                        "share_pct": (share.values * 100).round(1),
                        "budget_cr": (share.values * budget_cr).round(2)})
    reserve = pd.DataFrame([{"sport": "Digital, activation & measurement", "score": np.nan,
                             "share_pct": float(reserve_pct),
                             "budget_cr": round(budget_cr * reserve_pct / 100, 2)}])
    return pd.concat([out, reserve], ignore_index=True)


if __name__ == "__main__":
    d = load_data()
    wa, cr = ahp_weights()
    print("AHP weights:\n", wa.round(3), "\nConsistency ratio:", round(cr, 3))
    print(rank_table(d, wa, entropy_weights(d)).round(1).to_string())
