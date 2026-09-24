"""Run the full SponsorFit pipeline:  python run_analysis.py
1) preprocess raw indicators -> modelling dataset  2) score, rank, test robustness  3) charts + tables"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import preprocess
from core import (score_noise_test, BASE, CRITERIA, LABELS, load_data, ahp_weights, entropy_weights, rank_table,
                  wsm_score, monte_carlo, one_at_a_time, allocate_budget, weight_sweep, break_even)

CH, RS = BASE / "charts", BASE / "results"
CH.mkdir(exist_ok=True); RS.mkdir(exist_ok=True)
PAL = {"Cricket": "#E4572E", "Kabaddi": "#17BEBB", "Football": "#76B041",
       "Badminton": "#FFC914", "Hockey": "#2E86AB", "Running": "#7B4B94"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout(); fig.savefig(CH / name, dpi=200); plt.close(fig)


def main():
    preprocess.build()
    df = load_data()
    w_ahp, cr = ahp_weights()
    w_ent = entropy_weights(df)
    ranks = rank_table(df, w_ahp, w_ent)
    scores = wsm_score(df, w_ahp)                       # PRIMARY: brand-priority (AHP) weights
    prob, mean_rank = monte_carlo(df, w_ahp, n=10000, concentration=30)
    prob_wide, _ = monte_carlo(df, w_ahp, n=10000, concentration=8)
    oat = one_at_a_time(df, w_ahp)
    sweep = weight_sweep(df, w_ahp, "reach")
    be = {s: break_even(sweep, s) for s in df["sport"]}
    alloc = allocate_budget(df, scores, budget_cr=10.0, top_k=2)
    noise = pd.DataFrame({f"+-{n} pts": score_noise_test(df, w_ahp, noise=n) for n in (1.0, 1.5, 2.0)})
    noise.round(4).to_csv(RS / "rating_noise_test.csv")

    pd.DataFrame({"AHP": w_ahp, "Entropy": w_ent}).round(4).to_csv(RS / "weights.csv")
    ranks.round(2).to_csv(RS / "rankings.csv", index=False)
    prob.round(4).to_csv(RS / "monte_carlo_rank_probability.csv")
    pd.DataFrame({"P_rank1_moderate": prob["Rank 1"], "P_rank1_high": prob_wide["Rank 1"],
                  "mean_rank_moderate": mean_rank}).round(3).to_csv(RS / "monte_carlo_summary.csv")
    oat.to_csv(RS / "sensitivity_one_at_a_time.csv", index=False)
    alloc.to_csv(RS / "budget_allocation.csv", index=False)
    pd.Series(be, name="reach_weight_needed_to_rank_first").to_csv(RS / "break_even_reach_weight.csv")
    pd.Series({"ahp_cr": cr, "base_reach_weight": w_ahp["reach"]}).to_csv(RS / "model_meta.csv")

    print(f"AHP consistency ratio: {cr:.3f} (acceptable < 0.10)")
    print(ranks.round(1).to_string(index=False))
    print("\nP(rank 1) moderate / high weight uncertainty:")
    print(pd.DataFrame({"moderate": prob["Rank 1"], "high": prob_wide["Rank 1"]}).round(3).to_string())
    print("\nOne-at-a-time winners:", oat.top_sport.value_counts().to_dict())
    print("Break-even reach weight (min weight at which sport is #1):", {k: (round(v, 2) if v is not None else None) for k, v in be.items()})
    print("\nP(rank 1) when expert ratings are perturbed:\n", noise.round(3).to_string())
    print("\n", alloc.to_string(index=False))

    # 01 heatmap
    m = df.set_index("sport")[CRITERIA]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(m.values, cmap="YlGnBu", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(range(len(CRITERIA)), [LABELS[c] for c in CRITERIA], rotation=30, ha="right")
    ax.set_yticks(range(len(m)), m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.1f}".rstrip("0").rstrip("."), ha="center", va="center", color="white" if v > 6 else "black")
    ax.set_title("Criteria scores by sport (1-10, higher = better for the brand)")
    fig.colorbar(im, ax=ax, fraction=0.03); save(fig, "01_criteria_heatmap.png")

    # 02 weights (AHP vs entropy)
    fig, ax = plt.subplots(figsize=(8, 3.8)); x = np.arange(len(CRITERIA)); wd = 0.38
    ax.bar(x - wd / 2, w_ahp.values * 100, wd, label="AHP - brand priorities (used)", color="#17BEBB")
    ax.bar(x + wd / 2, w_ent.values * 100, wd, label="Entropy - data dispersion (cross-check)", color="#E4572E")
    ax.set_xticks(x, [LABELS[c] for c in CRITERIA], rotation=30, ha="right")
    ax.set_ylabel("Weight (%)"); ax.legend(frameon=False); ax.set_title("Criteria weights by method")
    save(fig, "02_weights.png")

    # 03 ranking (AHP primary)
    order = ranks.sort_values("WSM_AHP")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(order["sport"], order["WSM_AHP"], color=[PAL[s] for s in order["sport"]])
    for i, v in enumerate(order["WSM_AHP"]):
        ax.text(v + 0.8, i, f"{v:.1f}", va="center")
    ax.set_xlabel("Brand-Sport Fit Index (0-100)"); ax.set_xlim(0, 100)
    ax.set_title("SponsorFit ranking (brand-priority AHP weights)"); save(fig, "03_ranking.png")

    # 04 positioning
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for _, r in df.iterrows():
        ax.scatter(r.cost_efficiency, r.reach, s=r.audience_fit ** 2 * 14, alpha=0.75, color=PAL[r.sport], edgecolor="white")
        ax.annotate(r.sport, (r.cost_efficiency, r.reach), xytext={"Hockey": (10, 14), "Running": (16, -18), "Badminton": (-62, -18)}.get(r.sport, (8, 8)), textcoords="offset points")
    ax.axvline(5, color="grey", lw=0.6, ls="--"); ax.axhline(5.5, color="grey", lw=0.6, ls="--")
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 11)
    ax.set_xlabel("Cost efficiency (10 = cheapest per unit reach)"); ax.set_ylabel("Audience reach (data-derived, 1-10)")
    ax.set_title("Reach vs cost efficiency (bubble = audience-brand fit)"); save(fig, "04_positioning.png")

    # 05 radar top 3
    top3 = ranks["sport"].head(3).tolist()
    ang = np.linspace(0, 2 * np.pi, len(CRITERIA), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(5.6, 5.6), subplot_kw=dict(polar=True))
    for s in top3:
        v = df.set_index("sport").loc[s, CRITERIA].tolist(); v += v[:1]
        ax.plot(ang, v, color=PAL[s], lw=2, label=s); ax.fill(ang, v, color=PAL[s], alpha=0.12)
    ax.set_xticks(ang[:-1], [LABELS[c].replace(" ", "\n") for c in CRITERIA], fontsize=8)
    ax.set_ylim(0, 10); ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), frameon=False)
    ax.set_title("Top-3 profiles", pad=20); save(fig, "05_radar_top3.png")

    # 06 Monte Carlo
    fig, ax = plt.subplots(figsize=(8, 3.8)); idx = np.arange(len(df)); sp = df["sport"].tolist()
    ax.bar(idx - 0.2, [prob.loc[s, "Rank 1"] * 100 for s in sp], 0.4, label="Moderate weight uncertainty", color="#17BEBB")
    ax.bar(idx + 0.2, [prob_wide.loc[s, "Rank 1"] * 100 for s in sp], 0.4, label="High weight uncertainty", color="#7B4B94")
    ax.set_xticks(idx, sp); ax.set_ylabel("% of 10,000 simulations ranked #1"); ax.legend(frameon=False)
    ax.set_title("Monte Carlo: how often does each sport come first?"); save(fig, "06_monte_carlo.png")

    # 07 budget
    fig, ax = plt.subplots(figsize=(5.2, 4))
    lab = [f"{s.replace('Digital, activation & measurement', 'Digital &' + chr(10) + 'activation')}\nRs {b} cr" for s, b in zip(alloc.sport, alloc.budget_cr)]
    ax.pie(alloc["budget_cr"], labels=lab, colors=[PAL.get(s, "#9AA5B1") for s in alloc.sport], startangle=90, wedgeprops=dict(edgecolor="white"))
    ax.set_title("Suggested split of a Rs 10 crore budget"); save(fig, "07_budget.png")

    # 09 method comparison
    fig, ax = plt.subplots(figsize=(8, 3.8)); order = ranks["sport"].tolist(); x = np.arange(len(order)); wd = 0.27
    for k, (col, name, c) in enumerate([("WSM_AHP", "Weighted sum (AHP) - primary", "#17BEBB"),
                                        ("TOPSIS_AHP", "TOPSIS (AHP weights)", "#2E86AB"),
                                        ("WSM_Entropy", "Weighted sum (entropy weights)", "#E4572E")]):
        ax.bar(x + (k - 1) * wd, ranks[col], wd, label=name, color=c)
    ax.set_xticks(x, order); ax.set_ylabel("Score (0-100)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Same data, three scoring methods"); save(fig, "09_method_comparison.png")

    # 10 reach-weight sweep (decision map)
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in df["sport"]:
        ax.plot(sweep["weight"] * 100, sweep[s], color=PAL[s], lw=2.2 if s in top3 else 1.2, label=s)
    ax.axvline(w_ahp["reach"] * 100, color="black", ls="--", lw=1)
    ax.text(w_ahp["reach"] * 100 + 1, 3, "brand's weight\non reach", fontsize=8)
    if be.get("Cricket") is not None:
        ax.axvline(be["Cricket"] * 100, color=PAL["Cricket"], ls=":", lw=1.2)
        ax.text(be["Cricket"] * 100 + 1, 10, "cricket takes\nfirst place", fontsize=8, color=PAL["Cricket"])
    ax.set_xlabel("Weight given to audience reach (%)"); ax.set_ylabel("Fit Index (0-100)")
    ax.set_ylim(0, 100); ax.legend(frameon=False, ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.set_title("Decision map: how much must reach matter before cricket wins?"); save(fig, "10_reach_sweep.png")

    # 11 audience base data (Ormax) with imputed running flagged
    fig, ax = plt.subplots(figsize=(8, 3.6)); d2 = df.sort_values("audience_base_m")
    cols = [PAL[s] for s in d2["sport"]]
    bars = ax.barh(d2["sport"], d2["audience_base_m"], color=cols)
    for b, (_, r) in zip(bars, d2.iterrows()):
        if r.reach_imputed:
            b.set_hatch("//"); b.set_alpha(0.45)
        ax.text(r.audience_base_m + 6, b.get_y() + b.get_height() / 2,
                f"{r.audience_base_m:.0f}M" + ("  (imputed: participation sport)" if r.reach_imputed else ""), va="center", fontsize=9)
    ax.set_xlim(0, 760); ax.set_xlabel("Audience base (million people)")
    ax.set_title("Raw data used for the reach criterion (Ormax Sports Audience Report 2024)"); save(fig, "11_audience_data.png")
    print("\nCharts saved to", CH)


if __name__ == "__main__":
    main()
