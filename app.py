"""SponsorFit dashboard.  Run:  streamlit run app.py"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from core import (BASE, CRITERIA, LABELS, load_data, ahp_weights, entropy_weights,
                  wsm_score, topsis_score, monte_carlo, allocate_budget)

st.set_page_config(page_title="SponsorFit", page_icon="🏆", layout="wide")
PAL = {"Cricket": "#E4572E", "Kabaddi": "#17BEBB", "Football": "#76B041",
       "Badminton": "#FFC914", "Hockey": "#2E86AB", "Running": "#7B4B94"}

df = load_data()
w_ahp, cr = ahp_weights()
w_ent = entropy_weights(df)

st.title("🏆 SponsorFit — Choosing the right brand–sport partnership")
st.caption("Scenario: FitFuel, a hypothetical mid-size Indian health-drink brand targeting 18–34 year olds. "
           "Scores are analyst estimates informed by public industry data — see the Data tab.")

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Brand priorities")
mode = st.sidebar.radio("Weighting method", ["AHP (judgement)", "Entropy (data-driven)", "Hybrid", "Custom sliders"])
base = {"AHP (judgement)": w_ahp, "Entropy (data-driven)": w_ent,
        "Hybrid": (w_ahp + w_ent) / (w_ahp + w_ent).sum()}.get(mode, (w_ahp + w_ent) / (w_ahp + w_ent).sum())
if mode == "Custom sliders":
    raw = {c: st.sidebar.slider(LABELS[c], 0, 100, int(round(base[c] * 100))) for c in CRITERIA}
    tot = sum(raw.values()) or 1
    w = pd.Series({c: v / tot for c, v in raw.items()})
else:
    w = base
budget = st.sidebar.number_input("Budget (Rs crore)", 1.0, 100.0, 10.0, 0.5)
reserve = st.sidebar.slider("Reserve for digital/activation (%)", 0, 40, 20)
st.sidebar.caption(f"AHP consistency ratio: {cr:.3f} (acceptable < 0.10)")

scores = wsm_score(df, w)
tops = topsis_score(df, w) * 100
res = pd.DataFrame({"Sport": df["sport"], "Property": df["property"],
                    "Fit Index (WSM)": scores.round(1), "TOPSIS": tops.round(1)}
                   ).sort_values("Fit Index (WSM)", ascending=False).reset_index(drop=True)
res.index += 1

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ranking", "Compare", "Robustness", "Budget", "Data & survey"])

with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        o = res.iloc[::-1]
        ax.barh(o["Sport"], o["Fit Index (WSM)"], color=[PAL[s] for s in o["Sport"]])
        for i, v in enumerate(o["Fit Index (WSM)"]):
            ax.text(v + 0.8, i, f"{v:.1f}", va="center")
        ax.set_xlim(0, 100); ax.set_xlabel("Brand-Sport Fit Index (0-100)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
    with c2:
        st.metric("Top recommendation", res.loc[1, "Sport"], f"{res.loc[1, 'Fit Index (WSM)']} / 100")
        st.dataframe(res[["Sport", "Fit Index (WSM)", "TOPSIS"]], use_container_width=True)

with tab2:
    pick = st.multiselect("Sports to compare", df["sport"].tolist(), default=res["Sport"].head(3).tolist())
    ang = np.linspace(0, 2 * np.pi, len(CRITERIA), endpoint=False).tolist(); ang += ang[:1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    for s in pick:
        v = df.set_index("sport").loc[s, CRITERIA].tolist(); v += v[:1]
        ax.plot(ang, v, color=PAL[s], lw=2, label=s); ax.fill(ang, v, color=PAL[s], alpha=0.12)
    ax.set_xticks(ang[:-1], [LABELS[c].replace(" ", "\n") for c in CRITERIA], fontsize=8)
    ax.set_ylim(0, 10); ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False)
    st.pyplot(fig)

with tab3:
    conc = st.slider("Uncertainty about weights (lower = more uncertain)", 4, 60, 30)
    prob, mean_rank = monte_carlo(df, w, n=5000, concentration=conc)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(prob.index, prob["Rank 1"] * 100, color=[PAL[s] for s in prob.index])
    ax.set_ylabel("% of simulations ranked #1"); ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    st.dataframe((prob * 100).round(1), use_container_width=True)

with tab4:
    a = allocate_budget(df, scores, budget_cr=budget, top_k=2, reserve_pct=reserve)
    st.dataframe(a, use_container_width=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.pie(a["budget_cr"], labels=a["sport"], colors=[PAL.get(s, "#9AA5B1") for s in a["sport"]],
           autopct="%1.0f%%", startangle=90, wedgeprops=dict(edgecolor="white"))
    st.pyplot(fig)

with tab5:
    st.subheader("Scoring dataset (1–10 scale)")
    st.dataframe(df, use_container_width=True)
    surv = BASE / "results" / "survey_summary.csv"
    if surv.exists():
        st.subheader("Primary survey results")
        st.dataframe(pd.read_csv(surv, index_col=0), use_container_width=True)
        img = BASE / "charts" / "08_survey_interest_intent.png"
        if img.exists():
            st.image(str(img))
    else:
        st.info("Run survey_analysis.py on your Google Forms export to add primary-data results here.")
