# SponsorFit – Brand-Sport Partnership Selection (free tools only)

Case: Sports Sponsorship Strategy – Selecting the Right Brand-Sport Partnership.
Scenario: hypothetical mid-size health-drink brand "FitFuel", 18-34 target, Rs 10 crore budget.

## Quickest way to see it (no install)
Open `dashboard/index.html` in any browser (double-click it, or right-click > "Open with Live Server" in VS Code).

## Rebuild everything (Python 3.10+)
    pip install pandas numpy scipy scikit-learn matplotlib
    python preprocess.py          # raw indicators -> data/sports_scores.csv + results/preprocessing_log.txt
    python run_analysis.py        # charts -> charts/, tables -> results/
    python build_dashboard.py     # rebuilds dashboard/index.html from the latest data

Optional: `pip install streamlit` then `streamlit run app.py` (older, simpler dashboard).

## Pipeline
1. data/raw_indicators.csv  – every sourced number: value, unit, year, quality grade, source, link
2. data/expert_scores.csv   – expert ratings (audience fit, engagement, image, cost efficiency, growth, risk) with rationale
3. preprocess.py            – cleaning: common yardstick, conflict resolution, missing-data flag, scaling, join
4. core.py                  – AHP, entropy weights, weighted sum, TOPSIS, Monte Carlo, break-even, rating-noise test, budget split
5. run_analysis.py          – produces all report charts and result tables
6. build_dashboard.py + dashboard_template.html – standalone interactive dashboard

Other files: survey_form_questions.md and survey_analysis.py (your own fan survey), fetch_trends.py (optional Google Trends).

## Honest limitations (say these in your viva)
- Only ONE of seven criteria (reach) is built from measured data. The other six are expert ratings backed by cited evidence.
- Running's reach is imputed (participation sport), flagged in the data and deliberately conservative.
- Cost efficiency is a relative index, not rate-card prices; the PKL title price is from 2017.
- No survey data is included. Collect your own (80+ responses) and run survey_analysis.py.
