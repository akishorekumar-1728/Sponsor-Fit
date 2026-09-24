"""OPTIONAL (needs internet): pull 5-year Google Trends interest for each sport in India.
pip install pytrends ; python fetch_trends.py
Google Trends is free. Output: data/google_trends_india.csv (relative interest 0-100)."""
import pandas as pd
from pytrends.request import TrendReq

TERMS = ["IPL", "Pro Kabaddi", "ISL football", "badminton", "hockey india", "marathon"]
pt = TrendReq(hl="en-IN", tz=330)
pt.build_payload(TERMS[:5], timeframe="today 5-y", geo="IN")
a = pt.interest_over_time().drop(columns="isPartial")
pt.build_payload(["IPL", TERMS[5]], timeframe="today 5-y", geo="IN")   # keep IPL as anchor term
b = pt.interest_over_time().drop(columns=["isPartial", "IPL"])
out = a.join(b)
out.to_csv("data/google_trends_india.csv")
print(out.tail())
# Use: compare 2025-26 average vs 2021-22 average per term as a data-backed 'growth' indicator.
