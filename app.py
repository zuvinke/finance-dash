import os
import math
import time

from dateutil.parser import parse as dtparse  # (kept, even if unused for now)
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

# ----------------------------
# Settings
# ----------------------------
st.set_page_config(page_title="Finance Dash (SEC + Earnings)", layout="wide")

USER_AGENT = os.getenv("SEC_USER_AGENT", "FinanceDash/1.0 (contact: you@example.com)")
FMP_API_KEY = os.getenv("FMP_API_KEY", "").strip()

SEC_BASE = "https://data.sec.gov"
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"  # ticker -> cik

# ----------------------------
# Helpers
# ----------------------------
def _sec_headers():
    return {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }

def _get_json(url: str, headers: dict, sleep_s: float = 0.12):
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code == 429:
        time.sleep(1.0)
        resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    time.sleep(sleep_s)
    return resp.json()

@st.cache_data(ttl=24 * 3600)
def load_ticker_map():
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    data = _get_json(SEC_TICKER_MAP_URL, headers=headers, sleep_s=0.12)

    rows = []
    for _, v in data.items():
        rows.append({
            "ticker": str(v.get("ticker", "")).upper(),
            "cik": int(v.get("cik_str")),
            "name": v.get("title", "")
        })
    df = pd.DataFrame(rows).dropna()
    df = df[df["ticker"] != ""].drop_duplicates("ticker").reset_index(drop=True)
    return df

def cik_pad(cik: int) -> str:
    return str(int(cik)).zfill(10)

@st.cache_data(ttl=6 * 3600)
def load_company_facts(cik: int):
    url = f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik_pad(cik)}.json"
    return _get_json(url, headers=_sec_headers())

def pick_unit_series(facts: dict, taxonomy: str, tag: str, preferred_units=("USD", "shares", "USD/shares")):
    try:
        units = facts["facts"][taxonomy][tag]["units"]
    except KeyError:
        return None, None

    for u in preferred_units:
        if u in units:
            return u, units[u]

    any_unit = next(iter(units.keys()), None)
    return any_unit, units.get(any_unit) if any_unit else (None, None)

def to_df(series_list, metric_name: str):
    if not series_list:
        return pd.DataFrame(columns=["end", "fy", "fp", "form", "filed", "val", "metric"])

    df = pd.DataFrame(series_list).copy()

    for col in ["end", "fy", "fp", "form", "filed", "val"]:
        if col not in df.columns:
            df[col] = np.nan

    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.dropna(subset=["end", "val"])
    df["metric"] = metric_name
    return df

def keep_latest_per_period(df: pd.DataFrame):
    if df.empty:
        return df
    df = df.sort_values(["end", "filed"]).dropna(subset=["end"])
    df = df.groupby(["end", "fy", "fp"], as_index=False).tail(1)
    return df.sort_values("end")

def build_quarterly(df: pd.DataFrame):
    if df.empty:
        return df
    q = df[df["fp"].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
    return keep_latest_per_period(q)

def build_annual(df: pd.DataFrame):
    if df.empty:
        return df
    a = df[df["fp"].isin(["FY"])].copy()
    return keep_latest_per_period(a)

FP_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5}

def ytd_to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Many SEC income statement items for 10-Qs are reported YTD for Q2/Q3.
    Convert to single-quarter values by differencing within the same FY.
    """
    if df.empty:
        return df

    out = []
    d = df[df["fp"].isin(["Q1", "Q2", "Q3", "Q4", "FY"])].copy()
    d["fp_order"] = d["fp"].map(FP_ORDER)
    d = keep_latest_per_period(d).sort_values(["fy", "fp_order", "end"])

    for fy, g in d.groupby("fy", dropna=True):
        g = g.sort_values("fp_order").copy()

        q1 = g[g["fp"] == "Q1"].tail(1)
        q2 = g[g["fp"] == "Q2"].tail(1)
        q3 = g[g["fp"] == "Q3"].tail(1)
        q4 = g[g["fp"] == "Q4"].tail(1)
        fy_row = g[g["fp"] == "FY"].tail(1)

        def val(row):
            return float(row["val"].iloc[0]) if not row.empty else None

        def end(row):
            return row["end"].iloc[0] if not row.empty else None

        v1, v2, v3 = val(q1), val(q2), val(q3)

        if v1 is not None and end(q1) is not None:
            r = q1.iloc[0].to_dict()
            r["val"] = v1
            r["fp"] = "Q1"
            out.append(r)

        if v2 is not None and end(q2) is not None:
            r = q2.iloc[0].to_dict()
            r["val"] = v2 - v1 if (v1 is not None and v2 is not None) else v2
            r["fp"] = "Q2"
            out.append(r)

        if v3 is not None and end(q3) is not None:
            r = q3.iloc[0].to_dict()
            r["val"] = v3 - v2 if (v2 is not None and v3 is not None) else v3
            r["fp"] = "Q3"
            out.append(r)

        if not fy_row.empty and end(fy_row) is not None:
            vfy = val(fy_row)
            if vfy is not None:
                r = fy_row.iloc[0].to_dict()
                r["val"] = vfy - v3 if (v3 is not None) else vfy
                r["fp"] = "Q4"
                out.append(r)
        elif not q4.empty and end(q4) is not None:
            out.append(q4.iloc[0].to_dict())

    out_df = pd.DataFrame(out)
    if out_df.empty:
        return out_df

    out_df = out_df.drop(columns=[c for c in ["fp_order"] if c in out_df.columns], errors="ignore")
    out_df = out_df.sort_values("end")
    return keep_latest_per_period(out_df)

def yoy_growth(series: pd.Series, periods: int = 4):
    return series.pct_change(periods=periods) * 100.0

def compute_ttm(quarterly_df: pd.DataFrame, value_col="val"):
    if quarterly_df.empty:
        return quarterly_df.assign(ttm=np.nan)
    q = quarterly_df.sort_values("end").copy()
    q["ttm"] = q[value_col].rolling(4).sum()
    return q

# ----------------------------
# Optional: FMP surprises
# ----------------------------
@st.cache_data(ttl=24 * 3600)
def fmp_earnings_surprises(ticker: str, api_key: str):
    url = f"https://financialmodelingprep.com/api/v3/earnings-surprises/{ticker.upper()}?apikey={api_key}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    return r.json()

def surprises_to_df(raw):
    if not raw or not isinstance(raw, list):
        return pd.DataFrame(columns=["date", "epsEstimated", "epsActual", "surprise", "surprisePercent"])
    df = pd.DataFrame(raw).copy()
    for c in ["date", "epsEstimated", "epsActual", "surprise", "surprisePercent"]:
        if c not in df.columns:
            df[c] = np.nan
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date", ascending=False)
    return df

# ----------------------------
# UI
# ----------------------------
st.title("Finance Dash — SEC Fundamentals + Optional Earnings Surprises")

with st.sidebar:
    st.header("Settings")
    ticker_map = load_ticker_map()

    ticker = st.text_input("Ticker (US)", value="AAPL").strip().upper()
    show_surprises = st.toggle("Show historical surprises (optional)", value=True)

    st.caption("SEC requires a proper User-Agent header. Set env var `SEC_USER_AGENT` when deploying.")
    if show_surprises:
        if not FMP_API_KEY:
            st.warning("No FMP_API_KEY set. Surprises panel will show as unavailable.")
        else:
            st.success("FMP_API_KEY detected (surprises enabled).")

row = ticker_map[ticker_map["ticker"] == ticker]
if row.empty:
    st.error("Ticker not found in SEC mapping. Try another US-listed ticker.")
    st.stop()

cik = int(row.iloc[0]["cik"])
company_name = row.iloc[0]["name"]
st.subheader(f"{company_name} ({ticker}) — CIK {cik_pad(cik)}")

with st.spinner("Loading SEC fundamentals…"):
    facts = load_company_facts(cik)

REVENUE_TAG_CANDIDATES = [
    ("us-gaap", "Revenues"),
    ("us-gaap", "SalesRevenueNet"),
    ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
    ("us-gaap", "SalesRevenueServicesNet"),
    ("us-gaap", "SalesRevenueGoodsNet"),
    ("us-gaap", "TotalRevenuesAndOtherIncome"),
]

NET_INCOME_CANDIDATES = [("us-gaap", "NetIncomeLoss")]
EPS_DILUTED_CANDIDATES = [("us-gaap", "EarningsPerShareDiluted")]
GROSS_PROFIT_CANDIDATES = [("us-gaap", "GrossProfit")]
OPERATING_INCOME_CANDIDATES = [("us-gaap", "OperatingIncomeLoss")]

def load_metric_df(label, candidates, preferred_units):
    for tax, tag in candidates:
        unit, series = pick_unit_series(facts, tax, tag, preferred_units=preferred_units)
        if series:
            df = to_df(series, label)
            return keep_latest_per_period(df), (tax, tag, unit)
    return pd.DataFrame(columns=["end", "fy", "fp", "form", "filed", "val", "metric"]), (None, None, None)

rev_df, _ = load_metric_df("Revenue", REVENUE_TAG_CANDIDATES, preferred_units=("USD",))
ni_df, _ = load_metric_df("Net Income", NET_INCOME_CANDIDATES, preferred_units=("USD",))
eps_df, _ = load_metric_df("EPS (Diluted)", EPS_DILUTED_CANDIDATES, preferred_units=("USD/shares",))
gp_df, _ = load_metric_df("Gross Profit", GROSS_PROFIT_CANDIDATES, preferred_units=("USD",))
op_df, _ = load_metric_df("Operating Income", OPERATING_INCOME_CANDIDATES, preferred_units=("USD",))

# Annual
rev_a = build_annual(rev_df)
eps_a = build_annual(eps_df)
ni_a = build_annual(ni_df)

# Quarterly (YTD -> true quarterly)
rev_q = ytd_to_quarterly(rev_df)
ni_q = ytd_to_quarterly(ni_df)

# EPS usually already quarterly
eps_q = build_quarterly(eps_df)

gp_q = build_quarterly(gp_df)
op_q = build_quarterly(op_df)

if not rev_q.empty:
    rev_q = rev_q.sort_values("end").reset_index(drop=True)
    rev_q["Revenue YoY %"] = yoy_growth(rev_q["val"], periods=4)

if not eps_q.empty:
    eps_q = eps_q.sort_values("end").reset_index(drop=True)
    eps_q["EPS YoY %"] = yoy_growth(eps_q["val"], periods=4)

# Margins
margins_q = None
if not rev_q.empty:
    margins_q = pd.DataFrame({"end": rev_q["end"], "revenue": rev_q["val"]})

    if not gp_q.empty:
        tmp = gp_q[["end", "val"]].rename(columns={"val": "gross_profit"})
        margins_q = margins_q.merge(tmp, on="end", how="left")
        margins_q["Gross Margin %"] = (margins_q["gross_profit"] / margins_q["revenue"]) * 100.0

    if not op_q.empty:
        tmp = op_q[["end", "val"]].rename(columns={"val": "operating_income"})
        margins_q = margins_q.merge(tmp, on="end", how="left")
        margins_q["Operating Margin %"] = (margins_q["operating_income"] / margins_q["revenue"]) * 100.0

    if not ni_q.empty:
        tmp = ni_q[["end", "val"]].rename(columns={"val": "net_income"})
        margins_q = margins_q.merge(tmp, on="end", how="left")
        margins_q["Net Margin %"] = (margins_q["net_income"] / margins_q["revenue"]) * 100.0

# TTM
rev_ttm = compute_ttm(rev_q, "val") if not rev_q.empty else rev_q
ni_ttm = compute_ttm(ni_q, "val") if not ni_q.empty else ni_q

eps_ttm = None
if not eps_q.empty:
    eps_ttm = eps_q.sort_values("end").copy()
    eps_ttm["ttm_eps"] = eps_ttm["val"].rolling(4).sum()

# ----------------------------
# KPI Cards
# ----------------------------
def latest_val(df):
    if df is None or df.empty:
        return np.nan, None
    d = df.sort_values("end").iloc[-1]
    return float(d["val"]), pd.to_datetime(d["end"]).date()

k1, d1 = latest_val(rev_q)
k2, d2 = latest_val(eps_q)
k3, d3 = latest_val(ni_q)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Latest Quarterly Revenue", value="N/A" if np.isnan(k1) else f"{k1:,.0f}")
    if d1: st.caption(f"Period end: {d1}")
with c2:
    st.metric("Latest Quarterly EPS (Diluted)", value="N/A" if np.isnan(k2) else f"{k2:.2f}")
    if d2: st.caption(f"Period end: {d2}")
with c3:
    st.metric("Latest Quarterly Net Income", value="N/A" if np.isnan(k3) else f"{k3:,.0f}")
    if d3: st.caption(f"Period end: {d3}")
with c4:
    st.metric("Data Source", value="SEC EDGAR")

# ----------------------------
# Charts
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Earnings Trends", "Margins & Growth", "Earnings Surprises (Optional)"])

with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Revenue (Quarterly) + TTM")

        if rev_q.empty:
            st.info("Revenue not found yet — we can expand SEC tag fallbacks if needed.")
        else:
            plot_df = (
                rev_q[["end", "val"]]
                .rename(columns={"val": "Quarterly Revenue"})
                .merge(
                    compute_ttm(rev_q, "val")[["end", "ttm"]].rename(columns={"ttm": "TTM Revenue"}),
                    on="end",
                    how="left",
                )
                .dropna(subset=["end"])
                .sort_values("end")
            )

            base = alt.Chart(plot_df).encode(x=alt.X("end:T", title="Period End"))

            bars = base.mark_bar().encode(
                y=alt.Y("Quarterly Revenue:Q", title="Quarterly Revenue")
            )

            line = base.mark_line(strokeWidth=3).encode(
                y=alt.Y("TTM Revenue:Q", title="TTM Revenue")
            )

            st.altair_chart(
                alt.layer(bars, line).resolve_scale(y="independent"),
                use_container_width=True,
            )

    with right:
        st.subheader("EPS (Diluted) (Quarterly) + TTM (approx)")
        if eps_q.empty:
            st.info("EPS diluted not found for this ticker in SEC tags (rare).")
        else:
            plot_df = eps_q[["end", "val"]].rename(columns={"val": "Quarterly EPS"})
            if eps_ttm is not None:
                plot_df = plot_df.merge(
                    eps_ttm[["end", "ttm_eps"]].rename(columns={"ttm_eps": "TTM EPS"}),
                    on="end",
                    how="left"
                )
            plot_df = plot_df.set_index("end")
            st.line_chart(plot_df)

    st.subheader("Net Income (Quarterly) + TTM")
    if ni_q.empty:
        st.info("Net income not found for this ticker in SEC tags (rare).")
    else:
        plot_df = ni_q[["end", "val"]].rename(columns={"val": "Quarterly Net Income"})
        plot_df = plot_df.merge(
            ni_ttm[["end", "ttm"]].rename(columns={"ttm": "TTM Net Income"}),
            on="end",
            how="left"
        )
        plot_df = plot_df.set_index("end")
        st.line_chart(plot_df)

with tab2:
    left, right = st.columns(2)

    with left:
        st.subheader("YoY Growth (Quarterly)")
        if rev_q.empty and eps_q.empty:
            st.info("No revenue/EPS data to compute YoY.")
        else:
            out = pd.DataFrame({"end": pd.Series(dtype="datetime64[ns]")})
            if not rev_q.empty and "Revenue YoY %" in rev_q.columns:
                out = rev_q[["end", "Revenue YoY %"]].copy()
            if not eps_q.empty and "EPS YoY %" in eps_q.columns:
                e = eps_q[["end", "EPS YoY %"]].copy()
                out = out.merge(e, on="end", how="outer")
            out = out.sort_values("end").set_index("end")
            st.line_chart(out)

    with right:
        st.subheader("Margins (Quarterly)")
        if margins_q is None or margins_q.empty:
            st.info("Margins not available yet (needs revenue + profit tags).")
        else:
            cols = [c for c in ["Gross Margin %", "Operating Margin %", "Net Margin %"] if c in margins_q.columns]
            if not cols:
                st.info("Profit/income tags missing for margin calculation.")
            else:
                plot_df = margins_q[["end"] + cols].sort_values("end").set_index("end")
                st.line_chart(plot_df)

    st.subheader("Raw SEC Data (last 12 quarters)")

    ends = []
    if not rev_q.empty:
        ends.append(rev_q["end"])
    if not eps_q.empty:
        ends.append(eps_q["end"])
    if not ni_q.empty:
        ends.append(ni_q["end"])

    if not ends:
        st.info("No quarterly series available to show raw data.")
    else:
        show_df = pd.DataFrame({"end": pd.concat(ends).dropna().drop_duplicates().sort_values()})

        if not rev_q.empty:
            show_df = show_df.merge(rev_q[["end", "val"]].rename(columns={"val": "revenue"}), on="end", how="left")
        if not eps_q.empty:
            show_df = show_df.merge(eps_q[["end", "val"]].rename(columns={"val": "eps_diluted"}), on="end", how="left")
        if not ni_q.empty:
            show_df = show_df.merge(ni_q[["end", "val"]].rename(columns={"val": "net_income"}), on="end", how="left")

        show_df = show_df.sort_values("end", ascending=False).head(12)
        st.dataframe(show_df, use_container_width=True)

with tab3:
    st.subheader("Historical Earnings Surprises (EPS Actual vs Estimate)")
    if not show_surprises:
        st.info("Toggle surprises on in the sidebar to enable this panel.")
    elif not FMP_API_KEY:
        st.warning("Set env var `FMP_API_KEY` to enable historical surprises (free tier works).")
    else:
        with st.spinner("Loading surprises…"):
            raw = fmp_earnings_surprises(ticker, FMP_API_KEY)

        s_df = surprises_to_df(raw)
        if s_df.empty:
            st.info("No surprise data returned (could be coverage or rate limit).")
        else:
            plot = s_df[["date", "surprisePercent"]].dropna().sort_values("date").set_index("date")
            st.line_chart(plot)

            st.dataframe(
                s_df[["date", "epsEstimated", "epsActual", "surprise", "surprisePercent"]].head(20),
                use_container_width=True
            )

# ----------------------------
# Notes
# ----------------------------
st.divider()
st.subheader("Notes")
default_note = st.session_state.get(f"note_{ticker}", "")
note = st.text_area("Your thesis / notes for this company", value=default_note, height=120)
st.session_state[f"note_{ticker}"] = note
st.caption("Notes are stored in your current session. We can upgrade to persistent storage (file/db) in a later update.")

