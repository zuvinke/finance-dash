import os
import time

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

FP_ORDER = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5}

def ytd_to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Convert YTD income statement values to true quarterly by differencing within FY."""
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
        fy_row = g[g["fp"] == "FY"].tail(1)

        def val(row):
            return float(row["val"].iloc[0]) if not row.empty else None

        v1, v2, v3 = val(q1), val(q2), val(q3)

        if not q1.empty and v1 is not None:
            r = q1.iloc[0].to_dict()
            r["val"] = v1
            r["fp"] = "Q1"
            out.append(r)

        if not q2.empty and v2 is not None:
            r = q2.iloc[0].to_dict()
            r["val"] = v2 - v1 if (v1 is not None) else v2
            r["fp"] = "Q2"
            out.append(r)

        if not q3.empty and v3 is not None:
            r = q3.iloc[0].to_dict()
            r["val"] = v3 - v2 if (v2 is not None) else v3
            r["fp"] = "Q3"
            out.append(r)

        if not fy_row.empty:
            vfy = val(fy_row)
            if vfy is not None:
                r = fy_row.iloc[0].to_dict()
                r["val"] = vfy - v3 if (v3 is not None) else vfy
                r["fp"] = "Q4"
                out.append(r)

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

def latest_val(df: pd.DataFrame):
    if df is None or df.empty:
        return np.nan, "N/A"
    d = df.sort_values("end").iloc[-1]
    return float(d["val"]), d["end"].date().isoformat()

def fmt_billions(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x/1e9:,.2f}B"

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
# Chart helpers (bar-based)
# ----------------------------
def bar_plus_line(df, x_col, bar_col, line_col, bar_title, line_title):
    base = alt.Chart(df).encode(
        x=alt.X(f"{x_col}:T", title="Period End"),
        tooltip=[alt.Tooltip(f"{x_col}:T", title="End"),
                 alt.Tooltip(f"{bar_col}:Q", title=bar_title),
                 alt.Tooltip(f"{line_col}:Q", title=line_title)]
    )
    bars = base.mark_bar().encode(
        y=alt.Y(f"{bar_col}:Q", title=bar_title)
    )
    line = base.mark_line(strokeWidth=3).encode(
        y=alt.Y(f"{line_col}:Q", title=line_title)
    )
    st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent"), use_container_width=True)

def grouped_bars_long(df_long, y_title):
    chart = alt.Chart(df_long).mark_bar().encode(
        x=alt.X("end:T", title="Period End"),
        y=alt.Y("value:Q", title=y_title),
        color=alt.Color("series:N", title="Series"),
        xOffset="series:N",
        tooltip=[alt.Tooltip("end:T", title="End"),
                 alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("value:Q", title=y_title)]
    )
    st.altair_chart(chart, use_container_width=True)

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
            return keep_latest_per_period(df)
    return pd.DataFrame(columns=["end", "fy", "fp", "form", "filed", "val", "metric"])

rev_df = load_metric_df("Revenue", REVENUE_TAG_CANDIDATES, preferred_units=("USD",))
ni_df  = load_metric_df("Net Income", NET_INCOME_CANDIDATES, preferred_units=("USD",))
eps_df = load_metric_df("EPS (Diluted)", EPS_DILUTED_CANDIDATES, preferred_units=("USD/shares",))
gp_df  = load_metric_df("Gross Profit", GROSS_PROFIT_CANDIDATES, preferred_units=("USD",))
op_df  = load_metric_df("Operating Income", OPERATING_INCOME_CANDIDATES, preferred_units=("USD",))

# ----------------------------
# Quarterly series (KEY FIX)
# Revenue is often YTD/FY-only in SEC facts, so default to YTD→Quarterly.
# If that fails, fallback to raw quarterly.
# ----------------------------
rev_q = ytd_to_quarterly(rev_df)
if rev_q.empty:
    rev_q = build_quarterly(rev_df)

# Other income statement items: YTD→Quarterly tends to be best
ni_q  = ytd_to_quarterly(ni_df)
gp_q  = ytd_to_quarterly(gp_df)
op_q  = ytd_to_quarterly(op_df)

# EPS usually already quarterly
eps_q = build_quarterly(eps_df)

# Derived metrics
if not rev_q.empty:
    rev_q = rev_q.sort_values("end").reset_index(drop=True)
    rev_q["Revenue YoY %"] = yoy_growth(rev_q["val"], periods=4)

if not eps_q.empty:
    eps_q = eps_q.sort_values("end").reset_index(drop=True)
    eps_q["EPS YoY %"] = yoy_growth(eps_q["val"], periods=4)

# TTM
rev_ttm = compute_ttm(rev_q, "val") if not rev_q.empty else rev_q
ni_ttm  = compute_ttm(ni_q, "val") if not ni_q.empty else ni_q

eps_ttm = None
if not eps_q.empty:
    eps_ttm = eps_q.sort_values("end").copy()
    eps_ttm["ttm_eps"] = eps_ttm["val"].rolling(4).sum()

# Margins
margins_q = None
if not rev_q.empty:
    margins_q = pd.DataFrame({"end": rev_q["end"], "revenue": rev_q["val"]})
    if not gp_q.empty:
        margins_q = margins_q.merge(gp_q[["end", "val"]].rename(columns={"val": "gross_profit"}), on="end", how="left")
        margins_q["Gross Margin %"] = (margins_q["gross_profit"] / margins_q["revenue"]) * 100.0
    if not op_q.empty:
        margins_q = margins_q.merge(op_q[["end", "val"]].rename(columns={"val": "operating_income"}), on="end", how="left")
        margins_q["Operating Margin %"] = (margins_q["operating_income"] / margins_q["revenue"]) * 100.0
    if not ni_q.empty:
        margins_q = margins_q.merge(ni_q[["end", "val"]].rename(columns={"val": "net_income"}), on="end", how="left")
        margins_q["Net Margin %"] = (margins_q["net_income"] / margins_q["revenue"]) * 100.0

# KPI (independent latest)
k_rev, d_rev = latest_val(rev_q)
k_eps, d_eps = latest_val(eps_q)
k_ni,  d_ni  = latest_val(ni_q)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Latest Quarterly Revenue", value=fmt_billions(k_rev))
    st.caption(f"Period end: {d_rev}")
with c2:
    st.metric("Latest Quarterly EPS (Diluted)", value="N/A" if np.isnan(k_eps) else f"{k_eps:.2f}")
    st.caption(f"Period end: {d_eps}")
with c3:
    st.metric("Latest Quarterly Net Income", value=fmt_billions(k_ni))
    st.caption(f"Period end: {d_ni}")
with c4:
    st.metric("Data Source", value="SEC EDGAR")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Earnings Trends", "Margins & Growth", "Earnings Surprises (Optional)"])

with tab1:
    left, right = st.columns(2)

    with left:
        st.subheader("Revenue (Quarterly) + TTM")
        if rev_q.empty:
            st.info("Revenue not found yet — we can expand SEC tag fallbacks if needed.")
        else:
            plot = (
                rev_q[["end", "val"]]
                .rename(columns={"val": "q"})
                .merge(rev_ttm[["end", "ttm"]].rename(columns={"ttm": "ttm"}), on="end", how="left")
                .sort_values("end")
            )
            plot["Quarterly Revenue ($B)"] = plot["q"] / 1e9
            plot["TTM Revenue ($B)"] = plot["ttm"] / 1e9
            bar_plus_line(plot, "end", "Quarterly Revenue ($B)", "TTM Revenue ($B)",
                          "Quarterly Revenue ($B)", "TTM Revenue ($B)")

    with right:
        st.subheader("EPS (Diluted) (Quarterly) + TTM (approx)")
        if eps_q.empty:
            st.info("EPS diluted not found for this ticker in SEC tags (rare).")
        else:
            plot = eps_q[["end", "val"]].rename(columns={"val": "q"}).sort_values("end")
            if eps_ttm is not None:
                plot = plot.merge(eps_ttm[["end", "ttm_eps"]].rename(columns={"ttm_eps": "ttm"}), on="end", how="left")
            else:
                plot["ttm"] = np.nan
            plot["Quarterly EPS"] = plot["q"]
            plot["TTM EPS"] = plot["ttm"]
            bar_plus_line(plot, "end", "Quarterly EPS", "TTM EPS", "Quarterly EPS", "TTM EPS")

    st.subheader("Net Income (Quarterly) + TTM")
    if ni_q.empty:
        st.info("Net income not found for this ticker in SEC tags (rare).")
    else:
        plot = (
            ni_q[["end", "val"]]
            .rename(columns={"val": "q"})
            .merge(ni_ttm[["end", "ttm"]].rename(columns={"ttm": "ttm"}), on="end", how="left")
            .sort_values("end")
        )
        plot["Quarterly Net Income ($B)"] = plot["q"] / 1e9
        plot["TTM Net Income ($B)"] = plot["ttm"] / 1e9
        bar_plus_line(plot, "end", "Quarterly Net Income ($B)", "TTM Net Income ($B)",
                      "Quarterly Net Income ($B)", "TTM Net Income ($B)")

with tab2:
    left, right = st.columns(2)

    with left:
        st.subheader("YoY Growth (Quarterly)")
        rows = []
        if not rev_q.empty and "Revenue YoY %" in rev_q.columns:
            rows.append(rev_q[["end", "Revenue YoY %"]].rename(columns={"Revenue YoY %": "value"}).assign(series="Revenue YoY %"))
        if not eps_q.empty and "EPS YoY %" in eps_q.columns:
            rows.append(eps_q[["end", "EPS YoY %"]].rename(columns={"EPS YoY %": "value"}).assign(series="EPS YoY %"))
        if not rows:
            st.info("No revenue/EPS data to compute YoY.")
        else:
            yoy_long = pd.concat(rows, ignore_index=True).dropna(subset=["end", "value"]).sort_values("end")
            grouped_bars_long(yoy_long, "YoY Growth (%)")

    with right:
        st.subheader("Margins (Quarterly)")
        if margins_q is None or margins_q.empty:
            st.info("Margins not available yet (needs revenue + profit tags).")
        else:
            cols = [c for c in ["Gross Margin %", "Operating Margin %", "Net Margin %"] if c in margins_q.columns]
            if not cols:
                st.info("Profit/income tags missing for margin calculation.")
            else:
                m_long = margins_q[["end"] + cols].melt(id_vars=["end"], var_name="series", value_name="value")
                m_long = m_long.dropna(subset=["end", "value"]).sort_values("end")
                grouped_bars_long(m_long, "Margin (%)")

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
            plot = s_df[["date", "surprisePercent"]].dropna().sort_values("date").rename(
                columns={"date": "end", "surprisePercent": "value"}
            )
            plot["series"] = "Surprise %"
            grouped_bars_long(plot, "Surprise (%)")

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
