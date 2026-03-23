
# =================================================================
#  AI Trading Screener v5
#  - Live auto-refresh (30s / 1min / 5min)
#  - Intradenne grafy: 1m, 5m, 15m, 1h, 1d s range-selectorem
#  - Lepsi AI model: multi-timeframe trenovani
#  - Kompletni redesign UI
# =================================================================
import os, math, time, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(
    page_title="AI Trader v5 LIVE",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background: #1a1a2e; border-radius: 10px;
    padding: 12px; border: 1px solid #16213e;
}
div[data-testid="stRadio"] > div { flex-direction: row; gap: 8px; }
.live-badge {
    display:inline-block; background:#00c7b7; color:#000;
    padding:2px 10px; border-radius:20px; font-weight:bold; font-size:0.8rem;
}
.signal-card {
    padding:1.2rem; border-radius:12px; background:#1a1a2e;
    margin:0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Konstanty ────────────────────────────────────────────────────
PRESETS = {
    "── AKCIE ──": None,
    "Apple (AAPL)":        "AAPL",
    "Microsoft (MSFT)":    "MSFT",
    "NVIDIA (NVDA)":       "NVDA",
    "Tesla (TSLA)":        "TSLA",
    "Amazon (AMZN)":       "AMZN",
    "Meta (META)":         "META",
    "Google (GOOGL)":      "GOOGL",
    "AMD":                 "AMD",
    "Palantir (PLTR)":     "PLTR",
    "── INDEXY / ETF ──":  None,
    "S&P 500 (SPY)":       "SPY",
    "Nasdaq (QQQ)":        "QQQ",
    "Russell 2000 (IWM)":  "IWM",
    "── KOMODITY ──":      None,
    "Zlato (GC=F)":        "GC=F",
    "Stribro (SI=F)":      "SI=F",
    "Ropa WTI (CL=F)":     "CL=F",
    "── KRYPTO ──":        None,
    "Bitcoin (BTC-USD)":   "BTC-USD",
    "Ethereum (ETH-USD)":  "ETH-USD",
    "Solana (SOL-USD)":    "SOL-USD",
}

SECTOR_ETFS = {
    "Technology":            "XLK",
    "Health Care":           "XLV",
    "Financials":            "XLF",
    "Consumer Discretionary":"XLY",
    "Communication Services":"XLC",
    "Industrials":           "XLI",
    "Consumer Staples":      "XLP",
    "Energy":                "XLE",
}

TF_PARAMS = {
    "1m":  {"period":"5d",   "interval":"1m",  "label":"1 minuta",  "bars":390},
    "5m":  {"period":"5d",   "interval":"5m",  "label":"5 minut",   "bars":200},
    "15m": {"period":"60d",  "interval":"15m", "label":"15 minut",  "bars":200},
    "1h":  {"period":"730d", "interval":"1h",  "label":"1 hodina",  "bars":200},
    "1d":  {"period":"5y",   "interval":"1d",  "label":"1 den",     "bars":200},
}

FCOLS = [
    "r1","r5","r20","ma5","ma20","ma50",
    "rsi","stoch_k","stoch_d","willi",
    "macd","bb","cci","atr",
    "obv","donch","adx","v5","v20","vc",
]

# ── Indikatory ───────────────────────────────────────────────────
def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _stoch(high, low, close, k=14, d=3):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    sk = 100 * (close - lo) / (hi - lo + 1e-9)
    return sk, sk.rolling(d).mean()

def _williams(high, low, close, p=14):
    hi = high.rolling(p).max()
    lo = low.rolling(p).min()
    return -100 * (hi - close) / (hi - lo + 1e-9)

def _macd(s, fast=12, slow=26, sig=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    m  = ef - es
    return (m - m.ewm(span=sig, adjust=False).mean()) / (s.abs() + 1e-9)

def _bb(s, p=20):
    ma  = s.rolling(p).mean()
    std = s.rolling(p).std()
    return (s - (ma - 2*std)) / (4*std + 1e-9)

def _cci(high, low, close, p=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(p).mean()
    md = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - ma) / (0.015 * md + 1e-9) / 100

def _atr(high, low, close, p=14):
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean() / (close + 1e-9)

def _obv(close, volume, p=10):
    obv = (np.sign(close.diff()) * volume).cumsum()
    return obv.pct_change(p)

def _donch(close, p=20):
    hi = close.rolling(p).max()
    lo = close.rolling(p).min()
    return (close - lo) / (hi - lo + 1e-9)

def _adx(high, low, close, p=14):
    tr   = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr_ = tr.ewm(span=p, adjust=False).mean()
    up   = high.diff()
    dn   = -low.diff()
    pdm  = pd.Series(np.where((up > dn) & (up > 0), up, 0), index=close.index)
    ndm  = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), index=close.index)
    pdi  = 100 * pdm.ewm(span=p, adjust=False).mean() / (atr_ + 1e-9)
    ndi  = 100 * ndm.ewm(span=p, adjust=False).mean() / (atr_ + 1e-9)
    dx   = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    return dx.ewm(span=p, adjust=False).mean() / 100

def pivot_levels(high, low, close, n=20):
    h = float(high.tail(n).max())
    l = float(low.tail(n).min())
    c = float(close.iloc[-1])
    p = (h + l + c) / 3
    return {
        "R2": round(p + (h - l), 4),
        "R1": round(2*p - l, 4),
        "PP": round(p, 4),
        "S1": round(2*p - h, 4),
        "S2": round(p - (h - l), 4),
    }

def featurize(df):
    c  = df["Close"]
    hi = df.get("High", c)
    lo = df.get("Low",  c)
    vo = df.get("Volume", pd.Series(1.0, index=c.index))
    fr = c.shift(-1) / c - 1
    sk, sd = _stoch(hi, lo, c)
    feat = pd.DataFrame({
        "r1":     c.pct_change(),
        "r5":     c.pct_change(5),
        "r20":    c.pct_change(20),
        "ma5":    c.rolling(5).mean()  / c - 1,
        "ma20":   c.rolling(20).mean() / c - 1,
        "ma50":   c.rolling(50).mean() / c - 1,
        "rsi":    _rsi(c) / 100,
        "stoch_k":sk / 100,
        "stoch_d":sd / 100,
        "willi":  _williams(hi, lo, c) / 100,
        "macd":   _macd(c),
        "bb":     _bb(c),
        "cci":    _cci(hi, lo, c),
        "atr":    _atr(hi, lo, c),
        "obv":    _obv(c, vo),
        "donch":  _donch(c),
        "adx":    _adx(hi, lo, c),
        "v5":     c.pct_change().rolling(5).std(),
        "v20":    c.pct_change().rolling(20).std(),
        "vc":     vo.pct_change(),
        "date":   df["date"],
        "close":  c,
        "high":   hi,
        "low":    lo,
        "open":   df.get("Open", c),
        "volume": vo,
    })
    feat["target"] = (fr > 0).astype(int)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return feat, fr

# ── Data loader ──────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def load_intraday(ticker, tf="1d"):
    try:
        import yfinance as yf
        params = TF_PARAMS[tf]
        if tf == "1d":
            df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
        else:
            df = yf.download(ticker, period=params["period"],
                             interval=params["interval"],
                             auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        date_col = "Datetime" if "Datetime" in df.columns else "Date"
        if date_col in df.columns:
            df.rename(columns={date_col: "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"]).sort_values("date").reset_index(drop=True)
        if "Volume" not in df.columns or df["Volume"].isna().all():
            df["Volume"] = 1.0
        df["Volume"] = df["Volume"].replace(0, np.nan).ffill().fillna(1.0)
        for col in ["High","Low","Open"]:
            if col not in df.columns:
                df[col] = df["Close"]
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_daily(ticker, period="5y"):
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame(), {}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"]).sort_values("date").reset_index(drop=True)
        if "Volume" not in df.columns or df["Volume"].isna().all():
            df["Volume"] = 1.0
        df["Volume"] = df["Volume"].replace(0, np.nan).ffill().fillna(1.0)
        for col in ["High","Low","Open"]:
            if col not in df.columns:
                df[col] = df["Close"]
        info = {}
        try:
            tk  = yf.Ticker(ticker)
            raw = tk.info
            info = {
                "name":         raw.get("longName", ticker),
                "sector":       raw.get("sector", "N/A"),
                "industry":     raw.get("industry", "N/A"),
                "mktcap":       raw.get("marketCap", 0),
                "pe":           raw.get("trailingPE", 0),
                "eps":          raw.get("trailingEps", 0),
                "beta":         raw.get("beta", 0),
                "52w_high":     raw.get("fiftyTwoWeekHigh", 0),
                "52w_low":      raw.get("fiftyTwoWeekLow", 0),
                "avg_volume":   raw.get("averageVolume", 0),
                "div_yield":    raw.get("dividendYield", 0),
                "target_price": raw.get("targetMeanPrice", 0),
            }
        except Exception:
            pass
        return df, info
    except Exception:
        return pd.DataFrame(), {}

@st.cache_resource(show_spinner=False)
def train_model(ticker, period="5y"):
    from sklearn.ensemble import (RandomForestClassifier,
                                  HistGradientBoostingClassifier,
                                  VotingClassifier)
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    df, _ = load_daily(ticker, period)
    if df.empty or len(df) < 200:
        return None, None, None, None

    feat, fr = featurize(df)
    N  = len(feat)
    ve = int(N * 0.85)
    X, y = feat[FCOLS].values, feat["target"].values
    sc   = StandardScaler()
    Xtv  = sc.fit_transform(X[:ve])

    rf  = RandomForestClassifier(
        n_estimators=250, max_depth=7, min_samples_leaf=4,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=250, max_depth=5, learning_rate=0.04,
        min_samples_leaf=8, random_state=42,
    )
    vote = VotingClassifier([("rf", rf), ("hgb", hgb)], voting="soft", n_jobs=-1)
    cal  = CalibratedClassifierCV(vote, cv=3, method="isotonic")
    cal.fit(Xtv, y[:ve])
    return cal, sc, feat, fr

# ── Backtest ─────────────────────────────────────────────────────
def run_backtest(probs, rets, uth, leverage=1.0, tc=0.0005,
                 margin_rate=0.06, sl=None, use_sizing=True):
    n     = min(len(probs), len(rets))
    probs = np.nan_to_num(np.array(probs[:n], dtype=float), nan=0.5)
    rets  = np.nan_to_num(np.array(rets[:n],  dtype=float), nan=0.0)
    lth   = 1 - uth
    pos   = np.where(probs >= uth, 1.0, np.where(probs <= lth, -1.0, 0.0))
    if use_sizing:
        conf = np.abs(probs - 0.5) * 2
        pos  = pos * np.clip(conf, 0.3, 1.0)
    daily_margin = margin_rate / 252 * max(leverage - 1.0, 0.0)
    equity = [1.0]
    trades, wins = [], []
    in_trade, entry_eq = False, 1.0
    cur_pos = 0.0
    for i in range(n):
        p = pos[i]; r = rets[i]
        cost      = abs(p - cur_pos) * tc
        lev_ret   = p * r * leverage - daily_margin * abs(p)
        new_eq    = equity[-1] * (1.0 + lev_ret - cost)
        if sl is not None and in_trade:
            liq = sl / leverage
            if new_eq / entry_eq - 1 < -liq:
                new_eq = entry_eq * (1.0 - liq)
                p = 0.0; in_trade = False
        if p != 0.0 and not in_trade:
            in_trade = True; entry_eq = equity[-1]
        elif p == 0.0 and in_trade:
            trades.append(new_eq / entry_eq - 1.0)
            wins.append(new_eq >= entry_eq)
            in_trade = False
        equity.append(max(new_eq, 1e-9))
        cur_pos = p
    equity = np.array(equity[1:])
    yr     = max(len(equity) / 252, 1e-9)
    cagr   = float(equity[-1]**(1.0 / yr) - 1.0)
    daily  = np.diff(np.log(np.maximum(equity, 1e-9)))
    sh     = float(daily.mean() / daily.std() * math.sqrt(252)) if daily.std() > 0 else 0.0
    rm     = np.maximum.accumulate(equity)
    mdd    = float((equity / rm - 1.0).min())
    win_r  = float(np.mean(wins)) if wins else 0.0
    profits= [t for t in trades if t > 0]
    losses = [abs(t) for t in trades if t < 0]
    pf     = sum(profits) / sum(losses) if losses and sum(losses) > 0 else 99.0
    return dict(cagr=cagr, sharpe=sh, max_dd=mdd, final_eq=float(equity[-1]),
                equity=equity, pos=pos, rets=pos*rets,
                win_rate=win_r, profit_factor=min(float(pf), 50.0),
                avg_trade=float(np.mean(trades)) if trades else 0.0,
                n_trades=len(trades))

def best_threshold(probs_v, rets_v):
    best_sh, best_uth = -999.0, 0.55
    for uth in np.arange(0.50, 0.68, 0.01):
        try:
            bt = run_backtest(probs_v, rets_v, float(uth),
                              leverage=1.0, tc=0.0, use_sizing=False)
            if bt["sharpe"] > best_sh and bt["n_trades"] > 3:
                best_sh = bt["sharpe"]; best_uth = uth
        except Exception:
            pass
    return round(float(best_uth), 2)

def sig_info(prob, uth):
    if prob >= uth:
        return "LONG",  "#00c7b7", "🟢"
    elif prob <= 1 - uth:
        return "SHORT", "#ff4b4b", "🔴"
    return "FLAT", "#888888", "⚪"

def fmt_large(n):
    if not n:
        return "N/A"
    for div, suf in [(1e12,"T"),(1e9,"B"),(1e6,"M"),(1e3,"K")]:
        if abs(n) >= div:
            return str(round(n / div, 2)) + suf
    return str(round(n, 2))

def monthly_pnl(dates, rets):
    df_m = pd.DataFrame({"date": pd.to_datetime(dates), "ret": rets})
    df_m["yr"] = df_m["date"].dt.year
    df_m["mo"] = df_m["date"].dt.month
    piv = df_m.groupby(["yr","mo"])["ret"].sum().unstack(fill_value=0) * 100
    mn  = {1:"Led",2:"Uno",3:"Bze",4:"Dub",5:"Kve",6:"Cvn",
           7:"Cvc",8:"Srp",9:"Zar",10:"Rib",11:"Lis",12:"Pro"}
    piv.columns = [mn.get(m, str(m)) for m in piv.columns]
    return piv.round(2)

# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚀 AI Trader v5")
    st.markdown("---")

    preset_label = st.selectbox("Instrument", list(PRESETS.keys()),
                                 index=list(PRESETS.keys()).index("Apple (AAPL)"))
    preset_val   = PRESETS.get(preset_label)
    custom       = st.text_input("Vlastni ticker (PLTR, AMD, BRK-B...)", value="")
    ticker       = (custom.strip().upper() if custom.strip()
                    else (preset_val if preset_val else "AAPL"))

    st.markdown("#### Graf – timeframe")
    tf_selected = st.radio("", list(TF_PARAMS.keys()), index=4,
                            horizontal=True, label_visibility="collapsed")

    st.markdown("#### Live refresh")
    live_mode    = st.toggle("Auto-refresh", value=False)
    refresh_int  = st.selectbox("Interval", ["30s","1 min","5 min"], index=1,
                                 disabled=not live_mode)

    st.markdown("---")
    st.markdown("#### Model & Rizeni")
    period_train = st.selectbox("Trenovaci data", ["2y","3y","5y","max"], index=2)
    leverage     = st.slider("Paka", 1.0, 10.0, 1.0, 0.5)
    margin_rate  = st.slider("Margin p.a. (%)", 0.0, 12.0, 6.0, 0.5) / 100
    use_sl       = st.toggle("Stop-Loss", value=True)
    sl_val       = st.slider("Stop-Loss (%)", 1, 20, 5) / 100 if use_sl else None
    use_sizing   = st.toggle("Confidence sizing", value=True)
    tc_bps       = st.number_input("Naklady (bps)", 0, 50, 5)
    auto_uth     = st.toggle("Auto-prah", value=True)
    man_uth      = st.slider("Rucni prah", 0.50, 0.70, 0.55, 0.01)
    st.markdown("---")
    st.caption("RF + HistGB | 20 indikatoru | Yahoo Finance")

tc = tc_bps / 10000

# ── Auto-refresh logika ───────────────────────────────────────────
if live_mode:
    intervals = {"30s": 30, "1 min": 60, "5 min": 300}
    wait_sec  = intervals.get(refresh_int, 60)
    st.markdown(
        '<span class="live-badge">🔴 LIVE – refresh za '
        + str(wait_sec) + 's</span>',
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════
#  NACITANI DAT A MODELU
# ════════════════════════════════════════════════════════════════
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.markdown("## " + ticker + " – AI Trading Screener")
with col_h2:
    refresh_btn = st.button("🔄 Refresh", use_container_width=True)

if refresh_btn:
    st.cache_data.clear()
    st.rerun()

now_str = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
st.caption("Posledni aktualizace: " + now_str)

with st.spinner("Nacitam denni data a trenuju model..."):
    df_daily, info = load_daily(ticker, period_train)
    model, sc, feat, fr = train_model(ticker, period_train)

with st.spinner("Nacitam " + TF_PARAMS[tf_selected]["label"] + " grafy..."):
    df_chart = load_intraday(ticker, tf_selected)

if df_daily.empty or model is None:
    st.error("Nepodarilo se nacist data pro " + ticker + ". Zkus jiny symbol.")
    st.stop()

X     = feat[FCOLS].values
probs = model.predict_proba(sc.transform(X))[:, 1]
N2    = len(feat)
ve2   = int(N2 * 0.80)

val_p = probs[ve2:int(N2*0.90)]
val_r = fr.iloc[ve2:int(N2*0.90)].values[:len(val_p)]
uth_  = best_threshold(val_p, val_r) if auto_uth else man_uth

p_now = float(probs[-1])
lbl, color, icon = sig_info(p_now, uth_)

# ════════════════════════════════════════════════════════════════
#  HORNI METRIKOVY PAS
# ════════════════════════════════════════════════════════════════
price_now  = round(float(feat["close"].iloc[-1]), 2)
price_prev = round(float(feat["close"].iloc[-2]), 2)
price_chg  = round((price_now / price_prev - 1) * 100, 2)
rsi_now    = round(float(feat["rsi"].iloc[-1] * 100), 1)
adx_now    = round(float(feat["adx"].iloc[-1] * 100), 1)
bb_now     = round(float(feat["bb"].iloc[-1]), 2)

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("Cena", str(price_now), delta=str(price_chg) + "%")
m2.metric("AI Signal", icon + " " + lbl, delta="p=" + str(round(p_now*100,1)) + "%")
m3.metric("RSI (14)", str(rsi_now),
          delta="Prekoup." if rsi_now > 70 else ("Prep." if rsi_now < 30 else "Neutral"))
m4.metric("ADX", str(adx_now), delta="Silny trend" if adx_now > 25 else "Slaby")
m5.metric("BB pozice", str(bb_now))
m6.metric("Prah", str(uth_))
m7.metric("Paka", str(leverage) + "x")

if info:
    mktcap_s = fmt_large(info.get("mktcap", 0))
    pe_s     = str(round(info.get("pe", 0), 1)) if info.get("pe") else "N/A"
    beta_s   = str(round(info.get("beta", 0), 2)) if info.get("beta") else "N/A"
    hi52_s   = str(round(info.get("52w_high", 0), 2)) if info.get("52w_high") else "N/A"
    lo52_s   = str(round(info.get("52w_low", 0), 2)) if info.get("52w_low") else "N/A"
    tgt_s    = str(round(info.get("target_price", 0), 2)) if info.get("target_price") else "N/A"
    f1,f2,f3,f4,f5,f6 = st.columns(6)
    f1.metric("Trzni kap.", mktcap_s)
    f2.metric("P/E", pe_s)
    f3.metric("Beta", beta_s)
    f4.metric("52T Max", hi52_s)
    f5.metric("52T Min", lo52_s)
    f6.metric("Cil cena anal.", tgt_s)
    sec_s = (info.get("sector","") + " / " + info.get("industry","")).strip(" /")
    if sec_s:
        st.caption("Sektor: " + sec_s + "  |  Nastroj: " + ticker)

st.markdown("---")

# ════════════════════════════════════════════════════════════════
#  STRÁNKY
# ════════════════════════════════════════════════════════════════
PAGE = st.radio("", ["Graf", "Backtest & Paka", "Trh & Sektor", "Screener"],
                horizontal=True, label_visibility="collapsed")
st.markdown("---")

# ────────────────────────────────────────────────────────────────
#  PAGE 1 – GRAF (s intradennimi daty)
# ────────────────────────────────────────────────────────────────
if PAGE == "Graf":

    use_df = df_chart if not df_chart.empty else df_daily
    disp_n = TF_PARAMS[tf_selected]["bars"]
    disp   = use_df.tail(disp_n).reset_index(drop=True)

    # Indikatory pro zobrazeni
    c_disp = disp["Close"]
    ma20   = c_disp.rolling(20).mean()
    ma50   = c_disp.rolling(50).mean()
    bb_mid = c_disp.rolling(20).mean()
    bb_std = c_disp.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std
    rsi_d  = _rsi(c_disp)
    vol_d  = disp.get("Volume", pd.Series(1.0, index=disp.index))

    # AI signaly na dennim modelu – pouze pro 1d timeframe
    if tf_selected == "1d":
        d_probs = probs[max(0, N2 - disp_n):]
        pos_arr = np.where(d_probs >= uth_, 1, np.where(d_probs <= 1-uth_, -1, 0))
        long_i  = np.where(pos_arr == 1)[0]
        short_i = np.where(pos_arr == -1)[0]
    else:
        d_probs = None
        long_i  = np.array([])
        short_i = np.array([])

    pivots = pivot_levels(disp["High"], disp["Low"], disp["Close"])

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        vertical_spacing=0.02,
        subplot_titles=("Cena   [" + TF_PARAMS[tf_selected]["label"] + "]",
                        "Volume", "RSI (14)", "MACD"),
    )

    # Svicky
    fig.add_trace(go.Candlestick(
        x=disp["date"], open=disp["Open"], high=disp["High"],
        low=disp["Low"],  close=disp["Close"],
        increasing_line_color="#00c7b7", decreasing_line_color="#ff4b4b",
        name="Cena"), row=1, col=1)

    # MA
    fig.add_trace(go.Scatter(x=disp["date"], y=ma20, name="MA20",
                              line=dict(color="#ffd700", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=disp["date"], y=ma50, name="MA50",
                              line=dict(color="#ff9900", width=1.2, dash="dot")), row=1, col=1)

    # Bollingerova pasma
    fig.add_trace(go.Scatter(x=disp["date"], y=bb_up, name="BB+",
                              line=dict(color="rgba(100,100,255,0.5)", width=0.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=disp["date"], y=bb_dn, name="BB-",
                              line=dict(color="rgba(100,100,255,0.5)", width=0.8),
                              fill="tonexty",
                              fillcolor="rgba(100,100,255,0.05)"), row=1, col=1)

    # Pivot uroven
    for lvl_name, lvl_val in pivots.items():
        lc = "#00c7b7" if "R" in lvl_name else ("#ff4b4b" if "S" in lvl_name else "#ffffff")
        fig.add_hline(y=lvl_val, line_dash="dash", line_color=lc,
                      line_width=0.6, row=1, col=1,
                      annotation_text=lvl_name,
                      annotation_position="right",
                      annotation_font_size=9)

    # AI signaly (jen na 1d)
    if tf_selected == "1d":
        if len(long_i) > 0:
            fig.add_trace(go.Scatter(
                x=disp["date"].iloc[long_i],
                y=disp["Low"].iloc[long_i] * 0.995,
                mode="markers",
                marker=dict(symbol="triangle-up", size=10, color="#00c7b7"),
                name="LONG"), row=1, col=1)
        if len(short_i) > 0:
            fig.add_trace(go.Scatter(
                x=disp["date"].iloc[short_i],
                y=disp["High"].iloc[short_i] * 1.005,
                mode="markers",
                marker=dict(symbol="triangle-down", size=10, color="#ff4b4b"),
                name="SHORT"), row=1, col=1)

    # Volume
    vol_colors = ["#00c7b7" if c >= o else "#ff4b4b"
                  for c, o in zip(disp["Close"], disp["Open"])]
    fig.add_trace(go.Bar(x=disp["date"], y=vol_d,
                         marker_color=vol_colors, name="Vol"), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=disp["date"], y=rsi_d,
                              line=dict(color="#a78bfa", width=1.5), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ff4b4b", line_width=0.7, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#00c7b7", line_width=0.7, row=3, col=1)
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)",
                  line_width=0, row=3, col=1)

    # MACD
    macd_line = _macd(c_disp) * c_disp.abs()
    fig.add_trace(go.Scatter(x=disp["date"], y=macd_line,
                              line=dict(color="#f0c040", width=1.2), name="MACD"), row=4, col=1)
    fig.add_hline(y=0, line_color="#888", line_width=0.5, row=4, col=1)

    # Range selector
    range_buttons = [
        dict(count=1,  label="1D",  step="day",   stepmode="backward"),
        dict(count=5,  label="5D",  step="day",   stepmode="backward"),
        dict(count=1,  label="1M",  step="month", stepmode="backward"),
        dict(count=3,  label="3M",  step="month", stepmode="backward"),
        dict(count=6,  label="6M",  step="month", stepmode="backward"),
        dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
        dict(step="all", label="Max"),
    ]
    fig.update_layout(
        height=760,
        showlegend=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="#ffffff",
        margin=dict(t=30, b=10, l=10, r=80),
        xaxis=dict(
            rangeselector=dict(
                buttons=range_buttons,
                bgcolor="#1a1a2e",
                activecolor="#00c7b7",
                font=dict(color="#ffffff"),
            ),
            rangeslider=dict(visible=False),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Pivot tabulka
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Pivot S&R uroven**")
        piv_df = pd.DataFrame([
            {"Uroven": k, "Cena": v,
             "Typ": "Odpor" if "R" in k else ("Podpora" if "S" in k else "Pivot")}
            for k, v in pivots.items()
        ])
        st.dataframe(piv_df, use_container_width=True, hide_index=True)
    with cb:
        st.markdown("**Indikatory (posledni svieka)**")
        ind_df = pd.DataFrame([
            {"Indikator": "RSI (14)",        "Hodnota": str(rsi_now),
             "Signal": "Prekoupeno" if rsi_now > 70 else ("Preprodano" if rsi_now < 30 else "Neutral")},
            {"Indikator": "ADX (14)",        "Hodnota": str(adx_now),
             "Signal": "Silny trend" if adx_now > 25 else "Slaby trend"},
            {"Indikator": "BB pozice",       "Hodnota": str(bb_now),
             "Signal": "Horni pasmo" if bb_now > 0.8 else ("Dolni pasmo" if bb_now < 0.2 else "Stred")},
            {"Indikator": "AI Prob(up)",     "Hodnota": str(round(p_now*100,1)) + "%",
             "Signal": lbl},
            {"Indikator": "Stoch K",
             "Hodnota": str(round(float(feat["stoch_k"].iloc[-1]*100),1)),
             "Signal": "N/A"},
            {"Indikator": "Williams %R",
             "Hodnota": str(round(float(feat["willi"].iloc[-1]*100),1)),
             "Signal": "N/A"},
        ])
        st.dataframe(ind_df, use_container_width=True, hide_index=True)

    # Signal karta
    st.markdown(
        '<div class="signal-card" style="border-left: 7px solid '
        + color + ';">'
        + '<b style="font-size:1.4rem;">' + icon + " " + lbl + "</b><br><br>"
        + "Prob(up): <b>" + str(round(p_now*100,1)) + "%</b> &nbsp;|&nbsp; "
        + "Cena: <b>" + str(price_now) + "</b> &nbsp;|&nbsp; "
        + "Datum: <b>" + feat["date"].iloc[-1].strftime("%d.%m.%Y %H:%M") + "</b>"
        + "</div>",
        unsafe_allow_html=True,
    )

# ────────────────────────────────────────────────────────────────
#  PAGE 2 – BACKTEST & PAKA
# ────────────────────────────────────────────────────────────────
elif PAGE == "Backtest & Paka":
    st.subheader("Backtest – " + ticker + " | Paka: " + str(leverage) + "x")

    test_idx = int(N2 * 0.90)
    te_probs = probs[test_idx:]
    te_rets  = fr.iloc[test_idx:test_idx + len(te_probs)].fillna(0).values
    bt1      = run_backtest(te_probs, te_rets, uth_, 1.0,      tc, margin_rate, sl_val, use_sizing)
    btL      = run_backtest(te_probs, te_rets, uth_, leverage,  tc, margin_rate, sl_val, use_sizing)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("##### 1x (bez paky)")
        c1,c2,c3 = st.columns(3)
        c1.metric("CAGR",    str(round(bt1["cagr"]*100,1))+"%")
        c2.metric("Sharpe",  str(round(bt1["sharpe"],2)))
        c3.metric("Max DD",  str(round(bt1["max_dd"]*100,1))+"%")
        c1b,c2b,c3b = st.columns(3)
        c1b.metric("Win rate",     str(round(bt1["win_rate"]*100,1))+"%")
        c2b.metric("Profit factor",str(round(bt1["profit_factor"],2)))
        c3b.metric("Obchodu",      str(bt1["n_trades"]))
    with t2:
        st.markdown("##### " + str(leverage) + "x (s pakou)")
        c1,c2,c3 = st.columns(3)
        c1.metric("CAGR",    str(round(btL["cagr"]*100,1))+"%",
                  delta=str(round((btL["cagr"]-bt1["cagr"])*100,1))+"%")
        c2.metric("Sharpe",  str(round(btL["sharpe"],2)))
        c3.metric("Max DD",  str(round(btL["max_dd"]*100,1))+"%",
                  delta=str(round((btL["max_dd"]-bt1["max_dd"])*100,1))+"%",
                  delta_color="inverse")
        c1b,c2b,c3b = st.columns(3)
        c1b.metric("Win rate",     str(round(btL["win_rate"]*100,1))+"%")
        c2b.metric("Profit factor",str(round(btL["profit_factor"],2)))
        c3b.metric("Final equity",  str(round(btL["final_eq"],3))+"x")

    # Equity curve
    all_rets = fr.iloc[:N2].fillna(0).values
    all_bt1  = run_backtest(probs, all_rets, uth_, 1.0,      tc, margin_rate, sl_val, use_sizing)
    all_btL  = run_backtest(probs, all_rets, uth_, leverage,  tc, margin_rate, sl_val, use_sizing)
    bh_norm  = feat["close"].values[:len(all_bt1["equity"])] / feat["close"].values[0]
    dates_eq = feat["date"].iloc[:len(all_bt1["equity"])].values

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=dates_eq, y=all_btL["equity"],
                                 mode="lines", name=str(leverage)+"x",
                                 line=dict(color="#ffd700", width=2)))
    fig_eq.add_trace(go.Scatter(x=dates_eq, y=all_bt1["equity"],
                                 mode="lines", name="AI 1x",
                                 line=dict(color="#00c7b7", width=2)))
    fig_eq.add_trace(go.Scatter(x=dates_eq, y=bh_norm,
                                 mode="lines", name="Buy&Hold",
                                 line=dict(color="#888", dash="dot", width=1)))
    fig_eq.add_vrect(
        x0=str(feat["date"].iloc[test_idx].date()),
        x1=str(feat["date"].iloc[-1].date()),
        fillcolor="rgba(255,255,100,0.06)", line_width=0,
        annotation_text="Test perioda", annotation_position="top left",
    )
    fig_eq.update_layout(
        title="Equity curve",
        height=380,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font_color="#fff", margin=dict(t=45),
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # Mesicni heatmapa
    st.subheader("Mesicni P&L (%)")
    m_pnl = monthly_pnl(
        feat["date"].iloc[:len(all_bt1["rets"])].values,
        all_bt1["rets"]
    )
    fig_hm = px.imshow(m_pnl,
                        color_continuous_scale=["#ff4b4b","#0e1117","#00c7b7"],
                        color_continuous_midpoint=0, text_auto=True, aspect="auto")
    fig_hm.update_layout(height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          font_color="#fff", margin=dict(t=10))
    st.plotly_chart(fig_hm, use_container_width=True)

    # Export
    te_dates = feat["date"].iloc[test_idx:test_idx+len(te_probs)].values
    pos_exp  = np.where(te_probs >= uth_, 1, np.where(te_probs <= 1-uth_, -1, 0))
    exp_df   = pd.DataFrame({
        "date":    te_dates,
        "prob_up": te_probs,
        "signal":  np.where(pos_exp==1,"LONG",np.where(pos_exp==-1,"SHORT","FLAT")),
        "ret_1x":  bt1["rets"],
        "ret_lev": btL["rets"],
    })
    st.download_button("Stahnout backtest CSV",
                        exp_df.to_csv(index=False).encode("utf-8"),
                        "backtest_" + ticker + ".csv", "text/csv")

# ────────────────────────────────────────────────────────────────
#  PAGE 3 – TRH & SEKTOR
# ────────────────────────────────────────────────────────────────
elif PAGE == "Trh & Sektor":
    sector     = info.get("sector","") if info else ""
    sector_etf = SECTOR_ETFS.get(sector, "XLK")
    st.subheader("Relativni sila – " + ticker + " vs SPY vs " + sector_etf)

    with st.spinner("Nacitam SPY a " + sector_etf + "..."):
        df_spy, _ = load_daily("SPY",      period_train)
        df_sec, _ = load_daily(sector_etf, period_train)

    if not df_spy.empty and not df_sec.empty:
        def norm_series(dff):
            c_ = dff["Close"].dropna()
            return c_ / c_.iloc[0]

        combined = pd.concat([
            norm_series(df_daily).rename(ticker),
            norm_series(df_spy).rename("SPY"),
            norm_series(df_sec).rename(sector_etf),
        ], axis=1).ffill().dropna()

        fig_r = go.Figure()
        colors_r = ["#00c7b7","#888888","#ffd700"]
        for j, col in enumerate(combined.columns):
            fig_r.add_trace(go.Scatter(x=combined.index, y=combined[col],
                                        mode="lines", name=col,
                                        line=dict(color=colors_r[j], width=2)))
        fig_r.update_layout(title="Normalizovana cena (start=1.0)",
                             height=360, paper_bgcolor="#0e1117",
                             plot_bgcolor="#0e1117", font_color="#fff",
                             margin=dict(t=45))
        st.plotly_chart(fig_r, use_container_width=True)

        rs_line = combined[ticker] / combined["SPY"]
        fig_rs  = go.Figure()
        fig_rs.add_trace(go.Scatter(x=rs_line.index, y=rs_line,
                                     mode="lines", name="Relat. sila",
                                     line=dict(color="#a78bfa", width=1.5)))
        fig_rs.add_hline(y=1.0, line_dash="dash", line_color="#888")
        fig_rs.update_layout(title="Relativni sila " + ticker + " / SPY",
                              height=240, paper_bgcolor="#0e1117",
                              plot_bgcolor="#0e1117", font_color="#fff",
                              margin=dict(t=45))
        st.plotly_chart(fig_rs, use_container_width=True)

        st.subheader("Vykonnost za obdobi")
        periods_ = {"1T":5,"1M":21,"3M":63,"6M":126,"1Y":252,"2Y":504}
        rows_p   = []
        for pname, pdays in periods_.items():
            row_ = {"Obdobi": pname}
            for sym, dff2 in [(ticker, df_daily),(sector_etf, df_sec),("SPY", df_spy)]:
                if len(dff2) >= pdays:
                    c_  = dff2["Close"].dropna()
                    ret = round(float(c_.iloc[-1] / c_.iloc[-pdays] - 1) * 100, 1)
                    row_[sym] = str(ret) + "%"
                else:
                    row_[sym] = "N/A"
            rows_p.append(row_)
        st.dataframe(pd.DataFrame(rows_p), use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────
#  PAGE 4 – SCREENER
# ────────────────────────────────────────────────────────────────
elif PAGE == "Screener":
    st.subheader("Multi-ticker Screener")
    default_t = "AAPL,MSFT,NVDA,TSLA,META,GOOGL,AMZN,AMD,PLTR,SPY,QQQ,GC=F,ETH-USD,BTC-USD,SOL-USD"
    t_input   = st.text_area("Tickery (oddelene carkou)", value=default_t, height=70)
    run_sc    = st.button("Spustit screener", type="primary")

    if run_sc:
        t_list = [t.strip().upper() for t in t_input.split(",") if t.strip()]
        rows_s = []
        prog   = st.progress(0)
        for idx_s, tk_s in enumerate(t_list):
            prog.progress((idx_s+1) / len(t_list), text="Analyzuji " + tk_s + "...")
            try:
                df_s, info_s = load_daily(tk_s, "1y")
                if df_s.empty or len(df_s) < 100:
                    continue
                feat_s, _ = featurize(df_s)
                if len(feat_s) < 60:
                    continue
                c_s    = feat_s["close"]
                rsi_s  = round(float(feat_s["rsi"].iloc[-1] * 100), 1)
                adx_s  = round(float(feat_s["adx"].iloc[-1] * 100), 1)
                ret1m  = round(float(c_s.iloc[-1]/c_s.iloc[-22]-1)*100,1) if len(c_s)>22 else 0
                ret3m  = round(float(c_s.iloc[-1]/c_s.iloc[-63]-1)*100,1) if len(c_s)>63 else 0
                vol_r  = round(float(feat_s["v20"].iloc[-1]*100*math.sqrt(252)), 1)
                if rsi_s < 30 and adx_s > 20:
                    sig_s = "🟢 Potencial. LONG"
                elif rsi_s > 70 and adx_s > 20:
                    sig_s = "🔴 Potencial. SHORT"
                elif adx_s < 15:
                    sig_s = "⚪ Sideways"
                else:
                    sig_s = "⚪ Neutral"
                rows_s.append({
                    "Ticker":    tk_s,
                    "Sektor":    info_s.get("sector","N/A") if info_s else "N/A",
                    "Trz. kap.": fmt_large(info_s.get("mktcap",0)) if info_s else "N/A",
                    "P/E":       str(round(info_s.get("pe",0),1)) if info_s and info_s.get("pe") else "N/A",
                    "1M ret.":   str(ret1m) + "%",
                    "3M ret.":   str(ret3m) + "%",
                    "RSI":       rsi_s,
                    "ADX":       adx_s,
                    "Volatilita":str(vol_r) + "%",
                    "Signal":    sig_s,
                })
            except Exception:
                continue
        prog.empty()
        if rows_s:
            st.dataframe(pd.DataFrame(rows_s), use_container_width=True, hide_index=True)
            st.download_button("Stahnout CSV",
                                pd.DataFrame(rows_s).to_csv(index=False).encode("utf-8"),
                                "screener.csv", "text/csv")
        else:
            st.warning("Zadna data nebyla nactena.")

# ── Auto-refresh (na konci skriptu) ───────────────────────────────
if live_mode:
    intervals_ = {"30s": 30, "1 min": 60, "5 min": 300}
    wait_       = intervals_.get(refresh_int, 60)
    time.sleep(wait_)
    st.rerun()

st.markdown("---")
st.caption("AI Trading Screener v5 | Pouze demonstrace – neni to financni poradenstvi.")
