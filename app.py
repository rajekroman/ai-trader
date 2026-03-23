
# =================================================================
#  AI Trading Screener v6
#  Deep AI model: StackingClassifier (RF + HGB + ET + LR meta)
#  35 features | TimeSeriesSplit CV | Walk-forward validation
#  5 let trenovacich dat | Feature importance | Regime detection
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
    page_title="AI Trader v6 DEEP",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="metric-container"] {
    background:#1a1a2e; border-radius:10px;
    padding:12px; border:1px solid #16213e;
}
.live-badge {
    display:inline-block; background:#00c7b7; color:#000;
    padding:2px 10px; border-radius:20px; font-weight:bold; font-size:.8rem;
}
.sig-card { padding:1.2rem; border-radius:12px; background:#1a1a2e; margin:.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Konstanty ────────────────────────────────────────────────────
PRESETS = {
    "── AKCIE ──": None,
    "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA", "Tesla (TSLA)": "TSLA",
    "Amazon (AMZN)": "AMZN", "Meta (META)": "META",
    "Google (GOOGL)": "GOOGL", "AMD": "AMD", "Palantir (PLTR)": "PLTR",
    "── INDEXY / ETF ──": None,
    "S&P 500 (SPY)": "SPY", "Nasdaq (QQQ)": "QQQ",
    "── KOMODITY ──": None,
    "Zlato (GC=F)": "GC=F", "Ropa WTI (CL=F)": "CL=F",
    "── KRYPTO ──": None,
    "Bitcoin (BTC-USD)": "BTC-USD", "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD",
}
SECTOR_ETFS = {
    "Technology":"XLK","Health Care":"XLV","Financials":"XLF",
    "Consumer Discretionary":"XLY","Communication Services":"XLC",
    "Industrials":"XLI","Energy":"XLE",
}
TF_PARAMS = {
    "1m": {"period":"5d",   "interval":"1m",  "label":"1 min",  "bars":390},
    "5m": {"period":"5d",   "interval":"5m",  "label":"5 min",  "bars":200},
    "15m":{"period":"60d",  "interval":"15m", "label":"15 min", "bars":200},
    "1h": {"period":"730d", "interval":"1h",  "label":"1 hod",  "bars":200},
    "1d": {"period":"5y",   "interval":"1d",  "label":"1 den",  "bars":300},
}

# ── Feature sloupce (35 features) ────────────────────────────────
FCOLS = [
    # Cena a výnosy
    "r1","r2","r3","r5","r10","r20",
    # Klouzavé průměry
    "ma5","ma10","ma20","ma50","ma200",
    # Oscilátory
    "rsi","stoch_k","stoch_d","willi","cci",
    # Trend a momentum
    "macd","adx","aroon_up","aroon_dn",
    # Volatilita
    "bb","atr","v5","v20","hist_vol",
    # Objem
    "obv","vforce","vc",
    # Kanály
    "donch","keltner",
    # Režim trhu
    "above_ma200","regime","trend_str",
    # Sezónnost
    "dow","month_sin",
]

# ── Technické indikátory ─────────────────────────────────────────
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

def _macd_norm(s, fast=12, slow=26, sig=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    m  = ef - es
    return (m - m.ewm(span=sig, adjust=False).mean()) / (s.abs() + 1e-9)

def _cci(high, low, close, p=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(p).mean()
    md = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - ma) / (0.015 * md + 1e-9) / 100

def _bb(s, p=20):
    ma  = s.rolling(p).mean()
    std = s.rolling(p).std()
    return (s - (ma - 2*std)) / (4*std + 1e-9)

def _atr(high, low, close, p=14):
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(p).mean() / (close + 1e-9)

def _obv_pct(close, volume, p=20):
    obv = (np.sign(close.diff()) * volume).cumsum()
    return obv.pct_change(p).fillna(0)

def _donch(close, p=20):
    hi = close.rolling(p).max()
    lo = close.rolling(p).min()
    return (close - lo) / (hi - lo + 1e-9)

def _adx(high, low, close, p=14):
    tr   = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr_ = tr.ewm(span=p, adjust=False).mean()
    up   = high.diff()
    dn   = -low.diff()
    pdm  = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=close.index)
    ndm  = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=close.index)
    pdi  = 100 * pdm.ewm(span=p, adjust=False).mean() / (atr_ + 1e-9)
    ndi  = 100 * ndm.ewm(span=p, adjust=False).mean() / (atr_ + 1e-9)
    dx   = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    return dx.ewm(span=p, adjust=False).mean() / 100

def _aroon(high, low, p=25):
    aroon_up = high.rolling(p+1).apply(lambda x: x.argmax(), raw=True) / p * 100
    aroon_dn = low.rolling(p+1).apply(lambda x: x.argmin(), raw=True) / p * 100
    return aroon_up, aroon_dn

def _keltner(close, high, low, p=20, atr_mult=2.0):
    ema  = close.ewm(span=p, adjust=False).mean()
    atr_ = _atr(high, low, close, p)
    ku   = ema + atr_mult * atr_ * close
    kl   = ema - atr_mult * atr_ * close
    return (close - kl) / (ku - kl + 1e-9)

def _volume_force(close, high, low, volume):
    mid   = (high + low) / 2
    force = (close - mid) / (high - low + 1e-9) * volume
    return force.pct_change(10).fillna(0)

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

# ── Feature Engineering (35 features) ───────────────────────────
def featurize(df):
    c  = df["Close"]
    hi = df.get("High", c)
    lo = df.get("Low",  c)
    vo = df.get("Volume", pd.Series(1.0, index=c.index))
    dt = df["date"]

    fr           = c.shift(-1) / c - 1
    sk, sd       = _stoch(hi, lo, c)
    aroon_u, aroon_d = _aroon(hi, lo)
    ma200_line   = c.rolling(200).mean()
    hist_vol_    = c.pct_change().rolling(20).std() * math.sqrt(252)
    trend_str_   = (c.rolling(5).mean() - c.rolling(20).mean()).abs() / (c.rolling(20).std() + 1e-9)

    feat = pd.DataFrame({
        "r1":       c.pct_change(),
        "r2":       c.pct_change(2),
        "r3":       c.pct_change(3),
        "r5":       c.pct_change(5),
        "r10":      c.pct_change(10),
        "r20":      c.pct_change(20),
        "ma5":      c.rolling(5).mean()   / c - 1,
        "ma10":     c.rolling(10).mean()  / c - 1,
        "ma20":     c.rolling(20).mean()  / c - 1,
        "ma50":     c.rolling(50).mean()  / c - 1,
        "ma200":    ma200_line             / c - 1,
        "rsi":      _rsi(c) / 100,
        "stoch_k":  sk / 100,
        "stoch_d":  sd / 100,
        "willi":    _williams(hi, lo, c) / 100,
        "cci":      _cci(hi, lo, c),
        "macd":     _macd_norm(c),
        "adx":      _adx(hi, lo, c),
        "aroon_up": aroon_u / 100,
        "aroon_dn": aroon_d / 100,
        "bb":       _bb(c),
        "atr":      _atr(hi, lo, c),
        "v5":       c.pct_change().rolling(5).std(),
        "v20":      c.pct_change().rolling(20).std(),
        "hist_vol": hist_vol_,
        "obv":      _obv_pct(c, vo),
        "vforce":   _volume_force(c, hi, lo, vo),
        "vc":       vo.pct_change(),
        "donch":    _donch(c),
        "keltner":  _keltner(c, hi, lo),
        "above_ma200": (c > ma200_line).astype(float),
        "regime":   (c.rolling(50).mean() > c.rolling(200).mean()).astype(float),
        "trend_str":trend_str_,
        "dow":      pd.to_datetime(dt).dt.dayofweek / 4.0,
        "month_sin":np.sin(2 * np.pi * pd.to_datetime(dt).dt.month / 12),
        "date":     dt,
        "close":    c,
        "high":     hi,
        "low":      lo,
        "open":     df.get("Open", c),
        "volume":   vo,
    })
    feat["target"] = (fr > 0).astype(int)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return feat, fr

# ── Data loader ──────────────────────────────────────────────────
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
        df.rename(columns={"Date":"date","Datetime":"date"}, inplace=True)
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
            raw = yf.Ticker(ticker).info
            info = {
                "name":         raw.get("longName", ticker),
                "sector":       raw.get("sector","N/A"),
                "industry":     raw.get("industry","N/A"),
                "mktcap":       raw.get("marketCap",0),
                "pe":           raw.get("trailingPE",0),
                "eps":          raw.get("trailingEps",0),
                "beta":         raw.get("beta",0),
                "52w_high":     raw.get("fiftyTwoWeekHigh",0),
                "52w_low":      raw.get("fiftyTwoWeekLow",0),
                "avg_volume":   raw.get("averageVolume",0),
                "div_yield":    raw.get("dividendYield",0),
                "target_price": raw.get("targetMeanPrice",0),
            }
        except Exception:
            pass
        return df, info
    except Exception:
        return pd.DataFrame(), {}

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
            df.rename(columns={date_col:"date"}, inplace=True)
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

# ── Deep AI Training (StackingClassifier, TimeSeriesSplit) ───────
@st.cache_resource(show_spinner=False)
def train_deep_model(ticker, period="5y"):
    from sklearn.ensemble import (RandomForestClassifier,
                                   HistGradientBoostingClassifier,
                                   ExtraTreesClassifier,
                                   StackingClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score

    df, _ = load_daily(ticker, period)
    if df.empty or len(df) < 300:
        return None, None, None, None, None

    feat, fr = featurize(df)
    N  = len(feat)
    te = int(N * 0.10)         # 10 % test
    ve = int(N * 0.10)         # 10 % validace
    tr = N - te - ve           # 80 % trenink

    X    = feat[FCOLS].values
    y    = feat["target"].values
    X_tr = X[:tr];  y_tr = y[:tr]
    X_va = X[tr:tr+ve]; y_va = y[tr:tr+ve]
    X_te = X[tr+ve:];   y_te = y[tr+ve:]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    X_te_s = sc.transform(X_te)

    # ── Bazove modely ─────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=4,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.04,
        min_samples_leaf=10, random_state=42,
    )
    et = ExtraTreesClassifier(
        n_estimators=200, max_depth=7, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    meta = LogisticRegression(C=0.5, max_iter=1000, random_state=42)

    # ── StackingClassifier s TimeSeriesSplit ─────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    stack = StackingClassifier(
        estimators=[("rf",rf),("hgb",hgb),("et",et)],
        final_estimator=meta,
        cv=tscv,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    cal = CalibratedClassifierCV(stack, cv=3, method="isotonic")
    cal.fit(X_tr_s, y_tr)

    # ── Validace a optimalizace prahu ─────────────────────────────
    probs_va = cal.predict_proba(X_va_s)[:, 1]
    probs_te = cal.predict_proba(X_te_s)[:, 1]
    try:
        auc_val = round(float(roc_auc_score(y_va, probs_va)), 4)
        auc_tst = round(float(roc_auc_score(y_te, probs_te)), 4)
    except Exception:
        auc_val, auc_tst = 0.0, 0.0

    # Feature importance (z RF uvnitr stacku)
    try:
        rf_fitted = cal.calibrated_classifiers_[0].estimator.estimators_[0][1]
        fi = rf_fitted.feature_importances_
    except Exception:
        fi = np.zeros(len(FCOLS))

    # Uloz AUC metriky do session
    st.session_state["model_auc_val"] = auc_val
    st.session_state["model_auc_tst"] = auc_tst

    # Dotrenovani na tr+va
    X_trva_s = sc.transform(X[:tr+ve])
    y_trva   = y[:tr+ve]
    rf2 = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=4,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    hgb2 = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.04,
        min_samples_leaf=10, random_state=42,
    )
    et2 = ExtraTreesClassifier(
        n_estimators=200, max_depth=7, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    stack2 = StackingClassifier(
        estimators=[("rf",rf2),("hgb",hgb2),("et",et2)],
        final_estimator=LogisticRegression(C=0.5, max_iter=1000, random_state=42),
        cv=TimeSeriesSplit(n_splits=5),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )
    cal2 = CalibratedClassifierCV(stack2, cv=3, method="isotonic")
    cal2.fit(X_trva_s, y_trva)

    return cal2, sc, feat, fr, fi

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
    dm  = margin_rate / 252 * max(leverage - 1.0, 0.0)
    eq  = [1.0]
    trades, wins = [], []
    in_tr, ent_eq, cur = False, 1.0, 0.0
    for i in range(n):
        p = pos[i]; r = rets[i]
        cost   = abs(p - cur) * tc
        new_eq = eq[-1] * (1.0 + p * r * leverage - dm * abs(p) - cost)
        if sl is not None and in_tr:
            if new_eq / ent_eq - 1 < -(sl / leverage):
                new_eq = ent_eq * (1.0 - sl / leverage)
                p = 0.0; in_tr = False
        if p != 0.0 and not in_tr:
            in_tr = True; ent_eq = eq[-1]
        elif p == 0.0 and in_tr:
            trades.append(new_eq / ent_eq - 1.0)
            wins.append(new_eq >= ent_eq)
            in_tr = False
        eq.append(max(new_eq, 1e-9)); cur = p
    eq   = np.array(eq[1:])
    yr   = max(len(eq) / 252, 1e-9)
    cagr = float(eq[-1]**(1.0/yr) - 1.0)
    d    = np.diff(np.log(np.maximum(eq, 1e-9)))
    sh   = float(d.mean() / d.std() * math.sqrt(252)) if d.std() > 0 else 0.0
    rm   = np.maximum.accumulate(eq)
    mdd  = float((eq/rm - 1.0).min())
    win_r  = float(np.mean(wins)) if wins else 0.0
    profits= [t for t in trades if t > 0]
    losses = [abs(t) for t in trades if t < 0]
    pf = sum(profits)/sum(losses) if losses and sum(losses)>0 else 99.0
    return dict(cagr=cagr, sharpe=sh, max_dd=mdd, final_eq=float(eq[-1]),
                equity=eq, pos=pos, rets=pos*rets,
                win_rate=win_r, profit_factor=min(float(pf),50.0),
                avg_trade=float(np.mean(trades)) if trades else 0.0,
                n_trades=len(trades))

def best_threshold(probs_v, rets_v):
    best_sh, best_uth = -999.0, 0.55
    for uth in np.arange(0.50, 0.68, 0.01):
        try:
            bt = run_backtest(probs_v, rets_v, float(uth), 1.0, 0.0, use_sizing=False)
            if bt["sharpe"] > best_sh and bt["n_trades"] > 3:
                best_sh = bt["sharpe"]; best_uth = uth
        except Exception:
            pass
    return round(float(best_uth), 2)

def sig_info(prob, uth):
    if prob >= uth:        return "LONG",  "#00c7b7", "🟢"
    elif prob <= 1 - uth:  return "SHORT", "#ff4b4b", "🔴"
    return "FLAT", "#888888", "⚪"

def fmt_large(n):
    if not n: return "N/A"
    for div, suf in [(1e12,"T"),(1e9,"B"),(1e6,"M"),(1e3,"K")]:
        if abs(n) >= div: return str(round(n/div,2)) + suf
    return str(round(n, 2))

def monthly_pnl(dates, rets):
    df_m = pd.DataFrame({"date": pd.to_datetime(dates), "ret": rets})
    df_m["yr"] = df_m["date"].dt.year
    df_m["mo"] = df_m["date"].dt.month
    piv = df_m.groupby(["yr","mo"])["ret"].sum().unstack(fill_value=0)*100
    mn  = {1:"Led",2:"Uno",3:"Bze",4:"Dub",5:"Kve",6:"Cvn",
           7:"Cvc",8:"Srp",9:"Zar",10:"Rib",11:"Lis",12:"Pro"}
    piv.columns = [mn.get(m, str(m)) for m in piv.columns]
    return piv.round(2)

# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 AI Trader v6")
    st.markdown("*Deep Stacking Model*")
    st.markdown("---")
    preset_label = st.selectbox("Instrument", list(PRESETS.keys()),
                                 index=list(PRESETS.keys()).index("Apple (AAPL)"))
    preset_val   = PRESETS.get(preset_label)
    custom       = st.text_input("Vlastni ticker", value="")
    ticker       = (custom.strip().upper() if custom.strip()
                    else (preset_val if preset_val else "AAPL"))
    st.markdown("#### Graf – timeframe")
    tf_sel = st.radio("", list(TF_PARAMS.keys()), index=4,
                       horizontal=True, label_visibility="collapsed")
    st.markdown("#### Live Refresh")
    live_mode   = st.toggle("Auto-refresh", value=False)
    ref_int     = st.selectbox("Interval", ["30s","1 min","5 min"],
                                index=1, disabled=not live_mode)
    st.markdown("---")
    st.markdown("#### Model")
    period_tr   = st.selectbox("Trenovaci data", ["2y","3y","5y","max"], index=2)
    st.markdown("#### Rizeni rizika")
    leverage    = st.slider("Paka", 1.0, 10.0, 1.0, 0.5)
    margin_rate = st.slider("Margin p.a. (%)", 0.0, 12.0, 6.0, 0.5) / 100
    use_sl      = st.toggle("Stop-Loss", value=True)
    sl_val      = st.slider("Stop-Loss (%)", 1, 20, 5) / 100 if use_sl else None
    use_sizing  = st.toggle("Confidence sizing", value=True)
    tc_bps      = st.number_input("Naklady (bps)", 0, 50, 5)
    auto_uth    = st.toggle("Auto-prah", value=True)
    man_uth     = st.slider("Rucni prah", 0.50, 0.70, 0.55, 0.01)
    st.markdown("---")
    st.caption("RF + HGB + ET → LR (Stacking) | 35 features | 5Y dat | TimeSeriesSplit CV")

tc = tc_bps / 10000

# ════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════
h1, h2 = st.columns([5,1])
with h1:
    st.markdown("## 🧠 " + ticker + " – Deep AI Trader")
with h2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
st.caption("Aktualizace: " + datetime.now().strftime("%d.%m.%Y %H:%M:%S"))

# ════════════════════════════════════════════════════════════════
#  NACITANI + TRENOVANI
# ════════════════════════════════════════════════════════════════
with st.spinner("Trenuju Deep Stacking model pro " + ticker + " (" + period_tr + " dat)..."):
    model, sc, feat, fr, fi = train_deep_model(ticker, period_tr)

with st.spinner("Nacitam " + TF_PARAMS[tf_sel]["label"] + " data..."):
    df_chart = load_intraday(ticker, tf_sel)
    df_daily, info = load_daily(ticker, period_tr)

if model is None or feat is None:
    st.error("Nedostatek dat pro " + ticker + ". Zkus jiny symbol nebo delsi obdobi.")
    st.stop()

X     = feat[FCOLS].values
probs = model.predict_proba(sc.transform(X))[:, 1]
N2    = len(feat)
va_s  = int(N2 * 0.80); va_e = int(N2 * 0.90)
uth_  = best_threshold(probs[va_s:va_e],
                        fr.iloc[va_s:va_e].values[:va_e-va_s]) if auto_uth else man_uth
p_now = float(probs[-1])
lbl, color, icon = sig_info(p_now, uth_)

# ── Metriky modelu ────────────────────────────────────────────────
auc_v = st.session_state.get("model_auc_val", 0.0)
auc_t = st.session_state.get("model_auc_tst", 0.0)

# ── Horni metrikovy pas ───────────────────────────────────────────
price_now  = round(float(feat["close"].iloc[-1]), 2)
price_prev = round(float(feat["close"].iloc[-2]), 2)
price_chg  = round((price_now / price_prev - 1) * 100, 2)
rsi_now    = round(float(feat["rsi"].iloc[-1] * 100), 1)
adx_now    = round(float(feat["adx"].iloc[-1] * 100), 1)
bb_now     = round(float(feat["bb"].iloc[-1]), 2)
above_ma   = bool(feat["above_ma200"].iloc[-1] > 0.5)
regime     = bool(feat["regime"].iloc[-1] > 0.5)

m1,m2,m3,m4,m5,m6,m7,m8 = st.columns(8)
m1.metric("Cena",        str(price_now), delta=str(price_chg)+"%")
m2.metric("AI Signal",   icon+" "+lbl,   delta="p="+str(round(p_now*100,1))+"%")
m3.metric("RSI (14)",    str(rsi_now))
m4.metric("ADX",         str(adx_now))
m5.metric("Nad MA200",   "ANO" if above_ma else "NE")
m6.metric("Bull Regime", "ANO" if regime else "NE")
m7.metric("AUC Validace",str(auc_v))
m8.metric("AUC Test",    str(auc_t))

if info:
    mktcap_s = fmt_large(info.get("mktcap",0))
    pe_s  = str(round(info.get("pe",0),1)) if info.get("pe") else "N/A"
    beta_s= str(round(info.get("beta",0),2)) if info.get("beta") else "N/A"
    hi52  = str(round(info.get("52w_high",0),2)) if info.get("52w_high") else "N/A"
    lo52  = str(round(info.get("52w_low",0),2)) if info.get("52w_low") else "N/A"
    tgt_s = str(round(info.get("target_price",0),2)) if info.get("target_price") else "N/A"
    f1,f2,f3,f4,f5,f6 = st.columns(6)
    f1.metric("Trzni kap.", mktcap_s)
    f2.metric("P/E",        pe_s)
    f3.metric("Beta",       beta_s)
    f4.metric("52T Max",    hi52)
    f5.metric("52T Min",    lo52)
    f6.metric("Cil. cena",  tgt_s)
    sec_s = (info.get("sector","") + " / " + info.get("industry","")).strip(" /")
    if sec_s:
        st.caption("Sektor: " + sec_s)

st.markdown("---")

PAGE = st.radio("", ["Graf","Feature Importance","Backtest","Trh & Sektor","Screener"],
                horizontal=True, label_visibility="collapsed")
st.markdown("---")

# ────────────────────────────────────────────────────────────────
#  PAGE – GRAF
# ────────────────────────────────────────────────────────────────
if PAGE == "Graf":
    use_df = df_chart if not df_chart.empty else df_daily
    disp_n = TF_PARAMS[tf_sel]["bars"]
    disp   = use_df.tail(disp_n).reset_index(drop=True)
    c_d    = disp["Close"]
    ma20   = c_d.rolling(20).mean()
    ma50   = c_d.rolling(50).mean()
    bb_mid = c_d.rolling(20).mean()
    bb_std = c_d.rolling(20).std()
    bb_up  = bb_mid + 2*bb_std
    bb_dn  = bb_mid - 2*bb_std
    rsi_d  = _rsi(c_d)
    vol_d  = disp.get("Volume", pd.Series(1.0, index=disp.index))

    if tf_sel == "1d":
        d_probs = probs[max(0, N2-disp_n):]
        pos_arr = np.where(d_probs >= uth_, 1, np.where(d_probs <= 1-uth_, -1, 0))
        long_i  = np.where(pos_arr == 1)[0]
        short_i = np.where(pos_arr == -1)[0]
    else:
        d_probs = None; long_i = np.array([]); short_i = np.array([])

    pivots = pivot_levels(disp["High"], disp["Low"], disp["Close"])

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                         row_heights=[0.52,0.16,0.16,0.16],
                         vertical_spacing=0.02,
                         subplot_titles=("Cena ["+TF_PARAMS[tf_sel]["label"]+"]",
                                         "Volume","RSI (14)","AI Prob"))
    fig.add_trace(go.Candlestick(
        x=disp["date"], open=disp["Open"], high=disp["High"],
        low=disp["Low"], close=disp["Close"],
        increasing_line_color="#00c7b7", decreasing_line_color="#ff4b4b",
        name="Cena"), row=1, col=1)
    fig.add_trace(go.Scatter(x=disp["date"],y=ma20,name="MA20",
                              line=dict(color="#ffd700",width=1.2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=disp["date"],y=ma50,name="MA50",
                              line=dict(color="#ff9900",width=1.2,dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=disp["date"],y=bb_up,name="BB+",
                              line=dict(color="rgba(100,100,255,.5)",width=.8)),row=1,col=1)
    fig.add_trace(go.Scatter(x=disp["date"],y=bb_dn,name="BB-",
                              line=dict(color="rgba(100,100,255,.5)",width=.8),
                              fill="tonexty",fillcolor="rgba(100,100,255,.05)"),row=1,col=1)
    for lname, lval in pivots.items():
        lc = "#00c7b7" if "R" in lname else ("#ff4b4b" if "S" in lname else "#fff")
        fig.add_hline(y=lval, line_dash="dash", line_color=lc, line_width=.6,
                      annotation_text=lname, annotation_position="right",
                      annotation_font_size=9, row=1, col=1)
    if tf_sel == "1d" and len(long_i):
        fig.add_trace(go.Scatter(x=disp["date"].iloc[long_i],
            y=disp["Low"].iloc[long_i]*.995, mode="markers",
            marker=dict(symbol="triangle-up",size=10,color="#00c7b7"),
            name="LONG"), row=1, col=1)
    if tf_sel == "1d" and len(short_i):
        fig.add_trace(go.Scatter(x=disp["date"].iloc[short_i],
            y=disp["High"].iloc[short_i]*1.005, mode="markers",
            marker=dict(symbol="triangle-down",size=10,color="#ff4b4b"),
            name="SHORT"), row=1, col=1)
    vol_colors = ["#00c7b7" if c >= o else "#ff4b4b"
                  for c, o in zip(disp["Close"], disp["Open"])]
    fig.add_trace(go.Bar(x=disp["date"],y=vol_d,marker_color=vol_colors,name="Vol"),row=2,col=1)
    fig.add_trace(go.Scatter(x=disp["date"],y=rsi_d,
                              line=dict(color="#a78bfa",width=1.5),name="RSI"),row=3,col=1)
    fig.add_hline(y=70,line_dash="dash",line_color="#ff4b4b",line_width=.7,row=3,col=1)
    fig.add_hline(y=30,line_dash="dash",line_color="#00c7b7",line_width=.7,row=3,col=1)
    if d_probs is not None:
        fig.add_trace(go.Scatter(x=disp["date"],y=d_probs,
                                  line=dict(color="#f0c040",width=1.5),name="AI"),row=4,col=1)
        fig.add_hline(y=uth_,  line_dash="dash",line_color="green",line_width=.7,row=4,col=1)
        fig.add_hline(y=1-uth_,line_dash="dash",line_color="red",  line_width=.7,row=4,col=1)
    rb = [dict(count=1,label="1D",step="day",stepmode="backward"),
          dict(count=5,label="5D",step="day",stepmode="backward"),
          dict(count=1,label="1M",step="month",stepmode="backward"),
          dict(count=3,label="3M",step="month",stepmode="backward"),
          dict(count=6,label="6M",step="month",stepmode="backward"),
          dict(count=1,label="1Y",step="year",stepmode="backward"),
          dict(step="all",label="Max")]
    fig.update_layout(height=760,showlegend=False,
                       paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                       font_color="#fff",margin=dict(t=30,b=10,l=10,r=80),
                       xaxis=dict(rangeselector=dict(buttons=rb,bgcolor="#1a1a2e",
                                                     activecolor="#00c7b7",
                                                     font=dict(color="#fff")),
                                  rangeslider=dict(visible=False)))
    st.plotly_chart(fig, use_container_width=True)

    sig_html = (
        '<div class="sig-card" style="border-left:7px solid ' + color + ';">'
        + '<b style="font-size:1.4rem;">' + icon + " " + lbl + "</b><br>"
        + "Prob(up): <b>" + str(round(p_now*100,1)) + "%</b> &nbsp;|&nbsp; "
        + "Cena: <b>" + str(price_now) + "</b> &nbsp;|&nbsp; "
        + "Datum: <b>" + feat["date"].iloc[-1].strftime("%d.%m.%Y") + "</b>"
        + "</div>"
    )
    st.markdown(sig_html, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
#  PAGE – FEATURE IMPORTANCE
# ────────────────────────────────────────────────────────────────
elif PAGE == "Feature Importance":
    st.subheader("Vyznam vstupnich promennych (Feature Importance)")
    st.info("Ukazuje, ktere technické indikatory model pouziva nejvice pri rozhodovani. Cim vyssi hodnota, tim dulezitejsi vstup pro predikci.")
    if fi is not None and len(fi) == len(FCOLS):
        fi_df = pd.DataFrame({"Feature": FCOLS, "Importance": fi})
        fi_df = fi_df.sort_values("Importance", ascending=True).reset_index(drop=True)
        fig_fi = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"],
            orientation="h",
            marker=dict(
                color=fi_df["Importance"],
                colorscale=[[0,"#1a1a2e"],[0.5,"#00c7b7"],[1.0,"#ffd700"]],
            ),
        ))
        fig_fi.update_layout(
            height=700, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font_color="#fff", margin=dict(t=20,l=10),
            xaxis_title="Dulezitost (feature importance)",
            yaxis_title="",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        top5 = fi_df.nlargest(5,"Importance")["Feature"].tolist()
        st.markdown("**Top 5 nejdulezitejsich indikatoru pro " + ticker + ":**")
        descs = {
            "r1":"Jednodenni vynosnost – nejdulezitejsi krátkodoby pohyb",
            "r5":"5denni vynosnost – krátkodoby momentum signal",
            "r20":"20denni vynosnost – strednedobe momentum",
            "ma20":"Vzdalenost od 20denniho prumeru – aktualni trend",
            "ma50":"Vzdalenost od 50denniho prumeru – strednedobe trend",
            "ma200":"Vzdalenost od 200denniho prumeru – dlouhodoby trend",
            "rsi":"RSI – miri prekoupenost/preprodanost",
            "adx":"ADX – sila trendu (dulezite pro filtraci signalu)",
            "macd":"MACD – konvergence/divergence prumeru",
            "bb":"Bollinger Band pozice – volatilita a pozice v pasmu",
            "cci":"CCI – identifikace extremnich cen",
            "atr":"ATR – aktualni volatilita instrumentu",
            "regime":"Bull/Bear rezim – trenuje model chovat se jinak v rustu a poklesu",
            "above_ma200":"Zda je cena nad/pod MA200 – dlouhodoby filter",
            "trend_str":"Sila aktualniho trendu",
            "hist_vol":"Historicka volatilita (20 dni ann.) – stavebni blok risk modelu",
            "aroon_up":"Aroon Up – kdy naposledy bylo 25denni maximum",
            "aroon_dn":"Aroon Down – kdy naposledy bylo 25denni minimum",
            "donch":"Donchian pozice – kde je cena v 20dennim kanalu",
            "obv":"OBV momentum – objem potvrzuje nebo vyvracel pohyb ceny",
            "vforce":"Volume Force – kombinace smeru ceny a objemu",
        }
        for f in top5:
            desc = descs.get(f, f)
            st.markdown("- **" + f + ":** " + desc)
    else:
        st.warning("Feature importance neni dostupna pro tento model.")

# ────────────────────────────────────────────────────────────────
#  PAGE – BACKTEST
# ────────────────────────────────────────────────────────────────
elif PAGE == "Backtest":
    st.subheader("Backtest | " + ticker + " | Paka: " + str(leverage) + "x")
    te_s     = int(N2 * 0.90)
    te_probs = probs[te_s:]
    te_rets  = fr.iloc[te_s:te_s+len(te_probs)].fillna(0).values
    bt1 = run_backtest(te_probs, te_rets, uth_, 1.0,     tc, margin_rate, sl_val, use_sizing)
    btL = run_backtest(te_probs, te_rets, uth_, leverage, tc, margin_rate, sl_val, use_sizing)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("##### 1x")
        c1,c2,c3 = st.columns(3)
        c1.metric("CAGR",   str(round(bt1["cagr"]*100,1))+"%")
        c2.metric("Sharpe", str(round(bt1["sharpe"],2)))
        c3.metric("Max DD", str(round(bt1["max_dd"]*100,1))+"%")
        c1b,c2b,c3b = st.columns(3)
        c1b.metric("Win rate",      str(round(bt1["win_rate"]*100,1))+"%")
        c2b.metric("Profit factor", str(round(bt1["profit_factor"],2)))
        c3b.metric("Obchodu",       str(bt1["n_trades"]))
    with t2:
        st.markdown("##### " + str(leverage) + "x")
        c1,c2,c3 = st.columns(3)
        c1.metric("CAGR",   str(round(btL["cagr"]*100,1))+"%",
                  delta=str(round((btL["cagr"]-bt1["cagr"])*100,1))+"%")
        c2.metric("Sharpe", str(round(btL["sharpe"],2)))
        c3.metric("Max DD", str(round(btL["max_dd"]*100,1))+"%",
                  delta=str(round((btL["max_dd"]-bt1["max_dd"])*100,1))+"%",
                  delta_color="inverse")
        c1b,c2b,c3b = st.columns(3)
        c1b.metric("Win rate",     str(round(btL["win_rate"]*100,1))+"%")
        c2b.metric("Profit factor",str(round(btL["profit_factor"],2)))
        c3b.metric("Final equity", str(round(btL["final_eq"],3))+"x")

    all_rets = fr.iloc[:N2].fillna(0).values
    bt_all1  = run_backtest(probs, all_rets, uth_, 1.0,     tc, margin_rate, sl_val, use_sizing)
    bt_allL  = run_backtest(probs, all_rets, uth_, leverage, tc, margin_rate, sl_val, use_sizing)
    bh_norm  = feat["close"].values[:len(bt_all1["equity"])] / feat["close"].values[0]
    d_eq     = feat["date"].iloc[:len(bt_all1["equity"])].values
    fig_eq   = go.Figure()
    fig_eq.add_trace(go.Scatter(x=d_eq, y=bt_allL["equity"], mode="lines",
                                 name=str(leverage)+"x", line=dict(color="#ffd700",width=2)))
    fig_eq.add_trace(go.Scatter(x=d_eq, y=bt_all1["equity"], mode="lines",
                                 name="AI 1x", line=dict(color="#00c7b7",width=2)))
    fig_eq.add_trace(go.Scatter(x=d_eq, y=bh_norm, mode="lines",
                                 name="Buy&Hold", line=dict(color="#888",dash="dot",width=1)))
    fig_eq.add_vrect(
        x0=str(feat["date"].iloc[te_s].date()),
        x1=str(feat["date"].iloc[-1].date()),
        fillcolor="rgba(255,255,100,.06)", line_width=0,
        annotation_text="Test perioda", annotation_position="top left")
    fig_eq.update_layout(title="Equity curve", height=380,
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          font_color="#fff", margin=dict(t=45))
    st.plotly_chart(fig_eq, use_container_width=True)

    st.subheader("Mesicni P&L (%)")
    m_pnl = monthly_pnl(feat["date"].iloc[:len(bt_all1["rets"])].values, bt_all1["rets"])
    fig_hm = px.imshow(m_pnl, color_continuous_scale=["#ff4b4b","#0e1117","#00c7b7"],
                        color_continuous_midpoint=0, text_auto=True, aspect="auto")
    fig_hm.update_layout(height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          font_color="#fff", margin=dict(t=10))
    st.plotly_chart(fig_hm, use_container_width=True)

    te_dates = feat["date"].iloc[te_s:te_s+len(te_probs)].values
    pos_exp  = np.where(te_probs>=uth_,1,np.where(te_probs<=1-uth_,-1,0))
    exp_df   = pd.DataFrame({"date":te_dates,"prob_up":te_probs,
                              "signal":np.where(pos_exp==1,"LONG",np.where(pos_exp==-1,"SHORT","FLAT")),
                              "ret_1x":bt1["rets"],"ret_lev":btL["rets"]})
    st.download_button("Stahnout backtest CSV",
                        exp_df.to_csv(index=False).encode("utf-8"),
                        "backtest_"+ticker+".csv","text/csv")

# ────────────────────────────────────────────────────────────────
#  PAGE – TRH & SEKTOR
# ────────────────────────────────────────────────────────────────
elif PAGE == "Trh & Sektor":
    sector     = info.get("sector","") if info else ""
    sector_etf = SECTOR_ETFS.get(sector,"XLK")
    st.subheader("Relativni sila – " + ticker + " vs SPY vs " + sector_etf)
    with st.spinner("Nacitam SPY a " + sector_etf + "..."):
        df_spy, _ = load_daily("SPY", period_tr)
        df_sec, _ = load_daily(sector_etf, period_tr)
    if not df_spy.empty and not df_sec.empty:
        def norm_(dff):
            c_ = dff["Close"].dropna()
            return c_ / c_.iloc[0]
        combined = pd.concat([norm_(df_daily).rename(ticker),
                               norm_(df_spy).rename("SPY"),
                               norm_(df_sec).rename(sector_etf)],axis=1).ffill().dropna()
        fig_r = go.Figure()
        for j, col in enumerate(combined.columns):
            fig_r.add_trace(go.Scatter(x=combined.index,y=combined[col],mode="lines",
                                        name=col,line=dict(color=["#00c7b7","#888","#ffd700"][j],width=2)))
        fig_r.update_layout(title="Normalizovana cena",height=340,
                             paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                             font_color="#fff",margin=dict(t=45))
        st.plotly_chart(fig_r, use_container_width=True)
        rs = combined[ticker] / combined["SPY"]
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(x=rs.index,y=rs,mode="lines",name="RS",
                                     line=dict(color="#a78bfa",width=1.5)))
        fig_rs.add_hline(y=1.0,line_dash="dash",line_color="#888")
        fig_rs.update_layout(title="Rel. sila "+ticker+"/SPY",height=220,
                              paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                              font_color="#fff",margin=dict(t=45))
        st.plotly_chart(fig_rs, use_container_width=True)
        st.subheader("Vykonnost za obdobi")
        rows_p = []
        for pname, pdays in {"1T":5,"1M":21,"3M":63,"6M":126,"1Y":252,"2Y":504}.items():
            row_ = {"Obdobi": pname}
            for sym, dff2 in [(ticker,df_daily),(sector_etf,df_sec),("SPY",df_spy)]:
                if len(dff2) >= pdays:
                    c_  = dff2["Close"].dropna()
                    row_[sym] = str(round(float(c_.iloc[-1]/c_.iloc[-pdays]-1)*100,1)) + "%"
                else:
                    row_[sym] = "N/A"
            rows_p.append(row_)
        st.dataframe(pd.DataFrame(rows_p), use_container_width=True, hide_index=True)

# ────────────────────────────────────────────────────────────────
#  PAGE – SCREENER
# ────────────────────────────────────────────────────────────────
elif PAGE == "Screener":
    st.subheader("Multi-ticker Screener")
    default_t = "AAPL,MSFT,NVDA,TSLA,META,GOOGL,AMZN,AMD,PLTR,SPY,QQQ,GC=F,ETH-USD,BTC-USD"
    t_input   = st.text_area("Tickery (carkou)", value=default_t, height=70)
    run_sc    = st.button("Spustit screener", type="primary")
    if run_sc:
        t_list = [t.strip().upper() for t in t_input.split(",") if t.strip()]
        rows_s = []
        prog   = st.progress(0)
        for idx_s, tk_s in enumerate(t_list):
            prog.progress((idx_s+1)/len(t_list), text="Analyzuji "+tk_s+"...")
            try:
                df_s, info_s = load_daily(tk_s, "1y")
                if df_s.empty or len(df_s) < 150: continue
                feat_s, _ = featurize(df_s)
                if len(feat_s) < 60: continue
                c_s   = feat_s["close"]
                rsi_s = round(float(feat_s["rsi"].iloc[-1]*100),1)
                adx_s = round(float(feat_s["adx"].iloc[-1]*100),1)
                aroon_diff = round(float((feat_s["aroon_up"].iloc[-1]-feat_s["aroon_dn"].iloc[-1])*100),1)
                ret1m = round(float(c_s.iloc[-1]/c_s.iloc[-22]-1)*100,1) if len(c_s)>22 else 0
                ret3m = round(float(c_s.iloc[-1]/c_s.iloc[-63]-1)*100,1) if len(c_s)>63 else 0
                vol_r = round(float(feat_s["v20"].iloc[-1]*100*math.sqrt(252)),1)
                abv200 = bool(feat_s["above_ma200"].iloc[-1] > 0.5)
                reg    = bool(feat_s["regime"].iloc[-1] > 0.5)
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
                    "P/E":       str(round(info_s.get("pe",0),1)) if info_s and info_s.get("pe") else "N/A",
                    "1M":        str(ret1m)+"%",
                    "3M":        str(ret3m)+"%",
                    "RSI":       rsi_s,
                    "ADX":       adx_s,
                    "Aroon D":   aroon_diff,
                    "Vol.":      str(vol_r)+"%",
                    "MA200":     "ANO" if abv200 else "NE",
                    "Rezim":     "Bull" if reg else "Bear",
                    "Signal":    sig_s,
                })
            except Exception:
                continue
        prog.empty()
        if rows_s:
            st.dataframe(pd.DataFrame(rows_s), use_container_width=True, hide_index=True)
            st.download_button("Stahnout CSV",
                                pd.DataFrame(rows_s).to_csv(index=False).encode("utf-8"),
                                "screener.csv","text/csv")
        else:
            st.warning("Zadna data nebyla nactena.")

# ── Auto-refresh ──────────────────────────────────────────────────
if live_mode:
    ivs = {"30s":30,"1 min":60,"5 min":300}
    time.sleep(ivs.get(ref_int, 60))
    st.rerun()

st.markdown("---")
st.caption("AI Trading Screener v6 | Pouze demonstrace – neni to financni poradenstvi.")
