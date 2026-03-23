
# ============================================================
#  app.py  –  AI Trading Screener v3
#  Model:  VotingClassifier (HistGB + RandomForest)
#  Data:   yfinance (ziva data)
#  Featury: 20 technickych indikatoru
# ============================================================
import os, math, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="AI Trading Screener v3",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# ── Konstanty ─────────────────────────────────────────────────
INSTRUMENTS = {
    "SPY":    {"label":"S&P 500 ETF", "emoji":"📊","category":"Akcie",    "yf":"SPY",     "period":"5y"},
    "GOLD":   {"label":"Zlato",       "emoji":"🥇","category":"Komodita", "yf":"GC=F",    "period":"5y"},
    "SILVER": {"label":"Stribro",     "emoji":"🥈","category":"Komodita", "yf":"SI=F",    "period":"5y"},
    "ETH":    {"label":"Ethereum",    "emoji":"🔷","category":"Krypto",   "yf":"ETH-USD", "period":"4y"},
    "SOL":    {"label":"Solana",      "emoji":"🟣","category":"Krypto",   "yf":"SOL-USD", "period":"3y"},
}

# ── Technické indikátory ──────────────────────────────────────
def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))

def _stoch(high, low, close, k=14, d=3):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    stoch_k = 100 * (close - lo) / (hi - lo + 1e-9)
    stoch_d = stoch_k.rolling(d).mean()
    return stoch_k, stoch_d

def _williams_r(high, low, close, p=14):
    hi = high.rolling(p).max()
    lo = low.rolling(p).min()
    return -100 * (hi - close) / (hi - lo + 1e-9)

def _macd(s, fast=12, slow=26, sig=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    m  = ef - es
    return (m - m.ewm(span=sig, adjust=False).mean()) / (s + 1e-9)

def _bollinger(s, p=20):
    ma  = s.rolling(p).mean()
    std = s.rolling(p).std()
    return (s - (ma - 2*std)) / (4*std + 1e-9)

def _cci(high, low, close, p=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(p).mean()
    md = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - ma) / (0.015 * md + 1e-9) / 100

def _atr(high, low, close, p=14):
    tr = pd.concat([(high-low),
                    (high-close.shift()).abs(),
                    (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean() / (close + 1e-9)

def _obv_trend(close, volume, p=10):
    obv = (np.sign(close.diff()) * volume).cumsum()
    return obv.pct_change(p)

def _donchian(close, p=20):
    hi = close.rolling(p).max()
    lo = close.rolling(p).min()
    return (close - lo) / (hi - lo + 1e-9)

def _adx(high, low, close, p=14):
    tr   = pd.concat([(high-low),(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
    atr_ = tr.ewm(span=p, adjust=False).mean()
    up   = high.diff(); dn = -low.diff()
    pdm  = pd.Series(np.where((up > dn) & (up > 0), up, 0), index=close.index)
    ndm  = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), index=close.index)
    pdi  = 100 * pdm.ewm(span=p,adjust=False).mean() / (atr_ + 1e-9)
    ndi  = 100 * ndm.ewm(span=p,adjust=False).mean() / (atr_ + 1e-9)
    dx   = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-9)
    return dx.ewm(span=p, adjust=False).mean() / 100

FCOLS = [
    "r1","r5","r20",
    "ma5","ma20","ma50",
    "rsi","stoch_k","stoch_d","willi",
    "macd","bb","cci","atr",
    "obv","donch","adx",
    "v5","v20","vc",
]

def featurize(df):
    c  = df["Close"]
    hi = df["High"] if "High" in df.columns else c
    lo = df["Low"]  if "Low"  in df.columns else c
    vo = df["Volume"]
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
        "willi":  _williams_r(hi, lo, c) / 100,
        "macd":   _macd(c),
        "bb":     _bollinger(c),
        "cci":    _cci(hi, lo, c),
        "atr":    _atr(hi, lo, c),
        "obv":    _obv_trend(c, vo),
        "donch":  _donchian(c),
        "adx":    _adx(hi, lo, c),
        "v5":     c.pct_change().rolling(5).std(),
        "v20":    c.pct_change().rolling(20).std(),
        "vc":     vo.pct_change(),
        "date":   df["date"],
        "close":  c,
        "high":   hi,
        "low":    lo,
        "open":   df["Open"] if "Open" in df.columns else c,
    })
    feat["target"] = (fr > 0).astype(int)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return feat, fr


# ── Data ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, period="5y"):
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
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
        if "High" not in df.columns:
            df["High"] = df["Close"]
        if "Low" not in df.columns:
            df["Low"] = df["Close"]
        if "Open" not in df.columns:
            df["Open"] = df["Close"]
        return df
    except Exception:
        return pd.DataFrame()


# ── Model ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(ticker, period):
    from sklearn.ensemble import (RandomForestClassifier,
                                  HistGradientBoostingClassifier,
                                  VotingClassifier)
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    df = load_data(ticker, period)
    if df.empty or len(df) < 300:
        return None, None, None, None

    feat, fr = featurize(df)
    N  = len(feat)
    ve = int(N * 0.85)

    X, y = feat[FCOLS].values, feat["target"].values
    sc   = StandardScaler()
    Xtv  = sc.fit_transform(X[:ve])

    rf  = RandomForestClassifier(
        n_estimators=300, max_depth=7,
        min_samples_leaf=4, max_features="sqrt",
        random_state=42, n_jobs=-1,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=300, max_depth=5,
        learning_rate=0.05, min_samples_leaf=8,
        random_state=42,
    )
    vote = VotingClassifier(
        estimators=[("rf", rf), ("hgb", hgb)],
        voting="soft", n_jobs=-1,
    )
    cal = CalibratedClassifierCV(vote, cv=3, method="isotonic")
    cal.fit(Xtv, y[:ve])
    return cal, sc, feat, fr


# ── Backtest ─────────────────────────────────────────────────
def run_backtest(probs, rets, uth, tc=0.0005, sl=None, use_sizing=True):
    lth = 1 - uth
    pos = np.where(probs >= uth, 1.0, np.where(probs <= lth, -1.0, 0.0))

    if use_sizing:
        conf = np.abs(probs - 0.5) * 2
        pos  = pos * np.clip(conf, 0.3, 1.0)

    equity = [1.0]
    trades, wins = [], []
    in_trade, entry_eq, entry_day = False, 1.0, 0
    cur_pos = 0.0

    for i, (p, r) in enumerate(zip(pos, rets)):
        trade_ret = p * r - abs(p - cur_pos) * tc
        new_eq    = equity[-1] * (1 + trade_ret)

        if sl is not None and in_trade:
            dd = new_eq / entry_eq - 1
            if dd < -sl:
                new_eq  = entry_eq * (1 - sl)
                p       = 0.0
                in_trade = False

        if p != 0 and not in_trade:
            in_trade  = True
            entry_eq  = equity[-1]
            entry_day = i
        elif p == 0 and in_trade:
            trade_pnl = new_eq / entry_eq - 1
            trades.append(trade_pnl)
            wins.append(trade_pnl > 0)
            in_trade = False

        equity.append(max(new_eq, 1e-6))
        cur_pos = p

    equity  = np.array(equity[1:])
    yr      = len(equity) / 252
    cagr    = float(equity[-1]**(1/yr) - 1) if yr > 0 else 0.0
    sr_     = pos * rets - abs(np.diff(np.concatenate([[0], pos]))) * tc
    sh      = float(sr_.mean() / sr_.std() * math.sqrt(252)) if sr_.std() > 0 else 0.0
    rm      = np.maximum.accumulate(equity)
    mdd     = float((equity / rm - 1).min())
    win_r   = float(np.mean(wins)) if wins else 0.0
    profits = [t for t in trades if t > 0]
    losses  = [t for t in trades if t < 0]
    pf      = abs(sum(profits) / sum(losses)) if losses and sum(losses) != 0 else 999.0
    avg_t   = float(np.mean(trades)) if trades else 0.0

    return dict(
        cagr=cagr, sharpe=sh, max_dd=mdd, final_eq=float(equity[-1]),
        equity=equity, pos=pos, rets=sr_,
        win_rate=win_r, profit_factor=min(pf, 50.0),
        avg_trade=avg_t, n_trades=len(trades),
    )


def best_threshold(probs_val, rets_val):
    best_sh, best_uth = -999, 0.55
    for uth in np.arange(0.50, 0.68, 0.01):
        try:
            bt = run_backtest(probs_val, rets_val, uth, use_sizing=False)
            if bt["sharpe"] > best_sh and bt["n_trades"] > 5:
                best_sh  = bt["sharpe"]
                best_uth = uth
        except Exception:
            pass
    return round(best_uth, 2)


def signal_info(prob, uth):
    if prob >= uth:
        return "LONG",  "green",  "🟢"
    elif prob <= 1 - uth:
        return "SHORT", "red",    "🔴"
    return "FLAT",  "#888888","⚪"


def monthly_returns(dates, rets):
    df_m = pd.DataFrame({"date": pd.to_datetime(dates), "ret": rets})
    df_m["year"]  = df_m["date"].dt.year
    df_m["month"] = df_m["date"].dt.month
    pivot = df_m.groupby(["year","month"])["ret"].sum().unstack(fill_value=0)
    return pivot


# ═══════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════
MONTH_NAMES = ["Led","Uno","Bze","Dub","Kve","Cvn","Cvc","Srp","Zar","Rib","Lis","Pro"]

st.markdown("""
<style>
[data-testid="metric-container"] {background:#1e1e2e;border-radius:8px;padding:10px;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/combo-chart.png", width=40)
    st.title("AI Trader v3")
    st.markdown("---")
    selected  = st.multiselect("Instrumenty", list(INSTRUMENTS.keys()),
                                default=list(INSTRUMENTS.keys()))
    auto_uth  = st.toggle("Auto-optimalizace prahu", value=True)
    man_uth   = st.slider("Rucni prah (pokud vypnuta auto)", 0.50, 0.70, 0.55, 0.01)
    use_sl    = st.toggle("Stop-Loss (3 %)", value=True)
    use_sizing= st.toggle("Confidence sizing pozice", value=True)
    tc_bps    = st.number_input("Transakc. naklady (bps)", 0, 50, 5)
    st.markdown("---")
    st.caption("Model: VotingClassifier (RF + HistGB)\nFeatury: 20 indikatoru\nData: Yahoo Finance")
    st.caption("Pouze demonstrace.")

tc = tc_bps / 10000
sl = 0.03 if use_sl else None

# ── Stránky ───────────────────────────────────────────────────
PAGE = st.radio("", ["Dashboard", "Detail & Backtest", "Portfolio Simulace"],
                horizontal=True, label_visibility="collapsed")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════
#  PAGE 1 – DASHBOARD
# ═══════════════════════════════════════════════════════════════
if PAGE == "Dashboard":
    st.subheader("Aktualni AI signaly")
    if not selected:
        st.info("Vyber alespon jeden instrument v levem panelu.")
        st.stop()

    cols = st.columns(len(selected))
    summary_rows = []

    for i, name in enumerate(selected):
        meta = INSTRUMENTS[name]
        with cols[i]:
            with st.spinner(meta["emoji"] + " " + name):
                model, sc, feat, fr = train_model(meta["yf"], meta["period"])
            if model is None:
                st.warning(name + " – chyba dat")
                continue

            X     = feat[FCOLS].values
            probs = model.predict_proba(sc.transform(X))[:, 1]
            N2    = len(feat); ve2 = int(N2 * 0.80)

            val_p = probs[ve2:int(N2*0.90)]
            val_r = fr.iloc[ve2:int(N2*0.90)].values[:len(val_p)]
            uth_  = best_threshold(val_p, val_r) if auto_uth else man_uth

            p_now = float(probs[-1])
            lbl, color, icon = signal_info(p_now, uth_)

            st.metric(
                label=meta["emoji"] + " " + meta["label"],
                value=icon + " " + lbl,
                delta="p=" + str(round(p_now*100,1)) + "% | th=" + str(uth_),
            )
            st.caption(meta["category"] + " · " + str(feat["date"].iloc[-1].strftime("%d.%m.%Y")))

            # Mini sparkline
            eq_vals = (1 + fr.iloc[ve2:].fillna(0).values).cumprod()
            fig_s = go.Figure(go.Scatter(y=eq_vals, mode="lines",
                                          line=dict(color=color, width=1.5)))
            fig_s.update_layout(height=80, margin=dict(l=0,r=0,t=0,b=0),
                                 xaxis=dict(visible=False), yaxis=dict(visible=False),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_s, use_container_width=True)

            test_rets = fr.iloc[int(N2*0.90):].fillna(0).values
            test_prbs = probs[int(N2*0.90):][:len(test_rets)]
            if len(test_prbs) > 5:
                bt = run_backtest(test_prbs, test_rets, uth_, tc, sl, use_sizing)
                summary_rows.append({
                    "Instrument": meta["emoji"] + " " + name,
                    "Signal":     icon + " " + lbl,
                    "Prob":       str(round(p_now*100,1)) + "%",
                    "CAGR":       str(round(bt["cagr"]*100,1)) + "%",
                    "Sharpe":     str(round(bt["sharpe"],2)),
                    "Win rate":   str(round(bt["win_rate"]*100,1)) + "%",
                    "Max DD":     str(round(bt["max_dd"]*100,1)) + "%",
                    "Obchody":    str(bt["n_trades"]),
                })

    if summary_rows:
        st.markdown("---")
        st.subheader("Srovnani (testovaci perioda)")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 – DETAIL & BACKTEST
# ═══════════════════════════════════════════════════════════════
elif PAGE == "Detail & Backtest":
    if not selected:
        st.info("Vyber instrument.")
        st.stop()

    name = st.selectbox("Instrument", selected)
    meta = INSTRUMENTS[name]

    with st.spinner("Trenuju " + name + "..."):
        model, sc, feat, fr = train_model(meta["yf"], meta["period"])

    if model is None:
        st.error("Nepodarilo se nacist data.")
        st.stop()

    X     = feat[FCOLS].values
    probs = model.predict_proba(sc.transform(X))[:, 1]
    N2    = len(feat)
    ve2   = int(N2 * 0.80)

    val_p = probs[ve2:int(N2*0.90)]
    val_r = fr.iloc[ve2:int(N2*0.90)].values[:len(val_p)]
    uth_  = best_threshold(val_p, val_r) if auto_uth else man_uth
    st.caption("Optimalizovany prah: **" + str(uth_) + "**")

    test_idx   = int(N2 * 0.90)
    test_prbs  = probs[test_idx:]
    test_rets  = fr.iloc[test_idx:test_idx+len(test_prbs)].fillna(0).values
    bt = run_backtest(test_prbs, test_rets, uth_, tc, sl, use_sizing)

    # Metriky
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("CAGR",         str(round(bt["cagr"]*100,1)) + "%")
    m2.metric("Sharpe",       str(round(bt["sharpe"],2)))
    m3.metric("Max DD",       str(round(bt["max_dd"]*100,1)) + "%")
    m4.metric("Win rate",     str(round(bt["win_rate"]*100,1)) + "%")
    m5.metric("Profit factor",str(round(bt["profit_factor"],2)))
    m6.metric("Obchodu",      str(bt["n_trades"]))
    st.markdown("---")

    # ── Svíčkový graf + signály ───────────────────────────────
    disp = feat.tail(120).reset_index(drop=True)
    d_probs = probs[max(0, N2-120):]
    pos_arr = np.where(d_probs >= uth_, 1, np.where(d_probs <= 1-uth_, -1, 0))

    fig_c = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           row_heights=[0.55, 0.25, 0.20],
                           vertical_spacing=0.03)
    fig_c.add_trace(go.Candlestick(
        x=disp["date"], open=disp["open"], high=disp["high"],
        low=disp["low"], close=disp["close"],
        increasing_line_color="#00c7b7", decreasing_line_color="#ff4b4b",
        name="Cena"), row=1, col=1)

    long_idx  = np.where(pos_arr == 1)[0]
    short_idx = np.where(pos_arr == -1)[0]
    if len(long_idx) > 0:
        fig_c.add_trace(go.Scatter(
            x=disp["date"].iloc[long_idx],
            y=disp["low"].iloc[long_idx] * 0.997,
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="#00c7b7"),
            name="LONG"), row=1, col=1)
    if len(short_idx) > 0:
        fig_c.add_trace(go.Scatter(
            x=disp["date"].iloc[short_idx],
            y=disp["high"].iloc[short_idx] * 1.003,
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="#ff4b4b"),
            name="SHORT"), row=1, col=1)

    fig_c.add_trace(go.Bar(
        x=disp["date"], y=disp["close"].pct_change(),
        marker_color=np.where(disp["close"].pct_change() >= 0, "#00c7b7", "#ff4b4b"),
        name="Vynost"), row=2, col=1)

    fig_c.add_trace(go.Scatter(
        x=disp["date"], y=d_probs,
        line=dict(color="#f0c040", width=1.5), name="Prob(up)"), row=3, col=1)
    fig_c.add_hline(y=uth_,   line_dash="dash", line_color="green", row=3, col=1)
    fig_c.add_hline(y=1-uth_, line_dash="dash", line_color="red",   row=3, col=1)

    fig_c.update_layout(height=620, showlegend=False,
                         title=name + " – poslednych 120 dni",
                         paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                         font_color="#ffffff", margin=dict(t=45, b=10))
    fig_c.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig_c, use_container_width=True)

    col_l, col_r = st.columns(2)

    # ── Equity curve ─────────────────────────────────────────
    with col_l:
        all_rets  = fr.iloc[:N2].fillna(0).values
        all_bt    = run_backtest(probs, all_rets, uth_, tc, sl, use_sizing)
        norm_bh   = feat["close"].values[:len(all_bt["equity"])] / feat["close"].values[0]
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=feat["date"].iloc[:len(all_bt["equity"])].values,
            y=all_bt["equity"], mode="lines", name="AI Strategie",
            line=dict(color="#00c7b7", width=2)))
        fig_eq.add_trace(go.Scatter(
            x=feat["date"].iloc[:len(all_bt["equity"])].values,
            y=norm_bh, mode="lines", name="Buy & Hold",
            line=dict(color="#888", dash="dot", width=1)))
        fig_eq.add_vrect(
            x0=str(feat["date"].iloc[test_idx].date()),
            x1=str(feat["date"].iloc[-1].date()),
            fillcolor="rgba(255,255,100,0.07)", line_width=0,
            annotation_text="Test", annotation_position="top left")
        fig_eq.update_layout(title="Equity curve", height=310,
                              paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                              font_color="#ffffff", margin=dict(t=45))
        st.plotly_chart(fig_eq, use_container_width=True)

    # ── Feature importance ────────────────────────────────────
    with col_r:
        try:
            rf_sub = model.calibrated_classifiers_[0].estimator.estimators_[0]
            imp    = rf_sub.feature_importances_
        except Exception:
            imp = np.ones(len(FCOLS)) / len(FCOLS)
        fi_df = pd.DataFrame({"Feature":FCOLS,"Importance":imp}).sort_values("Importance")
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        title="Dulezitost featur",
                        color="Importance", color_continuous_scale="teal")
        fig_fi.update_layout(height=310, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                              font_color="#ffffff", margin=dict(t=45), showlegend=False,
                              coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Mesicni heatmapa ──────────────────────────────────────
    st.subheader("Mesicni vynosy strategie (%)")
    dates_all = feat["date"].iloc[:len(all_bt["rets"])].values
    monthly   = monthly_returns(dates_all, all_bt["rets"])
    monthly.columns = [MONTH_NAMES[m-1] for m in monthly.columns]
    monthly_pct = (monthly * 100).round(2)

    fig_hm = px.imshow(
        monthly_pct,
        color_continuous_scale=["#ff4b4b","#0e1117","#00c7b7"],
        color_continuous_midpoint=0,
        text_auto=True,
        aspect="auto",
        title="Mesicni P&L (%)",
    )
    fig_hm.update_layout(height=300, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          font_color="#ffffff", margin=dict(t=45))
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── Aktualni signal ───────────────────────────────────────
    p_now = float(probs[-1])
    lbl, color, icon = signal_info(p_now, uth_)
    date_s  = str(feat["date"].iloc[-1].strftime("%d.%m.%Y"))
    close_s = str(round(float(feat["close"].iloc[-1]), 2))
    rsi_s   = str(round(float(feat["rsi"].iloc[-1]*100), 1))
    adx_s   = str(round(float(feat["adx"].iloc[-1]*100), 1))

    st.markdown(
        "<div style='padding:1.2rem;border-radius:10px;background:#1e1e2e;"
        "border-left:7px solid " + color + ";margin-top:1rem;'>"
        "<span style='font-size:1.6rem;font-weight:bold;'>" + icon + " " + lbl + "</span><br><br>"
        "<table style='color:#fff;width:100%;'>"
        "<tr><td>Pravdepodobnost rustu</td><td><b>" + str(round(p_now*100,1)) + "%</b></td>"
        "<td>Posledni cena</td><td><b>" + close_s + "</b></td></tr>"
        "<tr><td>Datum signalu</td><td><b>" + date_s + "</b></td>"
        "<td>RSI (14)</td><td><b>" + rsi_s + "</b></td></tr>"
        "<tr><td>ADX</td><td><b>" + adx_s + "</b></td>"
        "<td>Optimalizovany prah</td><td><b>" + str(uth_) + "</b></td></tr>"
        "</table></div>",
        unsafe_allow_html=True,
    )

    # Export
    pos_full = np.where(probs >= uth_, 1, np.where(probs <= 1-uth_, -1, 0))
    exp_df   = feat[["date","close"] + FCOLS].copy()
    exp_df["prob_up"] = probs
    exp_df["signal"]  = np.where(pos_full==1,"LONG",np.where(pos_full==-1,"SHORT","FLAT"))
    st.download_button(
        "Stahnout signaly CSV",
        exp_df.to_csv(index=False).encode("utf-8"),
        "signals_" + name + ".csv", "text/csv",
    )


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 – PORTFOLIO SIMULACE
# ═══════════════════════════════════════════════════════════════
elif PAGE == "Portfolio Simulace":
    st.subheader("Simulace portfolia – rovnomerne rozlozeni")
    if not selected:
        st.info("Vyber instrumenty.")
        st.stop()

    capital = st.number_input("Pocatecni kapital (USD)", 1000, 1000000, 10000, 1000)
    st.markdown("---")

    eq_curves  = {}
    tickers_ok = []

    for name in selected:
        meta = INSTRUMENTS[name]
        with st.spinner("Nacitam " + name + "..."):
            model, sc, feat, fr = train_model(meta["yf"], meta["period"])
        if model is None:
            continue
        X     = feat[FCOLS].values
        probs = model.predict_proba(sc.transform(X))[:, 1]
        N2    = len(feat)
        ve2   = int(N2 * 0.80)

        val_p = probs[ve2:int(N2*0.90)]
        val_r = fr.iloc[ve2:int(N2*0.90)].values[:len(val_p)]
        uth_  = best_threshold(val_p, val_r) if auto_uth else man_uth

        all_rets = fr.iloc[:N2].fillna(0).values
        bt       = run_backtest(probs, all_rets, uth_, tc, sl, use_sizing)

        eq_curves[name] = pd.Series(
            bt["equity"] * (capital / len(selected)),
            index=feat["date"].iloc[:len(bt["equity"])],
        )
        tickers_ok.append(name)

    if not eq_curves:
        st.error("Zadna data.")
        st.stop()

    combined = pd.concat(eq_curves.values(), axis=1)
    combined.columns = list(eq_curves.keys())
    combined = combined.fillna(method="ffill").dropna()
    portfolio = combined.sum(axis=1)

    fig_p = go.Figure()
    colors_ = ["#00c7b7","#ffd700","#c0c0c0","#6699ff","#ff66cc"]
    for j, col in enumerate(combined.columns):
        fig_p.add_trace(go.Scatter(x=combined.index, y=combined[col],
                                    mode="lines", name=col,
                                    line=dict(color=colors_[j % len(colors_)], width=1.5, dash="dot")))
    fig_p.add_trace(go.Scatter(x=portfolio.index, y=portfolio,
                                mode="lines", name="PORTFOLIO",
                                line=dict(color="#ffffff", width=3)))
    fig_p.update_layout(title="Simulace portfolia (USD)",
                         height=420, paper_bgcolor="#0e1117",
                         plot_bgcolor="#0e1117", font_color="#ffffff",
                         margin=dict(t=50))
    st.plotly_chart(fig_p, use_container_width=True)

    # Statistiky
    yr_   = len(portfolio) / 252
    cagr_ = float(portfolio.iloc[-1] / portfolio.iloc[0]) ** (1/yr_) - 1 if yr_ > 0 else 0
    rets_ = portfolio.pct_change().dropna()
    sh_   = float(rets_.mean() / rets_.std() * math.sqrt(252)) if rets_.std() > 0 else 0
    rm_   = portfolio.cummax()
    mdd_  = float(((portfolio - rm_) / rm_).min())
    final_= float(portfolio.iloc[-1])

    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Pocatecni kapital", str(capital) + " $")
    s2.metric("Konecna hodnota",   str(round(final_, 0)) + " $",
              delta=str(round(final_ - capital, 0)) + " $")
    s3.metric("CAGR portfolia",    str(round(cagr_*100, 1)) + "%")
    s4.metric("Sharpe portfolia",  str(round(sh_, 2)))

    st.metric("Max drawdown", str(round(mdd_*100, 1)) + "%")

    csv_p = portfolio.reset_index()
    csv_p.columns = ["date","portfolio_value"]
    st.download_button("Stahnout portfolio CSV",
                        csv_p.to_csv(index=False).encode("utf-8"),
                        "portfolio.csv","text/csv")
