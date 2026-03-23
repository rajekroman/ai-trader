
# ============================================================
#  app.py – AI Trading Screener v2
#  Ziva data: yfinance  |  Model: RandomForest (sklearn)
# ============================================================
import os, math, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="AI Trading Screener v2", layout="wide", page_icon="📈")

# ── Konstanty ────────────────────────────────────────────────
INSTRUMENTS = {
    "SPY":    {"label":"S&P 500 ETF",     "emoji":"📊", "category":"Akcie",    "yf":"SPY"},
    "GOLD":   {"label":"Zlato",           "emoji":"🥇", "category":"Komodita", "yf":"GC=F"},
    "SILVER": {"label":"Stribro",         "emoji":"🥈", "category":"Komodita", "yf":"SI=F"},
    "ETH":    {"label":"Ethereum",        "emoji":"🔷", "category":"Krypto",   "yf":"ETH-USD"},
    "SOL":    {"label":"Solana",          "emoji":"🟣", "category":"Krypto",   "yf":"SOL-USD"},
}

FCOLS = ["r1","r5","r20","ma5","ma20","v5","v20","vc",
         "rsi","bb_pos","macd_diff","atr_ratio"]

# ── Indikatory ───────────────────────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def bollinger_pos(series, period=20):
    ma  = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - (ma - 2*std)) / (4*std + 1e-9)

def macd_diff(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig

def atr_ratio(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([high-low,
                    (high-close.shift()).abs(),
                    (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean() / close


# ── Data z yfinance ──────────────────────────────────────────
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
        df.rename(columns={"Date":"date","index":"date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"]).sort_values("date").reset_index(drop=True)
        if "Volume" not in df.columns or df["Volume"].isna().all():
            df["Volume"] = 1.0
        df["Volume"] = df["Volume"].replace(0, np.nan).ffill().fillna(1.0)
        return df
    except Exception as e:
        return pd.DataFrame()


# ── Featury ──────────────────────────────────────────────────
def featurize(df):
    c = df["Close"]
    fr = c.shift(-1) / c - 1
    feat = pd.DataFrame({
        "r1":       c.pct_change(),
        "r5":       c.pct_change(5),
        "r20":      c.pct_change(20),
        "ma5":      c.rolling(5).mean() / c - 1,
        "ma20":     c.rolling(20).mean() / c - 1,
        "v5":       c.pct_change().rolling(5).std(),
        "v20":      c.pct_change().rolling(20).std(),
        "vc":       df["Volume"].pct_change(),
        "rsi":      rsi(c) / 100,
        "bb_pos":   bollinger_pos(c),
        "macd_diff":macd_diff(c),
        "atr_ratio":atr_ratio(df),
        "date":     df["date"],
        "close":    c,
    })
    feat["target"] = (fr > 0).astype(int)
    feat = feat.dropna().reset_index(drop=True)
    return feat, fr


# ── Trenovani (RandomForest) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(ticker, period="5y"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib

    df = load_data(ticker, period)
    if df.empty or len(df) < 200:
        return None, None, None, None

    feat, fr = featurize(df)
    N  = len(feat)
    te = int(N * 0.70)
    ve = int(N * 0.85)

    X, y = feat[FCOLS].values, feat["target"].values

    sc = StandardScaler()
    Xtv_s = sc.fit_transform(X[:ve])

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(Xtv_s, y[:ve])

    return rf, sc, feat, fr


# ── Backtest ─────────────────────────────────────────────────
def backtest(probs, rets, uth):
    lth = 1 - uth
    pos = np.where(probs >= uth, 1, np.where(probs <= lth, -1, 0))
    sr  = pos * rets
    eq  = (1 + sr).cumprod()
    yr  = len(sr) / 252
    cagr  = float(eq[-1] ** (1/yr) - 1) if yr > 0 else 0.0
    sh    = float(sr.mean() / sr.std() * math.sqrt(252)) if sr.std() > 0 else 0.0
    mdd   = float((eq / np.maximum.accumulate(eq) - 1).min())
    return dict(cagr=cagr, sharpe=sh, max_dd=mdd, final_eq=float(eq[-1]),
                equity=eq, pos=pos, rets=sr)


def signal_label(prob, uth):
    if prob >= uth:
        return "LONG", "green", "🟢"
    elif prob <= 1 - uth:
        return "SHORT", "red", "🔴"
    return "FLAT", "gray", "⚪"


# ────────────────────────────────────────────────────────────
#  UI
# ────────────────────────────────────────────────────────────
st.title("📈 AI Trading Screener v2")
st.caption("Ziva data: Yahoo Finance  |  Model: RandomForest (300 stromu, 12 featur)  |  Backtest: posledni 15 % dat")

with st.sidebar:
    st.header("Nastaveni")
    selected = st.multiselect("Instrumenty", list(INSTRUMENTS.keys()),
                               default=list(INSTRUMENTS.keys()))
    uth = st.slider("Prah signalu (model confidence)", 0.50, 0.70, 0.55, 0.01)
    period = st.selectbox("Historicke obdobi", ["2y","3y","5y","max"], index=2)
    st.markdown("---")
    st.info("Model se trenuje automaticky pri prvnim spusteni (~30 s).")
    st.caption("Pouze demonstrace – neni to financni poradenstvi.")

# ── Karty se signaly ─────────────────────────────────────────
st.subheader("Aktualni AI signaly")
if selected:
    col_cards = st.columns(len(selected))
    for i, name in enumerate(selected):
        meta = INSTRUMENTS[name]
        with col_cards[i]:
            with st.spinner("Trenuju " + name + "..."):
                rf, sc, feat, fr = train_model(meta["yf"], period)
            if rf is None:
                st.warning(meta["emoji"] + " " + name + " – chyba dat")
                continue
            probs = rf.predict_proba(sc.transform(feat[FCOLS].values))[:, 1]
            p     = float(probs[-1])
            label, color, icon = signal_label(p, uth)
            st.metric(
                label=meta["emoji"] + " " + name,
                value=icon + " " + label,
                delta="p=" + str(round(p*100, 1)) + "%  " + meta["category"],
            )

st.markdown("---")

# ── Detailni zalozky ──────────────────────────────────────────
if selected:
    tabs = st.tabs([INSTRUMENTS[n]["emoji"] + " " + n for n in selected])

    for tab, name in zip(tabs, selected):
        meta = INSTRUMENTS[name]
        with tab:
            st.subheader(meta["emoji"] + " " + meta["label"] + " (" + name + ")")

            rf, sc, feat, fr = train_model(meta["yf"], period)
            if rf is None:
                st.error("Nepodarilo se nacist data pro " + name + ".")
                continue

            X     = feat[FCOLS].values
            probs = rf.predict_proba(sc.transform(X))[:, 1]

            N2  = len(feat)
            ve2 = int(N2 * 0.85)
            test_rets  = fr.iloc[ve2:ve2+len(probs)-ve2].values if ve2 < len(probs) else fr.iloc[:len(probs)].values
            test_probs = probs[ve2:] if ve2 < len(probs) else probs
            mask = ~np.isnan(test_rets[:len(test_probs)])
            bt = backtest(test_probs[mask], test_rets[:len(test_probs)][mask], uth)

            # Metriky
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CAGR (test)",    str(round(bt["cagr"]*100, 1)) + "%")
            c2.metric("Sharpe ratio",   str(round(bt["sharpe"], 2)))
            c3.metric("Max drawdown",   str(round(bt["max_dd"]*100, 1)) + "%")
            c4.metric("Final equity",   str(round(bt["final_eq"], 3)) + "x")

            # Equity curve (cela historie)
            all_rets  = fr.iloc[:len(probs)].fillna(0).values
            all_bt    = backtest(probs, all_rets, uth)
            norm_bh   = feat["close"].values[:len(all_bt["equity"])] / feat["close"].values[0]
            dates_all = feat["date"].iloc[:len(all_bt["equity"])].values

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates_all, y=all_bt["equity"],
                                     mode="lines", name="AI Strategie",
                                     line=dict(color="#00c7b7", width=2)))
            fig.add_trace(go.Scatter(x=dates_all, y=norm_bh,
                                     mode="lines", name="Buy & Hold",
                                     line=dict(color="#888888", dash="dot", width=1)))
            fig.add_vrect(x0=str(feat["date"].iloc[ve2].date()),
                          x1=str(feat["date"].iloc[-1].date()),
                          fillcolor="rgba(255,255,100,0.06)", line_width=0,
                          annotation_text="Test perioda", annotation_position="top left")
            fig.update_layout(title="Equity curve – " + name + "  (zluta = test perioda)",
                              xaxis_title="Datum", yaxis_title="Hodnota (start=1)",
                              height=360, margin=dict(t=45),
                              legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig, use_container_width=True)

            # Dulezitost featur
            col_a, col_b = st.columns(2)
            with col_a:
                feat_imp = pd.DataFrame({
                    "Feature":    FCOLS,
                    "Importance": rf.feature_importances_,
                }).sort_values("Importance", ascending=True)
                fig2 = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                              title="Dulezitost featur (RandomForest)",
                              color="Importance",
                              color_continuous_scale="teal")
                fig2.update_layout(height=340, margin=dict(t=45), showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            with col_b:
                fig3 = px.histogram(x=probs, nbins=40,
                                    labels={"x":"Pravdepodobnost rustu"},
                                    title="Distribuce pravdepodobnosti modelu",
                                    color_discrete_sequence=["#00c7b7"])
                fig3.add_vline(x=uth,       line_dash="dash", line_color="green",
                               annotation_text="LONG")
                fig3.add_vline(x=1-uth,     line_dash="dash", line_color="red",
                               annotation_text="SHORT")
                fig3.update_layout(height=340, margin=dict(t=45))
                st.plotly_chart(fig3, use_container_width=True)

            # Aktualni signal
            p2 = float(probs[-1])
            label2, color2, icon2 = signal_label(p2, uth)
            date_str = str(feat["date"].iloc[-1].strftime("%d.%m.%Y"))
            close_val = str(round(float(feat["close"].iloc[-1]), 2))
            st.markdown(
                "<div style=\"padding:1rem;border-radius:0.5rem;"
                "background:#1e1e2e;border-left:6px solid " + color2 + ";\">"
                "<span style=\"font-size:1.4rem;font-weight:bold;\">"
                + icon2 + " " + label2 + "</span><br><br>"
                "Pravdepodobnost rustu:&nbsp;&nbsp;<strong>" + str(round(p2*100, 1)) + "%</strong><br>"
                "Posledni cena:&nbsp;&nbsp;<strong>" + close_val + "</strong><br>"
                "Datum signalu:&nbsp;&nbsp;<strong>" + date_str + "</strong>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.write("")

            # Export
            pos_arr = np.where(probs >= uth, 1, np.where(probs <= 1-uth, -1, 0))
            exp_df  = feat[["date","close"] + FCOLS].copy()
            exp_df["prob_up"] = probs
            exp_df["signal"]  = np.where(pos_arr==1,"LONG",np.where(pos_arr==-1,"SHORT","FLAT"))
            st.download_button(
                label="Stahnout signaly " + name + " (CSV)",
                data=exp_df.to_csv(index=False).encode("utf-8"),
                file_name="signals_" + name + ".csv",
                mime="text/csv",
            )

# ── Srovnavaci tabulka ────────────────────────────────────────
if len(selected) > 1:
    st.markdown("---")
    st.subheader("Srovnani vsech instrumentu")
    rows = []
    for name in selected:
        meta = INSTRUMENTS[name]
        rf, sc, feat, fr = train_model(meta["yf"], period)
        if rf is None:
            continue
        probs  = rf.predict_proba(sc.transform(feat[FCOLS].values))[:, 1]
        N2     = len(feat); ve2 = int(N2 * 0.85)
        tr2    = fr.iloc[ve2:ve2+len(probs[ve2:])].values
        p2     = probs[ve2:]; mask2 = ~np.isnan(tr2[:len(p2)])
        bt2    = backtest(p2[mask2], tr2[:len(p2)][mask2], uth)
        lbl, _, ico = signal_label(float(probs[-1]), uth)
        rows.append({
            "Instrument": meta["emoji"] + " " + name,
            "Kategorie":  meta["category"],
            "Signal":     ico + " " + lbl,
            "Prob. up":   str(round(probs[-1]*100, 1)) + "%",
            "CAGR":       str(round(bt2["cagr"]*100, 1)) + "%",
            "Sharpe":     str(round(bt2["sharpe"], 2)),
            "Max DD":     str(round(bt2["max_dd"]*100, 1)) + "%",
            "Final Eq.":  str(round(bt2["final_eq"], 3)) + "x",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Pouze demonstrace – neni to financni poradenstvi.")
