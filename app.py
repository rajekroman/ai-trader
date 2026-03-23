
# ============================================================
#  app.py  –  AI Trading Screener
#  Nástroje: Streamlit, scikit-learn, pandas, plotly
# ============================================================
import os, math, json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Trading Screener", layout="wide", page_icon="📈")

# ------------------------------------------------------------------
# Metadata a cesty
# ------------------------------------------------------------------
INSTRUMENTS = {
    "SPY":    {"label":"S&P 500 ETF",    "emoji":"📊", "category":"Akcie",    "url":"https://stooq.com/q/d/l/?s=spy.us&i=d"},
    "GOLD":   {"label":"Zlato (Gold)",   "emoji":"🥇", "category":"Komodita", "url":"https://stooq.com/q/d/l/?s=gc.f&i=d"},
    "SILVER": {"label":"Stříbro (Silver)","emoji":"🥈","category":"Komodita", "url":"https://stooq.com/q/d/l/?s=si.f&i=d"},
    "ETH":    {"label":"Ethereum (ETH)", "emoji":"🔷", "category":"Krypto",   "url":"https://stooq.com/q/d/l/?s=eth.v&i=d"},
    "SOL":    {"label":"Solana (SOL)",   "emoji":"🟣", "category":"Krypto",   "url":"https://stooq.com/q/d/l/?s=sol.v&i=d"},
}
FCOLS = ["r1","r5","r20","ma5","ma20","v5","vc"]
MODEL_DIR = os.path.dirname(__file__)

# ------------------------------------------------------------------
# Pomocné funkce
# ------------------------------------------------------------------

def load_model_scaler(name):
    m_path = os.path.join(MODEL_DIR, f"model_{name}.pkl")
    s_path = os.path.join(MODEL_DIR, f"scaler_{name}.pkl")
    if not os.path.exists(m_path) or not os.path.exists(s_path):
        return None, None
    return joblib.load(m_path), joblib.load(s_path)


@st.cache_data(ttl=3600, show_spinner=False)
def load_live_data(name, url):
    try:
        df = pd.read_csv(url)
        df.columns = [c.strip().title() for c in df.columns]
        df.rename(columns={"Date":"date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"])
        if "Volume" not in df.columns or df["Volume"].isna().all():
            df["Volume"] = 1.0
        df["Volume"] = df["Volume"].replace(0, np.nan).ffill().fillna(1.0)
        return df.tail(600).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def load_fallback_data(name):
    path = os.path.join(MODEL_DIR, f"raw_{name}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


def featurize(df):
    c = df["Close"]
    fr = c.shift(-1)/c - 1
    feat = pd.DataFrame({
        "r1": c.pct_change(), "r5": c.pct_change(5), "r20": c.pct_change(20),
        "ma5": c.rolling(5).mean()/c - 1, "ma20": c.rolling(20).mean()/c - 1,
        "v5": c.pct_change().rolling(5).std(), "vc": df["Volume"].pct_change(),
        "date": df["date"], "close": c
    })
    feat["target"] = (fr > 0).astype(int)
    feat = feat.dropna().reset_index(drop=True)
    return feat, fr


def backtest_full(probs, rets, uth):
    lth = 1 - uth
    pos = np.where(probs >= uth, 1, np.where(probs <= lth, -1, 0))
    sr  = pos * rets
    eq  = (1 + sr).cumprod()
    yr  = len(sr) / 252
    cagr  = float(eq[-1]**(1/yr) - 1) if yr > 0 else 0.0
    sh    = float(sr.mean()/sr.std()*math.sqrt(252)) if sr.std() > 0 else 0.0
    rm    = np.maximum.accumulate(eq)
    mdd   = float((eq/rm - 1).min())
    return dict(cagr=cagr, sharpe=sh, max_dd=mdd, final_eq=float(eq[-1]),
                equity=eq, pos=pos, rets=sr)


def signal_label(prob, uth):
    if prob >= uth:
        return "🟢 LONG", "green"
    elif prob <= 1 - uth:
        return "🔴 SHORT", "red"
    else:
        return "⚪ FLAT", "gray"


# ------------------------------------------------------------------
# Hlavní UI
# ------------------------------------------------------------------
st.title("📈 AI Trading Screener")
st.caption("Natrénovaný LogisticRegression model (technické indikátory) · backtest · živé signály")

with st.sidebar:
    st.header("⚙️ Nastavení")
    selected = st.multiselect(
        "Zobrazit instrumenty",
        list(INSTRUMENTS.keys()),
        default=list(INSTRUMENTS.keys()),
    )
    uth_global = st.slider("Prah pro LONG signal (model confidence)", 0.50, 0.70, 0.52, 0.01)
    show_live = st.toggle("Pokusit se o živá data (Stooq)", value=True)
    st.markdown("---")
    st.caption("Modely jsou natrénované na syntetických cenových řadách (GBM). "
               "Slouží k demonstraci – nejsou finančním poradenstoím.")

# ------------------------------------------------------------------
# Overview karet
# ------------------------------------------------------------------
st.subheader("🔎 Přehled – aktuální AI signály")
COLS = st.columns(len(selected)) if selected else []

for i, name in enumerate(selected):
    meta = INSTRUMENTS[name]
    model, scaler = load_model_scaler(name)

    with COLS[i]:
        if model is None:
            st.warning(f"{meta['emoji']} {name}
Model nenalezen")
            continue
        df = load_live_data(name, meta["url"]) if show_live else pd.DataFrame()
        if df.empty:
            df = load_fallback_data(name)
        if df.empty or len(df) < 25:
            st.warning(f"{meta['emoji']} {name}
Málo dat")
            continue
        feat, _ = featurize(df)
        if feat.empty:
            continue
        X = feat[FCOLS].values
        Xs = scaler.transform(X)
        probs = model.predict_proba(Xs)[:,1]
        latest_prob = probs[-1]
        label, color = signal_label(latest_prob, uth_global)
        st.metric(
            label=f"{meta['emoji']} {name}",
            value=label,
            delta=f"p={latest_prob:.2%}  |  {meta['category']}",
        )

st.markdown("---")

# ------------------------------------------------------------------
# Detailní záložky
# ------------------------------------------------------------------
tabs = st.tabs([f"{INSTRUMENTS[n]['emoji']} {n}" for n in selected]) if selected else []

for tab, name in zip(tabs, selected):
    meta = INSTRUMENTS[name]
    with tab:
        st.subheader(f"{meta['emoji']} {meta['label']} ({name})")

        model, scaler = load_model_scaler(name)
        if model is None:
            st.error("Model nebo scaler nebyl nalezen. Ujistěte se, že soubory model_*.pkl a scaler_*.pkl jsou ve stejné složce jako app.py.")
            continue

        df = load_live_data(name, meta["url"]) if show_live else pd.DataFrame()
        if df.empty:
            df = load_fallback_data(name)
        if df.empty or len(df) < 30:
            st.warning("Nedostatek dat.")
            continue

        feat, fr = featurize(df)
        if len(feat) < 30:
            st.warning("Nedostatek featur.")
            continue

        X    = feat[FCOLS].values
        Xs   = scaler.transform(X)
        probs = model.predict_proba(Xs)[:,1]

        # Backtest na celé dostupné historii
        rets_aligned = fr.iloc[:len(probs)].values
        mask = ~np.isnan(rets_aligned)
        bt = backtest_full(probs[mask], rets_aligned[mask], uth_global)

        # Metriky
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📈 CAGR", f"{bt['cagr']*100:.1f}%")
        c2.metric("⚖️ Sharpe ratio", f"{bt['sharpe']:.2f}")
        c3.metric("📉 Max drawdown", f"{bt['max_dd']*100:.1f}%")
        c4.metric("💰 Final equity", f"{bt['final_eq']:.3f}×")

        # Equity curve
        bt_df = pd.DataFrame({
            "date": feat["date"].iloc[:len(bt["equity"])].values,
            "equity": bt["equity"],
            "position": bt["pos"],
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt_df["date"], y=bt_df["equity"],
                                 mode="lines", name="Strategie", line=dict(color="#00c7b7")))
        # Cena normalizovaná
        norm_close = (feat["close"].values[:len(bt_df)] /
                      feat["close"].values[0])
        fig.add_trace(go.Scatter(x=bt_df["date"], y=norm_close,
                                 mode="lines", name="Buy & Hold",
                                 line=dict(color="#888888", dash="dot")))
        fig.update_layout(title=f"Equity curve – {name}",
                          xaxis_title="Datum", yaxis_title="Kumulativní hodnota (start=1)",
                          legend=dict(x=0.01, y=0.99), height=350, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

        # Histogram pravděpodobností
        col_a, col_b = st.columns(2)
        with col_a:
            fig2 = px.histogram(x=probs, nbins=30,
                                labels={"x":"Pravděpodobnost růstu"},
                                title="Distribuce predikovaných pravděpodobností",
                                color_discrete_sequence=["#00c7b7"])
            fig2.add_vline(x=uth_global, line_dash="dash", line_color="green",
                           annotation_text="LONG práh")
            fig2.add_vline(x=1-uth_global, line_dash="dash", line_color="red",
                           annotation_text="SHORT práh")
            fig2.update_layout(height=280, margin=dict(t=40))
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            # Denní výnosy vs. pozice
            ret_df = pd.DataFrame({
                "return": bt["rets"],
                "pos": bt["pos"].astype(str),
            })
            fig3 = px.scatter(ret_df, x=ret_df.index, y="return", color="pos",
                              color_discrete_map={"1":"green","-1":"red","0":"gray"},
                              title="Denní výnosy strategie (zelená=LONG, červená=SHORT)",
                              labels={"x":"Den","return":"Denní výnos"})
            fig3.update_layout(height=280, margin=dict(t=40), showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        # Aktuální signál
        latest_prob = float(probs[-1])
        label, color = signal_label(latest_prob, uth_global)
        st.markdown(
            f"""
            <div style="padding:1rem;border-radius:0.5rem;background:#1e1e1e;border-left:5px solid {color};">
            <strong style="font-size:1.2rem;">{label}</strong><br>
            Pravděpodobnost růstu: <strong>{latest_prob:.2%}</strong><br>
            Datum signálu: <strong>{feat["date"].iloc[-1].strftime('%d.%m.%Y')}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Export dat
        export_df = feat[["date","close"] + FCOLS].copy()
        export_df["prob_up"] = probs
        pos_arr = np.where(probs >= uth_global, 1, np.where(probs <= 1-uth_global, -1, 0))
        export_df["signal"] = np.where(pos_arr==1,"LONG",np.where(pos_arr==-1,"SHORT","FLAT"))
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"💾 Stáhnout {name} signály (CSV)",
            data=csv,
            file_name=f"signals_{name}.csv",
            mime="text/csv",
        )

# ------------------------------------------------------------------
# Srovnávací tabulka všech instrumentů
# ------------------------------------------------------------------
if len(selected) > 1:
    st.markdown("---")
    st.subheader("📊 Srovnání všech instrumentů")
    rows = []
    for name in selected:
        meta = INSTRUMENTS[name]
        model, scaler = load_model_scaler(name)
        if model is None: continue
        df = load_live_data(name, meta["url"]) if show_live else pd.DataFrame()
        if df.empty: df = load_fallback_data(name)
        if df.empty or len(df) < 30: continue
        feat, fr = featurize(df)
        if feat.empty: continue
        X = feat[FCOLS].values
        probs = model.predict_proba(scaler.transform(X))[:,1]
        rets_a = fr.iloc[:len(probs)].values
        mask = ~np.isnan(rets_a)
        bt = backtest_full(probs[mask], rets_a[mask], uth_global)
        label, _ = signal_label(float(probs[-1]), uth_global)
        rows.append({
            "Instrument": f"{meta['emoji']} {name}",
            "Kategorie": meta["category"],
            "Signál": label,
            "Prob. ↑": f"{probs[-1]:.2%}",
            "CAGR": f"{bt['cagr']*100:.1f}%",
            "Sharpe": f"{bt['sharpe']:.2f}",
            "Max DD": f"{bt['max_dd']*100:.1f}%",
            "Final Eq.": f"{bt['final_eq']:.3f}×",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("⚠️ Tato aplikace slouží POUZE k demonstraci AI/ML v obchodování. Není to finanční poradenství. Vždy proveďte vlastní analýzu a konzultujte s odborníkem.")
