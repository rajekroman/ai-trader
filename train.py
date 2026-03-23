#!/usr/bin/env python3
"""
train.py  –  Trenovaci skript AI Trader v6
Spust jednou lokalne:  python train.py
Vygeneruje:  models/<TICKER>_model.pkl + models/<TICKER>_meta.json
Pote nahraj cely priecinok models/ do Githubu.
"""
import os, json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib

TICKERS = [
    "SPY","QQQ","AAPL","MSFT","NVDA","TSLA",
    "AMZN","META","GOOGL","AMD","PLTR","GC=F","BTC-USD","ETH-USD"
]
PERIOD  = "5y"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

FCOLS = [
    "r1","r2","r3","r5","r10","r20",
    "ma5","ma10","ma20","ma50","ma200",
    "rsi","stoch_k","stoch_d","willi","cci",
    "macd","adx","aroon_up","aroon_dn",
    "bb","atr","v5","v20","hist_vol",
    "obv","vforce","vc","donch","keltner",
    "above_ma200","regime","trend_str","dow","month_sin",
]

# ── Indikatory ─────────────────────────────────────────────────────
def _rsi(s,p=14):
    d=s.diff(); g=d.clip(lower=0).rolling(p).mean()
    l=(-d.clip(upper=0)).rolling(p).mean()
    return 100-100/(1+g/l.replace(0,np.nan))
def _stoch(hi,lo,c,k=14,d=3):
    lom=lo.rolling(k).min(); him=hi.rolling(k).max()
    sk=100*(c-lom)/(him-lom+1e-9); return sk, sk.rolling(d).mean()
def _williams(hi,lo,c,p=14):
    return -100*(hi.rolling(p).max()-c)/(hi.rolling(p).max()-lo.rolling(p).min()+1e-9)
def _macd(s,f=12,sl=26,sg=9):
    m=s.ewm(span=f,adjust=False).mean()-s.ewm(span=sl,adjust=False).mean()
    return (m-m.ewm(span=sg,adjust=False).mean())/(s.abs()+1e-9)
def _bb(s,p=20):
    ma=s.rolling(p).mean(); std=s.rolling(p).std()
    return (s-(ma-2*std))/(4*std+1e-9)
def _cci(hi,lo,c,p=20):
    tp=(hi+lo+c)/3; ma=tp.rolling(p).mean()
    md=tp.rolling(p).apply(lambda x:np.mean(np.abs(x-x.mean())))
    return (tp-ma)/(0.015*md+1e-9)/100
def _atr(hi,lo,c,p=14):
    tr=pd.concat([(hi-lo),(hi-c.shift()).abs(),(lo-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()/(c+1e-9)
def _obv(c,vo,p=20):
    obv=(np.sign(c.diff())*vo).cumsum()
    return obv.pct_change(p).fillna(0)
def _donch(c,p=20):
    return (c-c.rolling(p).min())/(c.rolling(p).max()-c.rolling(p).min()+1e-9)
def _adx(hi,lo,c,p=14):
    tr=pd.concat([(hi-lo),(hi-c.shift()).abs(),(lo-c.shift()).abs()],axis=1).max(axis=1)
    atr_=tr.ewm(span=p,adjust=False).mean()
    up=hi.diff(); dn=-lo.diff()
    pdm=pd.Series(np.where((up>dn)&(up>0),up,0.),index=c.index)
    ndm=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=c.index)
    pdi=100*pdm.ewm(span=p,adjust=False).mean()/(atr_+1e-9)
    ndi=100*ndm.ewm(span=p,adjust=False).mean()/(atr_+1e-9)
    dx=100*(pdi-ndi).abs()/(pdi+ndi+1e-9)
    return dx.ewm(span=p,adjust=False).mean()/100
def _aroon(hi,lo,p=25):
    au=hi.rolling(p+1).apply(lambda x:x.argmax(),raw=True)/p*100
    ad=lo.rolling(p+1).apply(lambda x:x.argmin(),raw=True)/p*100
    return au,ad
def _keltner(c,hi,lo,p=20,m=2.0):
    ema=c.ewm(span=p,adjust=False).mean()
    atr_=_atr(hi,lo,c,p)
    ku=ema+m*atr_*c; kl=ema-m*atr_*c
    return (c-kl)/(ku-kl+1e-9)
def _vforce(c,hi,lo,vo):
    mid=(hi+lo)/2
    return ((c-mid)/(hi-lo+1e-9)*vo).pct_change(10).fillna(0)

def featurize(df):
    c=df["Close"]; hi=df.get("High",c); lo=df.get("Low",c)
    vo=df.get("Volume",pd.Series(1.0,index=c.index))
    fr=c.shift(-1)/c-1
    sk,sd=_stoch(hi,lo,c); au,ad=_aroon(hi,lo)
    ma200l=c.rolling(200).mean()
    hv=c.pct_change().rolling(20).std()*math.sqrt(252)
    ts=(c.rolling(5).mean()-c.rolling(20).mean()).abs()/(c.rolling(20).std()+1e-9)
    feat=pd.DataFrame({
        "r1":c.pct_change(),"r2":c.pct_change(2),"r3":c.pct_change(3),
        "r5":c.pct_change(5),"r10":c.pct_change(10),"r20":c.pct_change(20),
        "ma5":c.rolling(5).mean()/c-1,"ma10":c.rolling(10).mean()/c-1,
        "ma20":c.rolling(20).mean()/c-1,"ma50":c.rolling(50).mean()/c-1,
        "ma200":ma200l/c-1,
        "rsi":_rsi(c)/100,"stoch_k":sk/100,"stoch_d":sd/100,
        "willi":_williams(hi,lo,c)/100,"cci":_cci(hi,lo,c),
        "macd":_macd(c),"adx":_adx(hi,lo,c),
        "aroon_up":au/100,"aroon_dn":ad/100,
        "bb":_bb(c),"atr":_atr(hi,lo,c),
        "v5":c.pct_change().rolling(5).std(),
        "v20":c.pct_change().rolling(20).std(),
        "hist_vol":hv,"obv":_obv(c,vo),"vforce":_vforce(c,hi,lo,vo),
        "vc":vo.pct_change(),"donch":_donch(c),"keltner":_keltner(c,hi,lo),
        "above_ma200":(c>ma200l).astype(float),
        "regime":(c.rolling(50).mean()>ma200l).astype(float),
        "trend_str":ts,
        "dow":pd.to_datetime(df["date"]).dt.dayofweek/4.0,
        "month_sin":np.sin(2*np.pi*pd.to_datetime(df["date"]).dt.month/12),
        "date":df["date"],"close":c,
    })
    feat["target"]=(fr>0).astype(int)
    return feat.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)

def load_data(ticker, period="5y"):
    import yfinance as yf
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    dc = "Datetime" if "Datetime" in df.columns else "Date"
    df.rename(columns={dc:"date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"]).sort_values("date").reset_index(drop=True)
    if "Volume" not in df.columns or df["Volume"].isna().all():
        df["Volume"] = 1.0
    df["Volume"] = df["Volume"].replace(0,np.nan).ffill().fillna(1.0)
    for col in ["High","Low","Open"]:
        if col not in df.columns:
            df[col] = df["Close"]
    return df

def best_threshold(probs, rets):
    best_sh, best_u = -999.0, 0.55
    for u in np.arange(0.50, 0.68, 0.01):
        lth = 1 - u
        pos = np.where(probs>=u, 1.0, np.where(probs<=lth, -1.0, 0.0))
        r   = pos * rets
        if r.std() > 0:
            sh = float(r.mean() / r.std() * math.sqrt(252))
            if sh > best_sh and np.sum(np.abs(pos)>0) > 3:
                best_sh = sh; best_u = u
    return round(float(best_u), 2)

def train_ticker(ticker):
    from sklearn.ensemble import (RandomForestClassifier,
                                   HistGradientBoostingClassifier,
                                   ExtraTreesClassifier,
                                   VotingClassifier)
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score

    print(f"  [{ticker}] stahuji data...", end="", flush=True)
    df = load_data(ticker, PERIOD)
    if df.empty or len(df) < 300:
        print(" PRESKOCENO (malo dat)")
        return False

    feat = featurize(df)
    N = len(feat)
    if N < 300:
        print(f" PRESKOCENO (feat={N})")
        return False

    te = int(N * 0.10); ve = int(N * 0.10); tr = N - te - ve
    X = feat[FCOLS].values; y = feat["target"].values
    fr_vals = feat["close"].pct_change().shift(-1).fillna(0).values

    sc = StandardScaler()
    Xtr = sc.fit_transform(X[:tr]);   ytr = y[:tr]
    Xva = sc.transform(X[tr:tr+ve]);  yva = y[tr:tr+ve]
    Xte = sc.transform(X[tr+ve:]);    yte = y[tr+ve:]

    rf  = RandomForestClassifier(n_estimators=200, max_depth=7,
                                  min_samples_leaf=5, max_features="sqrt",
                                  random_state=42, n_jobs=1)
    hgb = HistGradientBoostingClassifier(max_iter=200, max_depth=5,
                                          learning_rate=0.05, min_samples_leaf=10,
                                          random_state=42)
    et  = ExtraTreesClassifier(n_estimators=150, max_depth=6,
                                min_samples_leaf=5, max_features="sqrt",
                                random_state=42, n_jobs=1)
    vote = VotingClassifier(estimators=[("rf",rf),("hgb",hgb),("et",et)],
                             voting="soft", n_jobs=1)
    cal  = CalibratedClassifierCV(vote, cv=3, method="isotonic", n_jobs=1)

    print(" trenuji...", end="", flush=True)
    cal.fit(Xtr, ytr)

    pva = cal.predict_proba(Xva)[:,1]
    pte = cal.predict_proba(Xte)[:,1]
    try:    auc_v = round(float(roc_auc_score(yva, pva)), 4)
    except: auc_v = 0.0
    try:    auc_t = round(float(roc_auc_score(yte, pte)), 4)
    except: auc_t = 0.0

    uth = best_threshold(pva, fr_vals[tr:tr+ve])

    # Dotrenink na tr+va
    Xtrva = sc.transform(X[:tr+ve]); ytrva = y[:tr+ve]
    rf2  = RandomForestClassifier(n_estimators=200, max_depth=7,
                                   min_samples_leaf=5, max_features="sqrt",
                                   random_state=42, n_jobs=1)
    hgb2 = HistGradientBoostingClassifier(max_iter=200, max_depth=5,
                                           learning_rate=0.05, min_samples_leaf=10,
                                           random_state=42)
    et2  = ExtraTreesClassifier(n_estimators=150, max_depth=6,
                                 min_samples_leaf=5, max_features="sqrt",
                                 random_state=42, n_jobs=1)
    vote2 = VotingClassifier(estimators=[("rf",rf2),("hgb",hgb2),("et",et2)],
                              voting="soft", n_jobs=1)
    cal2  = CalibratedClassifierCV(vote2, cv=3, method="isotonic", n_jobs=1)
    cal2.fit(Xtrva, ytrva)

    # Feature importance z RF
    fi = np.zeros(len(FCOLS))
    try:
        rff = cal2.calibrated_classifiers_[0].estimator.estimators_[0][1]
        fi  = rff.feature_importances_
    except Exception:
        pass

    safe = ticker.replace("=","_").replace("-","_")
    joblib.dump({"model": cal2, "scaler": sc, "fi": fi, "fcols": FCOLS},
                os.path.join(MODELS_DIR, f"{safe}_model.pkl"),
                compress=3)

    meta = {
        "ticker":   ticker,
        "trained":  pd.Timestamp.now().isoformat(),
        "period":   PERIOD,
        "n_rows":   N,
        "auc_val":  auc_v,
        "auc_test": auc_t,
        "uth":      uth,
        "fcols":    FCOLS,
    }
    with open(os.path.join(MODELS_DIR, f"{safe}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f" ✅  AUC_val={auc_v}  AUC_test={auc_t}  prah={uth}")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print(f"Trenuju {len(TICKERS)} modelu na {PERIOD} datech")
    print("=" * 60)
    ok, fail = 0, 0
    for t in TICKERS:
        try:
            if train_ticker(t): ok += 1
            else:               fail += 1
        except Exception as e:
            print(f"  [{t}] CHYBA: {e}"); fail += 1
    print("=" * 60)
    print(f"Hotovo: {ok} OK, {fail} chyb")
    print(f"Modely ulozeny v:  ./{MODELS_DIR}/")
    print("Nahraj cely adresar models/ do GitHubu vedle app.py")
