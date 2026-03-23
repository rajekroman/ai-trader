
import os, math, time, warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="AI Trader v6", layout="wide",
                   page_icon="🧠", initial_sidebar_state="expanded")
st.markdown("""
<style>
div[data-testid="metric-container"]{background:#1a1a2e;border-radius:10px;
  padding:12px;border:1px solid #16213e;}
</style>""", unsafe_allow_html=True)

PRESETS = {
    "──AKCIE──":None,
    "Apple (AAPL)":"AAPL","Microsoft (MSFT)":"MSFT","NVIDIA (NVDA)":"NVDA",
    "Tesla (TSLA)":"TSLA","Amazon (AMZN)":"AMZN","Meta (META)":"META",
    "Google (GOOGL)":"GOOGL","AMD":"AMD","Palantir (PLTR)":"PLTR",
    "──ETF──":None,
    "S&P 500 (SPY)":"SPY","Nasdaq (QQQ)":"QQQ","Russell (IWM)":"IWM",
    "──KOMODITY──":None,
    "Zlato (GC=F)":"GC=F","Ropa (CL=F)":"CL=F",
    "──KRYPTO──":None,
    "Bitcoin (BTC-USD)":"BTC-USD","Ethereum (ETH-USD)":"ETH-USD","Solana (SOL-USD)":"SOL-USD",
}
SECTOR_ETFS = {"Technology":"XLK","Health Care":"XLV","Financials":"XLF",
               "Consumer Discretionary":"XLY","Communication Services":"XLC",
               "Industrials":"XLI","Energy":"XLE"}
TF_PARAMS = {
    "1m": {"period":"5d",  "interval":"1m",  "label":"1 min", "bars":390},
    "5m": {"period":"5d",  "interval":"5m",  "label":"5 min", "bars":200},
    "15m":{"period":"60d", "interval":"15m", "label":"15 min","bars":200},
    "1h": {"period":"730d","interval":"1h",  "label":"1 hod", "bars":200},
    "1d": {"period":"5y",  "interval":"1d",  "label":"1 den", "bars":300},
}
FCOLS = [
    "r1","r2","r3","r5","r10","r20",
    "ma5","ma10","ma20","ma50","ma200",
    "rsi","stoch_k","stoch_d","willi","cci",
    "macd","adx","aroon_up","aroon_dn",
    "bb","atr","v5","v20","hist_vol",
    "obv","vforce","vc","donch","keltner",
    "above_ma200","regime","trend_str","dow","month_sin",
]

def _rsi(s,p=14):
    d=s.diff(); g=d.clip(lower=0).rolling(p).mean()
    l=(-d.clip(upper=0)).rolling(p).mean()
    return 100-100/(1+g/l.replace(0,np.nan))
def _stoch(hi,lo,c,k=14,d=3):
    lom=lo.rolling(k).min(); him=hi.rolling(k).max()
    sk=100*(c-lom)/(him-lom+1e-9); return sk,sk.rolling(d).mean()
def _williams(hi,lo,c,p=14):
    him=hi.rolling(p).max(); lom=lo.rolling(p).min()
    return -100*(him-c)/(him-lom+1e-9)
def _macd(s,fast=12,slow=26,sig=9):
    ef=s.ewm(span=fast,adjust=False).mean()
    es=s.ewm(span=slow,adjust=False).mean()
    m=ef-es; return (m-m.ewm(span=sig,adjust=False).mean())/(s.abs()+1e-9)
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
    pdm=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=c.index)
    ndm=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=c.index)
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
def pivot_levels(hi,lo,c,n=20):
    h=float(hi.tail(n).max()); l=float(lo.tail(n).min()); cv=float(c.iloc[-1])
    p=(h+l+cv)/3
    return {"R2":round(p+(h-l),4),"R1":round(2*p-l,4),"PP":round(p,4),
            "S1":round(2*p-h,4),"S2":round(p-(h-l),4)}

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
        "above_ma200":(c>ma200l).astype(float),"regime":(c.rolling(50).mean()>ma200l).astype(float),
        "trend_str":ts,
        "dow":pd.to_datetime(df["date"]).dt.dayofweek/4.0,
        "month_sin":np.sin(2*np.pi*pd.to_datetime(df["date"]).dt.month/12),
        "date":df["date"],"close":c,"high":hi,"low":lo,
        "open":df.get("Open",c),"volume":vo,
    })
    feat["target"]=(fr>0).astype(int)
    feat=feat.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
    return feat,fr

@st.cache_data(ttl=3600,show_spinner=False)
def load_daily(ticker,period="5y"):
    try:
        import yfinance as yf
        df=yf.download(ticker,period=period,auto_adjust=True,progress=False)
        if df.empty: return pd.DataFrame(),{}
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        df=df.reset_index()
        df.rename(columns={"Date":"date","Datetime":"date"},inplace=True)
        df["date"]=pd.to_datetime(df["date"])
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns: df[col]=pd.to_numeric(df[col],errors="coerce")
        df=df.dropna(subset=["Close"]).sort_values("date").reset_index(drop=True)
        if "Volume" not in df.columns or df["Volume"].isna().all(): df["Volume"]=1.0
        df["Volume"]=df["Volume"].replace(0,np.nan).ffill().fillna(1.0)
        for col in ["High","Low","Open"]:
            if col not in df.columns: df[col]=df["Close"]
        info={}
        try:
            raw=yf.Ticker(ticker).info
            info={"name":raw.get("longName",ticker),"sector":raw.get("sector","N/A"),
                  "industry":raw.get("industry","N/A"),"mktcap":raw.get("marketCap",0),
                  "pe":raw.get("trailingPE",0),"eps":raw.get("trailingEps",0),
                  "beta":raw.get("beta",0),"52w_high":raw.get("fiftyTwoWeekHigh",0),
                  "52w_low":raw.get("fiftyTwoWeekLow",0),"avg_volume":raw.get("averageVolume",0),
                  "div_yield":raw.get("dividendYield",0),"target_price":raw.get("targetMeanPrice",0)}
        except Exception: pass
        return df,info
    except Exception: return pd.DataFrame(),{}

@st.cache_data(ttl=60,show_spinner=False)
def load_intraday(ticker,tf="1d"):
    try:
        import yfinance as yf
        p=TF_PARAMS[tf]
        if tf=="1d": df=yf.download(ticker,period="5y",auto_adjust=True,progress=False)
        else: df=yf.download(ticker,period=p["period"],interval=p["interval"],auto_adjust=True,progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        df=df.reset_index()
        dc="Datetime" if "Datetime" in df.columns else "Date"
        if dc in df.columns: df.rename(columns={dc:"date"},inplace=True)
        df["date"]=pd.to_datetime(df["date"])
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns: df[col]=pd.to_numeric(df[col],errors="coerce")
        df=df.dropna(subset=["Close"]).sort_values("date").reset_index(drop=True)
        if "Volume" not in df.columns or df["Volume"].isna().all(): df["Volume"]=1.0
        df["Volume"]=df["Volume"].replace(0,np.nan).ffill().fillna(1.0)
        for col in ["High","Low","Open"]:
            if col not in df.columns: df[col]=df["Close"]
        return df
    except Exception: return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def train_model(ticker,period="5y"):
    from sklearn.ensemble import (RandomForestClassifier,
                                   HistGradientBoostingClassifier,
                                   ExtraTreesClassifier,
                                   VotingClassifier)
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score
    df,_=load_daily(ticker,period)
    if df.empty or len(df)<300: return None,None,None,None,None
    feat,fr=featurize(df)
    N=len(feat); te=int(N*0.10); ve=int(N*0.10); tr=N-te-ve
    X=feat[FCOLS].values; y=feat["target"].values
    sc=StandardScaler()
    Xtr=sc.fit_transform(X[:tr]); ytr=y[:tr]
    Xva=sc.transform(X[tr:tr+ve]); yva=y[tr:tr+ve]
    Xte=sc.transform(X[tr+ve:]); yte=y[tr+ve:]
    rf=RandomForestClassifier(n_estimators=150,max_depth=7,min_samples_leaf=5,
                               max_features="sqrt",random_state=42,n_jobs=1)
    hgb=HistGradientBoostingClassifier(max_iter=150,max_depth=5,learning_rate=0.05,
                                        min_samples_leaf=10,random_state=42)
    et=ExtraTreesClassifier(n_estimators=100,max_depth=6,min_samples_leaf=5,
                             max_features="sqrt",random_state=42,n_jobs=1)
    vote=VotingClassifier(estimators=[("rf",rf),("hgb",hgb),("et",et)],
                           voting="soft",n_jobs=1)
    cal=CalibratedClassifierCV(vote,cv=3,method="isotonic",n_jobs=1)
    cal.fit(Xtr,ytr)
    pva=cal.predict_proba(Xva)[:,1]
    pte=cal.predict_proba(Xte)[:,1]
    try: av=round(float(roc_auc_score(yva,pva)),4)
    except Exception: av=0.0
    try: at=round(float(roc_auc_score(yte,pte)),4)
    except Exception: at=0.0
    st.session_state["auc_val"]=av
    st.session_state["auc_tst"]=at
    fi=np.zeros(len(FCOLS))
    try:
        rff=cal.calibrated_classifiers_[0].estimator.estimators_[0][1]
        fi=rff.feature_importances_
    except Exception: pass
    Xtrva=sc.transform(X[:tr+ve]); ytrva=y[:tr+ve]
    rf2=RandomForestClassifier(n_estimators=150,max_depth=7,min_samples_leaf=5,
                                max_features="sqrt",random_state=42,n_jobs=1)
    hgb2=HistGradientBoostingClassifier(max_iter=150,max_depth=5,learning_rate=0.05,
                                         min_samples_leaf=10,random_state=42)
    et2=ExtraTreesClassifier(n_estimators=100,max_depth=6,min_samples_leaf=5,
                              max_features="sqrt",random_state=42,n_jobs=1)
    vote2=VotingClassifier(estimators=[("rf",rf2),("hgb",hgb2),("et",et2)],
                            voting="soft",n_jobs=1)
    cal2=CalibratedClassifierCV(vote2,cv=3,method="isotonic",n_jobs=1)
    cal2.fit(Xtrva,ytrva)
    return cal2,sc,feat,fr,fi

def run_backtest(probs,rets,uth,leverage=1.0,tc=0.0005,margin_rate=0.06,sl=None,use_sizing=True):
    n=min(len(probs),len(rets))
    probs=np.nan_to_num(np.array(probs[:n],dtype=float),nan=0.5)
    rets=np.nan_to_num(np.array(rets[:n],dtype=float),nan=0.0)
    lth=1-uth
    pos=np.where(probs>=uth,1.0,np.where(probs<=lth,-1.0,0.0))
    if use_sizing:
        conf=np.abs(probs-0.5)*2; pos=pos*np.clip(conf,0.3,1.0)
    dm=margin_rate/252*max(leverage-1.0,0.0)
    eq=[1.0]; trades,wins=[],[]; in_tr,ent_eq,cur=False,1.0,0.0
    for i in range(n):
        p=pos[i]; r=rets[i]
        new_eq=eq[-1]*(1.0+p*r*leverage-dm*abs(p)-abs(p-cur)*tc)
        if sl is not None and in_tr:
            if new_eq/ent_eq-1<-(sl/leverage): new_eq=ent_eq*(1.0-sl/leverage); p=0.0; in_tr=False
        if p!=0.0 and not in_tr: in_tr=True; ent_eq=eq[-1]
        elif p==0.0 and in_tr:
            trades.append(new_eq/ent_eq-1.0); wins.append(new_eq>=ent_eq); in_tr=False
        eq.append(max(new_eq,1e-9)); cur=p
    eq=np.array(eq[1:]); yr=max(len(eq)/252,1e-9)
    cagr=float(eq[-1]**(1.0/yr)-1.0)
    d=np.diff(np.log(np.maximum(eq,1e-9)))
    sh=float(d.mean()/d.std()*math.sqrt(252)) if d.std()>0 else 0.0
    rm=np.maximum.accumulate(eq); mdd=float((eq/rm-1.0).min())
    wr=float(np.mean(wins)) if wins else 0.0
    pros=[t for t in trades if t>0]; lss=[abs(t) for t in trades if t<0]
    pf=sum(pros)/sum(lss) if lss and sum(lss)>0 else 99.0
    return dict(cagr=cagr,sharpe=sh,max_dd=mdd,final_eq=float(eq[-1]),
                equity=eq,pos=pos,rets=pos*rets,win_rate=wr,
                profit_factor=min(float(pf),50.0),
                avg_trade=float(np.mean(trades)) if trades else 0.0,n_trades=len(trades))

def best_threshold(pv,rv):
    bs,bu=-999.0,0.55
    for u in np.arange(0.50,0.68,0.01):
        try:
            bt=run_backtest(pv,rv,float(u),1.0,0.0,use_sizing=False)
            if bt["sharpe"]>bs and bt["n_trades"]>3: bs=bt["sharpe"]; bu=u
        except Exception: pass
    return round(float(bu),2)

def sig_info(p,u):
    if p>=u: return "LONG","#00c7b7","🟢"
    elif p<=1-u: return "SHORT","#ff4b4b","🔴"
    return "FLAT","#888888","⚪"

def fmt_large(n):
    if not n: return "N/A"
    for d,s in [(1e12,"T"),(1e9,"B"),(1e6,"M"),(1e3,"K")]:
        if abs(n)>=d: return str(round(n/d,2))+s
    return str(round(n,2))

def monthly_pnl(dates,rets):
    dm=pd.DataFrame({"date":pd.to_datetime(dates),"ret":rets})
    dm["yr"]=dm["date"].dt.year; dm["mo"]=dm["date"].dt.month
    piv=dm.groupby(["yr","mo"])["ret"].sum().unstack(fill_value=0)*100
    mn={1:"Led",2:"Uno",3:"Bze",4:"Dub",5:"Kve",6:"Cvn",
        7:"Cvc",8:"Srp",9:"Zar",10:"Rib",11:"Lis",12:"Pro"}
    piv.columns=[mn.get(m,str(m)) for m in piv.columns]
    return piv.round(2)

# ════════════ SIDEBAR ═════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 AI Trader v6")
    st.caption("RF + HGB + ET | 35 features | 5Y dat")
    st.markdown("---")
    preset_label=st.selectbox("Instrument",list(PRESETS.keys()),
                               index=list(PRESETS.keys()).index("Apple (AAPL)"))
    preset_val=PRESETS.get(preset_label)
    custom=st.text_input("Vlastni ticker",value="")
    ticker=(custom.strip().upper() if custom.strip() else (preset_val if preset_val else "AAPL"))
    st.markdown("#### Graf")
    tf_sel=st.radio("Timeframe",list(TF_PARAMS.keys()),index=4,
                     horizontal=True,label_visibility="collapsed")
    st.markdown("#### Live Refresh")
    live_mode=st.toggle("Auto-refresh",value=False)
    ref_int=st.selectbox("Interval",["30s","1 min","5 min"],index=1,disabled=not live_mode)
    st.markdown("---")
    period_tr=st.selectbox("Trenovaci data",["2y","3y","5y","max"],index=2)
    leverage=st.slider("Paka",1.0,10.0,1.0,0.5)
    margin_rate=st.slider("Margin p.a. (%)",0.0,12.0,6.0,0.5)/100
    use_sl=st.toggle("Stop-Loss",value=True)
    sl_val=st.slider("Stop-Loss (%)",1,20,5)/100 if use_sl else None
    use_sizing=st.toggle("Confidence sizing",value=True)
    tc_bps=st.number_input("Naklady (bps)",0,50,5)
    auto_uth=st.toggle("Auto-prah",value=True)
    man_uth=st.slider("Rucni prah",0.50,0.70,0.55,0.01)

tc=tc_bps/10000

# ════════════ HEADER ══════════════════════════════════════════════
h1,h2=st.columns([5,1])
with h1: st.markdown("## 🧠 "+ticker+" – Deep AI Trader")
with h2:
    if st.button("🔄 Refresh",use_container_width=True):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
st.caption("Aktualizace: "+datetime.now().strftime("%d.%m.%Y %H:%M:%S"))

with st.spinner("Trenuju model pro "+ticker+"..."):
    model,sc,feat,fr,fi=train_model(ticker,period_tr)
with st.spinner("Nacitam "+TF_PARAMS[tf_sel]["label"]+" data..."):
    df_chart=load_intraday(ticker,tf_sel)
    df_daily,info=load_daily(ticker,period_tr)

if model is None or feat is None:
    st.error("Nedostatek dat pro "+ticker+". Zkus jiny symbol nebo delsi obdobi.")
    st.stop()

X=feat[FCOLS].values; probs=model.predict_proba(sc.transform(X))[:,1]
N2=len(feat); vas=int(N2*0.80); vae=int(N2*0.90)
uth_=best_threshold(probs[vas:vae],fr.iloc[vas:vae].values[:vae-vas]) if auto_uth else man_uth
p_now=float(probs[-1]); lbl,color,icon=sig_info(p_now,uth_)
auc_v=st.session_state.get("auc_val",0.0); auc_t=st.session_state.get("auc_tst",0.0)

price_now=round(float(feat["close"].iloc[-1]),2)
price_prev=round(float(feat["close"].iloc[-2]),2)
price_chg=round((price_now/price_prev-1)*100,2)
rsi_now=round(float(feat["rsi"].iloc[-1]*100),1)
adx_now=round(float(feat["adx"].iloc[-1]*100),1)
bb_now=round(float(feat["bb"].iloc[-1]),2)
above_ma=bool(feat["above_ma200"].iloc[-1]>0.5)
regime=bool(feat["regime"].iloc[-1]>0.5)

m1,m2,m3,m4,m5,m6,m7,m8=st.columns(8)
m1.metric("Cena",str(price_now),delta=str(price_chg)+"%")
m2.metric("AI Signal",icon+" "+lbl,delta="p="+str(round(p_now*100,1))+"%")
m3.metric("RSI (14)",str(rsi_now))
m4.metric("ADX",str(adx_now))
m5.metric("Nad MA200","ANO" if above_ma else "NE")
m6.metric("Bull Regime","ANO" if regime else "NE")
m7.metric("AUC Val.",str(auc_v))
m8.metric("AUC Test",str(auc_t))
if info:
    f1,f2,f3,f4,f5,f6=st.columns(6)
    f1.metric("Trzni kap.",fmt_large(info.get("mktcap",0)))
    f2.metric("P/E",str(round(info.get("pe",0),1)) if info.get("pe") else "N/A")
    f3.metric("Beta",str(round(info.get("beta",0),2)) if info.get("beta") else "N/A")
    f4.metric("52T Max",str(round(info.get("52w_high",0),2)) if info.get("52w_high") else "N/A")
    f5.metric("52T Min",str(round(info.get("52w_low",0),2)) if info.get("52w_low") else "N/A")
    f6.metric("Cil. cena",str(round(info.get("target_price",0),2)) if info.get("target_price") else "N/A")
    sec_s=(info.get("sector","")+"/"+info.get("industry","")).strip("/")
    if sec_s: st.caption("Sektor: "+sec_s)
st.markdown("---")

PAGE=st.radio("Stranka",["Graf","Feature Importance","Backtest","Trh & Sektor","Screener"],
               horizontal=True,label_visibility="collapsed")
st.markdown("---")

# ════════════ PAGE: GRAF ══════════════════════════════════════════
if PAGE=="Graf":
    use_df=df_chart if not df_chart.empty else df_daily
    disp_n=TF_PARAMS[tf_sel]["bars"]; disp=use_df.tail(disp_n).reset_index(drop=True)
    cd=disp["Close"]
    ma20=cd.rolling(20).mean(); ma50=cd.rolling(50).mean()
    bb_mid=cd.rolling(20).mean(); bb_std=cd.rolling(20).std()
    bb_up=bb_mid+2*bb_std; bb_dn=bb_mid-2*bb_std
    rsi_d=_rsi(cd); vol_d=disp.get("Volume",pd.Series(1.0,index=disp.index))
    pivots=pivot_levels(disp["High"],disp["Low"],disp["Close"])
    if tf_sel=="1d":
        dp=probs[max(0,N2-disp_n):]
        pa=np.where(dp>=uth_,1,np.where(dp<=1-uth_,-1,0))
        li=np.where(pa==1)[0]; si=np.where(pa==-1)[0]
    else:
        dp=None; li=np.array([]); si=np.array([])
    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[0.52,0.16,0.16,0.16],
                       vertical_spacing=0.02,
                       subplot_titles=("Cena ["+TF_PARAMS[tf_sel]["label"]+"]","Volume","RSI","AI Prob"))
    fig.add_trace(go.Candlestick(x=disp["date"],open=disp["Open"],high=disp["High"],
        low=disp["Low"],close=disp["Close"],
        increasing_line_color="#00c7b7",decreasing_line_color="#ff4b4b",name="Cena"),row=1,col=1)
    fig.add_trace(go.Scatter(x=disp["date"],y=ma20,name="MA20",
                              line=dict(color="#ffd700",width=1.2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=disp["date"],y=ma50,name="MA50",
                              line=dict(color="#ff9900",width=1.2,dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=disp["date"],y=bb_up,name="BB+",
                              line=dict(color="rgba(100,100,255,.5)",width=.8)),row=1,col=1)
    fig.add_trace(go.Scatter(x=disp["date"],y=bb_dn,name="BB-",
                              line=dict(color="rgba(100,100,255,.5)",width=.8),
                              fill="tonexty",fillcolor="rgba(100,100,255,.05)"),row=1,col=1)
    for ln,lv in pivots.items():
        lc="#00c7b7" if "R" in ln else ("#ff4b4b" if "S" in ln else "#fff")
        fig.add_hline(y=lv,line_dash="dash",line_color=lc,line_width=.6,
                      annotation_text=ln,annotation_position="right",
                      annotation_font_size=9,row=1,col=1)
    if tf_sel=="1d" and len(li):
        fig.add_trace(go.Scatter(x=disp["date"].iloc[li],y=disp["Low"].iloc[li]*.995,
            mode="markers",marker=dict(symbol="triangle-up",size=10,color="#00c7b7"),
            name="LONG"),row=1,col=1)
    if tf_sel=="1d" and len(si):
        fig.add_trace(go.Scatter(x=disp["date"].iloc[si],y=disp["High"].iloc[si]*1.005,
            mode="markers",marker=dict(symbol="triangle-down",size=10,color="#ff4b4b"),
            name="SHORT"),row=1,col=1)
    vc=["#00c7b7" if c>=o else "#ff4b4b" for c,o in zip(disp["Close"],disp["Open"])]
    fig.add_trace(go.Bar(x=disp["date"],y=vol_d,marker_color=vc,name="Vol"),row=2,col=1)
    fig.add_trace(go.Scatter(x=disp["date"],y=rsi_d,
                              line=dict(color="#a78bfa",width=1.5),name="RSI"),row=3,col=1)
    fig.add_hline(y=70,line_dash="dash",line_color="#ff4b4b",line_width=.7,row=3,col=1)
    fig.add_hline(y=30,line_dash="dash",line_color="#00c7b7",line_width=.7,row=3,col=1)
    if dp is not None:
        fig.add_trace(go.Scatter(x=disp["date"],y=dp,
                                  line=dict(color="#f0c040",width=1.5),name="AI"),row=4,col=1)
        fig.add_hline(y=uth_,line_dash="dash",line_color="green",line_width=.7,row=4,col=1)
        fig.add_hline(y=1-uth_,line_dash="dash",line_color="red",line_width=.7,row=4,col=1)
    rb=[dict(count=1,label="1D",step="day",stepmode="backward"),
        dict(count=5,label="5D",step="day",stepmode="backward"),
        dict(count=1,label="1M",step="month",stepmode="backward"),
        dict(count=3,label="3M",step="month",stepmode="backward"),
        dict(count=6,label="6M",step="month",stepmode="backward"),
        dict(count=1,label="1Y",step="year",stepmode="backward"),
        dict(step="all",label="Max")]
    fig.update_layout(height=760,showlegend=False,paper_bgcolor="#0e1117",
                       plot_bgcolor="#0e1117",font_color="#fff",
                       margin=dict(t=30,b=10,l=10,r=80),
                       xaxis=dict(rangeselector=dict(buttons=rb,bgcolor="#1a1a2e",
                                                     activecolor="#00c7b7",
                                                     font=dict(color="#fff")),
                                  rangeslider=dict(visible=False)))
    st.plotly_chart(fig,use_container_width=True)
    sig_html=('<div style="padding:1.2rem;border-radius:12px;background:#1a1a2e;'
              'border-left:7px solid '+color+';">'
              '<b style="font-size:1.4rem;">'+icon+" "+lbl+"</b><br>"
              "Prob(up): <b>"+str(round(p_now*100,1))+"%</b> &nbsp;|&nbsp; "
              "Cena: <b>"+str(price_now)+"</b> &nbsp;|&nbsp; "
              "Datum: <b>"+feat["date"].iloc[-1].strftime("%d.%m.%Y")+"</b>"
              "</div>")
    st.markdown(sig_html,unsafe_allow_html=True)
    ca,cb=st.columns(2)
    with ca:
        st.markdown("**Pivot S&R**")
        st.dataframe(pd.DataFrame([{"Uroven":k,"Cena":v,
            "Typ":"Odpor" if "R" in k else ("Podpora" if "S" in k else "Pivot")}
            for k,v in pivots.items()]),use_container_width=True,hide_index=True)
    with cb:
        st.markdown("**Indikatory**")
        st.dataframe(pd.DataFrame([
            {"Ind":"RSI","Val":str(rsi_now),"Sig":"Preprod." if rsi_now<30 else ("Prekoup." if rsi_now>70 else "OK")},
            {"Ind":"ADX","Val":str(adx_now),"Sig":"Silny" if adx_now>25 else "Slaby"},
            {"Ind":"BB","Val":str(bb_now),"Sig":"Horni" if bb_now>.8 else ("Dolni" if bb_now<.2 else "Stred")},
            {"Ind":"AI Prob","Val":str(round(p_now*100,1))+"%","Sig":lbl},
            {"Ind":"MA200","Val":"Nad" if above_ma else "Pod","Sig":"Bull" if above_ma else "Bear"},
            {"Ind":"Rezim","Val":"Bull" if regime else "Bear","Sig":""},
        ]),use_container_width=True,hide_index=True)

# ════════════ PAGE: FEATURE IMPORTANCE ════════════════════════════
elif PAGE=="Feature Importance":
    st.subheader("Dulezitost indikatoru – "+ticker)
    st.info("Cim vyssi hodnota, tim vice model pouziva dany indikator pri rozhodovani LONG/SHORT/FLAT.")
    if fi is not None and len(fi)==len(FCOLS):
        fi_df=pd.DataFrame({"Feature":FCOLS,"Importance":fi}).sort_values("Importance",ascending=True)
        fig_fi=go.Figure(go.Bar(x=fi_df["Importance"],y=fi_df["Feature"],orientation="h",
            marker=dict(color=fi_df["Importance"],
                        colorscale=[[0,"#1a1a2e"],[.5,"#00c7b7"],[1.0,"#ffd700"]])))
        fig_fi.update_layout(height=700,paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                              font_color="#fff",margin=dict(t=20,l=10),
                              xaxis_title="Feature importance")
        st.plotly_chart(fig_fi,use_container_width=True)
        top5=fi_df.nlargest(5,"Importance")["Feature"].tolist()
        descs={"r1":"1denni výnos","r5":"5denni výnos","r20":"20denni výnos",
               "ma20":"Vzdalenost od MA20","ma50":"Vzdalenost od MA50",
               "ma200":"Vzdalenost od MA200","rsi":"RSI – prekoupenost",
               "adx":"ADX – sila trendu","macd":"MACD","bb":"Bollinger Band",
               "cci":"CCI","atr":"ATR – volatilita","regime":"Bull/Bear rezim",
               "above_ma200":"Nad/Pod MA200","hist_vol":"Historicka volatilita",
               "aroon_up":"Aroon Up","aroon_dn":"Aroon Down",
               "donch":"Donchian kanal","obv":"OBV momentum","vforce":"Volume Force"}
        st.markdown("**Top 5 pro "+ticker+":**")
        for f in top5:
            st.markdown("- **"+f+":** "+descs.get(f,f))
    else:
        st.warning("Feature importance neni dostupna.")

# ════════════ PAGE: BACKTEST ═══════════════════════════════════════
elif PAGE=="Backtest":
    st.subheader("Backtest | "+ticker+" | Paka: "+str(leverage)+"x")
    tes=int(N2*0.90)
    tp=probs[tes:]; tr_=fr.iloc[tes:tes+len(tp)].fillna(0).values
    bt1=run_backtest(tp,tr_,uth_,1.0,tc,margin_rate,sl_val,use_sizing)
    btL=run_backtest(tp,tr_,uth_,leverage,tc,margin_rate,sl_val,use_sizing)
    t1,t2=st.columns(2)
    with t1:
        st.markdown("##### 1x")
        c1,c2,c3=st.columns(3)
        c1.metric("CAGR",str(round(bt1["cagr"]*100,1))+"%")
        c2.metric("Sharpe",str(round(bt1["sharpe"],2)))
        c3.metric("Max DD",str(round(bt1["max_dd"]*100,1))+"%")
        c1b,c2b,c3b=st.columns(3)
        c1b.metric("Win rate",str(round(bt1["win_rate"]*100,1))+"%")
        c2b.metric("Profit F.",str(round(bt1["profit_factor"],2)))
        c3b.metric("Obchodu",str(bt1["n_trades"]))
    with t2:
        st.markdown("##### "+str(leverage)+"x")
        c1,c2,c3=st.columns(3)
        c1.metric("CAGR",str(round(btL["cagr"]*100,1))+"%",
                  delta=str(round((btL["cagr"]-bt1["cagr"])*100,1))+"%")
        c2.metric("Sharpe",str(round(btL["sharpe"],2)))
        c3.metric("Max DD",str(round(btL["max_dd"]*100,1))+"%",
                  delta=str(round((btL["max_dd"]-bt1["max_dd"])*100,1))+"%",delta_color="inverse")
        c1b,c2b,c3b=st.columns(3)
        c1b.metric("Win rate",str(round(btL["win_rate"]*100,1))+"%")
        c2b.metric("Profit F.",str(round(btL["profit_factor"],2)))
        c3b.metric("Final eq.",str(round(btL["final_eq"],3))+"x")
    all_r=fr.iloc[:N2].fillna(0).values
    ba1=run_backtest(probs,all_r,uth_,1.0,tc,margin_rate,sl_val,use_sizing)
    baL=run_backtest(probs,all_r,uth_,leverage,tc,margin_rate,sl_val,use_sizing)
    bh=feat["close"].values[:len(ba1["equity"])]/feat["close"].values[0]
    de=feat["date"].iloc[:len(ba1["equity"])].values
    fig_eq=go.Figure()
    fig_eq.add_trace(go.Scatter(x=de,y=baL["equity"],mode="lines",name=str(leverage)+"x",
                                 line=dict(color="#ffd700",width=2)))
    fig_eq.add_trace(go.Scatter(x=de,y=ba1["equity"],mode="lines",name="AI 1x",
                                 line=dict(color="#00c7b7",width=2)))
    fig_eq.add_trace(go.Scatter(x=de,y=bh,mode="lines",name="Buy&Hold",
                                 line=dict(color="#888",dash="dot",width=1)))
    fig_eq.add_vrect(x0=str(feat["date"].iloc[tes].date()),x1=str(feat["date"].iloc[-1].date()),
                     fillcolor="rgba(255,255,100,.06)",line_width=0,
                     annotation_text="Test",annotation_position="top left")
    fig_eq.update_layout(title="Equity curve",height=380,paper_bgcolor="#0e1117",
                          plot_bgcolor="#0e1117",font_color="#fff",margin=dict(t=45))
    st.plotly_chart(fig_eq,use_container_width=True)
    st.subheader("Mesicni P&L (%)")
    mp=monthly_pnl(feat["date"].iloc[:len(ba1["rets"])].values,ba1["rets"])
    fhm=px.imshow(mp,color_continuous_scale=["#ff4b4b","#0e1117","#00c7b7"],
                   color_continuous_midpoint=0,text_auto=True,aspect="auto")
    fhm.update_layout(height=280,paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                       font_color="#fff",margin=dict(t=10))
    st.plotly_chart(fhm,use_container_width=True)
    td=feat["date"].iloc[tes:tes+len(tp)].values
    pe=np.where(tp>=uth_,1,np.where(tp<=1-uth_,-1,0))
    edf=pd.DataFrame({"date":td,"prob_up":tp,
                       "signal":np.where(pe==1,"LONG",np.where(pe==-1,"SHORT","FLAT")),
                       "ret_1x":bt1["rets"],"ret_lev":btL["rets"]})
    st.download_button("Stahnout CSV",edf.to_csv(index=False).encode("utf-8"),
                        "backtest_"+ticker+".csv","text/csv")

# ════════════ PAGE: TRH & SEKTOR ══════════════════════════════════
elif PAGE=="Trh & Sektor":
    sector=info.get("sector","") if info else ""
    setf=SECTOR_ETFS.get(sector,"XLK")
    st.subheader("Srovnani – "+ticker+" vs SPY vs "+setf)
    with st.spinner("Nacitam..."):
        dspy,_=load_daily("SPY",period_tr); dsec,_=load_daily(setf,period_tr)
    if not dspy.empty and not dsec.empty:
        def nm(dff):
            cc=dff["Close"].dropna(); return cc/cc.iloc[0]
        comb=pd.concat([nm(df_daily).rename(ticker),nm(dspy).rename("SPY"),
                         nm(dsec).rename(setf)],axis=1).ffill().dropna()
        frr=go.Figure()
        for j,col in enumerate(comb.columns):
            frr.add_trace(go.Scatter(x=comb.index,y=comb[col],mode="lines",name=col,
                                      line=dict(color=["#00c7b7","#888","#ffd700"][j],width=2)))
        frr.update_layout(title="Normalizovana cena",height=340,paper_bgcolor="#0e1117",
                           plot_bgcolor="#0e1117",font_color="#fff",margin=dict(t=45))
        st.plotly_chart(frr,use_container_width=True)
        rs=comb[ticker]/comb["SPY"]
        frs=go.Figure()
        frs.add_trace(go.Scatter(x=rs.index,y=rs,mode="lines",name="RS",
                                  line=dict(color="#a78bfa",width=1.5)))
        frs.add_hline(y=1.0,line_dash="dash",line_color="#888")
        frs.update_layout(title="Rel. sila "+ticker+"/SPY",height=220,
                           paper_bgcolor="#0e1117",plot_bgcolor="#0e1117",
                           font_color="#fff",margin=dict(t=45))
        st.plotly_chart(frs,use_container_width=True)
        rows_p=[]
        for pn,pd_ in {"1T":5,"1M":21,"3M":63,"6M":126,"1Y":252,"2Y":504}.items():
            row_={"Obdobi":pn}
            for sym,d2 in [(ticker,df_daily),(setf,dsec),("SPY",dspy)]:
                if len(d2)>=pd_:
                    cc=d2["Close"].dropna()
                    row_[sym]=str(round(float(cc.iloc[-1]/cc.iloc[-pd_]-1)*100,1))+"%"
                else: row_[sym]="N/A"
            rows_p.append(row_)
        st.dataframe(pd.DataFrame(rows_p),use_container_width=True,hide_index=True)

# ════════════ PAGE: SCREENER ══════════════════════════════════════
elif PAGE=="Screener":
    st.subheader("Multi-ticker Screener")
    def_t="AAPL,MSFT,NVDA,TSLA,META,GOOGL,AMZN,AMD,PLTR,SPY,QQQ,GC=F,ETH-USD,BTC-USD"
    ti=st.text_area("Tickery (carkou)",value=def_t,height=70)
    if st.button("Spustit screener",type="primary"):
        tl=[t.strip().upper() for t in ti.split(",") if t.strip()]
        rows_s=[]; prog=st.progress(0)
        for idx_s,tk_s in enumerate(tl):
            prog.progress((idx_s+1)/len(tl),text="Analyzuji "+tk_s+"...")
            try:
                ds,is_=load_daily(tk_s,"1y")
                if ds.empty or len(ds)<150: continue
                fs,_=featurize(ds)
                if len(fs)<60: continue
                cs=fs["close"]
                ri=round(float(fs["rsi"].iloc[-1]*100),1)
                ai=round(float(fs["adx"].iloc[-1]*100),1)
                r1m=round(float(cs.iloc[-1]/cs.iloc[-22]-1)*100,1) if len(cs)>22 else 0
                r3m=round(float(cs.iloc[-1]/cs.iloc[-63]-1)*100,1) if len(cs)>63 else 0
                vv=round(float(fs["v20"].iloc[-1]*100*math.sqrt(252)),1)
                abv=bool(fs["above_ma200"].iloc[-1]>0.5)
                reg=bool(fs["regime"].iloc[-1]>0.5)
                if ri<30 and ai>20: sg="🟢 Potencial. LONG"
                elif ri>70 and ai>20: sg="🔴 Potencial. SHORT"
                elif ai<15: sg="⚪ Sideways"
                else: sg="⚪ Neutral"
                rows_s.append({"Ticker":tk_s,
                    "Sektor":is_.get("sector","N/A") if is_ else "N/A",
                    "P/E":str(round(is_.get("pe",0),1)) if is_ and is_.get("pe") else "N/A",
                    "1M":str(r1m)+"%","3M":str(r3m)+"%",
                    "RSI":ri,"ADX":ai,"Vol.":str(vv)+"%",
                    "MA200":"ANO" if abv else "NE",
                    "Rezim":"Bull" if reg else "Bear","Signal":sg})
            except Exception: continue
        prog.empty()
        if rows_s:
            st.dataframe(pd.DataFrame(rows_s),use_container_width=True,hide_index=True)
            st.download_button("Stahnout CSV",
                                pd.DataFrame(rows_s).to_csv(index=False).encode("utf-8"),
                                "screener.csv","text/csv")
        else: st.warning("Zadna data.")

if live_mode:
    ivs={"30s":30,"1 min":60,"5 min":300}
    time.sleep(ivs.get(ref_int,60)); st.rerun()

st.markdown("---")
st.caption("AI Trading Screener v6 | Pouze demonstrace – neni to financni poradenstvi.")
