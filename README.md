# 📈 AI Trading Screener

Kompletní Streamlit aplikace s natrénovaným AI modelem (LogisticRegression)
pro 5 instrumentů: SPY, Gold, Silver, ETH, SOL.

## 📁 Soubory
| Soubor | Popis |
|--------|-------|
| app.py | Hlavní Streamlit aplikace |
| model_*.pkl | Natrénované ML modely (1 per instrument) |
| scaler_*.pkl | Normalizační scalery |
| raw_*.csv | Trénovací data (záložní) |
| backtest_*.csv | Backtestové equity křivky |
| all_results.json | Souhrn výsledků |

## 🚀 Spuštění
```bash
# 1. Nainstaluj závislosti
pip install -r requirements.txt

# 2. Zkopíruj VŠECHNY .pkl a .csv soubory do stejné složky jako app.py

# 3. Spusť
streamlit run app.py
```

## ⚙️ Jak model funguje
- **Model:** LogisticRegression (scikit-learn)  
- **Featury:** 1/5/20-denní výnosy, MA5/MA20, volatilita, změna objemu  
- **Target:** 1 = cena příští den vzroste, 0 = klesne  
- **Signál:** LONG pokud p(↑) ≥ práh, SHORT pokud p(↑) ≤ (1 − práh), jinak FLAT  
- **Práh:** nastavitelný v Streamlit UI (default 0.52)

## ⚠️ Upozornění
Modely jsou natrénované na syntetických datech (GBM simulace).
Aplikace slouží POUZE k demonstraci – není finančním poradenstvím.
