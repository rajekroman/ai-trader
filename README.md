# 📈 AI Trading Screener v2

**Ziva data z Yahoo Finance + RandomForest (300 stromu, 12 featur)**

## Spusteni
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Featury modelu
| Featura | Popis |
|---------|-------|
| r1/r5/r20 | 1/5/20-denni vynosy |
| ma5/ma20 | Klouzave prumery |
| v5/v20 | Kратkodobá / strednedoba volatilita |
| vc | Zmena objemu |
| rsi | RSI (14) normalizovany |
| bb_pos | Pozice v Bollingerových pasmech |
| macd_diff | MACD – Signal |
| atr_ratio | ATR / Close (normalizovana volatilita) |

## Instrumenty
- SPY – S&P 500 ETF
- GC=F – Zlato
- SI=F – Stribro
- ETH-USD – Ethereum
- SOL-USD – Solana

## Upozorneni
Pouze demonstrace – neni to financni poradenstvi.
