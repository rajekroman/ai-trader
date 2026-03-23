# AI Trader v6 - iphone postup (bez terminalu)

## Krok 1: Nahraj soubory na GitHub (pres mobil)

1. Otevri github.com -> prihlaseni -> tvuj repozitar ai-trader
2. Klikni "Add file" -> "Upload files"
3. Nahraj VSECHNY soubory z tohoto ZIPu:
   - app.py
   - train.py
   - requirements.txt
   - .github/workflows/train_models.yml
   - models/.gitkeep
4. Klikni "Commit changes"

## Krok 2: GitHub automaticky spusti trenink

- Po commitu se automaticky spusti GitHub Actions
- Trenovani trva ~10-15 minut
- Sleduj progress: repozitar -> zalozka "Actions"
- Po dokonceni Actions samy nahraji models/*.pkl zpet do repa

## Krok 3: Streamlit Cloud

- Streamlit Cloud detekuje zmeny v repu a znovu nasadi app
- App nacte predtrenovane modely -> funguje okamzite

## Kdyz chces rucne spustit trenink znovu

GitHub -> Actions -> "Train AI Models" -> "Run workflow" -> zelene tlacitko
