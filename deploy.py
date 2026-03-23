#!/usr/bin/env python3
"""
deploy.py – Automatické nasazení AI Trading Screener na GitHub
Spusť jedním příkazem: python deploy.py
"""

import os, base64, json, time
import urllib.request, urllib.error

# ──────────────────────────────────────────────
#  KONFIGURACE – vyplň tyto 2 hodnoty
# ──────────────────────────────────────────────
GITHUB_TOKEN   = "ZDE_VLOZ_SVUJ_TOKEN"   # viz návod níže
GITHUB_USER    = "ZDE_VLOZ_SVE_USERNAME"  # tvé GitHub uživatelské jméno
REPO_NAME      = "ai-trader"              # název repozitáře (může být cokoliv)
# ──────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
API_BASE   = "https://api.github.com"
HEADERS    = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Content-Type": "application/json",
}

SOUBORY = [
    "app.py", "requirements.txt", "README.md", ".gitignore",
    "all_results.json",
    "model_SPY.pkl","model_GOLD.pkl","model_SILVER.pkl","model_ETH.pkl","model_SOL.pkl",
    "scaler_SPY.pkl","scaler_GOLD.pkl","scaler_SILVER.pkl","scaler_ETH.pkl","scaler_SOL.pkl",
    "raw_SPY.csv","raw_GOLD.csv","raw_SILVER.csv","raw_ETH.csv","raw_SOL.csv",
    "backtest_SPY.csv","backtest_GOLD.csv","backtest_SILVER.csv","backtest_ETH.csv","backtest_SOL.csv",
]
STREAMLIT_CONFIG = (".streamlit/config.toml",
                    os.path.join(SCRIPT_DIR, ".streamlit", "config.toml"))


def api(method, path, data=None):
    url = API_BASE + path
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=HEADERS, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read()), r.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code


def upload_file(repo, path_in_repo, local_path):
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode()
    # check if file already exists (to get sha for update)
    existing, status = api("GET", f"/repos/{GITHUB_USER}/{repo}/contents/{path_in_repo}")
    payload = {"message": f"Add {path_in_repo}", "content": content}
    if status == 200 and "sha" in existing:
        payload["sha"] = existing["sha"]
    result, status = api("PUT", f"/repos/{GITHUB_USER}/{repo}/contents/{path_in_repo}", payload)
    return status in (200, 201)


def main():
    if GITHUB_TOKEN == "ZDE_VLOZ_SVUJ_TOKEN":
        print("\n❌  Nejdříve vyplň GITHUB_TOKEN a GITHUB_USER v tomto souboru!")
        print("    Návod: https://github.com/settings/tokens/new")
        print("    Zaškrtni: repo (full control)")
        return

    print(f"\n🚀 Nasazuji AI Trading Screener na GitHub ({GITHUB_USER}/{REPO_NAME})...\n")

    # 1. Vytvoř repozitář (pokud neexistuje)
    _, status = api("GET", f"/repos/{GITHUB_USER}/{REPO_NAME}")
    if status == 404:
        data = {"name": REPO_NAME, "description": "AI Trading Screener – SPY GOLD SILVER ETH SOL",
                "private": False, "auto_init": False}
        _, status = api("POST", "/user/repos", data)
        if status == 201:
            print(f"  ✅  Repozitář '{REPO_NAME}' vytvořen")
            time.sleep(2)
        else:
            print(f"  ❌  Nepodařilo se vytvořit repozitář (status {status})")
            return
    else:
        print(f"  ℹ️   Repozitář '{REPO_NAME}' již existuje, aktualizuji soubory...")

    # 2. Nahraj .streamlit/config.toml
    cfg_arc, cfg_local = STREAMLIT_CONFIG
    if os.path.isfile(cfg_local):
        ok = upload_file(REPO_NAME, cfg_arc, cfg_local)
        print(f"  {'✅' if ok else '❌'}  {cfg_arc}")

    # 3. Nahraj všechny ostatní soubory
    for fn in SOUBORY:
        local = os.path.join(SCRIPT_DIR, fn)
        if not os.path.isfile(local):
            print(f"  ⚠️   CHYBÍ: {fn}")
            continue
        ok = upload_file(REPO_NAME, fn, local)
        print(f"  {'✅' if ok else '❌'}  {fn}")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  ✅  HOTOVO! Všechny soubory jsou na GitHubu.            ║
║                                                          ║
║  Teď nasaď aplikaci:                                     ║
║  1. Jdi na https://share.streamlit.io                    ║
║  2. Přihlas se přes GitHub                               ║
║  3. New app → {GITHUB_USER}/{REPO_NAME} → app.py       ║
║  4. Deploy! → za 3 min máš URL pro iPhone               ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
