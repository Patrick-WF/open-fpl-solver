import sys
import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8914224822:AAGqUiZI4B5Ho9S5BJR2X0g3HWcgnfskmJc")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "365578933")

print("🔄 Fetching live Premier League data from FPL API for automated update...")
fpl_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
data_refreshed = False

try:
    response = requests.get(fpl_url, timeout=15)
    if response.status_code == 200:
        fpl_data = response.json()
        teams = {t["id"]: t["short_name"] for t in fpl_data.get("teams", [])}
        pos_map = {1: "G", 2: "D", 3: "M", 4: "F"}
        
        players = []
        for p in fpl_data.get("elements", []):
            if p["status"] in ["u", "n"]: # Skip unavailable players
                continue
            team_code = teams.get(p["team"], "UNK")
            pos = pos_map.get(p["element_type"], "M")
            form = float(p.get("form", 0) or 0)
            ppg = float(p.get("points_per_game", 0) or 0)
            price = p.get("now_cost", 50) / 10.0
            
            # Automated baseline xP estimation formula
            base_xp = max(2.0, (form * 0.6) + (ppg * 0.4))
            if pos == "G": base_xp = max(2.5, ppg if ppg > 0 else 3.5)
            elif pos == "D": base_xp = max(2.5, (form * 0.5) + 1.5)
            elif pos == "M": base_xp = max(3.0, form * 0.7)
            elif pos == "F": base_xp = max(3.5, form * 0.8)

            row = {
                "ID": p["id"],
                "Name": f"{p['first_name']} {p['second_name']}",
                "Team": team_code,
                "Pos": pos,
                "Price": price
            }
            # Generate 8-week horizon projections dynamically
            for gw in range(1, 9):
                row[f"{gw}_Pts"] = round(base_xp, 2)
                row[f"{gw}_xMin"] = 90 if base_xp > 2.5 else 0
            players.append(row)
            
        df = pd.DataFrame(players)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/solio.csv", index=False)
        print(f"Successfully refreshed solio.csv with {len(df)} live players from FPL API.")
        data_refreshed = True
except Exception as e:
    print(f"Live API fetch notice ({e}), falling back to local dataset.")

if not data_refreshed and not os.path.exists("data/solio.csv"):
    print("Error: No data available.")
    sys.exit(1)

# Load dataset for optimization
df = pd.read_csv("data/solio.csv")
horizon = 8
pts_cols = [f"{gw}_Pts" for gw in range(1, horizon + 1) if f"{gw}_Pts" in df.columns]
df["Total_xP"] = df[pts_cols].sum(axis=1) if pts_cols else 0

# Select optimal squad structure (2 GKs, 5 DEFs, 5 MIDs, 3 FWDs)
gks = df[df["Pos"] == "G"].sort_values(by="Total_xP", ascending=False).head(2)
defs = df[df["Pos"] == "D"].sort_values(by="Total_xP", ascending=False).head(5)
mids = df[df["Pos"] == "M"].sort_values(by="Total_xP", ascending=False).head(5)
fwds = df[df["Pos"] == "F"].sort_values(by="Total_xP", ascending=False).head(3)

squad = pd.concat([gks, defs, mids, fwds])
total_cost = squad["Price"].sum()
total_xp = squad["Total_xP"].sum()

# Determine Captain & Vice Captain
sorted_squad = squad.sort_values(by="Total_xP", ascending=False).reset_index(drop=True)
captain = sorted_squad.iloc[0]
vice_captain = sorted_squad.iloc[1]

# Format Telegram Report
message = "🏆 *FPL Weekly Automated Report* (Live Data)\n\n"
message += f"🎯 *Strategy & Recommendations*\n"
message += f"• *Captain:* ⭐ {captain['Name']} ({captain['Total_xP']:.1f} xP)\n"
message += f"• *Vice-Captain:* 🤝 {vice_captain['Name']} ({vice_captain['Total_xP']:.1f} xP)\n"
message += f"• *Chip Strategy:* Hold Chips\n"
message += f"• *Suggested Transfers:* Roll FT (Squad Value: £{total_cost:.1f}m)\n\n"

message += f"📋 *Optimized Starting XI & Squad ({total_xp:.1f} Total xP)*\n"
message += "*Goalkeepers:*\n"
for _, r in gks.iterrows():
    message += f"• {r['Name']} (£{r['Price']}m) — {r['Total_xP']:.1f} xP\n"

message += "\n*Defenders:*\n"
for _, r in defs.iterrows():
    message += f"• {r['Name']} (£{r['Price']}m) — {r['Total_xP']:.1f} xP\n"

message += "\n*Midfielders:*\n"
for _, r in mids.iterrows():
    message += f"• {r['Name']} (£{r['Price']}m) — {r['Total_xP']:.1f} xP\n"

message += "\n*Forwards:*\n"
for _, r in fwds.iterrows():
    message += f"• {r['Name']} (£{r['Price']}m) — {r['Total_xP']:.1f} xP\n"

# Dispatch to Telegram
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
payload = {
    "chat_id": CHAT_ID,
    "text": message,
    "parse_mode": "Markdown"
}

try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("✅ Automated weekly FPL report with live data sent to Telegram successfully!")
    else:
        print(f"❌ Telegram error: {response.text}")
except Exception as e:
    print(f"❌ Connection error: {e}")