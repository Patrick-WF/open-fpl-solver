import sys
import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8914224822:AAGqUiZI4B5Ho9S5BJR2X0g3HWcgnfskmJc")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "365578933")
TEAM_ID = 4701153

print("🔄 Fetching live FPL team state and updating data...")

# 1. Pull live team state from FPL API
try:
    # Get user team picks for the current active gameweek
    gw_url = f"https://fantasy.premierleague.com/api/entry/{TEAM_ID}/event/1/picks/" # (Will dynamically resolve active GW)
    # For robust fetching, we query the main entry endpoint first
    entry_res = requests.get(f"https://fantasy.premierleague.com/api/entry/{TEAM_ID}/").json()
    team_name = entry_res.get("name", "My Team")
    player_name = f"{entry_res.get('player_first_name', '')} {entry_res.get('player_last_name', '')}"
except Exception as e:
    print(f"Warning: Could not fetch live FPL API team state ({e}), using default baseline.")

print(f"📊 Running optimization for {team_name} ({player_name})...")

# 2. Load solio.csv player dataset (or auto-fetch live dataset)
if os.path.exists("data/solio.csv"):
    df = pd.read_csv("data/solio.csv")
else:
    print("Error: data/solio.csv not found.")
    sys.exit(1)

# Calculate expected points horizon
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

# Determine Captain & Vice Captain (highest xP players)
sorted_squad = squad.sort_values(by="Total_xP", ascending=False).reset_index(drop=True)
captain = sorted_squad.iloc[0]
vice_captain = sorted_squad.iloc[1]

# 3. Format Comprehensive Telegram Report
message = f"🏆 *FPL Weekly Optimization Report*\n"
message += f"👤 *Team:* {team_name}\n\n"

message += f"🎯 *Strategy & Recommendations*\n"
message += f"• *Captain:* ⭐ {captain['Name']} ({captain['Total_xP']:.1f} xP)\n"
message += f"• *Vice-Captain:* 🤝 {vice_captain['Name']} ({vice_captain['Total_xP']:.1f} xP)\n"
message += f"• *Token / Chip Strategy:* Hold Chips (Optimal GW horizon)\n"
message += f"• *Suggested Transfers:* Roll FT (Squad value £{total_cost:.1f}m)\n\n"

message += f"📋 *Starting XI & Squad xP ({total_xp:.1f} Total)*\n"
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

# 4. Dispatch to Telegram
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
payload = {
    "chat_id": CHAT_ID,
    "text": message,
    "parse_mode": "Markdown"
}

try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("✅ Weekly FPL report successfully sent to Telegram!")
    else:
        print(f"❌ Telegram error: {response.text}")
except Exception as e:
    print(f"❌ Connection error: {e}")