import sys
import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8914224822:AAGqUiZI4B5Ho9S5BJR2X0g3HWcgnfskmJc")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "365578933")

print("Running FPL Optimization & Squad Selection...")

if os.path.exists("data/solio.csv"):
    df = pd.read_csv("data/solio.csv")
else:
    print("Error: data/solio.csv not found.")
    sys.exit(1)

# Calculate horizon expected points
horizon = 8
pts_cols = [f"{gw}_Pts" for gw in range(1, horizon + 1) if f"{gw}_Pts" in df.columns]
df["Total_xP"] = df[pts_cols].sum(axis=1) if pts_cols else 0

# Sort players by xP descending to pick a valid squad respecting the 3-player per club limit
df_sorted = df.sort_values(by="Total_xP", ascending=False)

squad = []
club_counts = {}
pos_counts = {"G": 0, "D": 0, "M": 0, "F": 0}
pos_limits = {"G": 2, "D": 5, "M": 5, "F": 3}
max_budget = 100.0
total_cost = 0.0

for _, player in df_sorted.iterrows():
    pos = player["Pos"]
    club = player["Team"]
    price = player["Price"]
    
    # Check position limit
    if pos_counts[pos] >= pos_limits[pos]:
        continue
    # Check 3-player club limit
    if club_counts.get(club, 0) >= 3:
        continue
    # Check budget
    if total_cost + price > max_budget:
        if len(squad) < 14: # Allow flexibility for cheap bench fillers
            pass
        else:
            continue
            
    squad.append(player)
    pos_counts[pos] += 1
    club_counts[club] = club_counts.get(club, 0) + 1
    total_cost += price
    
    if len(squad) == 15:
        break

squad_df = pd.DataFrame(squad)
total_xp = squad_df["Total_xP"].sum()

# Separate positions for display
gks = squad_df[squad_df["Pos"] == "G"].sort_values(by="Total_xP", ascending=False)
defs = squad_df[squad_df["Pos"] == "D"].sort_values(by="Total_xP", ascending=False)
mids = squad_df[squad_df["Pos"] == "M"].sort_values(by="Total_xP", ascending=False)
fwds = squad_df[squad_df["Pos"] == "F"].sort_values(by="Total_xP", ascending=False)

sorted_squad = squad_df.sort_values(by="Total_xP", ascending=False).reset_index(drop=True)
captain = sorted_squad.iloc[0]
vice_captain = sorted_squad.iloc[1]

# Format Telegram Message
message = "🏆 *FPL Weekly Optimized Squad*\n\n"
message += f"⭐ *Captain:* {captain['Name']} ({captain['Total_xP']:.1f} xP)\n"
message += f"🤝 *Vice-Captain:* {vice_captain['Name']} ({vice_captain['Total_xP']:.1f} xP)\n"
message += f"💰 *Squad Cost:* £{total_cost:.1f}m | *Total xP:* {total_xp:.1f}\n\n"

message += "*Goalkeepers:*\n"
for _, r in gks.iterrows():
    message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['Total_xP']:.1f} xP\n"

message += "\n*Defenders:*\n"
for _, r in defs.iterrows():
    message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['Total_xP']:.1f} xP\n"

message += "\n*Midfielders:*\n"
for _, r in mids.iterrows():
    message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['Total_xP']:.1f} xP\n"

message += "\n*Forwards:*\n"
for _, r in fwds.iterrows():
    message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['Total_xP']:.1f} xP\n"

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
payload = {
    "chat_id": CHAT_ID,
    "text": message,
    "parse_mode": "Markdown"
}

res = requests.post(url, json=payload)
print("Telegram response:", res.status_code, res.text)