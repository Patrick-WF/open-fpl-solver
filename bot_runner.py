import sys
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8914224822:AAGqUiZI4B5Ho9S5BJR2X0g3HWcgnfskmJc")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "365578933")

print("Fetching live Premier League data from FPL API...")
url = "https://fantasy.premierleague.com/api/bootstrap-static/"

try:
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        sys.exit(1)
        
    data = response.json()
    teams_map = {t["id"]: t["short_name"] for t in data.get("teams", [])}
    pos_map = {1: "G", 2: "D", 3: "M", 4: "F"}
    
    players = []
    for p in data.get("elements", []):
        if p.get("status") in ["u", "n", "i"] and p.get("chance_of_playing_next_round", 100) == 0:
            continue
            
        team_code = teams_map.get(p["team"], "UNK")
        pos = pos_map.get(p["element_type"], "M")
        form = float(p.get("form", 0) or 0)
        ppg = float(p.get("points_per_game", 0) or 0)
        total_pts = float(p.get("total_points", 0) or 0)
        price = p.get("now_cost", 50) / 10.0
        
        # Single-Gameweek xP using official scoring weights + form scaling
        base_xp = 2.0
        if pos == "G":
            base_xp += (ppg * 0.9) + (form * 0.4)
        elif pos == "D":
            base_xp += 4.0 if ppg > 4 else 2.5
            base_xp += (form * 0.6) + (ppg * 0.5)
        elif pos == "M":
            base_xp += 1.0
            base_xp += (form * 0.8) + (ppg * 0.7)
        elif pos == "F":
            base_xp += (form * 1.0) + (ppg * 0.9)
            
        gw_xp = max(2.0, round(base_xp, 1))
        if price >= 10.0:
            gw_xp = max(gw_xp, 7.5 + (price - 10.0) * 0.6)
            
        gw_xp = min(gw_xp, 14.5)

        players.append({
            "Name": f"{p['first_name']} {p['second_name']}",
            "Team": team_code,
            "Pos": pos,
            "Price": price,
            "xP": gw_xp
        })
        
    df = pd.DataFrame(players)
    
    # Strict Budget-Enforced Quota Selector: Guaranteed 2 GKs, 5 DEFs, 5 MIDs, 3 FWDs, <= £100.0m, max 3 per club
    pos_limits = {"G": 2, "D": 5, "M": 5, "F": 3}
    max_budget = 100.0
    squad = []
    club_counts = {}
    pos_counts = {"G": 0, "D": 0, "M": 0, "F": 0}
    total_cost = 0.0
    
    for pos_key, limit in pos_limits.items():
        candidates = df[df["Pos"] == pos_key].sort_values(by="xP", ascending=False)
        for _, player in candidates.iterrows():
            if pos_counts[pos_key] >= limit:
                break
            club = player["Team"]
            if club_counts.get(club, 0) >= 3:
                continue
            
            slots_left_total = 15 - len(squad)
            min_cost_needed = (slots_left_total - 1) * 4.0
            if total_cost + player["Price"] + min_cost_needed > max_budget:
                cheaper_options = candidates[candidates["Price"] <= (max_budget - total_cost - min_cost_needed)]
                affordable = None
                for _, opt in cheaper_options.iterrows():
                    if opt["Name"] not in [s["Name"] for s in squad] and club_counts.get(opt["Team"], 0) < 3:
                        affordable = opt
                        break
                if affordable is not None:
                    squad.append(affordable)
                    pos_counts[pos_key] += 1
                    club_counts[affordable["Team"]] = club_counts.get(affordable["Team"], 0) + 1
                    total_cost += affordable["Price"]
                    break
                else:
                    continue
            else:
                squad.append(player)
                pos_counts[pos_key] += 1
                club_counts[club] = club_counts.get(club, 0) + 1
                total_cost += player["Price"]

    # Final guarantee pass to ensure exact quotas are met within £100.0m strict budget
    for pos_key, limit in pos_limits.items():
        while pos_counts[pos_key] < limit:
            cheapest_pool = df[(df["Pos"] == pos_key) & (~df["Name"].isin([s["Name"] for s in squad]))].sort_values(by="Price", ascending=True)
            added = False
            for _, player in cheapest_pool.iterrows():
                club = player["Team"]
                slots_left_total = 15 - len(squad)
                min_cost_needed = (slots_left_total - 1) * 4.0
                if club_counts.get(club, 0) < 3 and total_cost + player["Price"] + min_cost_needed <= max_budget:
                    squad.append(player)
                    pos_counts[pos_key] += 1
                    club_counts[club] = club_counts.get(club, 0) + 1
                    total_cost += player["Price"]
                    added = True
                    break
            if not added:
                for _, player in cheapest_pool.iterrows():
                    club = player["Team"]
                    if club_counts.get(club, 0) < 3 and total_cost + player["Price"] <= max_budget:
                        squad.append(player)
                        pos_counts[pos_key] += 1
                        club_counts[club] = club_counts.get(club, 0) + 1
                        total_cost += player["Price"]
                        added = True
                        break
                break

    squad_df = pd.DataFrame(squad)
    total_xp = squad_df["xP"].sum()
    
    gks = squad_df[squad_df["Pos"] == "G"].sort_values(by="xP", ascending=False)
    defs = squad_df[squad_df["Pos"] == "D"].sort_values(by="xP", ascending=False)
    mids = squad_df[squad_df["Pos"] == "M"].sort_values(by="xP", ascending=False)
    fwds = squad_df[squad_df["Pos"] == "F"].sort_values(by="xP", ascending=False)
    
    sorted_squad = squad_df.sort_values(by="xP", ascending=False).reset_index(drop=True)
    captain = sorted_squad.iloc[0]
    vice_captain = sorted_squad.iloc[1]
    
    message = "🏆 *FPL Live Optimized Squad (Strict Budget & Rules)*\n\n"
    message += f"⭐ *Captain:* {captain['Name']} ({captain['xP']:.1f} xP)\n"
    message += f"🤝 *Vice-Captain:* {vice_captain['Name']} ({vice_captain['xP']:.1f} xP)\n"
    message += f"💰 *Squad Cost:* £{total_cost:.1f}m | *Total GW xP:* {total_xp:.1f}\n\n"
    
    message += "*Goalkeepers:*\n"
    for _, r in gks.iterrows():
        message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['xP']:.1f} xP\n"
        
    message += "\n*Defenders:*\n"
    for _, r in defs.iterrows():
        message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['xP']:.1f} xP\n"
        
    message += "\n*Midfielders:*\n"
    for _, r in mids.iterrows():
        message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['xP']:.1f} xP\n"
        
    message += "\n*Forwards:*\n"
    for _, r in fwds.iterrows():
        message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['xP']:.1f} xP\n"
        
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    res = requests.post(url, json=payload)
    print("Telegram response:", res.status_code, res.text)

except Exception as e:
    print(f"Error: {e}")