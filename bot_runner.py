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
    
    # 1. Select 15-player squad strictly <= £100.0m with exact quotas (2 GKs, 5 DEFs, 5 MIDs, 3 FWDs, max 3 per club)
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

    # Final guarantee pass for 15-player squad
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
    
    # 2. Select Optimal Starting XI from the 15-player squad
    # Rules: 1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD (Total 11 players)
    starting_xi = []
    bench = []
    
    # Pick best GK for Starting XI
    gks_in_squad = squad_df[squad_df["Pos"] == "G"].sort_values(by="xP", ascending=False)
    starting_xi.append(gks_in_squad.iloc[0])
    bench.append(gks_in_squad.iloc[1])
    
    outfielders = squad_df[squad_df["Pos"] != "G"].sort_values(by="xP", ascending=False)
    
    def_count = 0
    mid_count = 0
    fwd_count = 0
    
    xi_outfielders = []
    bench_outfielders = []
    
    for _, player in outfielders.iterrows():
        pos = player["Pos"]
        if pos == "D":
            if def_count < 5 and (def_count < 3 or len(xi_outfielders) < 10):
                xi_outfielders.append(player)
                def_count += 1
            else:
                bench_outfielders.append(player)
        elif pos == "M":
            if mid_count < 5 and (mid_count < 3 or len(xi_outfielders) < 10):
                xi_outfielders.append(player)
                mid_count += 1
            else:
                bench_outfielders.append(player)
        elif pos == "F":
            if fwd_count < 3 and (fwd_count < 1 or len(xi_outfielders) < 10):
                xi_outfielders.append(player)
                fwd_count += 1
            else:
                bench_outfielders.append(player)
                
    starting_xi.extend(xi_outfielders)
    bench.extend(bench_outfielders)
    
    xi_df = pd.DataFrame(starting_xi)
    total_xi_xp = xi_df["xP"].sum()
    
    # Captain & Vice-Captain from Starting XI
    sorted_xi = xi_df.sort_values(by="xP", ascending=False).reset_index(drop=True)
    captain = sorted_xi.iloc[0]
    vice_captain = sorted_xi.iloc[1]
    
    # Format Telegram Message
    message = "🏆 *FPL Live Optimized Starting XI & Squad*\n\n"
    message += f"⭐ *Captain:* {captain['Name']} ({captain['xP']:.1f} xP)\n"
    message += f"🤝 *Vice-Captain:* {vice_captain['Name']} ({vice_captain['xP']:.1f} xP)\n"
    message += f"💰 *Squad Cost:* £{total_cost:.1f}m | *Starting XI xP:* {total_xi_xp:.1f}\n\n"
    
    message += "⚽ *Starting XI*\n"
    message += f"• *GK:* {starting_xi[0]['Name']} ({starting_xi[0]['Team']}) - £{starting_xi[0]['Price']}m | {starting_xi[0]['xP']} xP\n"
    
    xi_defs = [p for p in starting_xi if p['Pos'] == 'D']
    xi_mids = [p for p in starting_xi if p['Pos'] == 'M']
    xi_fwds = [p for p in starting_xi if p['Pos'] == 'F']
    
    message += "*Defenders:*\n"
    for r in xi_defs:
        message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['xP']} xP\n"
        
    message += "*Midfielders:*\n"
    for r in xi_mids:
        message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['xP']} xP\n"
        
    message += "*Forwards:*\n"
    for r in xi_fwds:
        message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['xP']} xP\n"
        
    message += "\n🛋️ *Substitutes (Bench)*\n"
    for r in bench:
        message += f"• [{r['Pos']}] {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['xP']} xP\n"
        
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    res = requests.post(url, json=payload)
    print("Telegram response:", res.status_code, res.text)

except Exception as e:
    print(f"Error: {e}")