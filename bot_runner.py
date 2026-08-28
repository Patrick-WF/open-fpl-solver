import sys
import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8914224822:AAGqUiZI4B5Ho9S5BJR2X0g3HWcgnfskmJc")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "365578933")
TEAM_ID = 4701153

print("Fetching live Premier League data and user team from FPL API...")
url_bootstrap = "https://fantasy.premierleague.com/api/bootstrap-static/"

try:
    response = requests.get(url_bootstrap, timeout=15)
    if response.status_code != 200:
        sys.exit(1)
        
    data = response.json()
    teams_map = {t["id"]: t["short_name"] for t in data.get("teams", [])}
    pos_map = {1: "G", 2: "D", 3: "M", 4: "F"}
    
    elements_map = {}
    players = []
    for p in data.get("elements", []):
        p_id = p["id"]
        team_code = teams_map.get(p["team"], "UNK")
        pos = pos_map.get(p["element_type"], "M")
        form = float(p.get("form", 0) or 0)
        ppg = float(p.get("points_per_game", 0) or 0)
        price = p.get("now_cost", 50) / 10.0
        
        base_xp = 2.0
        if pos == "G": base_xp += (ppg * 0.9) + (form * 0.4)
        elif pos == "D": base_xp += 4.0 if ppg > 4 else 2.5; base_xp += (form * 0.6) + (ppg * 0.5)
        elif pos == "M": base_xp += 1.0; base_xp += (form * 0.8) + (ppg * 0.7)
        elif pos == "F": base_xp += (form * 1.0) + (ppg * 0.9)
            
        gw_xp = max(2.0, round(base_xp, 1))
        if price >= 10.0: gw_xp = max(gw_xp, 7.5 + (price - 10.0) * 0.6)
        gw_xp = min(gw_xp, 14.5)

        p_info = {
            "ID": p_id,
            "Name": f"{p['first_name']} {p['second_name']}",
            "Team": team_code,
            "Pos": pos,
            "Price": price,
            "xP": gw_xp,
            "status": p.get("status")
        }
        elements_map[p_id] = p_info
        if p.get("status") not in ["u", "n", "i"] or p.get("chance_of_playing_next_round", 100) != 0:
            players.append(p_info)
            
    df = pd.DataFrame(players)

    # 1. Fetch user's actual current live team picks from FPL API
    current_squad_objs = []
    current_squad_total_cost = 0.0
    try:
        event_url = f"https://fantasy.premierleague.com/api/entry/{TEAM_ID}/"
        entry_res = requests.get(event_url, timeout=10).json()
        current_gw = entry_res.get("current_event", 1) or 1
        
        picks_url = f"https://fantasy.premierleague.com/api/entry/{TEAM_ID}/event/{current_gw}/picks/"
        picks_res = requests.get(picks_url, timeout=10).json()
        for pick in picks_res.get("picks", []):
            pid = pick["element"]
            if pid in elements_map:
                obj = elements_map[pid]
                current_squad_objs.append(obj)
                current_squad_total_cost += obj["Price"]
        print(f"Successfully fetched user team picks for Gameweek {current_gw} ({len(current_squad_objs)} players).")
    except Exception as e:
        print(f"Could not fetch user picks ({e}), generating optimal baseline squad.")

    # Fallback if API picks didn't load properly: build a valid 15-player squad
    if len(current_squad_objs) < 15:
        pos_limits = {"G": 2, "D": 5, "M": 5, "F": 3}
        max_budget = 100.0
        squad = []
        club_counts = {}
        pos_counts = {"G": 0, "D": 0, "M": 0, "F": 0}
        total_cost = 0.0
        
        for pos_key, limit in pos_limits.items():
            candidates = df[df["Pos"] == pos_key].sort_values(by="xP", ascending=False)
            for _, player in candidates.iterrows():
                if pos_counts[pos_key] >= limit: break
                club = player["Team"]
                if club_counts.get(club, 0) >= 3: continue
                
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
                    squad.append(player)
                    pos_counts[pos_key] += 1
                    club_counts[club] = club_counts.get(club, 0) + 1
                    total_cost += player["Price"]

        # Guarantee pass for 15-player squad
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
        total_cost = squad_df["Price"].sum()
    else:
        squad_df = pd.DataFrame(current_squad_objs)
        total_cost = current_squad_total_cost

    # 2. Select Optimal Starting XI (11 players) and Bench (4 players) from YOUR squad
    starting_xi = []
    bench = []
    
    gks_in_squad = squad_df[squad_df["Pos"] == "G"].sort_values(by="xP", ascending=False).reset_index(drop=True)
    if len(gks_in_squad) >= 2:
        starting_xi.append(gks_in_squad.iloc[0].to_dict())
        bench.append(gks_in_squad.iloc[1].to_dict())
    else:
        for _, g in gks_in_squad.iterrows():
            starting_xi.append(g.to_dict())

    outfielders = squad_df[squad_df["Pos"] != "G"].sort_values(by="xP", ascending=False).reset_index(drop=True)
    xi_outfielders = []
    def_count, mid_count, fwd_count = 0, 0, 0
    
    for _, player in outfielders.iterrows():
        p_dict = player.to_dict()
        pos = p_dict["Pos"]
        if len(xi_outfielders) < 10:
            if pos == "D" and def_count < 5 and (def_count < 3 or (10 - len(xi_outfielders) > (3 - min(3, fwd_count)) + (3 - min(3, mid_count)))):
                xi_outfielders.append(p_dict); def_count += 1
            elif pos == "M" and mid_count < 5 and (mid_count < 3 or (10 - len(xi_outfielders) > (3 - min(3, fwd_count)) + (3 - min(3, def_count)))):
                xi_outfielders.append(p_dict); mid_count += 1
            elif pos == "F" and fwd_count < 3 and (fwd_count < 1 or (10 - len(xi_outfielders) > (3 - min(3, def_count)) + (3 - min(3, mid_count)))):
                xi_outfielders.append(p_dict); fwd_count += 1
        else:
            break
            
    while len(xi_outfielders) < 10:
        for _, player in outfielders.iterrows():
            p_dict = player.to_dict()
            if p_dict['Name'] not in [p['Name'] for p in xi_outfielders]:
                pos = p_dict['Pos']
                if pos == 'D' and def_count < 5: xi_outfielders.append(p_dict); def_count += 1; break
                elif pos == 'M' and mid_count < 5: xi_outfielders.append(p_dict); mid_count += 1; break
                elif pos == 'F' and fwd_count < 3: xi_outfielders.append(p_dict); fwd_count += 1; break

    starting_xi.extend(xi_outfielders)
    
    bench_outfielders = [p.to_dict() for _, p in outfielders.iterrows() if p['Name'] not in [x['Name'] for x in starting_xi]]
    bench.extend(bench_outfielders)
    
    xi_df = pd.DataFrame(starting_xi)
    total_xi_xp = xi_df["xP"].sum()
    
    sorted_xi = xi_df.sort_values(by="xP", ascending=False).reset_index(drop=True)
    captain = sorted_xi.iloc[0]
    vice_captain = sorted_xi.iloc[1]
    
    # 3. Intelligent Transfer & Bench Advice (Comparing your squad vs top available market options)
    transfers_advice = "Roll Free Transfer (Hold Current Squad) 🔄"
    if len(current_squad_objs) >= 15:
        market_df = df.sort_values(by="xP", ascending=False)
        current_names = {p["Name"] for p in current_squad_objs}
        
        best_gain = 0
        best_out = None
        best_in = None
        
        # Check if benching a low-xP starter in favor of a bench player is better than a -4 hit transfer
        lowest_starter = xi_df.sort_values(by="xP", ascending=True).iloc[0]
        best_bencher = pd.DataFrame(bench).sort_values(by="xP", ascending=False).iloc[0] if bench else None
        
        if best_bencher and best_bencher["xP"] > lowest_starter["xP"]:
            transfers_advice = f"Bench {lowest_starter['Name']} ({lowest_starter['xP']} xP) ➡️ Play {best_bencher['Name']} ({best_bencher['xP']} xP) instead of taking a hit"
        else:
            # Look at market upgrades from your current squad
            for _, curr_p in squad_df.iterrows():
                for _, mkt_p in market_df.iterrows():
                    if mkt_p["Name"] not in current_names and mkt_p["Pos"] == curr_p["Pos"]:
                        if mkt_p["Price"] <= (curr_p["Price"] + 0.5): # budget headroom check
                            gain = mkt_p["xP"] - curr_p["xP"]
                            if gain > best_gain:
                                best_gain = gain
                                best_out = curr_p["Name"]
                                best_in = mkt_p["Name"]
            
            # Apply 4-point hit penalty rule: gain must strictly exceed 4.0 xP to justify a hit beyond free transfer
            if best_out and best_in:
                if best_gain > 4.0:
                    transfers_advice = f"Transfer Out: {best_out} ➡️ Transfer In: {best_in} (Net Gain: +{best_gain:.1f} xP, justifies -4 hit)"
                elif best_gain > 1.5:
                    transfers_advice = f"Transfer Out: {best_out} ➡️ Transfer In: {best_in} (Free Transfer, +{best_gain:.1f} xP gain)"
                else:
                    transfers_advice = "Roll Free Transfer (Gains do not outweigh hit penalty) 🔄"

    # 4. Chip Strategy Evaluation
    bench_xp = sum([b['xP'] for b in bench])
    chip_advice = "Hold Chips 🛡️ (Save Free Hit / Wildcards)"
    if captain['xP'] >= 11.5: chip_advice = "Triple Captain Recommended 🚀"
    elif bench_xp >= 22.0: chip_advice = "Bench Boost Recommended 📈"
    
    # 5. Format Telegram Message Report
    message = "🏆 *FPL Weekly Team Manager Report*\n\n"
    message += f"⭐ *Captain:* {captain['Name']} ({captain['xP']:.1f} xP)\n"
    message += f"🤝 *Vice-Captain:* {vice_captain['Name']} ({vice_captain['xP']:.1f} xP)\n"
    message += f"🔄 *Suggested Move:* {transfers_advice}\n"
    message += f"🎯 *Chip Strategy:* {chip_advice}\n"
    message += f"💰 *Squad Cost:* £{total_cost:.1f}m | *Starting XI xP:* {total_xi_xp:.1f}\n\n"
    
    message += "⚽ *Starting XI (11)*\n"
    message += f"• *GK:* {starting_xi[0]['Name']} ({starting_xi[0]['Team']}) - £{starting_xi[0]['Price']}m | {starting_xi[0]['xP']} xP\n"
    
    for r in [p for p in starting_xi if p['Pos'] == 'D']: message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['xP']} xP\n"
    for r in [p for p in starting_xi if p['Pos'] == 'M']: message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['xP']} xP\n"
    for r in [p for p in starting_xi if p['Pos'] == 'F']: message += f"• {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['xP']} xP\n"
        
    message += "\n🛋️ *Substitutes (4)*\n"
    for r in bench: message += f"• [{r['Pos']}] {r['Name']} ({r['Team']}) - £{r['Price']}m | {r['xP']} xP\n"
        
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    res = requests.post(url, json=payload)
    print("Telegram response:", res.status_code)

except Exception as e:
    print(f"Error: {e}")