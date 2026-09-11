import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8914224822:AAGqUiZI4B5Ho9S5BJR2X0g3HWcgnfskmJc")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "365578933")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "365578933")
TEAM_ID = int(os.environ.get("FPL_TEAM_ID", "4701153"))
FPL_BANK_OVERRIDE = os.environ.get("FPL_BANK_MILLIONS")
FPL_FREE_TRANSFERS = int(os.environ.get("FPL_FREE_TRANSFERS", "1"))

FPL_BASE = "https://fantasy.premierleague.com/api"
REQUEST_TIMEOUT = 15
MAX_SQUAD_SIZE = 15
MAX_PLAYERS_PER_CLUB = 3
UNAVAILABLE_STATUSES = {"u", "n", "i", "s"}
MODEL_VERSION = "fixture-aware-v2"

# How much weight to place on recent gameweeks versus season baseline.
RECENT_GW_WINDOW = 5
RECENT_WEIGHT = 0.65
SEASON_WEIGHT = 0.35

# xG/xA are much more predictive of attacking involvement than raw recent points.
# These are deliberately moderate weights so the model doesn't overfit a tiny sample.


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def get_json(session: requests.Session, url: str, *, timeout: int = REQUEST_TIMEOUT):
    response = session.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 FPL-manager/2.0"},
    )
    response.raise_for_status()
    return response.json()


def parse_team_fdr(team: dict) -> float:
    return safe_float(team.get("strength", 3), 3.0)


def build_fixture_context(fixtures: List[dict], teams: Dict[int, dict], next_gw: int) -> Dict[int, dict]:
    """Create a per-club next-GW fixture context.

    Uses official FPL fixture difficulty when present, plus opponent strength and
    home/away status. For a double gameweek the individual fixture scores are
    combined rather than throwing one fixture away.
    """
    context: Dict[int, List[dict]] = {}

    for fixture in fixtures:
        event = fixture.get("event")
        if event != next_gw:
            continue
        if fixture.get("finished"):
            continue

        home = fixture.get("team_h")
        away = fixture.get("team_a")
        if home is None or away is None:
            continue

        team_h = teams.get(home, {})
        team_a = teams.get(away, {})

        # FPL publishes a 1-5 difficulty rating per side. We invert it into
        # a friendliness score: 1 = easiest, 5 = hardest.
        h_diff = safe_float(fixture.get("team_h_difficulty"), 3.0)
        a_diff = safe_float(fixture.get("team_a_difficulty"), 3.0)

        context.setdefault(home, []).append({
            "opponent": away,
            "home": True,
            "difficulty": h_diff,
            "opp_attack": safe_float(team_a.get("strength_attack_away"), parse_team_fdr(team_a)),
            "opp_defence": safe_float(team_a.get("strength_defence_away"), parse_team_fdr(team_a)),
            "own_attack": safe_float(team_h.get("strength_attack_home"), parse_team_fdr(team_h)),
            "own_defence": safe_float(team_h.get("strength_defence_home"), parse_team_fdr(team_h)),
        })
        context.setdefault(away, []).append({
            "opponent": home,
            "home": False,
            "difficulty": a_diff,
            "opp_attack": safe_float(team_h.get("strength_attack_home"), parse_team_fdr(team_h)),
            "opp_defence": safe_float(team_h.get("strength_defence_home"), parse_team_fdr(team_h)),
            "own_attack": safe_float(team_a.get("strength_attack_away"), parse_team_fdr(team_a)),
            "own_defence": safe_float(team_a.get("strength_defence_away"), parse_team_fdr(team_a)),
        })

    result: Dict[int, dict] = {}
    for team_id, games in context.items():
        # A lower difficulty is better. Convert to a 0.65-1.35 multiplier.
        difficulty_scores = [1.30 - (g["difficulty"] - 1.0) * 0.16 for g in games]
        avg_diff = sum(g["difficulty"] for g in games) / len(games)
        fixture_multiplier = sum(difficulty_scores) / len(difficulty_scores)

        home_boost = sum(0.045 if g["home"] else -0.015 for g in games) / len(games)
        attack_matchup = []
        defence_matchup = []
        for g in games:
            own_attack = max(g["own_attack"], 1.0)
            own_def = max(g["own_defence"], 1.0)
            opp_def = max(g["opp_defence"], 1.0)
            opp_attack = max(g["opp_attack"], 1.0)
            attack_matchup.append(own_attack / opp_def)
            defence_matchup.append(own_def / opp_attack)

        result[team_id] = {
            "fixtures": games,
            "n_fixtures": len(games),
            "avg_difficulty": avg_diff,
            "fixture_multiplier": fixture_multiplier + home_boost,
            "attack_matchup": sum(attack_matchup) / len(attack_matchup),
            "defence_matchup": sum(defence_matchup) / len(defence_matchup),
            "home_rate": sum(1 if g["home"] else 0 for g in games) / len(games),
        }

    return result


def history_features(history: List[dict]) -> dict:
    """Summarise recent player history without using raw recent points alone."""
    rows = [r for r in history if r.get("minutes", 0) and r.get("round")]
    rows = sorted(rows, key=lambda x: x.get("round", 0), reverse=True)[:RECENT_GW_WINDOW]
    if not rows:
        return {
            "recent_minutes": 0.0,
            "recent_points_per_90": 0.0,
            "recent_xg_per_90": 0.0,
            "recent_xa_per_90": 0.0,
            "recent_bps_per_90": 0.0,
            "starts_rate": 0.0,
            "clean_sheet_rate": 0.0,
            "goals_conceded_per_90": 0.0,
        }

    total_minutes = sum(safe_float(r.get("minutes")) for r in rows)
    total_minutes = max(total_minutes, 1.0)
    total_points = sum(safe_float(r.get("total_points")) for r in rows)
    goals = sum(safe_float(r.get("goals_scored")) for r in rows)
    assists = sum(safe_float(r.get("assists")) for r in rows)
    bps = sum(safe_float(r.get("bps")) for r in rows)
    xg = sum(safe_float(r.get("expected_goals")) for r in rows)
    xa = sum(safe_float(r.get("expected_assists")) for r in rows)
    clean_sheets = sum(1 for r in rows if safe_float(r.get("clean_sheets")) > 0)
    conceded = sum(safe_float(r.get("goals_conceded")) for r in rows)

    return {
        "recent_minutes": total_minutes / len(rows),
        "recent_points_per_90": total_points / total_minutes * 90,
        "recent_xg_per_90": xg / total_minutes * 90,
        "recent_xa_per_90": xa / total_minutes * 90,
        "recent_bps_per_90": bps / total_minutes * 90,
        "starts_rate": sum(1 for r in rows if safe_float(r.get("minutes")) >= 60) / len(rows),
        "clean_sheet_rate": clean_sheets / len(rows),
        "goals_conceded_per_90": conceded / total_minutes * 90,
    }


def minutes_probability(player: dict, recent: dict) -> float:
    status = player.get("status")
    chance = player.get("chance_of_playing_next_round")
    chance_factor = 1.0 if chance is None else max(0.0, min(1.0, safe_float(chance) / 100.0))

    # Recent starts provide a useful continuity signal; blend with official chance.
    starts = recent.get("starts_rate", 0.0)
    if recent.get("recent_minutes", 0) <= 0:
        starts = 0.70

    base = 0.45 + (0.45 * starts)
    if status == "d":
        base *= 0.70
    elif status in {"s", "u", "i", "n"}:
        base *= 0.10

    return max(0.05, min(1.0, 0.55 * base + 0.45 * chance_factor))


def calculate_enhanced_xp(player: dict, recent: dict, fixture: dict) -> float:
    """Project GW points from availability, underlying stats and fixtures."""
    pos = fixture["Pos"]
    price = fixture["Price"]

    ppg = safe_float(player.get("points_per_game"))
    form = safe_float(player.get("form"))
    xg90 = safe_float(player.get("expected_goals_per_90"))
    xa90 = safe_float(player.get("expected_assists_per_90"))
    xg = safe_float(player.get("expected_goals"))
    xa = safe_float(player.get("expected_assists"))
    minutes = safe_float(player.get("minutes"))
    starts = safe_float(player.get("starts"))
    threat = safe_float(player.get("threat"))
    creativity = safe_float(player.get("creativity"))
    ict = safe_float(player.get("ict_index"))

    # Bootstrap does not guarantee per-90 xG/xA fields, so use season totals as
    # fallback when needed.
    if xg90 <= 0 and minutes > 0:
        xg90 = xg / minutes * 90
    if xa90 <= 0 and minutes > 0:
        xa90 = xa / minutes * 90

    recent_xg90 = recent.get("recent_xg_per_90", 0.0)
    recent_xa90 = recent.get("recent_xa_per_90", 0.0)
    recent_pp90 = recent.get("recent_points_per_90", 0.0)

    blended_xg90 = RECENT_WEIGHT * recent_xg90 + SEASON_WEIGHT * xg90 if recent_xg90 else xg90
    blended_xa90 = RECENT_WEIGHT * recent_xa90 + SEASON_WEIGHT * xa90 if recent_xa90 else xa90
    blended_pp90 = RECENT_WEIGHT * recent_pp90 + SEASON_WEIGHT * ppg if recent_pp90 else ppg

    minutes_prob = minutes_probability(player, recent)

    fixture_mult = fixture["fixture_multiplier"]
    attack_mult = fixture["attack_matchup"]
    defence_mult = fixture["defence_matchup"]
    games = max(1, fixture["n_fixtures"])

    # Home/away and FPL difficulty are already represented in fixture_multiplier.
    # Attack/defence matchup adds information from published team strengths.
    attacking_signal = (blended_xg90 * 5.0 + blended_xa90 * 3.0)
    attacking_signal += max(0.0, blended_pp90 - 3.0) * 0.35
    attacking_signal += math.log1p(max(threat, 0)) * 0.10
    attacking_signal += math.log1p(max(creativity, 0)) * 0.06
    attacking_signal += math.log1p(max(ict, 0)) * 0.05

    # Position-specific base expected return. These aren't official FPL rates;
    # they provide a stable point scale around which player signals can move.
    if pos == "G":
        baseline = 2.2 + 1.15 * defence_mult + 0.35 * recent.get("clean_sheet_rate", 0.0)
        baseline += max(0.0, 0.08 * safe_float(player.get("saves")))
    elif pos == "D":
        baseline = 2.1 + 1.15 * defence_mult + 0.55 * attacking_signal
        baseline += 0.45 * recent.get("clean_sheet_rate", 0.0)
    elif pos == "M":
        baseline = 2.2 + 0.95 * attacking_signal
    else:
        baseline = 2.25 + 1.08 * attacking_signal

    # Form and recent points matter, but only as secondary signals.
    form_signal = 0.12 * max(0.0, form - 3.0) + 0.08 * max(0.0, blended_pp90 - 3.0)
    experience_signal = 0.05 * min(1.0, starts / max(1.0, minutes / 90.0)) if minutes > 0 else 0.0

    score = baseline * fixture_mult * (0.90 + 0.25 * attack_mult) + form_signal + experience_signal
    score *= minutes_prob

    # Double Gameweek gets a benefit, but not a simple 2x multiplier because
    # the minutes probability applies once to the player rather than to each fixture.
    if games >= 2:
        score *= 1.0 + 0.40 * (games - 1)

    # Keep numbers useful and comparable in the report without pretending they
    # are official FPL projections.
    return round(max(1.5, min(16.0, score)), 1)


def fetch_player_history(session: requests.Session, player_id: int) -> dict:
    return get_json(session, f"{FPL_BASE}/element-summary/{player_id}/")


def enrich_players(
    session: requests.Session,
    raw_players: List[dict],
    fixture_context: Dict[int, dict],
    teams: Dict[int, dict],
) -> pd.DataFrame:
    """Enrich the market using player histories, fetched concurrently.

    The FPL API's element-summary endpoint gives per-GW history and upcoming
    fixtures. We fetch the entire player pool with a conservative worker count.
    If an individual summary fails, the player remains in the pool using bootstrap
    data rather than disappearing silently.
    """
    results: Dict[int, dict] = {}
    max_workers = int(os.environ.get("FPL_HISTORY_WORKERS", "8"))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_player_history, session, p["id"]): p for p in raw_players}
        for future in as_completed(futures):
            p = futures[future]
            try:
                results[p["id"]] = future.result()
            except requests.RequestException:
                results[p["id"]] = {}

    output = []
    pos_map = {1: "G", 2: "D", 3: "M", 4: "F"}
    for p in raw_players:
        team_id = p["team"]
        pos = pos_map.get(p["element_type"], "M")
        team_info = teams.get(team_id, {})
        context = fixture_context.get(team_id)

        # If a club has no fixture entry for next GW, give the player a very
        # conservative placeholder instead of accidentally treating it as free.
        if not context:
            context = {
                "fixtures": [],
                "n_fixtures": 0,
                "avg_difficulty": 5.0,
                "fixture_multiplier": 0.70,
                "attack_matchup": 0.80,
                "defence_matchup": 0.80,
                "home_rate": 0.0,
            }

        history = results.get(p["id"], {})
        recent = history_features(history.get("history", []))
        fixture_for_player = {
            "Pos": pos,
            "Price": safe_float(p.get("now_cost"), 50.0) / 10.0,
            **context,
        }
        model_xp = calculate_enhanced_xp(p, recent, fixture_for_player)

        chance = p.get("chance_of_playing_next_round")
        status = p.get("status")
        output.append({
            "ID": p["id"],
            "Name": f"{p['first_name']} {p['second_name']}",
            "Team": team_info.get("short_name", "UNK"),
            "TeamID": team_id,
            "Pos": pos,
            "Price": safe_float(p.get("now_cost"), 50.0) / 10.0,
            "xP": model_xp,
            "status": status,
            "chance": 100 if chance is None else safe_float(chance),
            "form": safe_float(p.get("form")),
            "PPG": safe_float(p.get("points_per_game")),
            "xG": safe_float(p.get("expected_goals")),
            "xA": safe_float(p.get("expected_assists")),
            "minutes": safe_float(p.get("minutes")),
            "starts": safe_float(p.get("starts")),
            "fixture_difficulty": context["avg_difficulty"],
            "fixture_games": context["n_fixtures"],
            "fixture_mult": context["fixture_multiplier"],
            "recent_xG90": recent["recent_xg_per_90"],
            "recent_xA90": recent["recent_xa_per_90"],
            "recent_PP90": recent["recent_points_per_90"],
        })

    df = pd.DataFrame(output)
    # Only unavailable players are removed; doubtful players remain but receive
    # a reduced minutes projection.
    df = df[(~df["status"].isin(UNAVAILABLE_STATUSES)) & (df["chance"] != 0)].reset_index(drop=True)
    return df


def build_players_from_bootstrap(bootstrap: dict) -> Tuple[Dict[int, dict], List[dict], Dict[int, dict]]:
    teams = {t["id"]: t for t in bootstrap.get("teams", [])}
    elements_map = {p["id"]: p for p in bootstrap.get("elements", [])}
    return elements_map, bootstrap.get("elements", []), teams


def fetch_current_team(session: requests.Session, team_id: int, elements_map: Dict[int, dict]):
    entry = get_json(session, f"{FPL_BASE}/entry/{team_id}/")
    current_gw = entry.get("current_event")
    if not current_gw:
        raise RuntimeError("FPL entry API did not return a current gameweek.")

    picks = get_json(session, f"{FPL_BASE}/entry/{team_id}/event/{current_gw}/picks/")
    squad = []
    for pick in picks.get("picks", []):
        pid = pick.get("element")
        if pid not in elements_map:
            raise RuntimeError(f"Pick {pid} is missing from bootstrap data.")
        p = elements_map[pid]
        obj = {
            "ID": pid,
            "Name": f"{p['first_name']} {p['second_name']}",
            "TeamID": p["team"],
            "Pos": {1: "G", 2: "D", 3: "M", 4: "F"}.get(p["element_type"], "M"),
            "Price": safe_float(p.get("now_cost"), 50.0) / 10.0,
            "SellingPrice": safe_float(pick.get("selling_price"), safe_float(p.get("now_cost"), 50.0)) / 10.0,
            "multiplier": pick.get("multiplier", 1),
            "position": pick.get("position"),
            "status": p.get("status"),
        }
        squad.append(obj)

    if len(squad) != MAX_SQUAD_SIZE:
        raise RuntimeError(f"Expected {MAX_SQUAD_SIZE} players, received {len(squad)}.")

    history = get_json(session, f"{FPL_BASE}/entry/{team_id}/history/")
    used_chips = {c.get("name") for c in history.get("chips", []) if c.get("name")}
    return pd.DataFrame(squad), used_chips, {"entry": entry, "current_gw": current_gw, "history": history}


def validate_squad(squad_df: pd.DataFrame) -> None:
    if len(squad_df) != 15:
        raise ValueError(f"Squad must contain 15 players, found {len(squad_df)}.")
    counts = squad_df["Pos"].value_counts().to_dict()
    expected = {"G": 2, "D": 5, "M": 5, "F": 3}
    if counts != expected:
        raise ValueError(f"Squad must be 2/5/5/3, found {counts}.")
    if squad_df.groupby("TeamID").size().max() > MAX_PLAYERS_PER_CLUB:
        raise ValueError("Squad exceeds the three-player-per-club limit.")


def find_best_starting_xi(squad_df: pd.DataFrame):
    gks = squad_df[squad_df["Pos"] == "G"].sort_values("xP", ascending=False)
    defs = squad_df[squad_df["Pos"] == "D"].sort_values("xP", ascending=False)
    mids = squad_df[squad_df["Pos"] == "M"].sort_values("xP", ascending=False)
    fwds = squad_df[squad_df["Pos"] == "F"].sort_values("xP", ascending=False)

    best = None
    for d in range(3, 6):
        for m in range(2, 6):
            f = 10 - d - m
            if not 1 <= f <= 3:
                continue
            if len(defs) < d or len(mids) < m or len(fwds) < f:
                continue
            selected = pd.concat([gks.head(1), defs.head(d), mids.head(m), fwds.head(f)], ignore_index=True)
            score = float(selected["xP"].sum())
            candidate = (score, selected, f"{d}-{m}-{f}")
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        raise ValueError("No legal FPL formation could be generated.")

    xi = best[1]
    bench_df = squad_df[~squad_df["Name"].isin(set(xi["Name"]))].copy()
    bench_df = bench_df.sort_values(["xP", "Pos"], ascending=[True, True]).reset_index(drop=True)
    return xi, bench_df.to_dict("records"), best[2]


def find_best_transfer(squad_df, market_df, bank: Optional[float], free_transfers: int):
    if bank is None:
        return 0.0, None, None, "Roll Transfer (bank balance unavailable) 🔄"

    current_names = set(squad_df["Name"])
    club_counts = squad_df["TeamID"].value_counts().to_dict()
    best = (0.0, None, None)

    for _, out in squad_df.iterrows():
        sell_price = safe_float(out.get("SellingPrice"), safe_float(out["Price"]))
        candidates = market_df[(market_df["Pos"] == out["Pos"]) & (~market_df["Name"].isin(current_names))]
        for _, inc in candidates.iterrows():
            if inc["TeamID"] != out["TeamID"] and club_counts.get(inc["TeamID"], 0) >= MAX_PLAYERS_PER_CLUB:
                continue
            delta = float(inc["Price"]) - sell_price
            if delta > bank + 1e-9:
                continue
            gain = float(inc["xP"]) - float(out["xP"])
            if gain > best[0]:
                best = (gain, out.to_dict(), inc.to_dict())

    gain, out, inc = best
    if inc is None or gain <= 1.5:
        return gain, out, inc, "Roll Transfer (no sufficiently strong affordable improvement) 🔄"
    if free_transfers > 0:
        advice = f"Transfer Out: {out['Name']} ➡️ Transfer In: {inc['Name']} (Free Transfer, +{gain:.1f} model xP)"
    elif gain > 4.0:
        advice = f"Transfer Out: {out['Name']} ➡️ Transfer In: {inc['Name']} (Likely worth -4, +{gain:.1f} model xP)"
    else:
        advice = "Roll Transfer (gain does not justify a -4 hit) 🔄"
    return gain, out, inc, advice


def apply_transfer(squad_df, out_obj, in_obj, advice):
    if not out_obj or not in_obj or not ("Free Transfer" in advice or "Likely worth -4" in advice):
        return squad_df
    rows = [in_obj if r["Name"] == out_obj["Name"] else r for r in squad_df.to_dict("records")]
    return pd.DataFrame(rows)


def get_bank_and_free_transfers(entry: dict):
    if FPL_BANK_OVERRIDE is not None:
        bank = float(FPL_BANK_OVERRIDE)
    else:
        raw = entry.get("last_deadline_bank")
        bank = raw / 10.0 if isinstance(raw, (int, float)) else None
    return bank, max(0, FPL_FREE_TRANSFERS)


def select_captains(xi_df):
    ranked = xi_df.sort_values("xP", ascending=False).reset_index(drop=True)
    return ranked.iloc[0].to_dict(), ranked.iloc[1].to_dict()


def chip_advice(captain: dict, bench: List[dict], used_chips: set):
    bench_xp = sum(float(p["xP"]) for p in bench)
    if captain["xP"] >= 12.0 and "3xc" not in used_chips:
        return "Triple Captain Watch 🚀 (model captain ≥ 12.0)"
    if bench_xp >= 24.0 and "bboost" not in used_chips:
        return "Bench Boost Watch 📈 (bench ≥ 24.0 model xP)"
    if "3xc" in used_chips and captain["xP"] >= 12.0:
        return "Hold Chips 🛡️ (Triple Captain already used)"
    if "bboost" in used_chips and bench_xp >= 24.0:
        return "Hold Chips 🛡️ (Bench Boost already used)"
    return "Hold Chips 🛡️"


def format_report(xi, bench, captain, vice, transfer, chips, total_cost, formation, model_notes):
    msg = "🏆 *FPL Weekly Team Manager Report*\n\n"
    msg += f"🧠 *Model:* {MODEL_VERSION}\n"
    msg += f"⭐ *Captain:* {captain['Name']} ({captain['xP']:.1f} model xP)\n"
    msg += f"🤝 *Vice-Captain:* {vice['Name']} ({vice['xP']:.1f} model xP)\n"
    msg += f"🔄 *Suggested Move:* {transfer}\n"
    msg += f"🎯 *Chip Strategy:* {chips}\n"
    msg += f"💰 *Squad Cost:* £{total_cost:.1f}m | *XI model xP:* {xi['xP'].sum():.1f} | *Formation:* {formation}\n\n"
    msg += "⚽ *Starting XI*\n"
    for pos, label in [("G", "GK"), ("D", "DEF"), ("M", "MID"), ("F", "FWD")]:
        for _, p in xi[xi["Pos"] == pos].iterrows():
            msg += f"• *{label}:* {p['Name']} ({p['Team']}) - £{p['Price']:.1f}m | {p['xP']:.1f}\n"
    msg += "\n🛋️ *Substitutes*\n"
    for p in bench:
        msg += f"• [{p['Pos']}] {p['Name']} ({p['Team']}) - £{p['Price']:.1f}m | {p['xP']:.1f}\n"
    msg += "\n📌 *Projection notes:* " + model_notes
    return msg


def send_telegram(session, message):
    if not TOKEN or not CHAT_ID:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before sending a report.")
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    response = session.post(url, json={"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()


def main():
    print(f"Fetching live FPL data using {MODEL_VERSION}...")
    with requests.Session() as session:
        bootstrap = get_json(session, f"{FPL_BASE}/bootstrap-static/")
        elements_map, raw_players, teams = build_players_from_bootstrap(bootstrap)

        events = bootstrap.get("events", [])
        current_event = next((e for e in events if e.get("is_current")), None)
        next_event = next((e for e in events if e.get("is_next")), None)
        if current_event and next_event:
            target_gw = int(next_event["id"])
        elif current_event:
            target_gw = int(current_event["id"])
        else:
            target_gw = 1

        print(f"Projection target: GW{target_gw}")
        fixtures = get_json(session, f"{FPL_BASE}/fixtures/")
        fixture_context = build_fixture_context(fixtures, teams, target_gw)

        # Fetch the actual user's team first. This deliberately happens before
        # expensive player enrichment so an invalid team/endpoint fails quickly.
        squad_df, used_chips, meta = fetch_current_team(session, TEAM_ID, elements_map)
        validate_squad(squad_df)

        market_df = enrich_players(session, raw_players, fixture_context, teams)
        market_df = market_df.rename(columns={"Team": "Team"})

        # Enrich current squad from market model so its xP uses exactly the same
        # projection method as transfer candidates.
        model_lookup = market_df.set_index("ID").to_dict("index")
        for col in ["xP", "Team", "fixture_difficulty", "fixture_games", "fixture_mult", "recent_xG90", "recent_xA90", "recent_PP90"]:
            squad_df[col] = squad_df["ID"].map(lambda pid: model_lookup.get(pid, {}).get(col))
        if squad_df["xP"].isna().any():
            missing = squad_df.loc[squad_df["xP"].isna(), "Name"].tolist()
            raise RuntimeError(f"Could not build projection for current squad players: {missing}")

        bank, free_transfers = get_bank_and_free_transfers(meta["entry"])
        _, out_obj, in_obj, transfer_advice = find_best_transfer(squad_df, market_df, bank, free_transfers)
        updated_squad = apply_transfer(squad_df, out_obj, in_obj, transfer_advice)
        validate_squad(updated_squad)

        xi, bench, formation = find_best_starting_xi(updated_squad)
        captain, vice = select_captains(xi)
        chips = chip_advice(captain, bench, used_chips)
        total_cost = float(updated_squad["Price"].sum())

        fixtures_count = sum(1 for v in fixture_context.values() if v.get("n_fixtures"))
        notes = (
            f"Uses FPL fixture difficulty, home/away, team strength, player xG/xA, recent 5-GW history, "
            f"and availability. {fixtures_count} clubs have a GW{target_gw} fixture in the fixture feed. "
            "Scores are model projections, not official FPL xP."
        )

        report = format_report(xi, bench, captain, vice, transfer_advice, chips, total_cost, formation, notes)
        print("\n" + report)
        send_telegram(session, report)
        print("Successfully sent report to Telegram!")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"API/network error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Critical Error encountered: {exc}")
        sys.exit(1)
