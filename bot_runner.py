import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8914224822:AAGqUiZI4B5Ho9S5BJR2X0g3HWcgnfskmJc")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "365578933")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TEAM_ID = int(os.environ.get("FPL_TEAM_ID", "4701153"))

FPL_BANK_OVERRIDE = os.environ.get("FPL_BANK_MILLIONS")
FPL_FREE_TRANSFERS = int(
    os.environ.get("FPL_FREE_TRANSFERS", "1")
)

# Set this to true when using the Wildcard.
#
# Example:
# FPL_WILDCARD_THIS_WEEK=true
#
FPL_WILDCARD_THIS_WEEK = os.environ.get(
    "FPL_WILDCARD_THIS_WEEK",
    "true"
).lower() in {"1", "true", "yes", "y", "on"}

FPL_BASE = "https://fantasy.premierleague.com/api"

REQUEST_TIMEOUT = 15

MAX_SQUAD_SIZE = 15
MAX_PLAYERS_PER_CLUB = 3

UNAVAILABLE_STATUSES = {
    "u",
    "n",
    "i",
    "s",
}

MODEL_VERSION = "calibrated-v3"

RECENT_GW_WINDOW = 5

# Bayesian pseudo-minutes.
#
# The bigger this number, the more conservative the model is with tiny
# samples.
#
# A player needs a substantial number of minutes before observed xG/90
# can strongly move the projection away from the positional prior.
RATE_PRIOR_MINUTES = 900.0


# Positional priors.
#
# These are deliberately conservative. They are not intended to represent
# elite-player rates; they are a stabilising baseline.
#
XG90_PRIORS = {
    "G": 0.01,
    "D": 0.08,
    "M": 0.22,
    "F": 0.38,
}

XA90_PRIORS = {
    "G": 0.02,
    "D": 0.08,
    "M": 0.18,
    "F": 0.10,
}

PP90_PRIORS = {
    "G": 3.0,
    "D": 3.7,
    "M": 4.5,
    "F": 4.0,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def get_json(
    session: requests.Session,
    url: str,
    *,
    timeout: int = REQUEST_TIMEOUT,
):
    response = session.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0 FPL-manager/3.0"
        },
    )

    response.raise_for_status()

    return response.json()


def parse_team_fdr(team: dict) -> float:
    return safe_float(
        team.get("strength", 3),
        3.0,
    )


# ============================================================
# FIXTURE MODEL
# ============================================================

def build_fixture_context(
    fixtures: List[dict],
    teams: Dict[int, dict],
    next_gw: int,
) -> Dict[int, dict]:

    context: Dict[int, List[dict]] = {}

    for fixture in fixtures:

        if fixture.get("event") != next_gw:
            continue

        if fixture.get("finished"):
            continue

        home = fixture.get("team_h")
        away = fixture.get("team_a")

        if home is None or away is None:
            continue

        team_h = teams.get(home, {})
        team_a = teams.get(away, {})

        h_diff = safe_float(
            fixture.get("team_h_difficulty"),
            3.0,
        )

        a_diff = safe_float(
            fixture.get("team_a_difficulty"),
            3.0,
        )

        context.setdefault(home, []).append(
            {
                "opponent": away,
                "home": True,
                "difficulty": h_diff,

                "opp_attack": safe_float(
                    team_a.get("strength_attack_away"),
                    parse_team_fdr(team_a),
                ),

                "opp_defence": safe_float(
                    team_a.get("strength_defence_away"),
                    parse_team_fdr(team_a),
                ),

                "own_attack": safe_float(
                    team_h.get("strength_attack_home"),
                    parse_team_fdr(team_h),
                ),

                "own_defence": safe_float(
                    team_h.get("strength_defence_home"),
                    parse_team_fdr(team_h),
                ),
            }
        )

        context.setdefault(away, []).append(
            {
                "opponent": home,
                "home": False,
                "difficulty": a_diff,

                "opp_attack": safe_float(
                    team_h.get("strength_attack_home"),
                    parse_team_fdr(team_h),
                ),

                "opp_defence": safe_float(
                    team_h.get("strength_defence_home"),
                    parse_team_fdr(team_h),
                ),

                "own_attack": safe_float(
                    team_a.get("strength_attack_away"),
                    parse_team_fdr(team_a),
                ),

                "own_defence": safe_float(
                    team_a.get("strength_defence_away"),
                    parse_team_fdr(team_a),
                ),
            }
        )

    result: Dict[int, dict] = {}

    for team_id, games in context.items():

        difficulty_scores = [
            1.30 - (g["difficulty"] - 1.0) * 0.16
            for g in games
        ]

        avg_diff = (
            sum(g["difficulty"] for g in games)
            / len(games)
        )

        fixture_multiplier = (
            sum(difficulty_scores)
            / len(difficulty_scores)
        )

        home_boost = (
            sum(
                0.045 if g["home"] else -0.015
                for g in games
            )
            / len(games)
        )

        attack_matchup = []
        defence_matchup = []

        for g in games:

            own_attack = max(
                g["own_attack"],
                1.0,
            )

            own_defence = max(
                g["own_defence"],
                1.0,
            )

            opp_defence = max(
                g["opp_defence"],
                1.0,
            )

            opp_attack = max(
                g["opp_attack"],
                1.0,
            )

            attack_matchup.append(
                own_attack / opp_defence
            )

            defence_matchup.append(
                own_defence / opp_attack
            )

        result[team_id] = {
            "fixtures": games,

            "n_fixtures": len(games),

            "avg_difficulty": avg_diff,

            "fixture_multiplier": max(
                0.65,
                min(
                    1.35,
                    fixture_multiplier + home_boost,
                ),
            ),

            "attack_matchup": max(
                0.70,
                min(
                    1.35,
                    sum(attack_matchup)
                    / len(attack_matchup),
                ),
            ),

            "defence_matchup": max(
                0.70,
                min(
                    1.35,
                    sum(defence_matchup)
                    / len(defence_matchup),
                ),
            ),

            "home_rate": (
                sum(
                    1 if g["home"] else 0
                    for g in games
                )
                / len(games)
            ),
        }

    return result


# ============================================================
# HISTORY
# ============================================================

def aggregate_history(rows: List[dict]) -> dict:

    total_minutes = sum(
        safe_float(r.get("minutes"))
        for r in rows
    )

    return {
        "minutes": total_minutes,

        "points": sum(
            safe_float(r.get("total_points"))
            for r in rows
        ),

        "goals": sum(
            safe_float(r.get("goals_scored"))
            for r in rows
        ),

        "assists": sum(
            safe_float(r.get("assists"))
            for r in rows
        ),

        "xg": sum(
            safe_float(r.get("expected_goals"))
            for r in rows
        ),

        "xa": sum(
            safe_float(r.get("expected_assists"))
            for r in rows
        ),

        "bps": sum(
            safe_float(r.get("bps"))
            for r in rows
        ),
    }


def history_features(history: List[dict]) -> dict:
    """
    Current-season recent history.

    IMPORTANT:
    xG/90 and xA/90 are calculated from aggregate minutes, rather than
    averaging individual match rates.

    That prevents a one-minute appearance from producing a huge rate.
    """

    rows = [
        r
        for r in history
        if safe_float(r.get("minutes")) > 0
        and r.get("round")
    ]

    rows = sorted(
        rows,
        key=lambda x: safe_float(x.get("round")),
        reverse=True,
    )[:RECENT_GW_WINDOW]

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
            "matches": 0,
        }

    agg = aggregate_history(rows)

    minutes = max(
        agg["minutes"],
        1.0,
    )

    return {
        "recent_minutes": (
            minutes / len(rows)
        ),

        "recent_points_per_90": (
            agg["points"]
            / minutes
            * 90
        ),

        "recent_xg_per_90": (
            agg["xg"]
            / minutes
            * 90
        ),

        "recent_xa_per_90": (
            agg["xa"]
            / minutes
            * 90
        ),

        "recent_bps_per_90": (
            agg["bps"]
            / minutes
            * 90
        ),

        "starts_rate": (
            sum(
                1
                for r in rows
                if safe_float(r.get("minutes")) >= 60
            )
            / len(rows)
        ),

        "clean_sheet_rate": (
            sum(
                1
                for r in rows
                if safe_float(r.get("clean_sheets")) > 0
            )
            / len(rows)
        ),

        "goals_conceded_per_90": (
            sum(
                safe_float(r.get("goals_conceded"))
                for r in rows
            )
            / minutes
            * 90
        ),

        "matches": len(rows),
    }


def historical_features(history_past: List[dict]) -> dict:
    """
    Previous-season FPL history.

    history_past generally contains historical FPL totals and minutes.
    It is particularly useful as a stabilising historical points/90 signal.

    We do NOT assume historical seasons contain xG/xA, because the
    FPL history_past endpoint does not consistently provide those fields.
    """

    rows = [
        r
        for r in history_past
        if safe_float(r.get("minutes")) > 0
    ]

    if not rows:

        return {
            "minutes": 0.0,
            "points_per_90": 0.0,
            "goals_per_90": 0.0,
            "assists_per_90": 0.0,
            "seasons": 0,
        }

    agg = aggregate_history(rows)

    minutes = max(
        agg["minutes"],
        1.0,
    )

    return {
        "minutes": minutes,

        "points_per_90": (
            agg["points"]
            / minutes
            * 90
        ),

        "goals_per_90": (
            agg["goals"]
            / minutes
            * 90
        ),

        "assists_per_90": (
            agg["assists"]
            / minutes
            * 90
        ),

        "seasons": len(rows),
    }


# ============================================================
# RATE SHRINKAGE
# ============================================================

def shrink_rate(
    observed_rate: float,
    observed_minutes: float,
    prior_rate: float,
    prior_minutes: float = RATE_PRIOR_MINUTES,
) -> float:
    """
    Bayesian-style shrinkage.

    Example:

        observed = 5.0 xG/90
        minutes = 1

    will be pulled extremely strongly toward the prior.

    With 900+ minutes, the observed rate has much more influence.
    """

    observed_minutes = max(
        0.0,
        observed_minutes,
    )

    weight = (
        observed_minutes
        / (
            observed_minutes
            + prior_minutes
        )
    )

    return (
        prior_rate
        + weight
        * (
            observed_rate
            - prior_rate
        )
    )


def blended_rate(
    current_rate: float,
    current_minutes: float,
    historical_rate: float,
    historical_minutes: float,
    recent_rate: float,
    recent_minutes: float,
    prior_rate: float,
) -> float:

    current = shrink_rate(
        current_rate,
        current_minutes,
        prior_rate,
    )

    if historical_minutes > 0:

        historical = shrink_rate(
            historical_rate,
            historical_minutes,
            prior_rate,
        )

    else:
        historical = prior_rate

    if recent_minutes > 0:

        recent = shrink_rate(
            recent_rate,
            recent_minutes,
            prior_rate,
            600.0,
        )

    else:
        recent = current

    if historical_minutes > 0:

        return (
            0.45 * current
            + 0.30 * historical
            + 0.25 * recent
        )

    return (
        0.70 * current
        + 0.30 * recent
    )


# ============================================================
# MINUTES MODEL
# ============================================================

def minutes_probability(
    player: dict,
    recent: dict,
) -> Tuple[float, float]:
    """
    Returns:

        probability player plays
        expected minutes

    This prevents a one-minute substitute appearance from being interpreted
    as evidence that the player is a 90-minute starter.
    """

    status = player.get("status")

    chance = player.get(
        "chance_of_playing_next_round"
    )

    if chance is None:

        chance_factor = 1.0

    else:

        chance_factor = max(
            0.0,
            min(
                1.0,
                safe_float(chance) / 100.0,
            ),
        )

    season_minutes = safe_float(
        player.get("minutes")
    )

    season_starts = safe_float(
        player.get("starts")
    )

    recent_start_rate = recent.get(
        "starts_rate",
        0.0,
    )

    recent_avg_minutes = recent.get(
        "recent_minutes",
        0.0,
    )

    if season_minutes > 0:

        season_start_rate = min(
            1.0,
            season_starts
            / max(
                1.0,
                season_minutes / 90.0,
            ),
        )

    else:

        season_start_rate = 0.0

    if recent.get("matches", 0) > 0:

        start_prob = (
            0.65 * recent_start_rate
            + 0.35 * season_start_rate
        )

    elif season_minutes > 0:

        start_prob = season_start_rate

    else:

        start_prob = 0.50

    if status == "d":

        start_prob *= 0.70

    elif status in {
        "s",
        "u",
        "i",
        "n",
    }:

        start_prob *= 0.10

    play_prob = max(
        0.05,
        min(
            1.0,
            0.55 * chance_factor
            + 0.45 * max(
                0.10,
                start_prob,
            ),
        ),
    )

    typical_start_minutes = max(
        60.0,
        min(
            90.0,
            0.65
            * max(
                60.0,
                recent_avg_minutes,
            )
            + 0.35 * 85.0,
        ),
    )

    expected_minutes = play_prob * (
        start_prob
        * typical_start_minutes
        + (
            1.0
            - start_prob
        )
        * 20.0
    )

    expected_minutes = max(
        0.0,
        min(
            90.0,
            expected_minutes,
        ),
    )

    return (
        play_prob,
        expected_minutes,
    )


# ============================================================
# PROJECTION
# ============================================================

def estimate_clean_sheet_probability(
    pos: str,
    fixture: dict,
) -> float:

    if pos == "F":
        return 0.0

    base = (
        0.18
        + 0.12
        * (
            fixture["defence_mult"]
            - 0.70
        )
    )

    base += (
        0.08
        * (
            fixture["fixture_multiplier"]
            - 0.65
        )
    )

    return max(
        0.05,
        min(
            0.58,
            base,
        ),
    )


def calculate_enhanced_xp(
    player: dict,
    recent: dict,
    historical: dict,
    fixture: dict,
) -> Tuple[float, dict]:

    pos = fixture["Pos"]

    minutes = safe_float(
        player.get("minutes")
    )

    xg_total = safe_float(
        player.get("expected_goals")
    )

    xa_total = safe_float(
        player.get("expected_assists")
    )

    # --------------------------------------------------------
    # Current-season xG/xA
    # --------------------------------------------------------

    if minutes > 0:

        season_xg90 = (
            xg_total
            / minutes
            * 90
        )

        season_xa90 = (
            xa_total
            / minutes
            * 90
        )

    else:

        season_xg90 = XG90_PRIORS[pos]
        season_xa90 = XA90_PRIORS[pos]

    # --------------------------------------------------------
    # Recent sample size
    # --------------------------------------------------------

    recent_minutes_total = (
        recent.get(
            "recent_minutes",
            0.0,
        )
        * recent.get(
            "matches",
            0,
        )
    )

    # --------------------------------------------------------
    # Safely blend current xG/xA with recent performance
    # --------------------------------------------------------

    xg90 = blended_rate(
        season_xg90,
        minutes,
        0.0,
        0.0,
        recent.get(
            "recent_xg_per_90",
            0.0,
        ),
        recent_minutes_total,
        XG90_PRIORS[pos],
    )

    xa90 = blended_rate(
        season_xa90,
        minutes,
        0.0,
        0.0,
        recent.get(
            "recent_xa_per_90",
            0.0,
        ),
        recent_minutes_total,
        XA90_PRIORS[pos],
    )

    # --------------------------------------------------------
    # Historical points/90
    # --------------------------------------------------------

    total_points = safe_float(
        player.get("total_points")
    )

    if minutes > 0:

        season_pp90 = (
            total_points
            / minutes
            * 90
        )

    else:

        season_pp90 = PP90_PRIORS[pos]

    pp90 = blended_rate(
        season_pp90,
        minutes,
        historical.get(
            "points_per_90",
            0.0,
        ),
        historical.get(
            "minutes",
            0.0,
        ),
        recent.get(
            "recent_points_per_90",
            0.0,
        ),
        recent_minutes_total,
        PP90_PRIORS[pos],
    )

    # --------------------------------------------------------
    # Expected minutes
    # --------------------------------------------------------

    play_prob, expected_minutes = (
        minutes_probability(
            player,
            recent,
        )
    )

    minute_fraction = (
        expected_minutes
        / 90.0
    )

    games = max(
        1,
        fixture["n_fixtures"],
    )

    # --------------------------------------------------------
    # Fixture adjustments
    # --------------------------------------------------------

    attack_factor = max(
        0.75,
        min(
            1.25,
            fixture["fixture_multiplier"]
            * fixture["attack_mult"],
        ),
    )

    defence_factor = max(
        0.75,
        min(
            1.25,
            fixture["fixture_multiplier"]
            * fixture["defence_mult"],
        ),
    )

    # --------------------------------------------------------
    # Expected goals / assists
    # --------------------------------------------------------

    expected_goals = (
        xg90
        * minute_fraction
        * attack_factor
        * games
    )

    expected_assists = (
        xa90
        * minute_fraction
        * attack_factor
        * games
    )

    # --------------------------------------------------------
    # FPL scoring
    # --------------------------------------------------------

    if pos == "F":

        goal_points = (
            expected_goals
            * 4.0
        )

        assist_points = (
            expected_assists
            * 3.0
        )

    elif pos == "M":

        goal_points = (
            expected_goals
            * 5.0
        )

        assist_points = (
            expected_assists
            * 3.0
        )

    elif pos == "D":

        goal_points = (
            expected_goals
            * 6.0
        )

        assist_points = (
            expected_assists
            * 3.0
        )

    else:

        goal_points = (
            expected_goals
            * 10.0
        )

        assist_points = (
            expected_assists
            * 3.0
        )

    appearance_points = (
        play_prob
        * 2.0
    )

    # --------------------------------------------------------
    # Clean sheets
    # --------------------------------------------------------

    cs_prob = (
        estimate_clean_sheet_probability(
            pos,
            fixture,
        )
        * minute_fraction
    )

    cs_prob *= defence_factor

    cs_prob = max(
        0.0,
        min(
            0.58,
            cs_prob,
        ),
    )

    if pos in {"G", "D"}:

        cs_points = (
            cs_prob
            * 4.0
        )

    elif pos == "M":

        cs_points = (
            cs_prob
            * 1.0
        )

    else:

        cs_points = 0.0

    # --------------------------------------------------------
    # Bonus proxy
    # --------------------------------------------------------

    bonus_proxy = max(
        0.0,
        min(
            0.65,
            (
                pp90
                - 3.0
            )
            / 10.0,
        ),
    )

    bonus_points = (
        bonus_proxy
        * minute_fraction
    )

    # --------------------------------------------------------
    # Goalkeeper saves
    # --------------------------------------------------------

    save_points = 0.0

    if pos == "G" and minutes > 0:

        saves = safe_float(
            player.get("saves")
        )

        saves90 = (
            saves
            / minutes
            * 90.0
        )

        save_points = max(
            0.0,
            min(
                1.4,
                saves90
                / 3.0
                * minute_fraction,
            ),
        )

    # --------------------------------------------------------
    # Small form adjustment
    # --------------------------------------------------------

    form = safe_float(
        player.get("form")
    )

    form_adjustment = max(
        -0.25,
        min(
            0.35,
            (
                form
                - 4.0
            )
            * 0.08,
        ),
    )

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    raw_xp = (
        appearance_points
        + goal_points
        + assist_points
        + cs_points
        + bonus_points
        + save_points
        + form_adjustment
    )

    # DGW.
    #
    # We don't simply double the whole score because minutes probability
    # has already been handled.
    if games >= 2:

        raw_xp *= (
            1.0
            + 0.72
            * (
                games
                - 1
            )
        )

    # --------------------------------------------------------
    # LOW-MINUTE SANITY GUARD
    # --------------------------------------------------------

    # These are intentionally conservative.
    #
    # This is NOT supposed to be the main calibration mechanism.
    # It is a final safety net against tiny-sample statistical explosions.
    #
    if minutes < 180:

        raw_xp = min(
            raw_xp,
            6.5,
        )

    if minutes < 90:

        raw_xp = min(
            raw_xp,
            5.5,
        )

    # Final useful range.
    #
    # 12 is now a sanity ceiling, not the mechanism producing the score.
    xp = round(
        max(
            1.0,
            min(
                12.0,
                raw_xp,
            ),
        ),
        1,
    )

    diagnostics = {
        "expected_minutes": round(
            expected_minutes,
            1,
        ),

        "play_prob": round(
            play_prob,
            3,
        ),

        "blended_xG90": round(
            xg90,
            3,
        ),

        "blended_xA90": round(
            xa90,
            3,
        ),

        "historical_minutes": round(
            historical.get(
                "minutes",
                0.0,
            ),
            0,
        ),

        "historical_PP90": round(
            historical.get(
                "points_per_90",
                0.0,
            ),
            2,
        ),

        "expected_goals": round(
            expected_goals,
            3,
        ),

        "expected_assists": round(
            expected_assists,
            3,
        ),

        "clean_sheet_prob": round(
            cs_prob,
            3,
        ),
    }

    return (
        xp,
        diagnostics,
    )


# ============================================================
# PLAYER ENRICHMENT
# ============================================================

def fetch_player_history(
    session: requests.Session,
    player_id: int,
) -> dict:

    return get_json(
        session,
        f"{FPL_BASE}/element-summary/{player_id}/",
    )


def enrich_players(
    session: requests.Session,
    raw_players: List[dict],
    fixture_context: Dict[int, dict],
    teams: Dict[int, dict],
) -> pd.DataFrame:

    results: Dict[int, dict] = {}

    max_workers = int(
        os.environ.get(
            "FPL_HISTORY_WORKERS",
            "8",
        )
    )

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as pool:

        futures = {
            pool.submit(
                fetch_player_history,
                session,
                p["id"],
            ): p
            for p in raw_players
        }

        for future in as_completed(futures):

            p = futures[future]

            try:

                results[p["id"]] = (
                    future.result()
                )

            except requests.RequestException:

                results[p["id"]] = {}

    output = []

    pos_map = {
        1: "G",
        2: "D",
        3: "M",
        4: "F",
    }

    for p in raw_players:

        team_id = p["team"]

        pos = pos_map.get(
            p["element_type"],
            "M",
        )

        team_info = teams.get(
            team_id,
            {},
        )

        context = fixture_context.get(
            team_id
        )

        if not context:

            context = {
                "fixtures": [],
                "n_fixtures": 0,
                "avg_difficulty": 5.0,
                "fixture_multiplier": 0.65,
                "attack_matchup": 0.75,
                "defence_matchup": 0.75,
                "home_rate": 0.0,
            }

        history = results.get(
            p["id"],
            {},
        )

        recent = history_features(
            history.get(
                "history",
                [],
            )
        )

        historical = historical_features(
            history.get(
                "history_past",
                [],
            )
        )

        fixture_for_player = {
            "Pos": pos,

            "Price": (
                safe_float(
                    p.get("now_cost"),
                    50.0,
                )
                / 10.0
            ),

            "fixture_multiplier": (
                context[
                    "fixture_multiplier"
                ]
            ),

            "attack_mult": (
                context[
                    "attack_matchup"
                ]
            ),

            "defence_mult": (
                context[
                    "defence_matchup"
                ]
            ),

            "n_fixtures": (
                context[
                    "n_fixtures"
                ]
            ),
        }

        xp, diagnostics = (
            calculate_enhanced_xp(
                p,
                recent,
                historical,
                fixture_for_player,
            )
        )

        chance = p.get(
            "chance_of_playing_next_round"
        )

        status = p.get(
            "status"
        )

        output.append(
            {
                "ID": p["id"],

                "Name": (
                    f"{p['first_name']} "
                    f"{p['second_name']}"
                ),

                "Team": team_info.get(
                    "short_name",
                    "UNK",
                ),

                "TeamID": team_id,

                "Pos": pos,

                "Price": (
                    safe_float(
                        p.get("now_cost"),
                        50.0,
                    )
                    / 10.0
                ),

                "xP": xp,

                "status": status,

                "chance": (
                    100
                    if chance is None
                    else safe_float(chance)
                ),

                "form": safe_float(
                    p.get("form")
                ),

                "PPG": safe_float(
                    p.get(
                        "points_per_game"
                    )
                ),

                "xG": safe_float(
                    p.get(
                        "expected_goals"
                    )
                ),

                "xA": safe_float(
                    p.get(
                        "expected_assists"
                    )
                ),

                "minutes": safe_float(
                    p.get("minutes")
                ),

                "starts": safe_float(
                    p.get("starts")
                ),

                "total_points": safe_float(
                    p.get("total_points")
                ),

                "fixture_difficulty": (
                    context[
                        "avg_difficulty"
                    ]
                ),

                "fixture_games": (
                    context[
                        "n_fixtures"
                    ]
                ),

                "fixture_mult": (
                    context[
                        "fixture_multiplier"
                    ]
                ),

                "recent_xG90": (
                    recent[
                        "recent_xg_per_90"
                    ]
                ),

                "recent_xA90": (
                    recent[
                        "recent_xa_per_90"
                    ]
                ),

                "recent_PP90": (
                    recent[
                        "recent_points_per_90"
                    ]
                ),

                "historical_minutes": (
                    diagnostics[
                        "historical_minutes"
                    ]
                ),

                "historical_PP90": (
                    diagnostics[
                        "historical_PP90"
                    ]
                ),

                "expected_minutes": (
                    diagnostics[
                        "expected_minutes"
                    ]
                ),

                "play_prob": (
                    diagnostics[
                        "play_prob"
                    ]
                ),

                "blended_xG90": (
                    diagnostics[
                        "blended_xG90"
                    ]
                ),

                "blended_xA90": (
                    diagnostics[
                        "blended_xA90"
                    ]
                ),

                "expected_goals": (
                    diagnostics[
                        "expected_goals"
                    ]
                ),

                "expected_assists": (
                    diagnostics[
                        "expected_assists"
                    ]
                ),
            }
        )

    df = pd.DataFrame(output)

    df = df[
        (~df["status"].isin(
            UNAVAILABLE_STATUSES
        ))
        & (
            df["chance"] != 0
        )
    ].reset_index(
        drop=True
    )

    return df


# ============================================================
# BOOTSTRAP / TEAM
# ============================================================

def build_players_from_bootstrap(
    bootstrap: dict,
):

    teams = {
        t["id"]: t
        for t in bootstrap.get(
            "teams",
            [],
        )
    }

    elements_map = {
        p["id"]: p
        for p in bootstrap.get(
            "elements",
            [],
        )
    }

    return (
        elements_map,
        bootstrap.get(
            "elements",
            []
        ),
        teams,
    )


def fetch_current_team(
    session: requests.Session,
    team_id: int,
    elements_map: Dict[int, dict],
):

    entry = get_json(
        session,
        f"{FPL_BASE}/entry/{team_id}/",
    )

    current_gw = entry.get(
        "current_event"
    )

    if not current_gw:

        raise RuntimeError(
            "FPL entry API did not return "
            "a current gameweek."
        )

    picks = get_json(
        session,
        (
            f"{FPL_BASE}/entry/"
            f"{team_id}/event/"
            f"{current_gw}/picks/"
        ),
    )

    squad = []

    for pick in picks.get(
        "picks",
        [],
    ):

        pid = pick.get(
            "element"
        )

        if pid not in elements_map:

            raise RuntimeError(
                f"Pick {pid} is missing "
                "from bootstrap data."
            )

        p = elements_map[pid]

        squad.append(
            {
                "ID": pid,

                "Name": (
                    f"{p['first_name']} "
                    f"{p['second_name']}"
                ),

                "TeamID": p["team"],

                "Pos": {
                    1: "G",
                    2: "D",
                    3: "M",
                    4: "F",
                }.get(
                    p["element_type"],
                    "M",
                ),

                "Price": (
                    safe_float(
                        p.get("now_cost"),
                        50.0,
                    )
                    / 10.0
                ),

                "SellingPrice": (
                    safe_float(
                        pick.get(
                            "selling_price"
                        ),
                        safe_float(
                            p.get(
                                "now_cost"
                            ),
                            50.0,
                        ),
                    )
                    / 10.0
                ),

                "multiplier": pick.get(
                    "multiplier",
                    1,
                ),

                "position": pick.get(
                    "position"
                ),

                "status": p.get(
                    "status"
                ),
            }
        )

    if len(squad) != MAX_SQUAD_SIZE:

        raise RuntimeError(
            f"Expected {MAX_SQUAD_SIZE} "
            f"players, received {len(squad)}."
        )

    history = get_json(
        session,
        f"{FPL_BASE}/entry/{team_id}/history/",
    )

    used_chips = {
        c.get("name")
        for c in history.get(
            "chips",
            [],
        )
        if c.get("name")
    }

    return (
        pd.DataFrame(squad),
        used_chips,
        {
            "entry": entry,
            "current_gw": current_gw,
            "history": history,
        },
    )


# ============================================================
# SQUAD VALIDATION
# ============================================================

def validate_squad(
    squad_df: pd.DataFrame,
) -> None:

    if len(squad_df) != 15:

        raise ValueError(
            f"Squad must contain 15 players, "
            f"found {len(squad_df)}."
        )

    counts = (
        squad_df["Pos"]
        .value_counts()
        .to_dict()
    )

    expected = {
        "G": 2,
        "D": 5,
        "M": 5,
        "F": 3,
    }

    if counts != expected:

        raise ValueError(
            f"Squad must be 2/5/5/3, "
            f"found {counts}."
        )

    if (
        squad_df
        .groupby("TeamID")
        .size()
        .max()
        > MAX_PLAYERS_PER_CLUB
    ):

        raise ValueError(
            "Squad exceeds the "
            "three-player-per-club limit."
        )


# ============================================================
# STARTING XI OPTIMISATION
# ============================================================

def find_best_starting_xi(
    squad_df: pd.DataFrame,
):

    gks = (
        squad_df[
            squad_df["Pos"] == "G"
        ]
        .sort_values(
            "xP",
            ascending=False,
        )
    )

    defs = (
        squad_df[
            squad_df["Pos"] == "D"
        ]
        .sort_values(
            "xP",
            ascending=False,
        )
    )

    mids = (
        squad_df[
            squad_df["Pos"] == "M"
        ]
        .sort_values(
            "xP",
            ascending=False,
        )
    )

    fwds = (
        squad_df[
            squad_df["Pos"] == "F"
        ]
        .sort_values(
            "xP",
            ascending=False,
        )
    )

    best = None

    for d in range(3, 6):

        for m in range(2, 6):

            f = 10 - d - m

            if not 1 <= f <= 3:
                continue

            if len(defs) < d:
                continue

            if len(mids) < m:
                continue

            if len(fwds) < f:
                continue

            selected = pd.concat(
                [
                    gks.head(1),
                    defs.head(d),
                    mids.head(m),
                    fwds.head(f),
                ],
                ignore_index=True,
            )

            score = float(
                selected["xP"].sum()
            )

            candidate = (
                score,
                selected,
                f"{d}-{m}-{f}",
            )

            if (
                best is None
                or candidate[0]
                > best[0]
            ):

                best = candidate

    if best is None:

        raise ValueError(
            "No legal FPL formation "
            "could be generated."
        )

    xi = best[1]

    xi_names = set(
        xi["Name"]
    )

    bench_df = squad_df[
        ~squad_df["Name"].isin(
            xi_names
        )
    ].copy()

    bench_df = (
        bench_df
        .sort_values(
            [
                "xP",
                "Pos",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return (
        xi,
        bench_df.to_dict(
            "records"
        ),
        best[2],
    )


# ============================================================
# ATTACH MODEL DATA
# ============================================================

def attach_models_to_squad(
    squad_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:

    model_lookup = (
        market_df
        .set_index("ID")
        .to_dict("index")
    )

    columns = [
        "xP",
        "Team",
        "fixture_difficulty",
        "fixture_games",
        "fixture_mult",
        "recent_xG90",
        "recent_xA90",
        "recent_PP90",
        "historical_minutes",
        "historical_PP90",
        "expected_minutes",
        "play_prob",
        "blended_xG90",
        "blended_xA90",
        "expected_goals",
        "expected_assists",
    ]

    for col in columns:

        squad_df[col] = (
            squad_df["ID"]
            .map(
                lambda pid:
                model_lookup
                .get(
                    pid,
                    {}
                )
                .get(
                    col
                )
            )
        )

    if squad_df["xP"].isna().any():

        missing = (
            squad_df.loc[
                squad_df["xP"].isna(),
                "Name",
            ]
            .tolist()
        )

        raise RuntimeError(
            "Could not build projection "
            "for current squad players: "
            f"{missing}"
        )

    return squad_df


# ============================================================
# TRANSFER OPTIMISATION
# ============================================================

def find_best_transfer(
    squad_df,
    market_df,
    bank: Optional[float],
    free_transfers: int,
):
    """
    Important change:

    The old model did:

        incoming xP - outgoing xP

    This model does:

        current XI xP
            ->
        transfer
            ->
        optimise XI
            ->
        new XI xP

    The transfer gain is the difference between those two team totals.
    """

    if bank is None:

        return (
            0.0,
            None,
            None,
            "Roll Transfer "
            "(bank balance unavailable) 🔄",
        )

    current_names = set(
        squad_df["Name"]
    )

    club_counts = (
        squad_df["TeamID"]
        .value_counts()
        .to_dict()
    )

    current_xi, _, _ = (
        find_best_starting_xi(
            squad_df
        )
    )

    current_xi_score = float(
        current_xi["xP"].sum()
    )

    best = (
        0.0,
        None,
        None,
        current_xi_score,
    )

    for _, out in squad_df.iterrows():

        sell_price = safe_float(
            out.get(
                "SellingPrice"
            ),
            safe_float(
                out["Price"]
            ),
        )

        candidates = market_df[
            (
                market_df["Pos"]
                == out["Pos"]
            )
            & (
                ~market_df["Name"]
                .isin(
                    current_names
                )
            )
        ]

        for _, inc in candidates.iterrows():

            # Three-player club rule.
            #
            # If the incoming player is from the same club as the outgoing
            # player, the net club count does not increase.
            if (
                inc["TeamID"]
                != out["TeamID"]
                and club_counts.get(
                    inc["TeamID"],
                    0,
                )
                >= MAX_PLAYERS_PER_CLUB
            ):
                continue

            price_difference = (
                float(inc["Price"])
                - sell_price
            )

            if (
                price_difference
                > bank + 1e-9
            ):
                continue

            rows = [
                (
                    inc.to_dict()
                    if r["Name"]
                    == out["Name"]
                    else r
                )
                for r
                in squad_df.to_dict(
                    "records"
                )
            ]

            candidate_squad = pd.DataFrame(
                rows
            )

            try:

                candidate_xi, _, _ = (
                    find_best_starting_xi(
                        candidate_squad
                    )
                )

            except ValueError:

                continue

            new_xi_score = float(
                candidate_xi["xP"].sum()
            )

            xi_gain = (
                new_xi_score
                - current_xi_score
            )

            if (
                xi_gain
                > best[0] + 1e-9
            ):

                best = (
                    xi_gain,
                    out.to_dict(),
                    inc.to_dict(),
                    new_xi_score,
                )

    gain, out, inc, _ = best

    if (
        inc is None
        or gain <= 1.5
    ):

        return (
            gain,
            out,
            inc,
            "Roll Transfer "
            "(no sufficiently strong "
            "starting-XI improvement) 🔄",
        )

    if free_transfers > 0:

        advice = (
            f"Transfer Out: "
            f"{out['Name']} ➡️ "
            f"Transfer In: "
            f"{inc['Name']} "
            f"(Free Transfer, "
            f"+{gain:.1f} XI model xP)"
        )

    elif gain > 4.0:

        advice = (
            f"Transfer Out: "
            f"{out['Name']} ➡️ "
            f"Transfer In: "
            f"{inc['Name']} "
            f"(Likely worth -4, "
            f"+{gain:.1f} XI model xP)"
        )

    else:

        advice = (
            "Roll Transfer "
            "(gain does not justify "
            "a -4 hit) 🔄"
        )

    return (
        gain,
        out,
        inc,
        advice,
    )


def apply_transfer(
    squad_df,
    out_obj,
    in_obj,
    advice,
):

    if (
        not out_obj
        or not in_obj
    ):

        return squad_df

    if not (
        "Free Transfer"
        in advice
        or "Likely worth -4"
        in advice
    ):

        return squad_df

    rows = [
        (
            in_obj
            if r["Name"]
            == out_obj["Name"]
            else r
        )
        for r
        in squad_df.to_dict(
            "records"
        )
    ]

    return pd.DataFrame(
        rows
    )


# ============================================================
# BANK / TRANSFERS
# ============================================================

def get_bank_and_free_transfers(
    entry: dict,
):

    if FPL_BANK_OVERRIDE is not None:

        bank = float(
            FPL_BANK_OVERRIDE
        )

    else:

        raw = entry.get(
            "last_deadline_bank"
        )

        if isinstance(
            raw,
            (int, float),
        ):

            bank = (
                raw / 10.0
            )

        else:

            bank = None

    return (
        bank,
        max(
            0,
            FPL_FREE_TRANSFERS,
        ),
    )


# ============================================================
# WILDCARD
# ============================================================

def wildcard_budget(
    entry: dict,
    bank: Optional[float],
) -> Optional[float]:

    """
    FPL entry.value is squad value in tenths of a million.

    Wildcard budget is therefore approximately:

        current squad value + bank

    This gives the optimiser the ability to completely rebuild the squad.
    """

    raw_value = entry.get(
        "value"
    )

    if not isinstance(
        raw_value,
        (int, float),
    ):

        return None

    return (
        raw_value / 10.0
        + (
            bank or 0.0
        )
    )


def wildcard_select_squad(
    market_df: pd.DataFrame,
    budget: float,
) -> pd.DataFrame:
    """
    Construct an entirely new legal 15-man squad.

    The current squad is NOT used as a constraint.

    The search is intentionally heuristic rather than an enormous brute-force
    search across every FPL player.
    """

    required = {
        "G": 2,
        "D": 5,
        "M": 5,
        "F": 3,
    }

    candidates = {}

    for pos, count in required.items():

        df = market_df[
            market_df["Pos"]
            == pos
        ].copy()

        df["value_score"] = (
            df["xP"]
            / df["Price"].clip(
                lower=0.1
            )
        )

        # Retain enough players to provide meaningful squad construction
        # choices while keeping the search manageable.
        df = (
            df
            .sort_values(
                [
                    "xP",
                    "value_score",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .head(40)
        )

        candidates[pos] = (
            df.to_dict(
                "records"
            )
        )

    # State:
    #
    # players
    # cost
    # raw xP
    # positional counts
    # club counts
    #
    states = [
        (
            [],
            0.0,
            0.0,
            {
                "G": 0,
                "D": 0,
                "M": 0,
                "F": 0,
            },
            {},
        )
    ]

    for pos in [
        "G",
        "D",
        "M",
        "F",
    ]:

        target = required[pos]

        new_states = []

        for (
            players,
            cost,
            score,
            pos_counts,
            clubs,
        ) in states:

            for p in candidates[pos]:

                if (
                    pos_counts[pos]
                    >= target
                ):
                    break

                new_cost = (
                    cost
                    + float(
                        p["Price"]
                    )
                )

                if (
                    new_cost
                    > budget + 1e-9
                ):
                    continue

                club = int(
                    p["TeamID"]
                )

                if (
                    clubs.get(
                        club,
                        0,
                    )
                    >= MAX_PLAYERS_PER_CLUB
                ):
                    continue

                new_players = (
                    players
                    + [p]
                )

                new_pos_counts = dict(
                    pos_counts
                )

                new_pos_counts[pos] += 1

                new_clubs = dict(
                    clubs
                )

                new_clubs[club] = (
                    new_clubs.get(
                        club,
                        0,
                    )
                    + 1
                )

                new_states.append(
                    (
                        new_players,
                        new_cost,
                        score
                        + float(
                            p["xP"]
                        ),
                        new_pos_counts,
                        new_clubs,
                    )
                )

        # Highest raw squad xP first.
        new_states.sort(
            key=lambda s: s[2],
            reverse=True,
        )

        # Keep a broad set of states rather than just the single best one.
        #
        # This prevents a high-cost early selection from making the rest
        # of the squad impossible.
        states = new_states[
            :12000
        ]

        if not states:

            raise RuntimeError(
                "Wildcard optimiser "
                f"could not construct "
                f"a legal squad within "
                f"£{budget:.1f}m."
            )

    final_states = [
        state
        for state in states
        if (
            state[3]
            == required
        )
        and (
            state[1]
            <= budget + 1e-9
        )
    ]

    if not final_states:

        raise RuntimeError(
            "Wildcard optimiser "
            "found no legal final squad."
        )

    best_score = None
    best_squad = None

    # Evaluate the actual starting XI of the best candidate squads.
    for state in final_states[
        :5000
    ]:

        players = state[0]
        cost = state[1]

        squad = pd.DataFrame(
            players
        )

        xi, _, _ = (
            find_best_starting_xi(
                squad
            )
        )

        xi_score = float(
            xi["xP"].sum()
        )

        # Tiny preference for retaining a little flexibility.
        remaining_budget = max(
            0.0,
            budget - cost,
        )

        adjusted_score = (
            xi_score
            + min(
                0.05,
                remaining_budget
                * 0.01,
            )
        )

        if (
            best_score is None
            or adjusted_score
            > best_score
        ):

            best_score = (
                adjusted_score
            )

            best_squad = squad

    if best_squad is None:

        raise RuntimeError(
            "Wildcard optimiser "
            "failed to select "
            "a final squad."
        )

    validate_squad(
        best_squad
    )

    return best_squad


# ============================================================
# CAPTAIN / CHIPS
# ============================================================

def select_captains(
    xi_df,
):

    ranked = (
        xi_df
        .sort_values(
            "xP",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return (
        ranked.iloc[0].to_dict(),
        ranked.iloc[1].to_dict(),
    )


def chip_advice(
    captain: dict,
    bench: List[dict],
    used_chips: set,
):

    bench_xp = sum(
        float(p["xP"])
        for p in bench
    )

    if (
        captain["xP"] >= 12.0
        and "3xc"
        not in used_chips
    ):

        return (
            "Triple Captain Watch 🚀 "
            "(model captain ≥ 12.0)"
        )

    if (
        bench_xp >= 24.0
        and "bboost"
        not in used_chips
    ):

        return (
            "Bench Boost Watch 📈 "
            "(bench ≥ 24.0 model xP)"
        )

    if (
        "3xc" in used_chips
        and captain["xP"] >= 12.0
    ):

        return (
            "Hold Chips 🛡️ "
            "(Triple Captain already used)"
        )

    if (
        "bboost" in used_chips
        and bench_xp >= 24.0
    ):

        return (
            "Hold Chips 🛡️ "
            "(Bench Boost already used)"
        )

    return "Hold Chips 🛡️"


# ============================================================
# SANITY CHECKS
# ============================================================

def sanity_notes(
    squad_df: pd.DataFrame,
) -> str:

    low_minutes = (
        squad_df[
            squad_df["minutes"] < 180
        ]
        .sort_values(
            "xP",
            ascending=False,
        )
    )

    if low_minutes.empty:

        return (
            "No selected player has "
            "fewer than 180 current-season "
            "minutes."
        )

    warnings = []

    for _, p in low_minutes.head(
        3
    ).iterrows():

        warnings.append(
            f"{p['Name']} has "
            f"{p['minutes']:.0f} current-season "
            f"minutes and "
            f"{p['xP']:.1f} model xP"
        )

    return (
        "Low-minute checks: "
        + "; ".join(warnings)
        + "."
    )


# ============================================================
# REPORT
# ============================================================

def format_report(
    xi,
    bench,
    captain,
    vice,
    transfer,
    chips,
    total_cost,
    formation,
    model_notes,
):

    msg = (
        "🏆 *FPL Weekly Team "
        "Manager Report*\n\n"
    )

    msg += (
        f"🧠 *Model:* "
        f"{MODEL_VERSION}\n"
    )

    msg += (
        f"⭐ *Captain:* "
        f"{captain['Name']} "
        f"({captain['xP']:.1f} model xP)\n"
    )

    msg += (
        f"🤝 *Vice-Captain:* "
        f"{vice['Name']} "
        f"({vice['xP']:.1f} model xP)\n"
    )

    msg += (
        f"🔄 *Suggested Move:* "
        f"{transfer}\n"
    )

    msg += (
        f"🎯 *Chip Strategy:* "
        f"{chips}\n"
    )

    msg += (
        f"💰 *Squad Cost:* "
        f"£{total_cost:.1f}m | "
        f"*XI model xP:* "
        f"{xi['xP'].sum():.1f} | "
        f"*Formation:* "
        f"{formation}\n\n"
    )

    msg += "⚽ *Starting XI*\n"

    for pos, label in [
        ("G", "GK"),
        ("D", "DEF"),
        ("M", "MID"),
        ("F", "FWD"),
    ]:

        for _, p in xi[
            xi["Pos"] == pos
        ].iterrows():

            msg += (
                f"• *{label}:* "
                f"{p['Name']} "
                f"({p['Team']}) - "
                f"£{p['Price']:.1f}m | "
                f"{p['xP']:.1f}\n"
            )

    msg += "\n🛋️ *Substitutes*\n"

    for p in bench:

        msg += (
            f"• [{p['Pos']}] "
            f"{p['Name']} "
            f"({p['Team']}) - "
            f"£{p['Price']:.1f}m | "
            f"{p['xP']:.1f}\n"
        )

    msg += (
        "\n📌 *Projection notes:* "
        + model_notes
    )

    return msg


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(
    session,
    message,
):

    if not TOKEN or not CHAT_ID:

        raise RuntimeError(
            "Set TELEGRAM_BOT_TOKEN "
            "and TELEGRAM_CHAT_ID "
            "before sending a report."
        )

    url = (
        f"https://api.telegram.org/"
        f"bot{TOKEN}/sendMessage"
    )

    response = session.post(
        url,
        json={
            "chat_id": CHAT_ID,
            "text": message,
            "parse_mode": "Markdown",
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Fetching live FPL data "
        f"using {MODEL_VERSION}..."
    )

    with requests.Session() as session:

        # ----------------------------------------------------
        # Bootstrap
        # ----------------------------------------------------

        bootstrap = get_json(
            session,
            f"{FPL_BASE}/bootstrap-static/",
        )

        (
            elements_map,
            raw_players,
            teams,
        ) = build_players_from_bootstrap(
            bootstrap
        )

        # ----------------------------------------------------
        # Determine target GW
        # ----------------------------------------------------

        events = bootstrap.get(
            "events",
            []
        )

        current_event = next(
            (
                e
                for e in events
                if e.get(
                    "is_current"
                )
            ),
            None,
        )

        next_event = next(
            (
                e
                for e in events
                if e.get(
                    "is_next"
                )
            ),
            None,
        )

        if (
            current_event
            and next_event
        ):

            target_gw = int(
                next_event["id"]
            )

        elif current_event:

            target_gw = int(
                current_event["id"]
            )

        else:

            target_gw = 1

        print(
            f"Projection target: "
            f"GW{target_gw}"
        )

        # ----------------------------------------------------
        # Fixtures
        # ----------------------------------------------------

        fixtures = get_json(
            session,
            f"{FPL_BASE}/fixtures/",
        )

        fixture_context = (
            build_fixture_context(
                fixtures,
                teams,
                target_gw,
            )
        )

        # ----------------------------------------------------
        # Current team
        # ----------------------------------------------------

        (
            squad_df,
            used_chips,
            meta,
        ) = fetch_current_team(
            session,
            TEAM_ID,
            elements_map,
        )

        validate_squad(
            squad_df
        )

        # ----------------------------------------------------
        # Full player market
        # ----------------------------------------------------

        market_df = enrich_players(
            session,
            raw_players,
            fixture_context,
            teams,
        )

        # ----------------------------------------------------
        # Apply exact same projection model to current squad
        # ----------------------------------------------------

        squad_df = (
            attach_models_to_squad(
                squad_df,
                market_df,
            )
        )

        # ----------------------------------------------------
        # Bank / transfers
        # ----------------------------------------------------

        (
            bank,
            free_transfers,
        ) = get_bank_and_free_transfers(
            meta["entry"]
        )

        # ====================================================
        # WILDCARD MODE
        # ====================================================

        if FPL_WILDCARD_THIS_WEEK:

            budget = wildcard_budget(
                meta["entry"],
                bank,
            )

            if budget is None:

                raise RuntimeError(
                    "Wildcard mode is enabled "
                    "but FPL entry API did not "
                    "return squad value."
                )

            print(
                "Wildcard mode enabled."
            )

            print(
                f"Available Wildcard budget: "
                f"£{budget:.1f}m"
            )

            updated_squad = (
                wildcard_select_squad(
                    market_df,
                    budget,
                )
            )

            validate_squad(
                updated_squad
            )

            transfer_advice = (
                "Wildcard active: rebuilt "
                "the full 15-man squad "
                f"within £{budget:.1f}m."
            )

        # ====================================================
        # NORMAL TRANSFER MODE
        # ====================================================

        else:

            (
                _,
                out_obj,
                in_obj,
                transfer_advice,
            ) = find_best_transfer(
                squad_df,
                market_df,
                bank,
                free_transfers,
            )

            updated_squad = (
                apply_transfer(
                    squad_df,
                    out_obj,
                    in_obj,
                    transfer_advice,
                )
            )

            validate_squad(
                updated_squad
            )

            # Reattach model fields after transfer.
            updated_squad = (
                attach_models_to_squad(
                    updated_squad,
                    market_df,
                )
            )

        # ----------------------------------------------------
        # Optimise XI
        # ----------------------------------------------------

        (
            xi,
            bench,
            formation,
        ) = find_best_starting_xi(
            updated_squad
        )

        captain, vice = (
            select_captains(xi)
        )

        chips = chip_advice(
            captain,
            bench,
            used_chips,
        )

        total_cost = float(
            updated_squad[
                "Price"
            ].sum()
        )

        fixtures_count = sum(
            1
            for v
            in fixture_context.values()
            if v.get(
                "n_fixtures"
            )
        )

        # ----------------------------------------------------
        # Report notes
        # ----------------------------------------------------

        notes = (
            f"Uses current-season xG/xA, "
            f"recent {RECENT_GW_WINDOW}-GW "
            f"history, previous-season "
            f"history_past, minutes-weighted "
            f"shrinkage, expected minutes, "
            f"FPL availability, fixture "
            f"difficulty, home/away and "
            f"team strength. "
            f"{fixtures_count} clubs have a "
            f"GW{target_gw} fixture. "
            f"{sanity_notes(updated_squad)} "
            f"Projection is calibrated "
            f"heuristic xP, not official "
            f"FPL xP."
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        report = format_report(
            xi,
            bench,
            captain,
            vice,
            transfer_advice,
            chips,
            total_cost,
            formation,
            notes,
        )

        print(
            "\n"
            + report
        )

        send_telegram(
            session,
            report,
        )

        print(
            "Successfully sent report "
            "to Telegram!"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except requests.RequestException as exc:

        print(
            f"API/network error: {exc}"
        )

        sys.exit(1)

    except Exception as exc:

        print(
            f"Critical Error encountered: "
            f"{exc}"
        )

        sys.exit(1)