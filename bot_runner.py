import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

FPL_BASE = "https://fantasy.premierleague.com/api"
REQUEST_TIMEOUT = 20

MAX_SQUAD_SIZE = 15
MAX_PLAYERS_PER_CLUB = 3

# You explicitly want to Wildcard this week.
FPL_WILDCARD_THIS_WEEK = os.environ.get(
    "FPL_WILDCARD_THIS_WEEK",
    "true",
).strip().lower() in {"1", "true", "yes", "y"}

# Environment
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("CHAT_ID")
TEAM_ID = os.environ.get("FPL_TEAM_ID") or os.environ.get("TEAM_ID")

FPL_BANK_OVERRIDE = os.environ.get("FPL_BANK_OVERRIDE")
FPL_FREE_TRANSFERS = os.environ.get("FPL_FREE_TRANSFERS")

MODEL_VERSION = "calibrated-v4-wildcard"

RECENT_GW_WINDOW = 5

# Bayesian shrinkage exposure.
# This is deliberately large enough to stop tiny samples exploding.
RECENT_PRIOR_MINUTES = 720.0
SEASON_PRIOR_MINUTES = 1200.0
HISTORICAL_PRIOR_MINUTES = 1800.0

MAX_WORKERS = 12

# Wildcard optimisation parameters.
WILDCARD_CANDIDATES_PER_POSITION = 45
WILDCARD_POSITION_BEAM = 3500
WILDCARD_FINAL_BEAM = 6000
WILDCARD_FINAL_EVALUATIONS = 750

# Small safety guard against numerical/API anomalies.
# This is NOT a projection ceiling.
MIN_XP = 0.5


# ============================================================
# GENERIC HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "").strip()

        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def get_json(session, url, params=None):
    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


def normalise_price(value):
    """
    FPL prices are normally stored in tenths of a million.
    The script works in £m.
    """
    value = safe_float(value)

    if value > 20:
        return value / 10.0

    return value


# ============================================================
# FPL DATA
# ============================================================

def fetch_bootstrap():
    with requests.Session() as session:
        return get_json(
            session,
            f"{FPL_BASE}/bootstrap-static/",
        )


def fetch_fixtures():
    with requests.Session() as session:
        return get_json(
            session,
            f"{FPL_BASE}/fixtures/",
        )


def fetch_entry(team_id):
    with requests.Session() as session:
        return get_json(
            session,
            f"{FPL_BASE}/entry/{team_id}/",
        )


def fetch_entry_picks(team_id, gameweek):
    with requests.Session() as session:
        return get_json(
            session,
            f"{FPL_BASE}/entry/{team_id}/event/{gameweek}/picks/",
        )


def fetch_entry_history(team_id):
    with requests.Session() as session:
        return get_json(
            session,
            f"{FPL_BASE}/entry/{team_id}/history/",
        )


def fetch_player_summary(player_id):
    with requests.Session() as session:
        return get_json(
            session,
            f"{FPL_BASE}/element-summary/{player_id}/",
        )


# ============================================================
# FIXTURE MODEL
# ============================================================

def parse_team_fdr(value, default=3.0):
    """
    FPL fixture difficulty is normally 1-5.
    Convert defensively to a bounded value.
    """
    value = safe_float(value, default)

    if value <= 0:
        value = default

    return clamp(value, 1.0, 5.0)


def difficulty_to_multiplier(difficulty):
    """
    Convert FPL difficulty into a smooth multiplier.

    1 = very favourable
    5 = very difficult
    """
    difficulty = parse_team_fdr(difficulty)

    return clamp(
        1.25 - ((difficulty - 1.0) * 0.125),
        0.75,
        1.25,
    )


def build_fixture_context(fixtures, teams, next_gw):
    """
    Build a one-gameweek fixture context for each club.

    The model uses both FDR and the opponent/club strengths
    available from the FPL fixture endpoint.
    """

    team_lookup = {
        safe_int(team.get("id")): team
        for team in teams
    }

    context = {}

    for team in teams:
        team_id = safe_int(team.get("id"))

        context[team_id] = {
            "fixture_multiplier": 1.0,
            "attack_matchup": 1.0,
            "defence_matchup": 1.0,
            "home_rate": 0.5,
            "opponent_attack": 1.0,
            "opponent_defence": 1.0,
            "fixtures": 0,
        }

    for fixture in fixtures:
        if safe_int(fixture.get("event")) != next_gw:
            continue

        home_id = safe_int(fixture.get("team_h"))
        away_id = safe_int(fixture.get("team_a"))

        if not home_id or not away_id:
            continue

        home_fdr = parse_team_fdr(
            fixture.get("team_h_difficulty"),
            3.0,
        )

        away_fdr = parse_team_fdr(
            fixture.get("team_a_difficulty"),
            3.0,
        )

        home_team = team_lookup.get(home_id, {})
        away_team = team_lookup.get(away_id, {})

        home_attack = safe_float(
            home_team.get("strength_attack_home"),
            safe_float(home_team.get("strength_attack"), 1000),
        )

        away_attack = safe_float(
            away_team.get("strength_attack_away"),
            safe_float(away_team.get("strength_attack"), 1000),
        )

        home_defence = safe_float(
            home_team.get("strength_defence_home"),
            safe_float(home_team.get("strength_defence"), 1000),
        )

        away_defence = safe_float(
            away_team.get("strength_defence_away"),
            safe_float(away_team.get("strength_defence"), 1000),
        )

        # Normalise around 1000.
        home_attack_strength = clamp(home_attack / 1000.0, 0.70, 1.30)
        away_attack_strength = clamp(away_attack / 1000.0, 0.70, 1.30)

        home_defence_strength = clamp(home_defence / 1000.0, 0.70, 1.30)
        away_defence_strength = clamp(away_defence / 1000.0, 0.70, 1.30)

        home_fixture_multiplier = difficulty_to_multiplier(home_fdr)
        away_fixture_multiplier = difficulty_to_multiplier(away_fdr)

        # Attack is improved by favourable opposition defence.
        home_attack_matchup = clamp(
            home_fixture_multiplier
            * (1.05 / max(away_defence_strength, 0.70)),
            0.75,
            1.30,
        )

        away_attack_matchup = clamp(
            away_fixture_multiplier
            * (1.05 / max(home_defence_strength, 0.70)),
            0.75,
            1.30,
        )

        # Defensive clean-sheet expectation is stronger when the
        # opposition attacking strength is low.
        home_defence_matchup = clamp(
            home_fixture_multiplier
            * (1.05 / max(away_attack_strength, 0.70)),
            0.70,
            1.30,
        )

        away_defence_matchup = clamp(
            away_fixture_multiplier
            * (1.05 / max(home_attack_strength, 0.70)),
            0.70,
            1.30,
        )

        context[home_id] = {
            "fixture_multiplier": home_fixture_multiplier,
            "attack_matchup": home_attack_matchup,
            "defence_matchup": home_defence_matchup,
            "home_rate": 1.0,
            "opponent_attack": away_attack_strength,
            "opponent_defence": away_defence_strength,
            "fixtures": 1,
        }

        context[away_id] = {
            "fixture_multiplier": away_fixture_multiplier,
            "attack_matchup": away_attack_matchup,
            "defence_matchup": away_defence_matchup,
            "home_rate": 0.0,
            "opponent_attack": home_attack_strength,
            "opponent_defence": home_defence_strength,
            "fixtures": 1,
        }

    return context


# ============================================================
# STATISTICAL HELPERS
# ============================================================

POSITION_PRIORS = {
    1: {
        "xg90": 0.02,
        "xa90": 0.02,
        "pp90": 4.0,
        "bps90": 12.0,
    },
    2: {
        "xg90": 0.10,
        "xa90": 0.06,
        "pp90": 4.0,
        "bps90": 14.0,
    },
    3: {
        "xg90": 0.20,
        "xa90": 0.15,
        "pp90": 4.5,
        "bps90": 15.0,
    },
    4: {
        "xg90": 0.35,
        "xa90": 0.08,
        "pp90": 4.6,
        "bps90": 14.0,
    },
}


def position_prior(position_id, metric):
    return POSITION_PRIORS.get(
        position_id,
        POSITION_PRIORS[3],
    ).get(metric, 0.0)


def shrunk_rate(
    total_value,
    minutes,
    prior_rate_per90,
    prior_minutes,
):
    """
    Bayesian-style exposure shrinkage.

    This is the important fix for the Kalimuendo problem.

    A player with 1 minute and 1 xG does NOT get a 90 xG/90 rate.
    The observed value is combined with a positional prior according
    to the amount of playing-time evidence available.
    """

    total_value = safe_float(total_value)
    minutes = safe_float(minutes)
    prior_rate_per90 = safe_float(prior_rate_per90)
    prior_minutes = max(1.0, safe_float(prior_minutes))

    if minutes <= 0:
        return prior_rate_per90

    observed_rate = total_value / minutes * 90.0

    weight = minutes / (minutes + prior_minutes)

    return (
        observed_rate * weight
        + prior_rate_per90 * (1.0 - weight)
    )


def sum_history(history, field):
    return sum(
        safe_float(row.get(field))
        for row in history
    )


def history_features(history, position_id):
    """
    Current-season history.

    We retain both totals and rates so that tiny samples can be
    shrunk properly.
    """

    if not history:
        return {
            "recent_minutes": 0.0,
            "recent_points": 0.0,
            "recent_xg": 0.0,
            "recent_xa": 0.0,
            "recent_bps": 0.0,
            "recent_goals": 0.0,
            "recent_assists": 0.0,
            "recent_clean_sheets": 0.0,
            "recent_conceded": 0.0,
            "recent_starts": 0.0,
            "recent_appearances": 0.0,
            "season_minutes": 0.0,
            "season_points": 0.0,
            "season_xg": 0.0,
            "season_xa": 0.0,
            "season_bps": 0.0,
            "season_goals": 0.0,
            "season_assists": 0.0,
            "season_clean_sheets": 0.0,
            "season_conceded": 0.0,
            "season_starts": 0.0,
            "season_appearances": 0.0,
        }

    rows = [
        row
        for row in history
        if safe_int(row.get("round"), 0) > 0
    ]

    rows = sorted(
        rows,
        key=lambda x: safe_int(x.get("round"), 0),
    )

    recent = rows[-RECENT_GW_WINDOW:]

    def aggregate(rows_to_use):
        minutes = sum_history(rows_to_use, "minutes")

        appearances = sum(
            1.0
            for row in rows_to_use
            if safe_float(row.get("minutes")) > 0
        )

        starts = sum(
            safe_float(row.get("starts"))
            for row in rows_to_use
        )

        return {
            "minutes": minutes,
            "points": sum_history(rows_to_use, "total_points"),
            "xg": sum_history(rows_to_use, "expected_goals"),
            "xa": sum_history(rows_to_use, "expected_assists"),
            "bps": sum_history(rows_to_use, "bps"),
            "goals": sum_history(rows_to_use, "goals_scored"),
            "assists": sum_history(rows_to_use, "assists"),
            "clean_sheets": sum_history(rows_to_use, "clean_sheets"),
            "conceded": sum_history(rows_to_use, "goals_conceded"),
            "starts": starts,
            "appearances": appearances,
        }

    recent_data = aggregate(recent)
    season_data = aggregate(rows)

    return {
        "recent_minutes": recent_data["minutes"],
        "recent_points": recent_data["points"],
        "recent_xg": recent_data["xg"],
        "recent_xa": recent_data["xa"],
        "recent_bps": recent_data["bps"],
        "recent_goals": recent_data["goals"],
        "recent_assists": recent_data["assists"],
        "recent_clean_sheets": recent_data["clean_sheets"],
        "recent_conceded": recent_data["conceded"],
        "recent_starts": recent_data["starts"],
        "recent_appearances": recent_data["appearances"],

        "season_minutes": season_data["minutes"],
        "season_points": season_data["points"],
        "season_xg": season_data["xg"],
        "season_xa": season_data["xa"],
        "season_bps": season_data["bps"],
        "season_goals": season_data["goals"],
        "season_assists": season_data["assists"],
        "season_clean_sheets": season_data["clean_sheets"],
        "season_conceded": season_data["conceded"],
        "season_starts": season_data["starts"],
        "season_appearances": season_data["appearances"],
    }


def historical_points_features(history_past, position_id):
    """
    Previous-season FPL history.

    Historical points are deliberately used as a weak stabilising
    signal rather than as a primary driver.
    """

    if not history_past:
        return {
            "historical_minutes": 0.0,
            "historical_points": 0.0,
            "historical_pp90": position_prior(position_id, "pp90"),
        }

    total_minutes = sum_history(
        history_past,
        "total_minutes",
    )

    total_points = sum_history(
        history_past,
        "total_points",
    )

    prior_pp90 = position_prior(
        position_id,
        "pp90",
    )

    pp90 = shrunk_rate(
        total_points,
        total_minutes,
        prior_pp90,
        HISTORICAL_PRIOR_MINUTES,
    )

    return {
        "historical_minutes": total_minutes,
        "historical_points": total_points,
        "historical_pp90": pp90,
    }


# ============================================================
# MINUTES MODEL
# ============================================================

def minutes_probability(player, features):
    """
    Estimate probability of a meaningful appearance and expected minutes.

    The key principle is that a 1-minute cameo is evidence of almost
    nothing about a player's future starting role.

    Official chance + recent starts + season starts are combined.
    """

    position_id = safe_int(player.get("element_type"), 3)

    chance = player.get("chance_of_playing_next_round")

    if chance is None:
        availability = 1.0
    else:
        availability = clamp(
            safe_float(chance, 100.0) / 100.0,
            0.0,
            1.0,
        )

    recent_apps = features["recent_appearances"]
    season_apps = features["season_appearances"]

    recent_start_rate = (
        features["recent_starts"] / recent_apps
        if recent_apps > 0
        else 0.0
    )

    season_start_rate = (
        features["season_starts"] / season_apps
        if season_apps > 0
        else 0.0
    )

    # Bootstrap role information for players with no history.
    bootstrap_starts = safe_float(player.get("starts"))
    bootstrap_minutes = safe_float(player.get("minutes"))

    if season_apps <= 0 and bootstrap_minutes > 0:
        season_start_rate = clamp(
            bootstrap_starts / max(1.0, season_apps),
            0.0,
            1.0,
        )

    if recent_apps > 0:
        role_signal = (
            0.65 * recent_start_rate
            + 0.35 * season_start_rate
        )
    elif season_apps > 0:
        role_signal = season_start_rate
    else:
        # New players: don't assume automatic 90s.
        role_signal = 0.50

    role_signal = clamp(role_signal, 0.05, 0.95)

    start_probability = availability * role_signal

    # Expected minutes when starting.
    start_minutes = 75.0

    if features["season_starts"] > 0:
        start_minutes = (
            features["season_minutes"]
            / max(1.0, features["season_starts"])
        )

    elif features["recent_starts"] > 0:
        start_minutes = (
            features["recent_minutes"]
            / max(1.0, features["recent_starts"])
        )

    start_minutes = clamp(
        start_minutes,
        60.0,
        90.0,
    )

    # Expected minutes if used from bench.
    bench_minutes = 22.0

    if features["recent_appearances"] > 0:
        non_start_apps = max(
            0.0,
            features["recent_appearances"]
            - features["recent_starts"],
        )

        if non_start_apps > 0:
            estimated_bench_minutes = (
                features["recent_minutes"]
                - features["recent_starts"] * start_minutes
            ) / non_start_apps

            if estimated_bench_minutes > 0:
                bench_minutes = clamp(
                    estimated_bench_minutes,
                    10.0,
                    35.0,
                )

    p_60 = start_probability * clamp(
        start_minutes / 75.0,
        0.80,
        1.0,
    )

    p_1_59 = clamp(
        availability - p_60,
        0.0,
        1.0,
    )

    p_no_appearance = clamp(
        1.0 - p_60 - p_1_59,
        0.0,
        1.0,
    )

    expected_minutes = (
        p_60 * start_minutes
        + p_1_59 * bench_minutes
    )

    # Goalkeepers tend to have more stable starting roles.
    if position_id == 1 and features["season_starts"] >= 2:
        p_60 = max(
            p_60,
            availability * 0.82,
        )

        p_1_59 = max(
            0.0,
            availability - p_60,
        )

        p_no_appearance = max(
            0.0,
            1.0 - p_60 - p_1_59,
        )

        expected_minutes = (
            p_60 * start_minutes
            + p_1_59 * bench_minutes
        )

    return {
        "p_60": clamp(p_60, 0.0, 1.0),
        "p_1_59": clamp(p_1_59, 0.0, 1.0),
        "p_no_appearance": clamp(p_no_appearance, 0.0, 1.0),
        "expected_minutes": clamp(
            expected_minutes,
            0.0,
            90.0,
        ),
        "availability": availability,
        "start_probability": clamp(
            start_probability,
            0.0,
            1.0,
        ),
    }


# ============================================================
# EXPECTED POINTS MODEL
# ============================================================

def calculate_enhanced_xp(
    player,
    features,
    historical,
    fixture,
):
    """
    Calibrated expected-points model.

    This is deliberately event-based rather than:

        xG * arbitrary coefficient + xA * arbitrary coefficient

    The model estimates:

        appearance
        + goals
        + assists
        + clean sheets
        + saves
        + bonus/form

    Historical FPL points are included as a stabilising signal.

    There is no artificial 16-point projection ceiling.
    """

    position_id = safe_int(
        player.get("element_type"),
        3,
    )

    minutes = minutes_probability(
        player,
        features,
    )

    expected_minutes = minutes["expected_minutes"]

    season_xg90 = shrunk_rate(
        features["season_xg"],
        features["season_minutes"],
        position_prior(position_id, "xg90"),
        SEASON_PRIOR_MINUTES,
    )

    recent_xg90 = shrunk_rate(
        features["recent_xg"],
        features["recent_minutes"],
        season_xg90,
        RECENT_PRIOR_MINUTES,
    )

    season_xa90 = shrunk_rate(
        features["season_xa"],
        features["season_minutes"],
        position_prior(position_id, "xa90"),
        SEASON_PRIOR_MINUTES,
    )

    recent_xa90 = shrunk_rate(
        features["recent_xa"],
        features["recent_minutes"],
        season_xa90,
        RECENT_PRIOR_MINUTES,
    )

    season_pp90 = shrunk_rate(
        features["season_points"],
        features["season_minutes"],
        position_prior(position_id, "pp90"),
        SEASON_PRIOR_MINUTES,
    )

    recent_pp90 = shrunk_rate(
        features["recent_points"],
        features["recent_minutes"],
        season_pp90,
        RECENT_PRIOR_MINUTES,
    )

    historical_pp90 = historical["historical_pp90"]

    # Current season is the main signal.
    # Recent form is useful but deliberately shrunk.
    blended_xg90 = (
        0.65 * season_xg90
        + 0.35 * recent_xg90
    )

    blended_xa90 = (
        0.65 * season_xa90
        + 0.35 * recent_xa90
    )

    blended_pp90 = (
        0.55 * season_pp90
        + 0.25 * recent_pp90
        + 0.20 * historical_pp90
    )

    fixture_attack = clamp(
        safe_float(fixture.get("attack_matchup"), 1.0),
        0.75,
        1.30,
    )

    fixture_defence = clamp(
        safe_float(fixture.get("defence_matchup"), 1.0),
        0.70,
        1.30,
    )

    home_rate = safe_float(
        fixture.get("home_rate"),
        0.5,
    )

    home_advantage = (
        1.04
        if home_rate > 0.5
        else 1.0
        if home_rate < 0.5
        else 1.02
    )

    expected_minutes_ratio = expected_minutes / 90.0

    # --------------------------------------------------------
    # Appearance points
    # --------------------------------------------------------

    appearance_points = (
        minutes["p_60"] * 2.0
        + minutes["p_1_59"] * 1.0
    )

    # --------------------------------------------------------
    # Goals
    # --------------------------------------------------------

    goal_rate = (
        blended_xg90
        * fixture_attack
        * home_advantage
    )

    expected_goals = (
        goal_rate
        * expected_minutes_ratio
    )

    if position_id == 1:
        goal_points = expected_goals * 10.0
    elif position_id == 2:
        goal_points = expected_goals * 6.0
    elif position_id == 3:
        goal_points = expected_goals * 5.0
    else:
        goal_points = expected_goals * 4.0

    # --------------------------------------------------------
    # Assists
    # --------------------------------------------------------

    assist_rate = (
        blended_xa90
        * fixture_attack
        * home_advantage
    )

    expected_assists = (
        assist_rate
        * expected_minutes_ratio
    )

    assist_points = expected_assists * 3.0

    # --------------------------------------------------------
    # Clean sheets
    # --------------------------------------------------------

    clean_sheet_points = 0.0

    if position_id in {1, 2, 3}:

        base_cs_probability = {
            1: 0.34,
            2: 0.36,
            3: 0.20,
        }.get(position_id, 0.20)

        cs_probability = clamp(
            base_cs_probability
            * fixture_defence
            * (0.98 if home_rate < 0.5 else 1.03),
            0.05,
            0.60,
        )

        # Need a meaningful appearance for a clean sheet.
        cs_probability *= minutes["p_60"]

        if position_id in {1, 2}:
            cs_points_value = 4.0
        else:
            cs_points_value = 1.0

        clean_sheet_points = (
            cs_probability
            * cs_points_value
        )

    # --------------------------------------------------------
    # Goalkeeper saves
    # --------------------------------------------------------

    save_points = 0.0

    if position_id == 1:

        opponent_attack = clamp(
            safe_float(
                fixture.get("opponent_attack"),
                1.0,
            ),
            0.70,
            1.30,
        )

        expected_saves_per_90 = (
            2.8
            * opponent_attack
            * (1.0 + 0.12 * (1.0 - fixture_defence))
        )

        expected_saves = (
            expected_saves_per_90
            * expected_minutes_ratio
        )

        # One FPL point per three saves.
        save_points = expected_saves / 3.0

    # --------------------------------------------------------
    # Bonus
    # --------------------------------------------------------

    bps90 = shrunk_rate(
        features["season_bps"],
        features["season_minutes"],
        position_prior(position_id, "bps90"),
        SEASON_PRIOR_MINUTES,
    )

    recent_bps90 = shrunk_rate(
        features["recent_bps"],
        features["recent_minutes"],
        bps90,
        RECENT_PRIOR_MINUTES,
    )

    blended_bps90 = (
        0.70 * bps90
        + 0.30 * recent_bps90
    )

    # Expected bonus should be modest.
    bonus_signal = clamp(
        blended_bps90 / 25.0,
        0.0,
        1.0,
    )

    bonus_points = (
        minutes["p_60"]
        * 0.55
        * bonus_signal
    )

    # --------------------------------------------------------
    # Historical points / form stabiliser
    # --------------------------------------------------------

    positional_baseline = position_prior(
        position_id,
        "pp90",
    )

    historical_adjustment = clamp(
        (blended_pp90 - positional_baseline)
        * 0.045,
        -0.20,
        0.35,
    )

    # This is intentionally tiny.
    # Historical points should stop wild projections,
    # not manufacture them.
    historical_adjustment *= (
        0.5
        + 0.5 * clamp(
            historical["historical_minutes"] / 1800.0,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Minor discipline / miscellaneous deduction allowance
    # --------------------------------------------------------

    card_rate = safe_float(
        player.get("yellow_cards"),
        0.0,
    )

    card_adjustment = -0.015 * clamp(
        card_rate / max(
            1.0,
            features["season_appearances"],
        ),
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Final xP
    # --------------------------------------------------------

    xp = (
        appearance_points
        + goal_points
        + assist_points
        + clean_sheet_points
        + save_points
        + bonus_points
        + historical_adjustment
        + card_adjustment
    )

    # A player who has effectively no chance of playing should
    # not be presented as a valuable starter.
    xp *= (
        0.15
        + 0.85 * minutes["availability"]
    )

    return round(
        max(MIN_XP, xp),
        2,
    )


# ============================================================
# PLAYER ENRICHMENT
# ============================================================

def fetch_player_history(player_id):
    try:
        return fetch_player_summary(player_id)
    except Exception as exc:
        print(
            f"Warning: failed to fetch summary for player "
            f"{player_id}: {exc}"
        )

        return {
            "history": [],
            "history_past": [],
        }


def enrich_players(players_df, fixture_context):
    """
    Fetch player summaries concurrently and calculate model xP.
    """

    rows = players_df.to_dict("records")

    summaries = {}

    print(
        f"Fetching player histories for {len(rows)} players..."
    )

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_map = {
            executor.submit(
                fetch_player_history,
                safe_int(row.get("ID")),
            ): safe_int(row.get("ID"))
            for row in rows
        }

        for future in as_completed(future_map):
            player_id = future_map[future]

            try:
                summaries[player_id] = future.result()
            except Exception as exc:
                print(
                    f"Warning: player {player_id} "
                    f"history failed: {exc}"
                )

                summaries[player_id] = {
                    "history": [],
                    "history_past": [],
                }

    enriched = []

    for row in rows:
        player_id = safe_int(row.get("ID"))

        summary = summaries.get(
            player_id,
            {
                "history": [],
                "history_past": [],
            },
        )

        position_id = safe_int(
            row.get("PositionID"),
            3,
        )

        history = summary.get("history") or []
        history_past = summary.get("history_past") or []

        features = history_features(
            history,
            position_id,
        )

        historical = historical_points_features(
            history_past,
            position_id,
        )

        team_id = safe_int(
            row.get("TeamID"),
            0,
        )

        fixture = fixture_context.get(
            team_id,
            {
                "fixture_multiplier": 1.0,
                "attack_matchup": 1.0,
                "defence_matchup": 1.0,
                "home_rate": 0.5,
                "opponent_attack": 1.0,
                "opponent_defence": 1.0,
                "fixtures": 0,
            },
        )

        xp = calculate_enhanced_xp(
            row,
            features,
            historical,
            fixture,
        )

        minutes = minutes_probability(
            row,
            features,
        )

        new_row = dict(row)

        new_row.update({
            "xP": xp,

            "ExpectedMinutes": round(
                minutes["expected_minutes"],
                1,
            ),

            "StartProbability": round(
                minutes["start_probability"],
                3,
            ),

            "Availability": round(
                minutes["availability"],
                3,
            ),

            "SeasonPP90": round(
                shrunk_rate(
                    features["season_points"],
                    features["season_minutes"],
                    position_prior(position_id, "pp90"),
                    SEASON_PRIOR_MINUTES,
                ),
                2,
            ),

            "RecentPP90": round(
                shrunk_rate(
                    features["recent_points"],
                    features["recent_minutes"],
                    position_prior(position_id, "pp90"),
                    RECENT_PRIOR_MINUTES,
                ),
                2,
            ),

            "HistoricalPP90": round(
                historical["historical_pp90"],
                2,
            ),

            "SeasonXG90": round(
                shrunk_rate(
                    features["season_xg"],
                    features["season_minutes"],
                    position_prior(position_id, "xg90"),
                    SEASON_PRIOR_MINUTES,
                ),
                3,
            ),

            "RecentXG90": round(
                shrunk_rate(
                    features["recent_xg"],
                    features["recent_minutes"],
                    position_prior(position_id, "xg90"),
                    RECENT_PRIOR_MINUTES,
                ),
                3,
            ),

            "SeasonXA90": round(
                shrunk_rate(
                    features["season_xa"],
                    features["season_minutes"],
                    position_prior(position_id, "xa90"),
                    SEASON_PRIOR_MINUTES,
                ),
                3,
            ),

            "RecentXA90": round(
                shrunk_rate(
                    features["recent_xa"],
                    features["recent_minutes"],
                    position_prior(position_id, "xa90"),
                    RECENT_PRIOR_MINUTES,
                ),
                3,
            ),

            "RecentMinutes": features["recent_minutes"],
            "SeasonMinutes": features["season_minutes"],
            "HistoricalMinutes": historical["historical_minutes"],
        })

        enriched.append(new_row)

    result = pd.DataFrame(enriched)

    if result.empty:
        raise RuntimeError(
            "Player enrichment returned no players."
        )

    return result


# ============================================================
# PLAYER DATAFRAME
# ============================================================

def build_players_from_bootstrap(bootstrap):
    elements = bootstrap.get("elements", [])

    if not elements:
        raise RuntimeError(
            "FPL bootstrap returned no players."
        )

    rows = []

    for player in elements:

        status = str(
            player.get("status", "")
        ).lower()

        if status in {"u", "i"}:
            # Unavailable / injured.
            # Suspended is handled more carefully below.
            continue

        position_id = safe_int(
            player.get("element_type"),
            3,
        )

        team_id = safe_int(
            player.get("team"),
            0,
        )

        price = normalise_price(
            player.get("now_cost")
        )

        if price <= 0:
            continue

        rows.append({
            "ID": safe_int(player.get("id")),
            "Name": (
                f"{player.get('first_name', '').strip()} "
                f"{player.get('second_name', '').strip()}"
            ).strip(),
            "ShortName": (
                player.get("web_name")
                or player.get("second_name")
                or ""
            ),
            "PositionID": position_id,
            "TeamID": team_id,
            "Price": price,

            "Status": status,
            "Chance": player.get(
                "chance_of_playing_next_round"
            ),

            "SelectedByPercent": safe_float(
                player.get("selected_by_percent")
            ),

            "Form": safe_float(
                player.get("form")
            ),

            "TotalPoints": safe_float(
                player.get("total_points")
            ),

            "Minutes": safe_float(
                player.get("minutes")
            ),

            "Starts": safe_float(
                player.get("starts")
            ),

            "Goals": safe_float(
                player.get("goals_scored")
            ),

            "Assists": safe_float(
                player.get("assists")
            ),

            "xG": safe_float(
                player.get("expected_goals")
            ),

            "xA": safe_float(
                player.get("expected_assists")
            ),

            "BPS": safe_float(
                player.get("bps")
            ),

            "YellowCards": safe_float(
                player.get("yellow_cards")
            ),
        })

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError(
            "No usable FPL players found."
        )

    return result


# ============================================================
# CURRENT TEAM
# ============================================================

def fetch_current_team(team_id, current_gw):
    entry = fetch_entry(team_id)

    picks_data = fetch_entry_picks(
        team_id,
        current_gw,
    )

    picks = picks_data.get("picks", [])

    if len(picks) != MAX_SQUAD_SIZE:
        raise RuntimeError(
            f"FPL returned {len(picks)} current picks; "
            f"expected {MAX_SQUAD_SIZE}."
        )

    history = fetch_entry_history(team_id)

    chips = set()

    for chip in history.get("chips", []):
        name = chip.get("name")

        if name:
            chips.add(
                str(name).lower()
            )

    current_player_ids = [
        safe_int(pick.get("element"))
        for pick in picks
    ]

    return (
        entry,
        picks,
        current_player_ids,
        chips,
        history,
    )


# ============================================================
# SQUAD VALIDATION
# ============================================================

def validate_squad(
    squad_df,
    budget=None,
):
    if len(squad_df) != MAX_SQUAD_SIZE:
        return False, (
            f"Squad has {len(squad_df)} players; "
            f"expected {MAX_SQUAD_SIZE}."
        )

    position_counts = (
        squad_df["PositionID"]
        .value_counts()
        .to_dict()
    )

    required = {
        1: 2,
        2: 5,
        3: 5,
        4: 3,
    }

    for position, count in required.items():
        if position_counts.get(position, 0) != count:
            return False, (
                f"Position {position} has "
                f"{position_counts.get(position, 0)} "
                f"players; expected {count}."
            )

    club_counts = (
        squad_df["TeamID"]
        .value_counts()
        .to_dict()
    )

    if any(
        count > MAX_PLAYERS_PER_CLUB
        for count in club_counts.values()
    ):
        return False, (
            "Squad contains more than "
            f"{MAX_PLAYERS_PER_CLUB} players "
            "from one club."
        )

    if budget is not None:
        total_cost = float(
            squad_df["Price"].sum()
        )

        if total_cost > budget + 1e-9:
            return False, (
                f"Squad costs £{total_cost:.1f}m "
                f"but budget is £{budget:.1f}m."
            )

    return True, "OK"


# ============================================================
# STARTING XI
# ============================================================

def find_best_starting_xi(squad_df):
    """
    Enumerate all legal FPL formations and choose the XI with
    the highest total xP.

    Returns:
        xi_df
        bench_df
        formation
    """

    if len(squad_df) != MAX_SQUAD_SIZE:
        raise ValueError(
            "Starting XI optimisation requires 15 players."
        )

    gks = squad_df[
        squad_df["PositionID"] == 1
    ].sort_values(
        "xP",
        ascending=False,
    )

    defs = squad_df[
        squad_df["PositionID"] == 2
    ].sort_values(
        "xP",
        ascending=False,
    )

    mids = squad_df[
        squad_df["PositionID"] == 3
    ].sort_values(
        "xP",
        ascending=False,
    )

    fwds = squad_df[
        squad_df["PositionID"] == 4
    ].sort_values(
        "xP",
        ascending=False,
    )

    if len(gks) < 1:
        raise RuntimeError("No goalkeeper available.")

    best = None

    # FPL legal formations:
    # DEF = 3-5
    # MID = 2-5
    # FWD = 1-3
    for defenders in range(3, 6):

        for midfielders in range(2, 6):

            forwards = 10 - defenders - midfielders

            if forwards < 1 or forwards > 3:
                continue

            if defenders > len(defs):
                continue

            if midfielders > len(mids):
                continue

            if forwards > len(fwds):
                continue

            selected = pd.concat([
                gks.head(1),
                defs.head(defenders),
                mids.head(midfielders),
                fwds.head(forwards),
            ])

            if len(selected) != 11:
                continue

            score = float(
                selected["xP"].sum()
            )

            if (
                best is None
                or score > best["score"]
            ):
                best = {
                    "score": score,
                    "xi": selected.copy(),
                    "formation": (
                        defenders,
                        midfielders,
                        forwards,
                    ),
                }

    if best is None:
        raise RuntimeError(
            "Could not find a legal starting XI."
        )

    xi_ids = set(
        best["xi"]["ID"].astype(int)
    )

    bench = squad_df[
        ~squad_df["ID"].isin(xi_ids)
    ].copy()

    bench = bench.sort_values(
        "xP",
        ascending=False,
    )

    return (
        best["xi"].copy(),
        bench,
        best["formation"],
    )


def team_expected_score(squad_df):
    """
    Expected FPL score of the best XI including captain.

    Captain is chosen from the XI because the Wildcard objective
    should reflect actual expected gameweek points, not merely
    total squad xP.
    """

    xi, bench, formation = find_best_starting_xi(
        squad_df
    )

    if xi.empty:
        return {
            "score": 0.0,
            "xi": xi,
            "bench": bench,
            "formation": formation,
            "captain": None,
            "vice": None,
        }

    ordered = xi.sort_values(
        "xP",
        ascending=False,
    )

    captain = ordered.iloc[0]

    vice = (
        ordered.iloc[1]
        if len(ordered) > 1
        else ordered.iloc[0]
    )

    base_score = float(
        xi["xP"].sum()
    )

    # Captain gets an additional copy of his expected score.
    total_score = (
        base_score
        + float(captain["xP"])
    )

    return {
        "score": total_score,
        "xi": xi,
        "bench": bench,
        "formation": formation,
        "captain": captain,
        "vice": vice,
    }


# ============================================================
# WILDCARD BUDGET
# ============================================================

def get_bank_and_free_transfers(entry):
    if FPL_BANK_OVERRIDE is not None:
        bank = safe_float(
            FPL_BANK_OVERRIDE,
            0.0,
        )
    else:
        # FPL entry bank is normally tenths of £m.
        bank_raw = entry.get("bank")

        if bank_raw is None:
            bank = 0.0
        else:
            bank = safe_float(
                bank_raw,
                0.0,
            )

            if bank > 20:
                bank /= 10.0

    if FPL_FREE_TRANSFERS is not None:
        free_transfers = max(
            0,
            safe_int(
                FPL_FREE_TRANSFERS,
                1,
            ),
        )
    else:
        # This is not critical for Wildcard mode.
        free_transfers = 1

    return (
        round(bank, 1),
        free_transfers,
    )


def get_wildcard_budget(
    entry,
    current_squad_df,
    bank,
):
    """
    Determine Wildcard budget robustly.

    Preferred:
        entry['value'] / 10 + bank

    Fallback:
        sum of the current 15 player prices + bank

    This avoids the exact failure that occurred in GitHub.
    """

    entry_value = safe_float(
        entry.get("value"),
        0.0,
    )

    if entry_value > 0:
        squad_value = entry_value / 10.0

        return round(
            squad_value + bank,
            1,
        )

    if (
        current_squad_df is not None
        and not current_squad_df.empty
        and "Price" in current_squad_df.columns
    ):
        prices = pd.to_numeric(
            current_squad_df["Price"],
            errors="coerce",
        ).dropna()

        if len(prices) == MAX_SQUAD_SIZE:

            squad_value = float(
                prices.sum()
            )

            if squad_value > 0:
                return round(
                    squad_value + bank,
                    1,
                )

    # Final FPL-compatible fallback.
    entry_bank = safe_float(
        entry.get("bank"),
        0.0,
    )

    if entry_bank > 20:
        entry_bank /= 10.0

    if (
        current_squad_df is not None
        and len(current_squad_df) == MAX_SQUAD_SIZE
        and "Price" in current_squad_df.columns
    ):
        prices = pd.to_numeric(
            current_squad_df["Price"],
            errors="coerce",
        ).dropna()

        if len(prices) == MAX_SQUAD_SIZE:
            squad_value = float(
                prices.sum()
            )

            if squad_value > 0:
                return round(
                    squad_value + entry_bank,
                    1,
                )

    raise RuntimeError(
        "Wildcard mode is enabled but the script could not "
        "determine the available squad budget from the FPL "
        "entry response or the 15 current squad player prices."
    )


# ============================================================
# WILDCARD CANDIDATE REDUCTION
# ============================================================

def select_wildcard_candidates(players_df):
    """
    Reduce the ~600-player market while retaining:

      - highest xP
      - highest xP per £m
      - cheapest players
      - popular/form players

    This makes the combinatorial Wildcard search practical.
    """

    candidate_frames = []

    for position_id, count in {
        1: 2,
        2: 5,
        3: 5,
        4: 3,
    }.items():

        position_df = players_df[
            players_df["PositionID"] == position_id
        ].copy()

        if position_df.empty:
            raise RuntimeError(
                f"No players available for position "
                f"{position_id}."
            )

        position_df["ValueMetric"] = (
            position_df["xP"]
            / position_df["Price"].clip(lower=0.1)
        )

        # Best xP
        top_xp = position_df.nlargest(
            WILDCARD_CANDIDATES_PER_POSITION,
            "xP",
        )

        # Best value
        top_value = position_df.nlargest(
            max(
                15,
                WILDCARD_CANDIDATES_PER_POSITION // 2,
            ),
            "ValueMetric",
        )

        # Cheapest players keep budget-feasible solutions alive.
        cheapest = position_df.nsmallest(
            15,
            "Price",
        )

        # Form/popularity provide additional sensible alternatives.
        top_form = position_df.nlargest(
            15,
            "Form",
        )

        candidates = pd.concat([
            top_xp,
            top_value,
            cheapest,
            top_form,
        ]).drop_duplicates(
            subset=["ID"]
        )

        candidates = candidates.sort_values(
            "xP",
            ascending=False,
        )

        candidate_frames.append(
            candidates
        )

        print(
            f"Wildcard candidates - "
            f"position {position_id}: "
            f"{len(candidates)}"
        )

    return pd.concat(
        candidate_frames,
        ignore_index=True,
    )


# ============================================================
# WILDCARD POSITION BEAM
# ============================================================

def make_position_bundles(
    candidates,
    position_id,
    required_count,
):
    """
    Generate strong legal combinations for one position.

    Uses a beam search and retains multiple price/club structures,
    rather than just the highest xP players.
    """

    position_df = candidates[
        candidates["PositionID"] == position_id
    ].copy()

    position_df = position_df.sort_values(
        "xP",
        ascending=False,
    ).reset_index(drop=True)

    if len(position_df) < required_count:
        raise RuntimeError(
            f"Not enough candidates for position "
            f"{position_id}."
        )

    players = position_df.to_dict("records")

    states = [{
        "ids": (),
        "cost10": 0,
        "score": 0.0,
        "clubs": (),
        "last_index": -1,
    }]

    for _ in range(required_count):

        expanded = []

        for state in states:

            club_counts = dict(
                state["clubs"]
            )

            start_index = (
                state["last_index"] + 1
            )

            for idx in range(
                start_index,
                len(players),
            ):

                player = players[idx]

                player_id = safe_int(
                    player["ID"]
                )

                club_id = safe_int(
                    player["TeamID"]
                )

                if player_id in state["ids"]:
                    continue

                current_club_count = club_counts.get(
                    club_id,
                    0,
                )

                if (
                    current_club_count
                    >= MAX_PLAYERS_PER_CLUB
                ):
                    continue

                new_club_counts = dict(
                    club_counts
                )

                new_club_counts[club_id] = (
                    current_club_count + 1
                )

                new_ids = (
                    state["ids"]
                    + (player_id,)
                )

                new_cost10 = (
                    state["cost10"]
                    + int(
                        round(
                            safe_float(
                                player["Price"]
                            ) * 10
                        )
                    )
                )

                new_score = (
                    state["score"]
                    + safe_float(
                        player["xP"]
                    )
                )

                expanded.append({
                    "ids": new_ids,
                    "cost10": new_cost10,
                    "score": new_score,
                    "clubs": tuple(
                        sorted(
                            new_club_counts.items()
                        )
                    ),
                    "last_index": idx,
                })

        if not expanded:
            break

        # Deduplicate by cost + club structure.
        dedup = {}

        for state in expanded:

            key = (
                state["cost10"],
                state["clubs"],
            )

            old = dedup.get(key)

            if (
                old is None
                or state["score"] > old["score"]
            ):
                dedup[key] = state

        states = sorted(
            dedup.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:WILDCARD_POSITION_BEAM]

    return states


# ============================================================
# WILDCARD OPTIMISER
# ============================================================

def optimise_wildcard(
    players_df,
    budget,
):
    """
    Find the best legal 15-man Wildcard squad.

    Exact requirements:
        2 GK
        5 DEF
        5 MID
        3 FWD
        <= 3 per club
        <= available budget

    Final ranking is based on best legal starting XI
    INCLUDING captain expected points.
    """

    candidates = select_wildcard_candidates(
        players_df
    )

    position_requirements = [
        (1, 2),
        (2, 5),
        (3, 5),
        (4, 3),
    ]

    bundles = {}

    for position_id, count in position_requirements:

        print(
            f"Building Wildcard bundles for "
            f"position {position_id}..."
        )

        bundles[position_id] = make_position_bundles(
            candidates,
            position_id,
            count,
        )

        if not bundles[position_id]:
            raise RuntimeError(
                f"Could not create valid Wildcard "
                f"bundles for position {position_id}."
            )

        print(
            f"  {len(bundles[position_id])} "
            f"bundles retained"
        )

    # Combine position bundles.
    states = [{
        "ids": (),
        "cost10": 0,
        "score": 0.0,
        "clubs": (),
    }]

    budget10 = int(
        round(budget * 10)
    )

    for position_id, _ in position_requirements:

        new_states = []

        for state in states:

            state_clubs = dict(
                state["clubs"]
            )

            for bundle in bundles[position_id]:

                new_cost10 = (
                    state["cost10"]
                    + bundle["cost10"]
                )

                if new_cost10 > budget10:
                    continue

                bundle_clubs = dict(
                    bundle["clubs"]
                )

                valid = True
                combined_clubs = dict(
                    state_clubs
                )

                for club_id, count in bundle_clubs.items():

                    new_count = (
                        combined_clubs.get(
                            club_id,
                            0,
                        )
                        + count
                    )

                    if (
                        new_count
                        > MAX_PLAYERS_PER_CLUB
                    ):
                        valid = False
                        break

                    combined_clubs[club_id] = (
                        new_count
                    )

                if not valid:
                    continue

                combined_ids = (
                    state["ids"]
                    + bundle["ids"]
                )

                if len(set(combined_ids)) != len(
                    combined_ids
                ):
                    continue

                new_states.append({
                    "ids": combined_ids,
                    "cost10": new_cost10,
                    "score": (
                        state["score"]
                        + bundle["score"]
                    ),
                    "clubs": tuple(
                        sorted(
                            combined_clubs.items()
                        )
                    ),
                })

        if not new_states:
            raise RuntimeError(
                "Wildcard optimiser found no legal "
                f"solutions after adding position "
                f"{position_id}."
            )

        # Retain a range of costs and club structures.
        dedup = {}

        for state in new_states:

            key = (
                state["cost10"],
                state["clubs"],
            )

            old = dedup.get(key)

            if (
                old is None
                or state["score"] > old["score"]
            ):
                dedup[key] = state

        states = sorted(
            dedup.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:WILDCARD_FINAL_BEAM]

        print(
            f"After position {position_id}: "
            f"{len(states)} Wildcard squad states"
        )

    # Evaluate actual team score, including captain.
    states = sorted(
        states,
        key=lambda x: x["score"],
        reverse=True,
    )[:WILDCARD_FINAL_EVALUATIONS]

    player_lookup = players_df.set_index(
        "ID",
        drop=False,
    )

    best = None

    for state in states:

        try:
            squad = player_lookup.loc[
                list(state["ids"])
            ].copy()
        except KeyError:
            continue

        if len(squad) != MAX_SQUAD_SIZE:
            continue

        valid, reason = validate_squad(
            squad,
            budget=budget,
        )

        if not valid:
            continue

        team = team_expected_score(
            squad
        )

        if (
            best is None
            or team["score"] > best["score"]
        ):
            best = {
                "score": team["score"],
                "squad": squad.copy(),
                "team": team,
            }

    if best is None:
        raise RuntimeError(
            "Wildcard optimisation completed but "
            "no legal final squad survived the "
            "budget/formation/club constraints."
        )

    return best


# ============================================================
# TRANSFER OPTIMISATION
# ============================================================

def find_best_transfer(
    current_squad_df,
    market_df,
    bank,
    free_transfers,
):
    """
    Non-Wildcard mode.

    Optimises the entire starting XI after each transfer.

    This is fundamentally different from simply comparing:

        incoming xP - outgoing xP

    because the transfer can alter the formation and captain.
    """

    current_team = team_expected_score(
        current_squad_df
    )

    current_score = current_team["score"]

    best = None

    current_ids = set(
        current_squad_df["ID"].astype(int)
    )

    for _, outgoing in current_squad_df.iterrows():

        outgoing_id = safe_int(
            outgoing["ID"]
        )

        outgoing_position = safe_int(
            outgoing["PositionID"]
        )

        outgoing_price = safe_float(
            outgoing["Price"]
        )

        sell_budget = (
            outgoing_price
            + bank
        )

        incoming_candidates = market_df[
            (
                market_df["PositionID"]
                == outgoing_position
            )
            & (
                ~market_df["ID"].isin(
                    current_ids
                )
            )
            & (
                market_df["Price"]
                <= sell_budget + 1e-9
            )
        ].copy()

        # Only test realistic high-value candidates.
        incoming_candidates = incoming_candidates.sort_values(
            "xP",
            ascending=False,
        ).head(60)

        for _, incoming in incoming_candidates.iterrows():

            incoming_id = safe_int(
                incoming["ID"]
            )

            candidate = current_squad_df[
                current_squad_df["ID"] != outgoing_id
            ].copy()

            candidate = pd.concat([
                candidate,
                pd.DataFrame([incoming]),
            ], ignore_index=True)

            if len(candidate) != MAX_SQUAD_SIZE:
                continue

            valid, _ = validate_squad(
                candidate,
                budget=sell_budget,
            )

            if not valid:
                continue

            team = team_expected_score(
                candidate
            )

            gross_gain = (
                team["score"]
                - current_score
            )

            hit_cost = (
                0.0
                if free_transfers > 0
                else 4.0
            )

            net_gain = (
                gross_gain
                - hit_cost
            )

            if (
                best is None
                or net_gain > best["net_gain"]
            ):
                best = {
                    "outgoing": outgoing.copy(),
                    "incoming": incoming.copy(),
                    "candidate": candidate.copy(),
                    "team": team,
                    "gross_gain": gross_gain,
                    "hit_cost": hit_cost,
                    "net_gain": net_gain,
                }

    return {
        "current_team": current_team,
        "best": best,
    }


# ============================================================
# CHIP ADVICE
# ============================================================

def chip_advice(
    used_chips,
    wildcard_active,
):
    if wildcard_active:
        return (
            "Wildcard is being used this week. "
            "The full 15-man squad has been optimised."
        )

    if "wildcard" in used_chips:
        return "Wildcard has already been used."

    return (
        "No additional chip is recommended automatically. "
        "Use the strongest XI and reassess after the next deadline."
    )


# ============================================================
# CAPTAIN
# ============================================================

def select_captains(team_result):
    captain = team_result["captain"]
    vice = team_result["vice"]

    if captain is None:
        return None, None

    return captain, vice


# ============================================================
# REPORT HELPERS
# ============================================================

POSITION_NAMES = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}


def format_player_line(row):
    return (
        f"{row['Name']} "
        f"({POSITION_NAMES.get(int(row['PositionID']), '?')}) "
        f"£{safe_float(row['Price']):.1f}m "
        f"xP {safe_float(row['xP']):.2f} "
        f"mins {safe_float(row.get('ExpectedMinutes')):.0f}"
    )


def format_report(
    next_gw,
    squad,
    team_result,
    wildcard_active,
    budget,
    transfer_result=None,
    used_chips=None,
):
    used_chips = used_chips or set()

    lines = []

    lines.append(
        f"FPL Weekly Team Manager — GW{next_gw}"
    )

    lines.append(
        f"Model: {MODEL_VERSION}"
    )

    lines.append("")

    if wildcard_active:
        lines.append(
            "🃏 WILDCARD MODE ACTIVE"
        )

        lines.append(
            f"Wildcard budget: £{budget:.1f}m"
        )

    lines.append("")

    formation = team_result["formation"]

    lines.append(
        "STARTING XI "
        f"({formation[0]}-{formation[1]}-{formation[2]})"
    )

    for _, row in team_result["xi"].sort_values(
        ["PositionID", "xP"],
        ascending=[True, False],
    ).iterrows():

        captain_marker = ""

        captain = team_result["captain"]

        if (
            captain is not None
            and safe_int(row["ID"])
            == safe_int(captain["ID"])
        ):
            captain_marker = " ©"

        lines.append(
            f"• {format_player_line(row)}"
            f"{captain_marker}"
        )

    lines.append("")

    captain, vice = select_captains(
        team_result
    )

    if captain is not None:
        lines.append(
            f"Captain: {captain['Name']} "
            f"(xP {captain['xP']:.2f})"
        )

    if vice is not None:
        lines.append(
            f"Vice: {vice['Name']} "
            f"(xP {vice['xP']:.2f})"
        )

    lines.append("")

    lines.append(
        "BENCH"
    )

    for _, row in team_result["bench"].iterrows():
        lines.append(
            f"• {format_player_line(row)}"
        )

    lines.append("")

    lines.append(
        f"Projected XI + captain: "
        f"{team_result['score']:.2f}"
    )

    if wildcard_active:
        total_cost = float(
            squad["Price"].sum()
        )

        remaining = (
            budget - total_cost
        )

        lines.append(
            f"Squad cost: £{total_cost:.1f}m"
        )

        lines.append(
            f"Budget remaining: £{remaining:.1f}m"
        )

        lines.append(
            "Historical FPL points are included as "
            "a stabilising signal; tiny xG/xA samples "
            "are Bayesian-shrunk."
        )

        lines.append(
            "Wildcard selection optimises the full 15, "
            "legal formation, club limits and captain."
        )

    else:

        if transfer_result is not None:

            best = transfer_result.get("best")

            if best is None:
                lines.append(
                    "Transfer: No positive-value transfer found."
                )

            else:
                outgoing = best["outgoing"]
                incoming = best["incoming"]

                lines.append(
                    "TRANSFER"
                )

                lines.append(
                    f"OUT: {outgoing['Name']} "
                    f"£{outgoing['Price']:.1f}m"
                )

                lines.append(
                    f"IN: {incoming['Name']} "
                    f"£{incoming['Price']:.1f}m"
                )

                lines.append(
                    f"Gross XI gain: "
                    f"{best['gross_gain']:+.2f}"
                )

                lines.append(
                    f"Hit cost: "
                    f"-{best['hit_cost']:.0f}"
                )

                lines.append(
                    f"Net gain: "
                    f"{best['net_gain']:+.2f}"
                )

    lines.append("")

    lines.append(
        chip_advice(
            used_chips,
            wildcard_active,
        )
    )

    return "\n".join(lines)


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(
    token,
    chat_id,
    text,
):
    if not token or not chat_id:
        print(
            "Telegram credentials not configured; "
            "report will only be printed."
        )
        return

    url = (
        f"https://api.telegram.org/bot"
        f"{token}/sendMessage"
    )

    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": text,
        },
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        if not TEAM_ID:
            raise RuntimeError(
                "TEAM_ID / FPL_TEAM_ID is not configured."
            )

        team_id = safe_int(
            TEAM_ID
        )

        if not team_id:
            raise RuntimeError(
                "TEAM_ID is invalid."
            )

        print(
            f"Fetching live FPL data using "
            f"{MODEL_VERSION}..."
        )

        bootstrap = fetch_bootstrap()

        current_gw = safe_int(
            bootstrap.get("events", [{}])[0].get(
                "id",
                1,
            )
        )

        # Find the current/next deadline GW robustly.
        events = bootstrap.get("events", [])

        current_event = None

        for event in events:

            if event.get("is_current"):
                current_event = event
                break

        if current_event is None:

            for event in events:

                if event.get("is_next"):
                    current_event = event
                    break

        if current_event is None:
            current_event = (
                events[0]
                if events
                else {}
            )

        current_gw = safe_int(
            current_event.get("id"),
            current_gw,
        )

        next_gw = current_gw

        print(
            f"Projection target: GW{next_gw}"
        )

        fixtures = fetch_fixtures()

        teams = bootstrap.get(
            "teams",
            [],
        )

        fixture_context = build_fixture_context(
            fixtures,
            teams,
            next_gw,
        )

        # ----------------------------------------------------
        # Current squad
        # ----------------------------------------------------

        (
            entry,
            picks,
            current_ids,
            used_chips,
            entry_history,
        ) = fetch_current_team(
            team_id,
            current_gw,
        )

        players_df = build_players_from_bootstrap(
            bootstrap
        )

        # Current squad price information is available directly
        # from the bootstrap player market.
        current_squad_df = players_df[
            players_df["ID"].isin(current_ids)
        ].copy()

        if len(current_squad_df) != MAX_SQUAD_SIZE:
            raise RuntimeError(
                "Could not map all 15 current squad "
                "players to the FPL player market."
            )

        # ----------------------------------------------------
        # Bank
        # ----------------------------------------------------

        bank, free_transfers = (
            get_bank_and_free_transfers(
                entry
            )
        )

        print(
            f"Current bank: £{bank:.1f}m"
        )

        # ----------------------------------------------------
        # Enrich entire market
        # ----------------------------------------------------

        market_df = enrich_players(
            players_df,
            fixture_context,
        )

        # Remove players explicitly unavailable.
        market_df = market_df[
            ~market_df["Status"].isin(
                {"u", "i"}
            )
        ].copy()

        if market_df.empty:
            raise RuntimeError(
                "No available players remained after "
                "availability filtering."
            )

        # ----------------------------------------------------
        # WILDCARD
        # ----------------------------------------------------

        if FPL_WILDCARD_THIS_WEEK:

            print(
                "Wildcard mode enabled."
            )

            wildcard_budget = get_wildcard_budget(
                entry,
                current_squad_df,
                bank,
            )

            print(
                f"Wildcard budget: "
                f"£{wildcard_budget:.1f}m"
            )

            wildcard_result = optimise_wildcard(
                market_df,
                wildcard_budget,
            )

            final_squad = (
                wildcard_result["squad"]
                .copy()
            )

            valid, reason = validate_squad(
                final_squad,
                wildcard_budget,
            )

            if not valid:
                raise RuntimeError(
                    f"Wildcard optimizer produced "
                    f"an invalid squad: {reason}"
                )

            final_team = team_expected_score(
                final_squad
            )

            print(
                "Wildcard optimisation completed."
            )

            print(
                f"Projected team score: "
                f"{final_team['score']:.2f}"
            )

            transfer_result = None

        # ----------------------------------------------------
        # NORMAL MODE
        # ----------------------------------------------------

        else:

            print(
                "Wildcard mode disabled."
            )

            # Enrich current squad projections.
            current_squad_df = (
                current_squad_df[
                    ["ID"]
                ].merge(
                    market_df,
                    on="ID",
                    how="left",
                    suffixes=("", "_market"),
                )
            )

            # If merge created duplicate fields, reconstruct
            # from market data.
            current_squad_df = market_df[
                market_df["ID"].isin(
                    current_ids
                )
            ].copy()

            transfer_result = (
                find_best_transfer(
                    current_squad_df,
                    market_df,
                    bank,
                    free_transfers,
                )
            )

            best_transfer = (
                transfer_result.get("best")
            )

            if (
                best_transfer is not None
                and best_transfer["net_gain"] > 0
            ):
                final_squad = (
                    best_transfer["candidate"]
                    .copy()
                )

                print(
                    "Positive-value transfer found:"
                )

                print(
                    f"OUT: "
                    f"{best_transfer['outgoing']['Name']}"
                )

                print(
                    f"IN: "
                    f"{best_transfer['incoming']['Name']}"
                )

                print(
                    f"Net gain: "
                    f"{best_transfer['net_gain']:+.2f}"
                )

            else:
                final_squad = (
                    current_squad_df.copy()
                )

                print(
                    "No positive-value transfer found."
                )

            final_team = team_expected_score(
                final_squad
            )

            wildcard_budget = None

        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        valid, reason = validate_squad(
            final_squad,
            wildcard_budget
            if FPL_WILDCARD_THIS_WEEK
            else None,
        )

        if not valid:
            raise RuntimeError(
                f"Final squad failed validation: "
                f"{reason}"
            )

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        report = format_report(
            next_gw,
            final_squad,
            final_team,
            FPL_WILDCARD_THIS_WEEK,
            wildcard_budget
            if FPL_WILDCARD_THIS_WEEK
            else 0.0,
            transfer_result,
            used_chips,
        )

        print("")
        print("=" * 60)
        print(report)
        print("=" * 60)

        send_telegram(
            TOKEN,
            CHAT_ID,
            report,
        )

        print(
            "Run completed successfully."
        )

    except Exception as exc:

        print(
            f"Critical Error encountered: {exc}"
        )

        # Re-raise so GitHub Actions correctly marks
        # the workflow as failed.
        raise


if __name__ == "__main__":
    main()