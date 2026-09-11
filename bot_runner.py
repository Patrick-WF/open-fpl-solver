import math
import os
from itertools import combinations

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

# Wildcard is ON this week.
FPL_WILDCARD_THIS_WEEK = os.environ.get(
    "FPL_WILDCARD_THIS_WEEK",
    "true",
).strip().lower() in {"1", "true", "yes", "y"}

# GitHub Actions secrets
TOKEN = (
    os.environ.get("TELEGRAM_BOT_TOKEN")
    or os.environ.get("TOKEN")
)

CHAT_ID = (
    os.environ.get("TELEGRAM_CHAT_ID")
    or os.environ.get("CHAT_ID")
)

TEAM_ID = (
    os.environ.get("TEAM_ID")
    or os.environ.get("FPL_TEAM_ID")
)

# Optional manual overrides
FPL_BANK_OVERRIDE = os.environ.get("FPL_BANK_OVERRIDE")

MODEL_VERSION = "fast-calibrated-v5-wildcard"

RECENT_GW_WINDOW = 5

# Bayesian shrinkage.
# These stop tiny samples producing absurd rates.
RECENT_PRIOR_MINUTES = 720.0
SEASON_PRIOR_MINUTES = 1200.0
HISTORICAL_PRIOR_MINUTES = 1800.0

# Wildcard candidate pool.
#
# This is deliberately much smaller than the whole market.
# The old version fetched detailed histories for ~600 players.
WILDCARD_POOL_PER_POSITION = {
    1: 30,  # GK
    2: 65,  # DEF
    3: 80,  # MID
    4: 45,  # FWD
}

WILDCARD_BEAM_SIZE = 2500
FINAL_EVALUATIONS = 1500

MIN_XP = 0.5


POSITION_NAMES = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}


# ============================================================
# BASIC HELPERS
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


def normalise_price(value):
    """
    FPL now_cost is stored in tenths of £m.
    Example:
        75 -> £7.5m
    """
    value = safe_float(value)

    if value <= 0:
        return 0.0

    if value > 20:
        return value / 10.0

    return value


def fpl_tenths_to_millions(value):
    """
    FPL account values such as bank/value are stored
    in tenths of £m.

    Example:
        15 -> £1.5m
    """
    value = safe_float(value)

    if value <= 0:
        return 0.0

    return value / 10.0


def get_json(session, url, params=None):
    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# FPL API
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


# ============================================================
# FPL GAMEWEEK
# ============================================================

def determine_target_gameweek(bootstrap):
    """
    Prefer the next gameweek when available.

    If there is no next GW, use the current GW.

    This is safer for a weekly pre-deadline run.
    """

    events = bootstrap.get("events", [])

    next_event = next(
        (
            event
            for event in events
            if event.get("is_next")
        ),
        None,
    )

    if next_event:
        return safe_int(
            next_event.get("id")
        )

    current_event = next(
        (
            event
            for event in events
            if event.get("is_current")
        ),
        None,
    )

    if current_event:
        return safe_int(
            current_event.get("id")
        )

    for event in events:
        if not event.get("finished"):
            return safe_int(
                event.get("id")
            )

    return 1


# ============================================================
# FIXTURE MODEL
# ============================================================

def difficulty_multiplier(difficulty):
    """
    FPL difficulty 1-5 -> smooth multiplier.
    """

    difficulty = clamp(
        safe_float(difficulty, 3.0),
        1.0,
        5.0,
    )

    return clamp(
        1.25 - ((difficulty - 1.0) * 0.125),
        0.75,
        1.25,
    )


def build_fixture_context(
    fixtures,
    teams,
    target_gw,
):
    """
    Build next-GW fixture information.

    This uses FPL's own fixture difficulty plus club
    strength data.
    """

    team_lookup = {
        safe_int(team.get("id")): team
        for team in teams
    }

    context = {}

    for team in teams:

        team_id = safe_int(
            team.get("id")
        )

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

        if safe_int(
            fixture.get("event")
        ) != target_gw:
            continue

        home_id = safe_int(
            fixture.get("team_h")
        )

        away_id = safe_int(
            fixture.get("team_a")
        )

        if not home_id or not away_id:
            continue

        home_fdr = clamp(
            safe_float(
                fixture.get("team_h_difficulty"),
                3.0,
            ),
            1.0,
            5.0,
        )

        away_fdr = clamp(
            safe_float(
                fixture.get("team_a_difficulty"),
                3.0,
            ),
            1.0,
            5.0,
        )

        home_team = team_lookup.get(
            home_id,
            {},
        )

        away_team = team_lookup.get(
            away_id,
            {},
        )

        home_attack = safe_float(
            home_team.get("strength_attack_home"),
            safe_float(
                home_team.get("strength_attack"),
                1000,
            ),
        )

        away_attack = safe_float(
            away_team.get("strength_attack_away"),
            safe_float(
                away_team.get("strength_attack"),
                1000,
            ),
        )

        home_defence = safe_float(
            home_team.get("strength_defence_home"),
            safe_float(
                home_team.get("strength_defence"),
                1000,
            ),
        )

        away_defence = safe_float(
            away_team.get("strength_defence_away"),
            safe_float(
                away_team.get("strength_defence"),
                1000,
            ),
        )

        home_attack = clamp(
            home_attack / 1000.0,
            0.70,
            1.30,
        )

        away_attack = clamp(
            away_attack / 1000.0,
            0.70,
            1.30,
        )

        home_defence = clamp(
            home_defence / 1000.0,
            0.70,
            1.30,
        )

        away_defence = clamp(
            away_defence / 1000.0,
            0.70,
            1.30,
        )

        home_fixture = difficulty_multiplier(
            home_fdr
        )

        away_fixture = difficulty_multiplier(
            away_fdr
        )

        home_attack_matchup = clamp(
            home_fixture
            * (1.05 / away_defence),
            0.75,
            1.30,
        )

        away_attack_matchup = clamp(
            away_fixture
            * (1.05 / home_defence),
            0.75,
            1.30,
        )

        home_defence_matchup = clamp(
            home_fixture
            * (1.05 / away_attack),
            0.70,
            1.30,
        )

        away_defence_matchup = clamp(
            away_fixture
            * (1.05 / home_attack),
            0.70,
            1.30,
        )

        context[home_id] = {
            "fixture_multiplier": home_fixture,
            "attack_matchup": home_attack_matchup,
            "defence_matchup": home_defence_matchup,
            "home_rate": 1.0,
            "opponent_attack": away_attack,
            "opponent_defence": away_defence,
            "fixtures": 1,
        }

        context[away_id] = {
            "fixture_multiplier": away_fixture,
            "attack_matchup": away_attack_matchup,
            "defence_matchup": away_defence_matchup,
            "home_rate": 0.0,
            "opponent_attack": home_attack,
            "opponent_defence": home_defence,
            "fixtures": 1,
        }

    return context


# ============================================================
# POSITION PRIORS
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


def position_prior(position, metric):
    return POSITION_PRIORS.get(
        position,
        POSITION_PRIORS[3],
    ).get(
        metric,
        0.0,
    )


def shrunk_rate(
    total_value,
    minutes,
    prior_rate_per90,
    prior_minutes,
):
    """
    Bayesian-style shrinkage.

    This prevents a player with one tiny appearance
    from generating absurd per-90 numbers.
    """

    total_value = safe_float(total_value)
    minutes = safe_float(minutes)
    prior_rate_per90 = safe_float(
        prior_rate_per90
    )

    prior_minutes = max(
        1.0,
        safe_float(prior_minutes),
    )

    if minutes <= 0:
        return prior_rate_per90

    observed_rate = (
        total_value
        / minutes
        * 90.0
    )

    weight = (
        minutes
        / (minutes + prior_minutes)
    )

    return (
        observed_rate * weight
        + prior_rate_per90 * (1.0 - weight)
    )


# ============================================================
# PLAYER PROJECTION
# ============================================================

def estimate_minutes(player):
    """
    Fast minutes model using only bootstrap data.

    No per-player API calls required.
    """

    chance = player.get(
        "chance_of_playing_next_round"
    )

    if chance is None:
        availability = 1.0
    else:
        availability = clamp(
            safe_float(chance, 100.0)
            / 100.0,
            0.0,
            1.0,
        )

    minutes = safe_float(
        player.get("minutes")
    )

    starts = safe_float(
        player.get("starts")
    )

    appearances = safe_float(
        player.get("appearances")
    )

    # If FPL supplies starts/appearances, use them.
    if appearances > 0:
        start_rate = clamp(
            starts / appearances,
            0.0,
            1.0,
        )
    elif minutes > 0:
        # Conservative fallback.
        start_rate = clamp(
            minutes / 90.0,
            0.0,
            1.0,
        )
    else:
        start_rate = 0.50

    # Form and recent minutes are useful but secondary.
    form = safe_float(
        player.get("form")
    )

    status = str(
        player.get("status", "")
    ).lower()

    # FPL chance-of-playing overrides everything
    # when clearly negative.
    if availability <= 0.25:
        start_probability = availability * 0.25

    else:
        start_probability = (
            0.70 * start_rate
            + 0.30 * 0.50
        )

        # Small boost for players with good recent form.
        if form >= 5.0:
            start_probability += 0.05

        if form >= 7.0:
            start_probability += 0.03

        start_probability *= availability

    start_probability = clamp(
        start_probability,
        0.03,
        0.97,
    )

    # Expected starting minutes.
    if starts > 0:
        start_minutes = (
            minutes / starts
        )

        start_minutes = clamp(
            start_minutes,
            60.0,
            90.0,
        )

    else:
        start_minutes = 75.0

    bench_minutes = 20.0

    p60 = (
        start_probability
        * clamp(
            start_minutes / 75.0,
            0.80,
            1.0,
        )
    )

    # Remaining available probability becomes
    # short appearance.
    p1_59 = clamp(
        availability - p60,
        0.0,
        1.0,
    )

    p0 = clamp(
        1.0 - p60 - p1_59,
        0.0,
        1.0,
    )

    expected_minutes = (
        p60 * start_minutes
        + p1_59 * bench_minutes
    )

    return {
        "availability": availability,
        "start_probability": start_probability,
        "p60": p60,
        "p1_59": p1_59,
        "p0": p0,
        "expected_minutes": clamp(
            expected_minutes,
            0.0,
            90.0,
        ),
    }


def project_player(
    player,
    fixture,
):
    """
    Fast event-based expected-points model.

    Inputs come directly from bootstrap-static.

    Historical/current xG/xA are Bayesian-shrunk.
    """

    position = safe_int(
        player.get("PositionID"),
        3,
    )

    minutes_model = estimate_minutes(
        player
    )

    expected_minutes = (
        minutes_model["expected_minutes"]
    )

    minutes_ratio = (
        expected_minutes / 90.0
    )

    season_minutes = safe_float(
        player.get("Minutes")
    )

    total_points = safe_float(
        player.get("TotalPoints")
    )

    xg = safe_float(
        player.get("xG")
    )

    xa = safe_float(
        player.get("xA")
    )

    bps = safe_float(
        player.get("BPS")
    )

    # --------------------------------------------------------
    # Bayesian xG/xA
    # --------------------------------------------------------

    season_xg90 = shrunk_rate(
        xg,
        season_minutes,
        position_prior(
            position,
            "xg90",
        ),
        1200.0,
    )

    season_xa90 = shrunk_rate(
        xa,
        season_minutes,
        position_prior(
            position,
            "xa90",
        ),
        1200.0,
    )

    pp90 = shrunk_rate(
        total_points,
        season_minutes,
        position_prior(
            position,
            "pp90",
        ),
        1200.0,
    )

    bps90 = shrunk_rate(
        bps,
        season_minutes,
        position_prior(
            position,
            "bps90",
        ),
        1200.0,
    )

    attack_matchup = clamp(
        safe_float(
            fixture.get("attack_matchup"),
            1.0,
        ),
        0.75,
        1.30,
    )

    defence_matchup = clamp(
        safe_float(
            fixture.get("defence_matchup"),
            1.0,
        ),
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
    )

    # --------------------------------------------------------
    # Appearance
    # --------------------------------------------------------

    appearance_points = (
        minutes_model["p60"] * 2.0
        + minutes_model["p1_59"] * 1.0
    )

    # --------------------------------------------------------
    # Goals
    # --------------------------------------------------------

    expected_goals = (
        season_xg90
        * attack_matchup
        * home_advantage
        * minutes_ratio
    )

    goal_points_value = {
        1: 10.0,
        2: 6.0,
        3: 5.0,
        4: 4.0,
    }.get(
        position,
        4.0,
    )

    goal_points = (
        expected_goals
        * goal_points_value
    )

    # --------------------------------------------------------
    # Assists
    # --------------------------------------------------------

    expected_assists = (
        season_xa90
        * attack_matchup
        * home_advantage
        * minutes_ratio
    )

    assist_points = (
        expected_assists * 3.0
    )

    # --------------------------------------------------------
    # Clean sheets
    # --------------------------------------------------------

    clean_sheet_points = 0.0

    if position in {1, 2, 3}:

        base_probability = {
            1: 0.34,
            2: 0.36,
            3: 0.20,
        }[position]

        probability = clamp(
            base_probability
            * defence_matchup
            * (
                1.03
                if home_rate > 0.5
                else 1.0
            ),
            0.05,
            0.60,
        )

        probability *= (
            minutes_model["p60"]
        )

        points_value = (
            4.0
            if position in {1, 2}
            else 1.0
        )

        clean_sheet_points = (
            probability
            * points_value
        )

    # --------------------------------------------------------
    # Goalkeeper saves
    # --------------------------------------------------------

    save_points = 0.0

    if position == 1:

        opponent_attack = clamp(
            safe_float(
                fixture.get(
                    "opponent_attack"
                ),
                1.0,
            ),
            0.70,
            1.30,
        )

        expected_saves = (
            2.8
            * opponent_attack
            * minutes_ratio
        )

        save_points = (
            expected_saves / 3.0
        )

    # --------------------------------------------------------
    # Bonus
    # --------------------------------------------------------

    bonus_signal = clamp(
        bps90 / 25.0,
        0.0,
        1.0,
    )

    bonus_points = (
        minutes_model["p60"]
        * 0.55
        * bonus_signal
    )

    # --------------------------------------------------------
    # Historical/total FPL points stabiliser
    # --------------------------------------------------------

    baseline = position_prior(
        position,
        "pp90",
    )

    stabiliser = clamp(
        (pp90 - baseline)
        * 0.045,
        -0.20,
        0.35,
    )

    # --------------------------------------------------------
    # Small discipline adjustment
    # --------------------------------------------------------

    yellow_cards = safe_float(
        player.get("YellowCards")
    )

    card_rate = (
        yellow_cards
        / max(
            1.0,
            safe_float(
                player.get("Minutes")
            ) / 90.0,
        )
    )

    card_adjustment = -0.01 * clamp(
        card_rate,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    xp = (
        appearance_points
        + goal_points
        + assist_points
        + clean_sheet_points
        + save_points
        + bonus_points
        + stabiliser
        + card_adjustment
    )

    # Availability adjustment.
    xp *= (
        0.15
        + 0.85
        * minutes_model["availability"]
    )

    return max(
        MIN_XP,
        round(xp, 2),
    ), minutes_model


# ============================================================
# PLAYER MARKET
# ============================================================

def build_players(
    bootstrap,
    fixture_context,
):
    elements = bootstrap.get(
        "elements",
        [],
    )

    teams = bootstrap.get(
        "teams",
        [],
    )

    team_names = {
        safe_int(team.get("id")):
        team.get("short_name")
        or team.get("name")
        or "?"
        for team in teams
    }

    rows = []

    for player in elements:

        status = str(
            player.get("status", "")
        ).lower()

        # Unavailable / injured / not registered.
        if status in {"u", "i"}:
            continue

        player_id = safe_int(
            player.get("id")
        )

        position = safe_int(
            player.get("element_type"),
            3,
        )

        team_id = safe_int(
            player.get("team")
        )

        price = normalise_price(
            player.get("now_cost")
        )

        if not player_id or price <= 0:
            continue

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

        base_row = {
            "ID": player_id,

            "Name": (
                f"{player.get('first_name', '').strip()} "
                f"{player.get('second_name', '').strip()}"
            ).strip(),

            "ShortName": (
                player.get("web_name")
                or player.get("second_name")
                or ""
            ),

            "PositionID": position,
            "TeamID": team_id,

            "TeamName": team_names.get(
                team_id,
                "?",
            ),

            "Price": price,

            "Status": status,

            "Chance": safe_float(
                player.get(
                    "chance_of_playing_next_round"
                ),
                100.0,
            ),

            "SelectedByPercent": safe_float(
                player.get(
                    "selected_by_percent"
                )
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

            "Appearances": safe_float(
                player.get("appearances")
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

            "News": player.get(
                "news"
            ) or "",
        }

        xp, minutes_model = project_player(
            base_row,
            fixture,
        )

        base_row.update({
            "xP": xp,

            "ExpectedMinutes": round(
                minutes_model[
                    "expected_minutes"
                ],
                1,
            ),

            "StartProbability": round(
                minutes_model[
                    "start_probability"
                ],
                3,
            ),

            "FixtureMultiplier": round(
                safe_float(
                    fixture.get(
                        "fixture_multiplier"
                    ),
                    1.0,
                ),
                3,
            ),

            "AttackMatchup": round(
                safe_float(
                    fixture.get(
                        "attack_matchup"
                    ),
                    1.0,
                ),
                3,
            ),

            "DefenceMatchup": round(
                safe_float(
                    fixture.get(
                        "defence_matchup"
                    ),
                    1.0,
                ),
                3,
            ),

            "ValueMetric": round(
                xp / max(price, 0.1),
                3,
            ),
        })

        rows.append(base_row)

    result = pd.DataFrame(rows)

    if result.empty:
        raise RuntimeError(
            "No usable FPL players found."
        )

    return result


# ============================================================
# CURRENT SQUAD
# ============================================================

def fetch_current_team(
    team_id,
    current_gw,
):
    entry = fetch_entry(
        team_id
    )

    picks_data = fetch_entry_picks(
        team_id,
        current_gw,
    )

    picks = picks_data.get(
        "picks",
        [],
    )

    if len(picks) != 15:
        raise RuntimeError(
            f"FPL returned {len(picks)} picks; "
            "expected 15."
        )

    history = fetch_entry_history(
        team_id
    )

    chips = {
        str(chip.get("name"))
        .lower()
        for chip in history.get(
            "chips",
            [],
        )
        if chip.get("name")
    }

    player_ids = [
        safe_int(
            pick.get("element")
        )
        for pick in picks
    ]

    return (
        entry,
        picks,
        player_ids,
        chips,
        history,
    )


# ============================================================
# BANK / WILDCARD BUDGET
# ============================================================

def get_bank(entry):
    """
    IMPORTANT:
    FPL entry['bank'] is in tenths of £m.

    Example:
        15 = £1.5m
    """

    if FPL_BANK_OVERRIDE is not None:
        override = safe_float(
            FPL_BANK_OVERRIDE
        )

        # Manual override is assumed to be £m.
        return max(
            0.0,
            override,
        )

    return max(
        0.0,
        fpl_tenths_to_millions(
            entry.get("bank")
        ),
    )


def get_wildcard_budget(
    entry,
    current_squad,
    picks,
    bank,
):
    """
    Preferred source:
        entry.value + bank

    Fallback:
        sum of current player selling_price + bank

    Final fallback:
        sum of current market prices + bank
    """

    entry_value_raw = entry.get(
        "value"
    )

    if entry_value_raw is not None:
        entry_value = (
            fpl_tenths_to_millions(
                entry_value_raw
            )
        )

        if entry_value > 0:
            return round(
                entry_value + bank,
                1,
            )

    # FPL picks normally contain selling_price.
    selling_prices = []

    for pick in picks:

        selling_price = pick.get(
            "selling_price"
        )

        if selling_price is not None:

            price = (
                fpl_tenths_to_millions(
                    selling_price
                )
            )

            if price > 0:
                selling_prices.append(
                    price
                )

    if len(selling_prices) == 15:

        return round(
            sum(selling_prices) + bank,
            1,
        )

    # Last resort.
    if (
        current_squad is not None
        and len(current_squad) == 15
    ):

        prices = pd.to_numeric(
            current_squad["Price"],
            errors="coerce",
        ).dropna()

        if len(prices) == 15:

            return round(
                float(prices.sum())
                + bank,
                1,
            )

    raise RuntimeError(
        "Could not determine Wildcard budget."
    )


# ============================================================
# SQUAD VALIDATION
# ============================================================

def validate_squad(
    squad,
    budget=None,
):
    if len(squad) != 15:
        return False, (
            f"Squad contains {len(squad)} players."
        )

    required = {
        1: 2,
        2: 5,
        3: 5,
        4: 3,
    }

    counts = (
        squad["PositionID"]
        .value_counts()
        .to_dict()
    )

    for position, expected in required.items():

        actual = counts.get(
            position,
            0,
        )

        if actual != expected:

            return False, (
                f"{POSITION_NAMES[position]} "
                f"count is {actual}; "
                f"expected {expected}."
            )

    club_counts = (
        squad["TeamID"]
        .value_counts()
        .to_dict()
    )

    if any(
        count > MAX_PLAYERS_PER_CLUB
        for count in club_counts.values()
    ):

        return False, (
            "More than 3 players from "
            "one club."
        )

    if budget is not None:

        cost = float(
            squad["Price"].sum()
        )

        if cost > budget + 0.01:

            return False, (
                f"Squad costs £{cost:.1f}m; "
                f"budget is £{budget:.1f}m."
            )

    return True, "OK"


# ============================================================
# STARTING XI
# ============================================================

def find_best_starting_xi(
    squad,
):
    """
    Find the highest-xP legal FPL formation.
    """

    gks = squad[
        squad["PositionID"] == 1
    ].sort_values(
        "xP",
        ascending=False,
    )

    defs = squad[
        squad["PositionID"] == 2
    ].sort_values(
        "xP",
        ascending=False,
    )

    mids = squad[
        squad["PositionID"] == 3
    ].sort_values(
        "xP",
        ascending=False,
    )

    fwds = squad[
        squad["PositionID"] == 4
    ].sort_values(
        "xP",
        ascending=False,
    )

    if gks.empty:
        raise RuntimeError(
            "No goalkeeper available."
        )

    best = None

    for defenders in range(3, 6):

        for midfielders in range(2, 6):

            forwards = (
                10
                - defenders
                - midfielders
            )

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
            "Could not create legal XI."
        )

    xi_ids = set(
        best["xi"]["ID"].astype(int)
    )

    bench = squad[
        ~squad["ID"].isin(xi_ids)
    ].copy()

    bench = bench.sort_values(
        "xP",
        ascending=False,
    )

    return (
        best["xi"],
        bench,
        best["formation"],
    )


def team_expected_score(
    squad,
):
    xi, bench, formation = (
        find_best_starting_xi(
            squad
        )
    )

    ordered = xi.sort_values(
        "xP",
        ascending=False,
    )

    captain = ordered.iloc[0]
    vice = ordered.iloc[1]

    base = float(
        xi["xP"].sum()
    )

    # Captain doubles his expected score.
    total = (
        base
        + float(captain["xP"])
    )

    return {
        "score": total,
        "xi": xi,
        "bench": bench,
        "formation": formation,
        "captain": captain,
        "vice": vice,
    }


# ============================================================
# WILDCARD CANDIDATES
# ============================================================

def wildcard_candidates(
    players,
):
    """
    Select a manageable but broad candidate pool.

    We deliberately retain:
      - highest xP
      - best xP/£m
      - best form
      - cheapest players
      - current highly-selected players

    This is much faster than fetching detailed histories
    for every player.
    """

    pieces = []

    for position, limit in (
        WILDCARD_POOL_PER_POSITION.items()
    ):

        df = players[
            players["PositionID"]
            == position
        ].copy()

        if df.empty:
            continue

        df["ValueMetric"] = (
            df["xP"]
            / df["Price"].clip(
                lower=0.1
            )
        )

        selections = [
            df.nlargest(
                limit,
                "xP",
            ),

            df.nlargest(
                max(10, limit // 2),
                "ValueMetric",
            ),

            df.nlargest(
                max(10, limit // 3),
                "Form",
            ),

            df.nlargest(
                max(10, limit // 3),
                "SelectedByPercent",
            ),

            df.nsmallest(
                12,
                "Price",
            ),
        ]

        combined = pd.concat(
            selections,
            ignore_index=True,
        ).drop_duplicates(
            subset=["ID"]
        )

        # Final cap.
        combined = combined.sort_values(
            [
                "xP",
                "ValueMetric",
            ],
            ascending=False,
        ).head(
            limit
        )

        print(
            f"Wildcard pool "
            f"{POSITION_NAMES[position]}: "
            f"{len(combined)} players"
        )

        pieces.append(
            combined
        )

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    if result.empty:
        raise RuntimeError(
            "Wildcard candidate pool is empty."
        )

    return result


# ============================================================
# WILDCARD BEAM SEARCH
# ============================================================

def build_position_bundles(
    df,
    position,
    count,
):
    """
    Build legal position bundles.

    Uses combinations only within the reduced candidate pool.
    """

    position_df = df[
        df["PositionID"] == position
    ].copy()

    position_df = position_df.sort_values(
        "xP",
        ascending=False,
    )

    # Keep this small enough to prevent combinations exploding.
    if len(position_df) > 28 and count >= 5:
        position_df = position_df.head(28)

    if len(position_df) < count:
        raise RuntimeError(
            f"Not enough {POSITION_NAMES[position]} "
            "candidates."
        )

    players = position_df.to_dict(
        "records"
    )

    bundles = []

    for combo in combinations(
        players,
        count,
    ):

        ids = tuple(
            safe_int(p["ID"])
            for p in combo
        )

        clubs = {}

        valid = True

        for player in combo:

            club = safe_int(
                player["TeamID"]
            )

            clubs[club] = (
                clubs.get(club, 0)
                + 1
            )

            if (
                clubs[club]
                > MAX_PLAYERS_PER_CLUB
            ):
                valid = False
                break

        if not valid:
            continue

        cost10 = sum(
            int(
                round(
                    safe_float(
                        p["Price"]
                    ) * 10
                )
            )
            for p in combo
        )

        score = sum(
            safe_float(
                p["xP"]
            )
            for p in combo
        )

        bundles.append({
            "ids": ids,
            "cost10": cost10,
            "score": score,
            "clubs": tuple(
                sorted(
                    clubs.items()
                )
            ),
        })

    bundles.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # Retain the best plus cheaper structures.
    if len(bundles) > WILDCARD_BEAM_SIZE:
        bundles = bundles[
            :WILDCARD_BEAM_SIZE
        ]

    return bundles


def optimise_wildcard(
    players,
    budget,
):
    candidates = wildcard_candidates(
        players
    )

    requirements = [
        (1, 2),
        (2, 5),
        (3, 5),
        (4, 3),
    ]

    all_bundles = {}

    for position, count in requirements:

        print(
            f"Creating "
            f"{POSITION_NAMES[position]} "
            "bundles..."
        )

        bundles = build_position_bundles(
            candidates,
            position,
            count,
        )

        if not bundles:
            raise RuntimeError(
                f"No bundles found for "
                f"{POSITION_NAMES[position]}."
            )

        all_bundles[position] = bundles

        print(
            f"  {len(bundles)} bundles"
        )

    # Start with empty squad.
    states = [{
        "ids": (),
        "cost10": 0,
        "score": 0.0,
        "clubs": (),
    }]

    budget10 = int(
        round(
            budget * 10
        )
    )

    for position, _ in requirements:

        next_states = []

        bundles = all_bundles[
            position
        ]

        for state in states:

            current_clubs = dict(
                state["clubs"]
            )

            for bundle in bundles:

                new_cost = (
                    state["cost10"]
                    + bundle["cost10"]
                )

                if new_cost > budget10:
                    continue

                combined_clubs = dict(
                    current_clubs
                )

                valid = True

                for club, count in dict(
                    bundle["clubs"]
                ).items():

                    total = (
                        combined_clubs.get(
                            club,
                            0,
                        )
                        + count
                    )

                    if (
                        total
                        > MAX_PLAYERS_PER_CLUB
                    ):
                        valid = False
                        break

                    combined_clubs[club] = total

                if not valid:
                    continue

                new_ids = (
                    state["ids"]
                    + bundle["ids"]
                )

                if len(
                    set(new_ids)
                ) != len(new_ids):
                    continue

                next_states.append({
                    "ids": new_ids,
                    "cost10": new_cost,
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

        if not next_states:
            raise RuntimeError(
                "Wildcard optimiser found "
                "no legal states."
            )

        # Keep best states while preserving
        # different cost/club structures.
        dedup = {}

        for state in next_states:

            key = (
                state["cost10"],
                state["clubs"],
            )

            previous = dedup.get(
                key
            )

            if (
                previous is None
                or state["score"]
                > previous["score"]
            ):
                dedup[key] = state

        states = sorted(
            dedup.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:WILDCARD_BEAM_SIZE]

        print(
            f"After "
            f"{POSITION_NAMES[position]}: "
            f"{len(states)} states"
        )

    # Evaluate the best candidates using actual legal XI
    # and captain selection.
    states = sorted(
        states,
        key=lambda x: x["score"],
        reverse=True,
    )[:FINAL_EVALUATIONS]

    lookup = players.set_index(
        "ID",
        drop=False,
    )

    best = None

    for state in states:

        try:
            squad = lookup.loc[
                list(state["ids"])
            ].copy()

        except KeyError:
            continue

        if len(squad) != 15:
            continue

        valid, reason = validate_squad(
            squad,
            budget,
        )

        if not valid:
            continue

        team = team_expected_score(
            squad
        )

        if (
            best is None
            or team["score"]
            > best["score"]
        ):
            best = {
                "score": team["score"],
                "squad": squad.copy(),
                "team": team,
            }

    if best is None:
        raise RuntimeError(
            "Wildcard optimisation produced "
            "no valid final squad."
        )

    return best


# ============================================================
# REPORT
# ============================================================

def format_player(row):
    return (
        f"{row['ShortName']} "
        f"({POSITION_NAMES.get(int(row['PositionID']), '?')}) "
        f"£{row['Price']:.1f}m "
        f"xP {row['xP']:.2f} "
        f"mins {row['ExpectedMinutes']:.0f}"
    )


def build_report(
    gameweek,
    squad,
    team_result,
    budget,
    bank,
):
    lines = []

    lines.append(
        f"⚽ FPL Weekly Manager — GW{gameweek}"
    )

    lines.append(
        f"Model: {MODEL_VERSION}"
    )

    lines.append("")

    lines.append(
        "🃏 WILDCARD ACTIVE"
    )

    lines.append(
        f"Wildcard budget: £{budget:.1f}m"
    )

    lines.append(
        f"Bank: £{bank:.1f}m"
    )

    lines.append("")

    formation = team_result[
        "formation"
    ]

    lines.append(
        "STARTING XI "
        f"({formation[0]}-"
        f"{formation[1]}-"
        f"{formation[2]})"
    )

    captain = team_result[
        "captain"
    ]

    vice = team_result[
        "vice"
    ]

    for _, row in team_result[
        "xi"
    ].sort_values(
        ["PositionID", "xP"],
        ascending=[True, False],
    ).iterrows():

        marker = ""

        if (
            captain is not None
            and safe_int(row["ID"])
            == safe_int(captain["ID"])
        ):
            marker = " ©"

        lines.append(
            f"• {format_player(row)}"
            f"{marker}"
        )

    lines.append("")

    lines.append(
        f"Captain: "
        f"{captain['ShortName']} "
        f"({captain['xP']:.2f} xP)"
    )

    lines.append(
        f"Vice: "
        f"{vice['ShortName']} "
        f"({vice['xP']:.2f} xP)"
    )

    lines.append("")

    lines.append(
        "BENCH"
    )

    for _, row in team_result[
        "bench"
    ].iterrows():

        lines.append(
            f"• {format_player(row)}"
        )

    lines.append("")

    total_cost = float(
        squad["Price"].sum()
    )

    remaining = (
        budget
        - total_cost
    )

    lines.append(
        f"Squad cost: £{total_cost:.1f}m"
    )

    lines.append(
        f"Budget remaining: "
        f"£{remaining:.1f}m"
    )

    lines.append(
        f"Projected XI + captain: "
        f"{team_result['score']:.2f}"
    )

    lines.append("")

    lines.append(
        "Model notes:"
    )

    lines.append(
        "• Tiny xG/xA samples are Bayesian-shrunk."
    )

    lines.append(
        "• Historical FPL points act as a weak stabiliser."
    )

    lines.append(
        "• Starting XI is optimised across legal formations."
    )

    lines.append(
        "• Wildcard optimises the full 15-man squad."
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
            "Telegram credentials missing; "
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

        # ----------------------------------------------------
        # Credentials
        # ----------------------------------------------------

        if not TEAM_ID:
            raise RuntimeError(
                "TEAM_ID / FPL_TEAM_ID "
                "is not configured."
            )

        team_id = safe_int(
            TEAM_ID
        )

        if team_id <= 0:
            raise RuntimeError(
                "TEAM_ID is invalid."
            )

        print(
            f"Fetching live FPL data using "
            f"{MODEL_VERSION}..."
        )

        # ----------------------------------------------------
        # Bootstrap
        # ----------------------------------------------------

        bootstrap = fetch_bootstrap()

        target_gw = (
            determine_target_gameweek(
                bootstrap
            )
        )

        print(
            f"Projection target: GW{target_gw}"
        )

        # ----------------------------------------------------
        # Fixtures
        # ----------------------------------------------------

        fixtures = fetch_fixtures()

        fixture_context = (
            build_fixture_context(
                fixtures,
                bootstrap.get(
                    "teams",
                    [],
                ),
                target_gw,
            )
        )

        # ----------------------------------------------------
        # Current team
        # ----------------------------------------------------

        # We need the current squad.
        #
        # If the target GW is the next GW, FPL's entry picks
        # normally need the current completed/current GW.
        events = bootstrap.get(
            "events",
            [],
        )

        current_event = next(
            (
                event
                for event in events
                if event.get("is_current")
            ),
            None,
        )

        current_gw = (
            safe_int(
                current_event.get("id")
            )
            if current_event
            else target_gw
        )

        (
            entry,
            picks,
            current_ids,
            used_chips,
            history,
        ) = fetch_current_team(
            team_id,
            current_gw,
        )

        # ----------------------------------------------------
        # Bank
        # ----------------------------------------------------

        bank = get_bank(
            entry
        )

        print(
            f"Current bank: "
            f"£{bank:.1f}m"
        )

        # ----------------------------------------------------
        # Player market
        # ----------------------------------------------------

        print(
            "Building player projections "
            "from bootstrap data..."
        )

        players = build_players(
            bootstrap,
            fixture_context,
        )

        print(
            f"Projected "
            f"{len(players)} players."
        )

        # ----------------------------------------------------
        # Current squad
        # ----------------------------------------------------

        current_squad = players[
            players["ID"].isin(
                current_ids
            )
        ].copy()

        if len(current_squad) != 15:
            raise RuntimeError(
                "Could not map all 15 "
                "current squad players."
            )

        # ----------------------------------------------------
        # Wildcard
        # ----------------------------------------------------

        if FPL_WILDCARD_THIS_WEEK:

            print(
                "Wildcard mode enabled."
            )

            budget = (
                get_wildcard_budget(
                    entry,
                    current_squad,
                    picks,
                    bank,
                )
            )

            print(
                f"Wildcard budget: "
                f"£{budget:.1f}m"
            )

            result = optimise_wildcard(
                players,
                budget,
            )

            final_squad = (
                result["squad"]
                .copy()
            )

            final_team = (
                result["team"]
            )

        else:

            # This script is primarily configured
            # for your Wildcard week.
            #
            # If Wildcard is disabled, simply report
            # the current squad rather than performing
            # a full transfer search.

            print(
                "Wildcard mode disabled. "
                "Using current squad."
            )

            final_squad = (
                current_squad.copy()
            )

            budget = (
                float(
                    final_squad[
                        "Price"
                    ].sum()
                )
                + bank
            )

            final_team = (
                team_expected_score(
                    final_squad
                )
            )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        valid, reason = validate_squad(
            final_squad,
            budget,
        )

        if not valid:
            raise RuntimeError(
                f"Final squad failed validation: "
                f"{reason}"
            )

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        report = build_report(
            target_gw,
            final_squad,
            final_team,
            budget,
            bank,
        )

        print("")
        print(
            "=" * 70
        )
        print(report)
        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Telegram
        # ----------------------------------------------------

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
            f"Critical Error encountered: "
            f"{exc}"
        )

        raise


if __name__ == "__main__":
    main()