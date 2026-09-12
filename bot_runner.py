import math
import os
from collections import defaultdict

import pandas as pd
import requests
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

MODEL_VERSION = "fast-calibrated-v6-wildcard"

FPL_BASE = "https://fantasy.premierleague.com/api"

TEAM_ID = os.getenv("TEAM_ID") or os.getenv("FPL_TEAM_ID")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

FPL_WILDCARD_THIS_WEEK = (
    os.getenv("FPL_WILDCARD_THIS_WEEK", "true").lower()
    in {"1", "true", "yes", "y", "on"}
)

# Wildcard candidate pool sizes
WILDCARD_POOL_PER_POSITION = {
    1: 30,   # GK
    2: 65,   # DEF
    3: 80,   # MID
    4: 45,   # FWD
}

# Beam-search sizes
WILDCARD_POSITION_BEAM = 1800
WILDCARD_SQUAD_BEAM = 2500
FINAL_EVALUATIONS = 1200

MAX_CLUB_PLAYERS = 3

FORMATION_OPTIONS = [
    (3, 4, 3),
    (3, 5, 2),
    (4, 3, 3),
    (4, 4, 2),
    (4, 5, 1),
    (5, 3, 2),
    (5, 4, 1),
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def normalise_name(name):
    return str(name or "").strip()


# ============================================================
# API
# ============================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 FPL Weekly Manager "
            "GitHubActions/1.0"
        )
    }
)


def fpl_get(path, timeout=30):
    url = f"{FPL_BASE}/{path.lstrip('/')}"

    response = SESSION.get(
        url,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


def fetch_bootstrap():
    print(
        "Fetching bootstrap-static...",
        flush=True,
    )

    return fpl_get("bootstrap-static/")


def fetch_fixtures():
    print(
        "Fetching fixtures...",
        flush=True,
    )

    return fpl_get("fixtures/")


def fetch_current_team(team_id):
    print(
        f"Fetching FPL entry {team_id}...",
        flush=True,
    )

    entry = fpl_get(
        f"entry/{team_id}/"
    )

    # The picks endpoint requires the event number.
    # We initially retrieve event 1 as a fallback and then
    # try the current event below where possible.
    try:
        picks = fpl_get(
            f"entry/{team_id}/event/1/picks/"
        )
    except Exception:
        picks = {}

    try:
        history = fpl_get(
            f"entry/{team_id}/history/"
        )
    except Exception:
        history = {}

    return {
        "entry": entry,
        "picks": picks,
        "history": history,
    }


# ============================================================
# GAMEWEEK
# ============================================================

def determine_target_gameweek(events):
    if not events:
        return None

    # Prefer the official next gameweek.
    for event in events:
        if event.get("is_next"):
            return safe_int(
                event.get("id")
            )

    # Otherwise use the current gameweek.
    for event in events:
        if event.get("is_current"):
            return safe_int(
                event.get("id")
            )

    # Otherwise find the next unfinished gameweek.
    future = [
        safe_int(e.get("id"))
        for e in events
        if safe_int(e.get("id")) > 0
        and not e.get("finished")
    ]

    if future:
        return min(future)

    valid_ids = [
        safe_int(e.get("id"))
        for e in events
        if safe_int(e.get("id")) > 0
    ]

    if valid_ids:
        return max(valid_ids)

    return None


# ============================================================
# FIXTURE CONTEXT
# ============================================================

def build_fixture_context(
    fixtures,
    target_gw,
):
    """
    Creates a lightweight fixture difficulty adjustment.

    Lower FPL difficulty = easier fixture.
    """

    fixture_map = defaultdict(list)

    for fixture in fixtures:

        if safe_int(
            fixture.get("event")
        ) != target_gw:
            continue

        team_h = safe_int(
            fixture.get("team_h")
        )

        team_a = safe_int(
            fixture.get("team_a")
        )

        if team_h:

            fixture_map[team_h].append(
                {
                    "opponent": team_a,
                    "home": True,
                    "difficulty": safe_int(
                        fixture.get(
                            "team_h_difficulty",
                            3,
                        )
                    ),
                }
            )

        if team_a:

            fixture_map[team_a].append(
                {
                    "opponent": team_h,
                    "home": False,
                    "difficulty": safe_int(
                        fixture.get(
                            "team_a_difficulty",
                            3,
                        )
                    ),
                }
            )

    return fixture_map


def difficulty_multiplier(
    difficulty,
):
    """
    Conservative fixture multiplier.

    Fixture difficulty affects the model, but only modestly.
    """

    difficulty = safe_float(
        difficulty,
        3.0,
    )

    mapping = {
        1: 1.10,
        2: 1.06,
        3: 1.00,
        4: 0.94,
        5: 0.89,
    }

    return mapping.get(
        int(round(difficulty)),
        1.0,
    )


# ============================================================
# MINUTES MODEL
# ============================================================

def estimate_minutes(player):
    """
    Estimate expected minutes using the NORMALISED fields
    created by build_players():

        Chance
        Form
        Minutes
        Starts
        Appearances
        Status

    This fixes the previous problem where the model was looking
    for the raw API field names and therefore defaulting almost
    everyone to approximately 48 expected minutes.
    """

    chance = safe_float(
        player.get("Chance"),
        100.0,
    )

    availability = clamp(
        chance / 100.0,
        0.0,
        1.0,
    )

    minutes = safe_float(
        player.get("Minutes")
    )

    starts = safe_float(
        player.get("Starts")
    )

    appearances = safe_float(
        player.get("Appearances")
    )

    form = safe_float(
        player.get("Form")
    )

    status = str(
        player.get("Status", "")
    ).lower().strip()

    # FPL status:
    # u = unavailable
    # i = injured
    # s = suspended
    # n = not available / other
    if status in {
        "u",
        "i",
        "s",
        "n",
    }:
        availability = min(
            availability,
            0.25,
        )

    if chance <= 25:
        availability = min(
            availability,
            0.25,
        )

    # --------------------------------------------------------
    # Starting probability
    # --------------------------------------------------------

    if (
        starts > 0
        and appearances > 0
    ):

        start_rate = clamp(
            starts / appearances,
            0.0,
            1.0,
        )

    elif starts > 0:

        start_rate = clamp(
            starts
            / max(
                starts,
                minutes / 75.0,
                1.0,
            ),
            0.0,
            1.0,
        )

    elif minutes > 0:

        if minutes < 300:
            start_rate = 0.45

        elif minutes < 600:
            start_rate = 0.60

        else:
            start_rate = 0.65

    else:

        start_rate = 0.40

    # Strong evidence from larger samples.
    if (
        minutes >= 1200
        and starts >= 12
    ):
        start_rate = max(
            start_rate,
            0.80,
        )

    elif (
        minutes >= 800
        and starts >= 8
    ):
        start_rate = max(
            start_rate,
            0.70,
        )

    elif (
        minutes >= 400
        and starts >= 4
    ):
        start_rate = max(
            start_rate,
            0.60,
        )

    # Small form adjustment.
    if form >= 7.0:
        start_rate += 0.05

    elif form >= 5.0:
        start_rate += 0.025

    start_rate = clamp(
        start_rate,
        0.20,
        0.97,
    )

    # Blend player-specific probability with neutral prior.
    start_probability = (
        0.85 * start_rate
        + 0.15 * 0.50
    ) * availability

    start_probability = clamp(
        start_probability,
        0.03,
        0.97,
    )

    # --------------------------------------------------------
    # Expected minutes if starting
    # --------------------------------------------------------

    if starts > 0:

        start_minutes = clamp(
            minutes / starts,
            60.0,
            90.0,
        )

    elif minutes > 0:

        denominator = (
            appearances
            if appearances > 0
            else 10.0
        )

        start_minutes = clamp(
            minutes / denominator,
            60.0,
            85.0,
        )

    else:

        start_minutes = 70.0

    # --------------------------------------------------------
    # Probability of 60+ minutes
    # --------------------------------------------------------

    p60 = (
        start_probability
        * clamp(
            start_minutes / 75.0,
            0.80,
            1.0,
        )
    )

    p60 = clamp(
        p60,
        0.0,
        availability,
    )

    p1_59 = clamp(
        availability - p60,
        0.0,
        1.0,
    )

    p0 = clamp(
        1.0 - availability,
        0.0,
        1.0,
    )

    expected_minutes = (
        p60 * start_minutes
        + p1_59 * 20.0
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


# ============================================================
# BAYESIAN SHRINKAGE
# ============================================================

def shrink_rate(
    observed,
    prior,
    sample_size,
    prior_weight=8.0,
):
    """
    Bayesian-style shrinkage.

    Small samples are pulled towards the positional prior.
    """

    observed = safe_float(
        observed
    )

    prior = safe_float(
        prior
    )

    sample_size = max(
        0.0,
        safe_float(sample_size),
    )

    weight = sample_size / (
        sample_size + prior_weight
    )

    return (
        weight * observed
        + (1.0 - weight) * prior
    )


# ============================================================
# PLAYER PROJECTION
# ============================================================

def project_player(
    player,
    team_strengths,
    fixture_context,
):
    position = safe_int(
        player.get("ElementType")
    )

    price = safe_float(
        player.get("Price")
    )

    team_id = safe_int(
        player.get("Team")
    )

    minutes_info = estimate_minutes(
        player
    )

    expected_minutes = (
        minutes_info["expected_minutes"]
    )

    # --------------------------------------------------------
    # Season data
    # --------------------------------------------------------

    xg = safe_float(
        player.get("XG")
    )

    xa = safe_float(
        player.get("XA")
    )

    total_points = safe_float(
        player.get("TotalPoints")
    )

    bps = safe_float(
        player.get("BPS")
    )

    form = safe_float(
        player.get("Form")
    )

    appearances = safe_float(
        player.get("Appearances")
    )

    starts = safe_float(
        player.get("Starts")
    )

    # --------------------------------------------------------
    # Positional priors
    # --------------------------------------------------------

    position_prior = {
        1: {
            "xg90": 0.02,
            "xa90": 0.02,
        },
        2: {
            "xg90": 0.08,
            "xa90": 0.10,
        },
        3: {
            "xg90": 0.20,
            "xa90": 0.20,
        },
        4: {
            "xg90": 0.35,
            "xa90": 0.08,
        },
    }

    prior = position_prior.get(
        position,
        {
            "xg90": 0.10,
            "xa90": 0.10,
        },
    )

    # --------------------------------------------------------
    # Convert xG/xA into rates
    # --------------------------------------------------------

    season_90s = max(
        appearances * 0.75,
        starts * 0.85,
        expected_minutes / 90.0,
        1.0,
    )

    observed_xg90 = (
        xg / season_90s
    )

    observed_xa90 = (
        xa / season_90s
    )

    xg90 = shrink_rate(
        observed_xg90,
        prior["xg90"],
        season_90s,
        prior_weight=8.0,
    )

    xa90 = shrink_rate(
        observed_xa90,
        prior["xa90"],
        season_90s,
        prior_weight=8.0,
    )

    # --------------------------------------------------------
    # Fixture adjustment
    # --------------------------------------------------------

    fixtures = fixture_context.get(
        team_id,
        [],
    )

    if fixtures:

        fixture_mult = (
            sum(
                difficulty_multiplier(
                    f["difficulty"]
                )
                for f in fixtures
            )
            / len(fixtures)
        )

    else:

        fixture_mult = 1.0

    # Small home advantage.
    if fixtures:

        home_count = sum(
            1
            for f in fixtures
            if f.get("home")
        )

        if home_count:
            fixture_mult *= 1.015

    fixture_mult = clamp(
        fixture_mult,
        0.88,
        1.12,
    )

    # --------------------------------------------------------
    # Team strength
    # --------------------------------------------------------

    team_strength = team_strengths.get(
        team_id,
        {},
    )

    attack_strength = safe_float(
        team_strength.get(
            "attack",
            1.0,
        ),
        1.0,
    )

    defence_strength = safe_float(
        team_strength.get(
            "defence",
            1.0,
        ),
        1.0,
    )

    attack_strength = clamp(
        attack_strength,
        0.80,
        1.20,
    )

    defence_strength = clamp(
        defence_strength,
        0.80,
        1.20,
    )

    # --------------------------------------------------------
    # Adjust attacking rates
    # --------------------------------------------------------

    if position in {
        3,
        4,
    }:

        xg90 *= attack_strength
        xa90 *= attack_strength

    elif position == 2:

        xg90 *= (
            attack_strength * 0.90
        )

        xa90 *= attack_strength

    else:

        xg90 *= (
            attack_strength * 0.50
        )

        xa90 *= (
            attack_strength * 0.50
        )

    xg90 *= fixture_mult
    xa90 *= fixture_mult

    # --------------------------------------------------------
    # Expected events
    # --------------------------------------------------------

    mins_factor = (
        expected_minutes / 90.0
    )

    expected_goals = (
        xg90 * mins_factor
    )

    expected_assists = (
        xa90 * mins_factor
    )

    if position == 4:
        goal_points = 4.0
        assist_points = 3.0

    elif position == 3:
        goal_points = 5.0
        assist_points = 3.0

    elif position == 2:
        goal_points = 6.0
        assist_points = 3.0

    else:
        goal_points = 0.0
        assist_points = 0.0

    goal_score = (
        expected_goals
        * goal_points
    )

    assist_score = (
        expected_assists
        * assist_points
    )

    # --------------------------------------------------------
    # Appearance points
    # --------------------------------------------------------

    appearance_score = (
        minutes_info["p60"] * 2.0
        + minutes_info["p1_59"] * 1.0
    )

    # --------------------------------------------------------
    # Clean sheets
    # --------------------------------------------------------

    if position in {
        2,
        3,
    }:

        base_clean_sheet = {
            2: 0.42,
            3: 0.30,
        }[position]

        clean_sheet_probability = (
            base_clean_sheet
            * defence_strength
            * fixture_mult
        )

        clean_sheet_probability = clamp(
            clean_sheet_probability,
            0.05,
            0.65,
        )

        clean_sheet_score = (
            minutes_info["p60"]
            * clean_sheet_probability
            * 4.0
        )

    elif position == 1:

        clean_sheet_probability = clamp(
            0.42
            * defence_strength
            * fixture_mult,
            0.05,
            0.65,
        )

        clean_sheet_score = (
            minutes_info["p60"]
            * clean_sheet_probability
            * 4.0
        )

    else:

        clean_sheet_score = 0.0

    # --------------------------------------------------------
    # Goalkeeper saves
    # --------------------------------------------------------

    save_score = 0.0

    if position == 1:

        expected_saves = (
            2.8
            * mins_factor
        )

        save_score = (
            expected_saves / 3.0
        )

    # --------------------------------------------------------
    # Bonus / BPS
    # --------------------------------------------------------

    if appearances > 0:

        bps_per_app = (
            bps / appearances
        )

    else:

        bps_per_app = 0.0

    bonus_score = clamp(
        bps_per_app / 40.0,
        0.0,
        1.2,
    )

    # --------------------------------------------------------
    # Form adjustment
    # --------------------------------------------------------

    form_adjustment = clamp(
        (form - 5.0) * 0.05,
        -0.20,
        0.20,
    )

    # --------------------------------------------------------
    # Historical stabiliser
    # --------------------------------------------------------

    if appearances > 0:

        points_per_app = (
            total_points
            / appearances
        )

    else:

        points_per_app = 0.0

    historical_stabiliser = clamp(
        points_per_app * 0.04,
        0.0,
        0.35,
    )

    # --------------------------------------------------------
    # Final xP
    # --------------------------------------------------------

    expected_points = (
        appearance_score
        + goal_score
        + assist_score
        + clean_sheet_score
        + save_score
        + bonus_score
        + form_adjustment
        + historical_stabiliser
    )

    expected_points = clamp(
        expected_points,
        0.0,
        15.0,
    )

    return {
        **player,
        "xP": expected_points,
        "ExpectedMinutes": expected_minutes,
        "StartProbability": minutes_info[
            "start_probability"
        ],
        "P60": minutes_info["p60"],
        "FixtureMultiplier": fixture_mult,
        "XG90": xg90,
        "XA90": xa90,
        "ExpectedGoals": expected_goals,
        "ExpectedAssists": expected_assists,
        "CleanSheetScore": clean_sheet_score,
        "Value": (
            expected_points / price
            if price > 0
            else 0.0
        ),
    }


# ============================================================
# BUILD PLAYER DATA
# ============================================================

def build_players(
    bootstrap,
    fixtures,
    target_gw,
):
    elements = bootstrap.get(
        "elements",
        [],
    )

    teams = bootstrap.get(
        "teams",
        [],
    )

    team_strengths = {}

    for team in teams:

        team_id = safe_int(
            team.get("id")
        )

        attack_home = safe_float(
            team.get(
                "strength_attack_home",
                1000,
            ),
            1000,
        )

        attack_away = safe_float(
            team.get(
                "strength_attack_away",
                1000,
            ),
            1000,
        )

        defence_home = safe_float(
            team.get(
                "strength_defence_home",
                1000,
            ),
            1000,
        )

        defence_away = safe_float(
            team.get(
                "strength_defence_away",
                1000,
            ),
            1000,
        )

        team_strengths[team_id] = {
            "attack": clamp(
                (
                    attack_home
                    + attack_away
                ) / 2000.0,
                0.80,
                1.20,
            ),
            "defence": clamp(
                (
                    defence_home
                    + defence_away
                ) / 2000.0,
                0.80,
                1.20,
            ),
        }

    fixture_context = (
        build_fixture_context(
            fixtures,
            target_gw,
        )
    )

    players = []

    for raw in elements:

        status = str(
            raw.get(
                "status",
                "",
            )
        ).lower().strip()

        # Exclude unavailable players.
        if status in {
            "u",
            "i",
            "s",
            "n",
        }:
            continue

        chance = safe_float(
            raw.get(
                "chance_of_playing_next_round"
            ),
            100.0,
        )

        # Exclude players with very low chance.
        if chance <= 25:
            continue

        player = {
            "id": safe_int(
                raw.get("id")
            ),

            "Name": normalise_name(
                raw.get("web_name")
                or raw.get("second_name")
            ),

            "FirstName": normalise_name(
                raw.get("first_name")
            ),

            "SecondName": normalise_name(
                raw.get("second_name")
            ),

            "Team": safe_int(
                raw.get("team")
            ),

            "ElementType": safe_int(
                raw.get("element_type")
            ),

            "Price": (
                safe_float(
                    raw.get("now_cost")
                ) / 10.0
            ),

            "TotalPoints": safe_float(
                raw.get("total_points")
            ),

            "Form": safe_float(
                raw.get("form")
            ),

            "PointsPerGame": safe_float(
                raw.get("points_per_game")
            ),

            "BPS": safe_float(
                raw.get("bps")
            ),

            "XG": safe_float(
                raw.get("expected_goals")
            ),

            "XA": safe_float(
                raw.get("expected_assists")
            ),

            "Minutes": safe_float(
                raw.get("minutes")
            ),

            "Starts": safe_float(
                raw.get("starts")
            ),

            "Appearances": safe_float(
                raw.get("appearances")
            ),

            "Chance": chance,

            "Status": status,

            "Selected": safe_float(
                raw.get(
                    "selected_by_percent"
                )
            ),
        }

        projected = project_player(
            player,
            team_strengths,
            fixture_context,
        )

        players.append(
            projected
        )

    df = pd.DataFrame(
        players
    )

    if df.empty:
        raise RuntimeError(
            "No usable FPL players were returned."
        )

    return df


# ============================================================
# WILDCARD BUDGET
# ============================================================

def get_wildcard_budget(
    entry,
    picks,
    players_df,
):
    """
    FPL monetary fields are in £0.1m.

    Prefer entry.value when available.

    Otherwise calculate current squad market value
    as a fallback.
    """

    bank = (
        safe_float(
            entry.get("bank")
        ) / 10.0
    )

    entry_value_raw = (
        entry.get("value")
    )

    if (
        entry_value_raw is not None
        and safe_float(
            entry_value_raw
        ) > 0
    ):

        squad_value = (
            safe_float(
                entry_value_raw
            ) / 10.0
        )

        return (
            squad_value + bank,
            bank,
        )

    current_ids = {
        safe_int(
            p.get("element")
        )
        for p in picks.get(
            "picks",
            [],
        )
    }

    current_players = players_df[
        players_df["id"].isin(
            current_ids
        )
    ]

    if not current_players.empty:

        squad_value = current_players[
            "Price"
        ].sum()

        return (
            squad_value + bank,
            bank,
        )

    return (
        100.0 + bank,
        bank,
    )


# ============================================================
# STARTING XI
# ============================================================

def find_best_starting_xi(
    squad,
    captain_id=None,
):
    """
    Optimise the legal starting XI across all
    supported FPL formations.
    """

    if squad.empty:
        return None

    # --------------------------------------------------------
    # DEFENSIVE FIX:
    #
    # Some internal optimiser operations use player ID as
    # the DataFrame index. The rest of the model expects
    # squad["id"] to exist.
    #
    # Restore the index as a normal column if necessary.
    # --------------------------------------------------------

    if "id" not in squad.columns:

        if squad.index.name == "id":

            squad = squad.reset_index()

        else:

            return None

    best = None

    gks = squad[
        squad["ElementType"] == 1
    ].sort_values(
        "xP",
        ascending=False,
    )

    defs = squad[
        squad["ElementType"] == 2
    ].sort_values(
        "xP",
        ascending=False,
    )

    mids = squad[
        squad["ElementType"] == 3
    ].sort_values(
        "xP",
        ascending=False,
    )

    fwds = squad[
        squad["ElementType"] == 4
    ].sort_values(
        "xP",
        ascending=False,
    )

    if (
        len(gks) < 1
        or len(defs) < 3
        or len(mids) < 2
        or len(fwds) < 1
    ):
        return None

    for (
        def_count,
        mid_count,
        fwd_count,
    ) in FORMATION_OPTIONS:

        if len(defs) < def_count:
            continue

        if len(mids) < mid_count:
            continue

        if len(fwds) < fwd_count:
            continue

        selected = pd.concat(
            [
                gks.head(1),
                defs.head(
                    def_count
                ),
                mids.head(
                    mid_count
                ),
                fwds.head(
                    fwd_count
                ),
            ]
        ).copy()

        # ----------------------------------------------------
        # Captain
        # ----------------------------------------------------

        if captain_id is not None:

            captain_rows = selected[
                selected["id"]
                == captain_id
            ]

            if not captain_rows.empty:

                captain = (
                    captain_rows.iloc[0]
                )

            else:

                captain = (
                    selected.sort_values(
                        "xP",
                        ascending=False,
                    ).iloc[0]
                )

        else:

            captain = (
                selected.sort_values(
                    "xP",
                    ascending=False,
                ).iloc[0]
            )

        captain_id_actual = safe_int(
            captain["id"]
        )

        outfield_for_vice = selected[
            selected["id"]
            != captain_id_actual
        ]

        vice = (
            outfield_for_vice.sort_values(
                "xP",
                ascending=False,
            ).iloc[0]
        )

        vice_id = safe_int(
            vice["id"]
        )

        # Captain gets an additional copy of his
        # expected points.
        score = (
            selected["xP"].sum()
            + captain["xP"]
        )

        result = {
            "formation": (
                def_count,
                mid_count,
                fwd_count,
            ),
            "xi": selected,
            "captain": captain_id_actual,
            "vice": vice_id,
            "score": score,
        }

        if (
            best is None
            or result["score"]
            > best["score"]
        ):

            best = result

    return best


# ============================================================
# WILDCARD CANDIDATES
# ============================================================

def wildcard_candidates(
    players_df,
):
    candidates = []

    for (
        position,
        pool_size,
    ) in WILDCARD_POOL_PER_POSITION.items():

        df = players_df[
            players_df["ElementType"]
            == position
        ].copy()

        if df.empty:

            candidates.append(
                df
            )

            continue

        # Candidate score blends:
        # - projected points
        # - value
        # - form
        # - historical output
        df["CandidateScore"] = (
            df["xP"] * 0.65
            + df["Value"] * 4.0 * 0.15
            + df["Form"].clip(
                lower=0,
                upper=10,
            ) * 0.05
            + (
                df["TotalPoints"]
                .rank(pct=True)
                * 0.15
            )
        )

        top_xp = df.nlargest(
            min(
                pool_size,
                len(df),
            ),
            "xP",
        )

        top_value = df.nlargest(
            min(
                max(
                    10,
                    pool_size // 3,
                ),
                len(df),
            ),
            "Value",
        )

        combined = pd.concat(
            [
                top_xp,
                top_value,
            ]
        ).drop_duplicates(
            subset=["id"]
        )

        combined = (
            combined.sort_values(
                [
                    "CandidateScore",
                    "xP",
                ],
                ascending=False,
            )
            .head(pool_size)
        )

        candidates.append(
            combined
        )

    return candidates


# ============================================================
# FAST POSITION BEAM SEARCH
# ============================================================

def build_position_bundles(
    position_df,
    count,
    beam_size,
):
    """
    Incremental beam search.

    Avoids expensive combinations(players, count).

    For example:
        28 choose 5 = 98,280 combinations

    Beam search keeps only the strongest partial states.
    """

    if len(position_df) < count:
        return []

    players = (
        position_df.sort_values(
            [
                "xP",
                "Value",
            ],
            ascending=False,
        )
        .to_dict("records")
    )

    states = [
        {
            "ids": (),
            "cost10": 0,
            "score": 0.0,
            "clubs": {},
            "last_index": -1,
        }
    ]

    for slot in range(count):

        new_states = []

        for state in states:

            start_index = (
                state["last_index"]
                + 1
            )

            for idx in range(
                start_index,
                len(players),
            ):

                player = players[idx]

                player_id = safe_int(
                    player["id"]
                )

                team_id = safe_int(
                    player["Team"]
                )

                new_cost10 = (
                    state["cost10"]
                    + round(
                        safe_float(
                            player["Price"]
                        ) * 10
                    )
                )

                clubs = dict(
                    state["clubs"]
                )

                clubs[team_id] = (
                    clubs.get(
                        team_id,
                        0,
                    )
                    + 1
                )

                if (
                    clubs[team_id]
                    > MAX_CLUB_PLAYERS
                ):
                    continue

                new_states.append(
                    {
                        "ids": (
                            state["ids"]
                            + (player_id,)
                        ),
                        "cost10": new_cost10,
                        "score": (
                            state["score"]
                            + safe_float(
                                player["xP"]
                            )
                        ),
                        "clubs": clubs,
                        "last_index": idx,
                    }
                )

        if not new_states:
            return []

        # ----------------------------------------------------
        # Dominance filtering
        # ----------------------------------------------------

        best_by_key = {}

        for state in new_states:

            club_signature = tuple(
                sorted(
                    state[
                        "clubs"
                    ].items()
                )
            )

            cost_bucket = (
                state["cost10"] // 2
            )

            key = (
                cost_bucket,
                club_signature,
            )

            previous = (
                best_by_key.get(key)
            )

            if (
                previous is None
                or state["score"]
                > previous["score"]
            ):

                best_by_key[key] = state

        states = sorted(
            best_by_key.values(),
            key=lambda s: (
                s["score"]
                + 0.12
                * (
                    s["score"]
                    / max(
                        s["cost10"],
                        1,
                    )
                )
            ),
            reverse=True,
        )[:beam_size]

    return states


# ============================================================
# WILDCARD OPTIMISER
# ============================================================

def optimise_wildcard(
    players_df,
    budget,
):
    print(
        "Building Wildcard candidate pools...",
        flush=True,
    )

    candidate_frames = (
        wildcard_candidates(
            players_df
        )
    )

    by_position = {}

    for (
        position,
        df,
    ) in zip(
        [1, 2, 3, 4],
        candidate_frames,
    ):

        by_position[position] = df

        print(
            f"Position {position}: "
            f"{len(df)} candidates",
            flush=True,
        )

    required = {
        1: 2,
        2: 5,
        3: 5,
        4: 3,
    }

    # --------------------------------------------------------
    # Build position bundles
    # --------------------------------------------------------

    bundles = {}

    for (
        position,
        count,
    ) in required.items():

        print(
            f"Building position {position} "
            f"bundles ({count})...",
            flush=True,
        )

        bundles[position] = (
            build_position_bundles(
                by_position[position],
                count,
                WILDCARD_POSITION_BEAM,
            )
        )

        print(
            f"  -> "
            f"{len(bundles[position])} "
            f"bundles",
            flush=True,
        )

        if not bundles[position]:

            raise RuntimeError(
                f"Unable to build enough "
                f"players for position "
                f"{position}."
            )

    # --------------------------------------------------------
    # Combine position bundles
    # --------------------------------------------------------

    states = [
        {
            "ids": (),
            "cost10": 0,
            "score": 0.0,
            "clubs": {},
        }
    ]

    budget10 = round(
        budget * 10
    )

    for position in [
        1,
        2,
        3,
        4,
    ]:

        next_states = []

        for state in states:

            for bundle in bundles[position]:

                new_cost10 = (
                    state["cost10"]
                    + bundle["cost10"]
                )

                if (
                    new_cost10
                    > budget10
                ):
                    continue

                clubs = dict(
                    state["clubs"]
                )

                valid = True

                for (
                    club_id,
                    count,
                ) in bundle[
                    "clubs"
                ].items():

                    clubs[club_id] = (
                        clubs.get(
                            club_id,
                            0,
                        )
                        + count
                    )

                    if (
                        clubs[club_id]
                        > MAX_CLUB_PLAYERS
                    ):

                        valid = False
                        break

                if not valid:
                    continue

                ids = (
                    state["ids"]
                    + bundle["ids"]
                )

                next_states.append(
                    {
                        "ids": ids,
                        "cost10": new_cost10,
                        "score": (
                            state["score"]
                            + bundle["score"]
                        ),
                        "clubs": clubs,
                    }
                )

        if not next_states:

            raise RuntimeError(
                "Wildcard optimiser found "
                "no budget-feasible squad."
            )

        # ----------------------------------------------------
        # Deduplicate / beam
        # ----------------------------------------------------

        best_by_key = {}

        for state in next_states:

            cost_bucket = (
                state["cost10"] // 2
            )

            club_signature = tuple(
                sorted(
                    state[
                        "clubs"
                    ].items()
                )
            )

            key = (
                cost_bucket,
                club_signature,
            )

            old = (
                best_by_key.get(key)
            )

            if (
                old is None
                or state["score"]
                > old["score"]
            ):

                best_by_key[key] = state

        states = sorted(
            best_by_key.values(),
            key=lambda s: (
                s["score"]
                + 0.10
                * (
                    s["score"]
                    / max(
                        s["cost10"],
                        1,
                    )
                )
            ),
            reverse=True,
        )[:WILDCARD_SQUAD_BEAM]

        print(
            f"After position "
            f"{position}: "
            f"{len(states)} squad states",
            flush=True,
        )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    print(
        "Evaluating final Wildcard squads...",
        flush=True,
    )

    # IMPORTANT:
    # We deliberately create a normal DataFrame lookup.
    # The previous version set id as the index and then passed
    # the resulting DataFrame to functions expecting an "id"
    # column. That caused:
    #
    # KeyError: 'id'
    #
    # We now reset_index() immediately after selecting.
    player_lookup = (
        players_df.set_index("id")
    )

    best = None

    for state in states[
        :FINAL_EVALUATIONS
    ]:

        ids = list(
            state["ids"]
        )

        try:

            squad = player_lookup.loc[
                ids
            ].copy()

            # ------------------------------------------------
            # CRITICAL FIX
            # ------------------------------------------------
            # Restore id as a normal column.
            # ------------------------------------------------

            squad = squad.reset_index()

        except Exception:

            continue

        if len(squad) != 15:
            continue

        # ----------------------------------------------------
        # Budget
        # ----------------------------------------------------

        cost = safe_float(
            squad["Price"].sum()
        )

        if cost > budget + 1e-9:
            continue

        # ----------------------------------------------------
        # Exact squad composition
        # ----------------------------------------------------

        counts = (
            squad[
                "ElementType"
            ]
            .value_counts()
            .to_dict()
        )

        if counts.get(1, 0) != 2:
            continue

        if counts.get(2, 0) != 5:
            continue

        if counts.get(3, 0) != 5:
            continue

        if counts.get(4, 0) != 3:
            continue

        # ----------------------------------------------------
        # Club limit
        # ----------------------------------------------------

        if (
            squad["Team"]
            .value_counts()
            .max()
            > MAX_CLUB_PLAYERS
        ):
            continue

        # ----------------------------------------------------
        # Starting XI
        # ----------------------------------------------------

        starting = (
            find_best_starting_xi(
                squad
            )
        )

        if starting is None:
            continue

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        final_score = (
            starting["score"]
            + 0.04
            * (
                budget - cost
            )
        )

        if (
            best is None
            or final_score
            > best["score"]
        ):

            best = {
                "squad": squad,
                "starting": starting,
                "cost": cost,
                "score": final_score,
            }

    if best is None:

        raise RuntimeError(
            "Wildcard optimiser could not "
            "produce a legal squad."
        )

    return best


# ============================================================
# NORMAL MODE / ONE-TRANSFER OPTIMISER
# ============================================================

def optimise_existing_team(
    current_ids,
    players_df,
    bank,
):
    """
    Conservative one-transfer optimiser.

    Used when Wildcard mode is disabled.
    """

    current = players_df[
        players_df["id"].isin(
            current_ids
        )
    ].copy()

    if current.empty:

        raise RuntimeError(
            "Could not reconstruct current squad."
        )

    starting = (
        find_best_starting_xi(
            current
        )
    )

    if starting is None:

        raise RuntimeError(
            "Current squad cannot form "
            "a legal XI."
        )

    current_score = (
        starting["score"]
    )

    best = {
        "squad": current,
        "starting": starting,
        "cost": current["Price"].sum(),
        "score": current_score,
        "transfer": None,
    }

    current_ids_set = set(
        current_ids
    )

    for out_id in current_ids:

        out_row = current[
            current["id"] == out_id
        ]

        if out_row.empty:
            continue

        out_player = (
            out_row.iloc[0]
        )

        available_budget = (
            bank
            + safe_float(
                out_player["Price"]
            )
        )

        candidates = players_df[
            (
                ~players_df["id"].isin(
                    current_ids_set
                )
            )
            & (
                players_df["ElementType"]
                == out_player[
                    "ElementType"
                ]
            )
            & (
                players_df["Price"]
                <= available_budget
            )
        ].copy()

        candidates = candidates.nlargest(
            30,
            "xP",
        )

        for _, new_player in (
            candidates.iterrows()
        ):

            test = current[
                current["id"]
                != out_id
            ].copy()

            test = pd.concat(
                [
                    test,
                    pd.DataFrame(
                        [new_player]
                    ),
                ],
                ignore_index=True,
            )

            if len(test) != 15:
                continue

            if (
                test["Team"]
                .value_counts()
                .max()
                > MAX_CLUB_PLAYERS
            ):
                continue

            starting_test = (
                find_best_starting_xi(
                    test
                )
            )

            if starting_test is None:
                continue

            score = (
                starting_test["score"]
            )

            if score > best["score"]:

                best = {
                    "squad": test,
                    "starting": starting_test,
                    "cost": test[
                        "Price"
                    ].sum(),
                    "score": score,
                    "transfer": {
                        "out": out_player,
                        "in": new_player,
                    },
                }

    return best


# ============================================================
# REPORT
# ============================================================

def format_price(price):
    return f"£{price:.1f}m"


def position_name(position):
    return {
        1: "GK",
        2: "DEF",
        3: "MID",
        4: "FWD",
    }.get(
        safe_int(position),
        "?",
    )


def build_report(
    target_gw,
    result,
    budget,
    bank,
    used_chips,
):
    starting = result[
        "starting"
    ]

    squad = result[
        "squad"
    ]

    formation = starting[
        "formation"
    ]

    (
        def_count,
        mid_count,
        fwd_count,
    ) = formation

    formation_text = (
        f"{def_count}-"
        f"{mid_count}-"
        f"{fwd_count}"
    )

    lines = []

    lines.append(
        f"FPL Weekly Manager — GW{target_gw}"
    )

    lines.append(
        f"Model: {MODEL_VERSION}"
    )

    lines.append("")

    if FPL_WILDCARD_THIS_WEEK:

        lines.append(
            "🃏 WILDCARD ACTIVE"
        )

        lines.append(
            f"Wildcard budget: "
            f"{format_price(budget)}"
        )

        lines.append(
            f"Bank: "
            f"{format_price(bank)}"
        )

        lines.append("")

    lines.append(
        f"STARTING XI "
        f"({formation_text})"
    )

    xi = starting[
        "xi"
    ].copy()

    captain_id = starting[
        "captain"
    ]

    vice_id = starting[
        "vice"
    ]

    for _, player in xi.iterrows():

        captain_marker = ""

        player_id = safe_int(
            player["id"]
        )

        if player_id == captain_id:

            captain_marker = " ©"

        elif player_id == vice_id:

            captain_marker = " (V)"

        lines.append(
            f"• {player['Name']} "
            f"({position_name(player['ElementType'])}) "
            f"{format_price(player['Price'])} "
            f"xP {player['xP']:.2f} "
            f"mins {player['ExpectedMinutes']:.0f}"
            f"{captain_marker}"
        )

    captain_rows = xi[
        xi["id"] == captain_id
    ]

    vice_rows = xi[
        xi["id"] == vice_id
    ]

    if captain_rows.empty:

        captain_row = (
            xi.sort_values(
                "xP",
                ascending=False,
            ).iloc[0]
        )

    else:

        captain_row = (
            captain_rows.iloc[0]
        )

    if vice_rows.empty:

        vice_row = (
            xi.sort_values(
                "xP",
                ascending=False,
            ).iloc[1]
        )

    else:

        vice_row = (
            vice_rows.iloc[0]
        )

    lines.append("")

    lines.append(
        f"Captain: "
        f"{captain_row['Name']} "
        f"({captain_row['xP']:.2f} xP)"
    )

    lines.append(
        f"Vice: "
        f"{vice_row['Name']} "
        f"({vice_row['xP']:.2f} xP)"
    )

    # --------------------------------------------------------
    # Bench & Chip Tracking Logic
    # --------------------------------------------------------

    xi_ids = set(
        safe_int(x)
        for x in xi["id"].tolist()
    )

    bench = squad[
        ~squad["id"].isin(
            xi_ids
        )
    ].copy()
    
    bench_xp = safe_float(bench["xP"].sum())
    captain_xp = safe_float(captain_row["xP"])
    
    chip_advice = "Hold Chips 🛡️ (Save Free Hit / Wildcards)"
    if FPL_WILDCARD_THIS_WEEK:
        chip_advice = "Wildcard Active 🃏"
    elif captain_xp >= 11.5 and "3xc" not in used_chips:
        chip_advice = "Triple Captain Recommended 🚀"
    elif bench_xp >= 22.0 and "bboost" not in used_chips:
        chip_advice = "Bench Boost Recommended 📈"
    elif "3xc" in used_chips and captain_xp >= 11.5:
        chip_advice = "Hold Chips 🛡️ (Triple Captain already used)"
    elif "bboost" in used_chips and bench_xp >= 22.0:
        chip_advice = "Hold Chips 🛡️ (Bench Boost already used)"

    lines.append(f"🎯 Chip Strategy: {chip_advice}")

    bench["BenchOrder"] = (
        bench["ElementType"].map(
            {
                1: 0,
                2: 1,
                3: 2,
                4: 3,
            }
        )
    )

    bench = bench.sort_values(
        [
            "BenchOrder",
            "xP",
        ],
        ascending=[
            True,
            False,
        ],
    )

    lines.append("")
    lines.append("BENCH")

    for _, player in (
        bench.iterrows()
    ):

        lines.append(
            f"• {player['Name']} "
            f"({position_name(player['ElementType'])}) "
            f"{format_price(player['Price'])} "
            f"xP {player['xP']:.2f} "
            f"mins {player['ExpectedMinutes']:.0f}"
        )

    squad_cost = safe_float(
        squad["Price"].sum()
    )

    remaining = (
        budget - squad_cost
    )

    lines.append("")

    lines.append(
        f"Squad cost: "
        f"{format_price(squad_cost)}"
    )

    lines.append(
        f"Budget remaining: "
        f"{format_price(remaining)}"
    )

    lines.append(
        f"Projected XI + captain: "
        f"{starting['score']:.2f}"
    )

    # --------------------------------------------------------
    # Transfer information
    # --------------------------------------------------------

    if result.get(
        "transfer"
    ):

        transfer = result[
            "transfer"
        ]

        out_player = transfer[
            "out"
        ]

        in_player = transfer[
            "in"
        ]

        lines.append("")

        lines.append(
            "TRANSFER"
        )

        lines.append(
            f"OUT: "
            f"{out_player['Name']} "
            f"{format_price(out_player['Price'])}"
        )

        lines.append(
            f"IN: "
            f"{in_player['Name']} "
            f"{format_price(in_player['Price'])}"
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
        "• Minutes model uses starts, appearances, "
        "minutes, form and availability."
    )

    lines.append(
        "• Starting XI is optimised across legal formations."
    )

    if FPL_WILDCARD_THIS_WEEK:

        lines.append(
            "• Wildcard optimises the full 15-man squad."
        )

    return "\n".join(lines)


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(message):

    if not TELEGRAM_BOT_TOKEN:

        print(
            "TELEGRAM_BOT_TOKEN not configured; "
            "skipping Telegram.",
            flush=True,
        )

        return

    if not TELEGRAM_CHAT_ID:

        print(
            "TELEGRAM_CHAT_ID not configured; "
            "skipping Telegram.",
            flush=True,
        )

        return

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}"
        "/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
    }

    response = requests.post(
        url,
        json=payload,
        timeout=30,
    )

    response.raise_for_status()

    print(
        "Telegram report sent successfully.",
        flush=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Starting FPL Weekly Manager "
        f"{MODEL_VERSION}",
        flush=True,
    )

    if not TEAM_ID:

        raise RuntimeError(
            "TEAM_ID / FPL_TEAM_ID "
            "is not configured."
        )

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    bootstrap = (
        fetch_bootstrap()
    )

    fixtures = (
        fetch_fixtures()
    )

    events = bootstrap.get(
        "events",
        [],
    )

    target_gw = (
        determine_target_gameweek(
            events
        )
    )

    if not target_gw:

        raise RuntimeError(
            "Could not determine "
            "target gameweek."
        )

    print(
        f"Projection target: GW{target_gw}",
        flush=True,
    )

    current_team = (
        fetch_current_team(
            TEAM_ID
        )
    )

    entry = current_team[
        "entry"
    ]

    picks = current_team[
        "picks"
    ]
    
    history = current_team[
        "history"
    ]
    
    used_chips = set()
    for chip in history.get("chips", []):
        used_chips.add(chip.get("name"))

    # --------------------------------------------------------
    # Player projections
    # --------------------------------------------------------

    print(
        "Building player projections...",
        flush=True,
    )

    players_df = build_players(
        bootstrap,
        fixtures,
        target_gw,
    )

    print(
        f"Usable players: "
        f"{len(players_df)}",
        flush=True,
    )

    # --------------------------------------------------------
    # Wildcard
    # --------------------------------------------------------

    if FPL_WILDCARD_THIS_WEEK:

        budget, bank = (
            get_wildcard_budget(
                entry,
                picks,
                players_df,
            )
        )

        print(
            f"Wildcard budget: "
            f"£{budget:.1f}m",
            flush=True,
        )

        print(
            f"Bank: "
            f"£{bank:.1f}m",
            flush=True,
        )

        result = (
            optimise_wildcard(
                players_df,
                budget,
            )
        )

    # --------------------------------------------------------
    # Normal mode
    # --------------------------------------------------------

    else:

        current_ids = [
            safe_int(
                p.get("element")
            )
            for p in picks.get(
                "picks",
                [],
            )
        ]

        bank = (
            safe_float(
                entry.get("bank")
            ) / 10.0
        )

        result = (
            optimise_existing_team(
                current_ids,
                players_df,
                bank,
            )
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    if FPL_WILDCARD_THIS_WEEK:

        report_budget, report_bank = (
            get_wildcard_budget(
                entry,
                picks,
                players_df,
            )
        )

    else:

        report_budget = result[
            "cost"
        ]

        report_bank = bank

    report = build_report(
        target_gw,
        result,
        report_budget,
        report_bank,
        used_chips,
    )

    print("")
    print(report)
    print("")

    # --------------------------------------------------------
    # Telegram
    # --------------------------------------------------------

    send_telegram(
        report
    )

    print(
        "FPL Weekly Manager completed successfully.",
        flush=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        print(
            "Critical Error encountered:",
            flush=True,
        )

        print(
            str(exc),
            flush=True,
        )

        raise