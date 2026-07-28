import os
from dataclasses import dataclass, field
from typing import cast

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches

from paths import DATA_DIR

HIT_COST = 4

# Spacing and sizing
BOX_HEIGHT = 0.9
BOX_WIDTH = 9
PLAYER_SPACING = 1.2
PLAYER_NAME_FONT_SIZE = 11
STATS_FONT_SIZE = 9
GAMEWEEK_SPACING = 14
POSITION_BORDER_WIDTH = 0.12
CAPTAIN_BORDER_WIDTH = 0.2
CHIP_BACKGROUND_ZORDERS = {"FH": -1.0, "WC": -5.0, "BB": -5.0, "TC": -5.0}

# color scheme
CAPTAIN_COLOR = "#ffd700"
VICE_CAPTAIN_COLOR = "#c0c0c0"
BG_COLOR = "#0f0f0f"
CELL_BG_COLOR = "#1e1e1e"
BENCH_BG_COLOR = "#2a2a2a"
TEXT_COLOR = "#ffffff"
STATS_COLOR = "#c3c2b7"
CHIP_BACKGROUND_COLOR = "#1a1a1a"
CHIP_LABEL_COLOR = "#fbbf24"
TRANSFER_LINE_COLOR = "#60a5fa"

# Validated as CVD-safe and >=3:1 against BG_COLOR on a dark surface
# (node scripts/validate_palette.js "<hexes>" --mode dark)
POSITIONS = ["GKP", "DEF", "MID", "FWD"]
POSITION_COLORS = {"GKP": "#3987e5", "DEF": "#d95926", "MID": "#199e70", "FWD": "#c98500"}
BASE_Y = 16


@dataclass
class RenderContext:
    """Shared state for a single render, computed once instead of per-helper."""

    df: pd.DataFrame
    df_squad: pd.DataFrame
    df_base: pd.DataFrame
    current_squad: list
    statistics: dict
    base_week: int | None  # display week for the pre-transfer squad; None in preseason
    fh_week: int | None  # gameweek Free Hit was played, if any
    first_stats_week: int | None  # earliest week in `statistics`, which only has partial keys
    display_weeks: list
    player_indexes: dict = field(default_factory=dict)


def calculate_bezier(x_start, x_end, y_start, y_end):
    """A curve from (x_start, y_start) to (x_end, y_end), used to draw transfer lines."""
    x_control1 = x_start + (x_end - x_start) * 0.3
    x_control2 = x_start + (x_end - x_start) * 0.7
    y_control1 = y_start + (y_end - y_start) * 0.02
    y_control2 = y_start + (y_end - y_start) * 0.98

    path_data = [
        ((x_start, y_start), mpath.Path.MOVETO),
        ((x_control1, y_control1), mpath.Path.CURVE4),
        ((x_control2, y_control2), mpath.Path.CURVE4),
        ((x_end, y_end), mpath.Path.CURVE4),
    ]
    vertices, codes = zip(*path_data, strict=True)
    return patches.PathPatch(mpath.Path(vertices, codes), facecolor="none", edgecolor=TRANSFER_LINE_COLOR, alpha=0.8, linewidth=1.5, zorder=-3.0)


def _draw_player_cell(ax, gw_idx, player_idx, player):
    """Draw one player's box, position stripe, captaincy marker, name, and stats line."""
    x = gw_idx * GAMEWEEK_SPACING
    y = BASE_Y - player_idx * PLAYER_SPACING

    ax.add_patch(
        patches.Rectangle(
            (x - BOX_WIDTH / 2, y - BOX_HEIGHT / 2),
            BOX_WIDTH,
            BOX_HEIGHT,
            facecolor=CELL_BG_COLOR if player["lineup"] else BENCH_BG_COLOR,
            edgecolor="none",
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (x - BOX_WIDTH / 2, y - BOX_HEIGHT / 2 - POSITION_BORDER_WIDTH),
            BOX_WIDTH,
            POSITION_BORDER_WIDTH,
            facecolor=POSITION_COLORS[player["pos"]],
            edgecolor="none",
        )
    )

    marker_color = None
    if gw_idx > 0:
        marker_color = CAPTAIN_COLOR if player["captain"] == 1 else VICE_CAPTAIN_COLOR if player["vicecaptain"] == 1 else None
    if marker_color is not None:
        ax.add_patch(
            patches.Rectangle((x - BOX_WIDTH / 2, y - BOX_HEIGHT / 2), CAPTAIN_BORDER_WIDTH, BOX_HEIGHT, facecolor=marker_color, edgecolor="none")
        )

    ax.text(x, y + 0.2, player["name"], color=TEXT_COLOR, ha="center", va="center", fontsize=PLAYER_NAME_FONT_SIZE, weight="medium")
    ax.text(
        x, y - 0.25, f"{player['xP']:.1f} xPts • {int(player['xMin'])} xMin", color=STATS_COLOR, ha="center", va="center", fontsize=STATS_FONT_SIZE
    )


def _setup_figure_and_data(picks, current_squad, statistics):
    df = pd.DataFrame(picks)
    df_squad = cast(pd.DataFrame, df[df["squad"] == 1])
    df_base = cast(pd.DataFrame, df[df["week"] == min(df["week"])])
    gameweeks = sorted(int(w) for w in df_squad["week"].unique())

    # Preseason: earliest gameweek is 1, so there's no prior squad to show as a base
    base_week = None if gameweeks[0] == 1 else gameweeks[0] - 1

    fh_rows = df.loc[df["chip"] == "FH"]
    fh_week = int(fh_rows.iloc[0]["week"]) if len(fh_rows) > 0 else None

    # `statistics`' earliest entry only has "itb"/"ft" keys (see dev/solver.py), so it must be
    # skipped when rendering per-gameweek stats. This is a distinct concept from `base_week`
    # above - they usually land on the same gameweek, but aren't guaranteed to.
    first_stats_week = min(int(w) for w in statistics) if statistics else None

    fig, ax = plt.subplots(figsize=(26, 14))
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    return (
        fig,
        ax,
        RenderContext(
            df=df,
            df_squad=df_squad,
            df_base=df_base,
            current_squad=current_squad,
            statistics=statistics,
            base_week=base_week,
            fh_week=fh_week,
            first_stats_week=first_stats_week,
            display_weeks=[base_week, *gameweeks] if base_week is not None else gameweeks,
        ),
    )


def _get_week_players(week, ctx):
    if ctx.base_week is not None and week == ctx.base_week:
        gw_players = ctx.df_base[ctx.df_base["id"].isin(ctx.current_squad)].copy()
        gw_players["lineup"] = 1
        return gw_players
    return ctx.df_squad[ctx.df_squad["week"] == week]


def _add_week_header(ax, gw_idx, week, ctx, gw_players):
    if ctx.base_week is not None and week == ctx.base_week:
        ax.text(gw_idx * GAMEWEEK_SPACING, BASE_Y + 1.2, "Base", color=TEXT_COLOR, fontsize=13, ha="center", weight="bold")
        return

    ax.text(gw_idx * GAMEWEEK_SPACING, BASE_Y + 1.2, f"GW{week}", color=TEXT_COLOR, fontsize=13, ha="center", weight="bold")
    if "chip" not in gw_players.columns:
        return
    played_chips = gw_players["chip"].dropna()
    played_chips = played_chips[played_chips != ""]
    if not played_chips.empty:
        ax.text(gw_idx * GAMEWEEK_SPACING, BASE_Y + 0.8, played_chips.iloc[0], color=CHIP_LABEL_COLOR, fontsize=11, ha="center", weight="bold")


def _add_player_cells(ax, gw_idx, gw_players, week, ctx):
    """Draw the starting XI then the bench, and record each player's row for transfer lines."""
    ordered = pd.concat(
        [
            gw_players[gw_players["lineup"] == 1].sort_values(["type", "xP"], ascending=[True, False]),
            gw_players[gw_players["lineup"] == 0].sort_values(["type", "xP"], ascending=[True, False]),
        ]
    ).reset_index(drop=True)

    week_indexes = {}
    for player_idx, player in ordered.iterrows():
        player_idx = cast(int, player_idx)
        _draw_player_cell(ax, gw_idx, player_idx, player)
        week_indexes[player["id"]] = (BASE_Y - player_idx * PLAYER_SPACING, player["pos"])
    ctx.player_indexes[week] = week_indexes


def _add_transfers(ax, gw_idx, week, ctx):
    if week == 1:
        return

    prev_weeks = [w for w in ctx.player_indexes if w < week]
    prev_week = max(prev_weeks) if prev_weeks else week - 1
    skip_fh = int(prev_week == ctx.fh_week) if ctx.fh_week else 0
    prev_week_indexes = ctx.player_indexes.get(prev_week - skip_fh)

    transfers_in = ctx.df.loc[(ctx.df["week"] == week) & (ctx.df["transfer_in"] == 1)]
    transfers_out = ctx.df.loc[(ctx.df["week"] == week) & (ctx.df["transfer_out"] == 1)]

    for pos in POSITIONS:
        players_out = transfers_out.loc[transfers_out["pos"] == pos].to_dict(orient="records")
        players_in = transfers_in.loc[transfers_in["pos"] == pos].to_dict(orient="records")
        if len(players_out) != len(players_in):
            print(f"Warning: {len(players_out)} {pos} out vs {len(players_in)} in for GW{week} - skipping transfer lines for this position")
            continue

        for player_out, player_in in zip(players_out, players_in, strict=True):
            if prev_week_indexes is None or player_out["id"] not in prev_week_indexes or player_in["id"] not in ctx.player_indexes[week]:
                print(f"Warning: couldn't locate a transferred player's row for GW{week} - skipping this transfer line")
                continue
            x_start = (gw_idx - 1 - skip_fh) * GAMEWEEK_SPACING + BOX_WIDTH / 2
            x_end = gw_idx * GAMEWEEK_SPACING - BOX_WIDTH / 2
            y_start = prev_week_indexes[player_out["id"]][0]
            y_end = ctx.player_indexes[week][player_in["id"]][0]
            ax.add_patch(calculate_bezier(x_start, x_end, y_start, y_end))


def _add_gameweek_statistics(ax, gw_idx, week, ctx, player_idx):
    if week == ctx.first_stats_week:
        return

    stats = ctx.statistics
    x = gw_idx * GAMEWEEK_SPACING
    stats_y = BASE_Y - (player_idx + 0.5) * PLAYER_SPACING
    ax.text(x, stats_y - 0.5, f"{stats[week]['xP']:.2f} xPts", color=TEXT_COLOR, fontsize=11, ha="center", weight="medium")

    if week - 1 in stats:
        itb_text = f"{stats[week - 1]['itb']:.1f} → {stats[week]['itb']:.1f}"
    else:
        itb_text = f"{stats[week]['itb']:.1f}"
    ax.text(x, stats_y - 0.9, f"ITB: {itb_text}", color=STATS_COLOR, fontsize=9, ha="center")

    if week - 1 in stats and stats[week]["chip"] not in ["FH", "WC"]:
        transfer_str = f"FTs: {round(stats[week]['nt'])}/{round(stats[week]['ft'])}"
        if stats[week]["pt"] > 0:
            transfer_str += f" (-{stats[week]['pt'] * HIT_COST})"
        ax.text(x, stats_y - 1.3, transfer_str, color=STATS_COLOR, fontsize=9, ha="center")


def _add_chip_backgrounds(ax, ctx, bottom_limit, top_limit):
    chip_weeks = dict(ctx.df.loc[ctx.df["chip"] != ""][["week", "chip"]].drop_duplicates().values)

    for gw, chip in chip_weeks.items():
        x_center = (gw - ctx.base_week if ctx.base_week is not None else gw - 1) * GAMEWEEK_SPACING
        ax.add_patch(
            patches.FancyBboxPatch(
                (x_center - GAMEWEEK_SPACING / 2, bottom_limit),
                GAMEWEEK_SPACING,
                top_limit - bottom_limit,
                edgecolor="none",
                facecolor=CHIP_BACKGROUND_COLOR,
                zorder=CHIP_BACKGROUND_ZORDERS[chip],
                boxstyle=patches.BoxStyle("Round", pad=-0.3, rounding_size=2),
                alpha=0.85,
            )
        )


def _add_position_legend(ax):
    handles = [patches.Patch(facecolor=POSITION_COLORS[pos], label=pos) for pos in POSITIONS]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.045),
        ncol=len(POSITIONS),
        frameon=False,
        labelcolor=STATS_COLOR,
        fontsize=10,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.2,
    )


def create_squad_timeline(current_squad, statistics, picks, filename):
    """Render a timeline visualization of squad changes across gameweeks to data/images/<filename>.png."""
    _fig, ax, ctx = _setup_figure_and_data(picks, current_squad, statistics)

    for gw_idx, week in enumerate(ctx.display_weeks):
        gw_players = _get_week_players(week, ctx)
        _add_week_header(ax, gw_idx, week, ctx, gw_players)
        _add_player_cells(ax, gw_idx, gw_players, week, ctx)
        _add_transfers(ax, gw_idx, week, ctx)
        _add_gameweek_statistics(ax, gw_idx, week, ctx, len(gw_players) - 1)

    total_width = (len(ctx.display_weeks) - 1) * GAMEWEEK_SPACING + BOX_WIDTH
    ax.set_xlim(-6, total_width + 2)
    max_rows = max((len(v) for v in ctx.player_indexes.values()), default=15)
    bottom_limit = BASE_Y - (max_rows + 1.5) * PLAYER_SPACING
    top_limit = BASE_Y + 2.8
    ax.set_ylim(bottom_limit, top_limit)
    ax.axis("off")

    plt.title(filename, color=TEXT_COLOR, fontsize=14, weight="bold", pad=28)
    _add_chip_backgrounds(ax, ctx, bottom_limit, top_limit)
    _add_position_legend(ax)

    os.makedirs(DATA_DIR / "images", exist_ok=True)
    plt.savefig(DATA_DIR / "images" / f"{filename}.png", bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
