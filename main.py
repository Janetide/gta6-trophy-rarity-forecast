"""GTA VI trophy rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Reads every <id>.in case in the input directory and writes <id>.out with five
quantiles (5/25/50/75/95%) of each trophy's Exophase earn rate at the case's
target. Python 3.13, standard library only, no network, deterministic: two
runs on the same case write the same bytes.

Method (per trophy, from the case history alone)
------------------------------------------------
An earn rate is a ratio: earners / players tracked. The two move differently
over launch week, so the forecast models them separately and divides:

- Players tracked N(t): the tracked base grows fast at launch and then
  slows. Fit log N against time over the recent distinct reads and damp the
  growth: the effective growth window is TAU * (1 - exp(-H / TAU)) with
  TAU = 24 h, so a 4.5-day horizon only grows the base for about a day at
  the recent rate instead of extrapolating the launch surge forever.
- Earners E(t) = rate * N / 100: fit log E against time the same way, per
  trophy, so story trophies the early cohort is racing through grow fast
  and hard trophies barely move. Earners never shrink, so the fitted
  growth rate is floored at 0.
- Median rate at the target = 100 * E(target) / N(target), clamped to
  0.01..99.99%, then moved only three quarters of the way from the last
  rate toward that projection: recent growth over-extrapolates over a
  multi-day horizon, so the move is shrunk. Because N keeps growing while
  E saturates, hard trophies are forecast to fall or stay flat and easy
  ones to rise, which is what the launch-week dilution does to the page.

Spread: the quantiles sit on the logit scale around the median, with a
sigma built from the scatter of the recent log-E and log-N fits (how noisy
the recent reads were) grown with the horizon, over a floor of 0.5 logits:
even a smooth-looking history does not justify pretending to more
precision than that over a multi-day launch week. A trophy nobody has
earned yet gets a median at the floor and a one-sided spread upward,
wider for bronze than for platinum.

Frozen page: when the history holds fewer than 3 distinct reads, the
page has not refreshed since launch night, so the level is stale and a
refresh before the target is likely. Refreshes dilute rates (the tracked
base jumps while earners barely move), so the median is shifted down a
logit and the spread is widened a lot. This is the practice cases'
shape: a flat launch-night page whose answer is the first refreshed one.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only. -P keeps this file's directory
OFF sys.path, so a module shipped beside main.py imports only after the
sys.path line below; -s drops user site-packages.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at
    a time, with only that case's file there): JSON
    {"schema": "makefun/gta6-trophies-case/v1", "id", "at",
    "horizon_minutes", "target", "levels", "game", "trophies", "history",
    "notes"}, where `trophies` lists the case's trophies {id, name,
    description, tier, points, secret} and `history` every read before
    `at`, oldest first, each {slot, at, players_tracked,
    trophies_earned, rates}, `rates` in the order of `trophies` (null
    where that read lacked the trophy);
  - write <id>.out to the output directory:
    {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}, one
    row for each of the case's trophies, each id once, no other key, each
    q five finite numbers from 0 to 100 (percent), non-decreasing; at most
    65,536 bytes. Anything else is an invalid case, scored as if every
    quantile equalled the trophy's rate in the last read, and an entry
    with an invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file. (This entry ships none; the line
# is here so the layout stays the standard one.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Growth is damped over this timescale (hours): a trend is only trusted for
# about this long before it is assumed to saturate.
TAU_HOURS = 24.0
# The recent window the growth rates are fitted over (hours).
WINDOW_HOURS = 24.0
# Shrink on the forecast move: the median is the last rate moved only this
# fraction of the way toward the model's projection. Recent growth
# over-extrapolates over a multi-day horizon (the launch surge decays
# faster than the trend), so the move is taken at three quarters, not whole.
TREND_SHRINK = 0.75
# Caps on fitted daily log growth, per day: earners at most ~3x a day, the
# tracked base at most ~2x a day.
MAX_GROWTH_EARNERS = 1.1
MAX_GROWTH_PLAYERS = 0.7
# Spread floor (logit units) and cap. The floor is the honest minimum for a
# multi-day launch-week forecast: even when the recent reads look smooth,
# the model is not the true process, so the quantiles never pretend to
# more precision than about half a logit.
SIGMA_FLOOR = 0.50
SIGMA_CAP = 1.80
# Frozen page (fewer than this many distinct reads): shift the median down
# this many logits and use this spread instead of the fitted one.
FROZEN_MIN_READS = 3
FROZEN_SHIFT = 1.0
FROZEN_SIGMA = 1.3
# A trophy with no earners yet: the 95% quantile by tier (percent).
ZERO_EARNER_Q95 = {"bronze": 2.0, "silver": 1.0, "gold": 0.3, "platinum": 0.05}


def parse_time(text):
    """ISO 8601 with a Z suffix or an offset, to an aware datetime (UTC)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it. A whole number comes back as
    an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def least_squares(xs, ys):
    """Slope and residual standard deviation of y on x. Returns (slope,
    resid_sd, n). With fewer than 2 points, or no spread in x, slope 0."""
    n = len(xs)
    if n < 2:
        return 0.0, 0.0, n
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    sd = math.sqrt(sum(r * r for r in resid) / dof)
    return slope, sd, n


def distinct_reads(history):
    """The reads whose page differs from the previous read's (a refresh),
    in order. Between refreshes Exophase's page is frozen, so repeated
    reads carry no new information."""
    out = []
    prev = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != prev:
            out.append(read)
            prev = key
    return out


def damped_days(horizon_hours):
    """Effective growth time in days: the horizon damped over TAU_HOURS."""
    return (TAU_HOURS * (1.0 - math.exp(-horizon_hours / TAU_HOURS))) / 24.0


def forecast_trophy(trophy, series, horizon_hours, frozen):
    """series: list of (hours_since_first_read, players, rate_or_None) for
    this trophy, distinct reads only, oldest first. Returns the five
    quantiles (percent)."""
    tier = trophy.get("tier", "bronze")
    horizon_days = horizon_hours / 24.0

    # Last non-null rate and the players tracked at that read.
    last_rate = None
    last_players = None
    for _t, players, rate in reversed(series):
        if rate is not None:
            last_rate = rate
            last_players = players
            break
    if last_rate is None or last_players is None or last_players <= 0:
        # No usable read for this trophy: median at the floor, wide upside.
        q95 = ZERO_EARNER_Q95.get(tier, 1.0)
        return [0.01, 0.01, 0.01, round4(q95 / 4.0), round4(q95)]

    last_earners = last_rate * last_players / 100.0

    if frozen:
        # Stale level, a refresh is likely before the target and refreshes
        # dilute: shift the median down, spread wide.
        centre = logit(last_rate) - FROZEN_SHIFT
        sigma = FROZEN_SIGMA
        quantiles = [round4(clamp_rate(logistic(centre + z * sigma))) for z in Z]
        return enforce_order(quantiles)

    # Recent window: the last WINDOW_HOURS of distinct reads.
    last_t = series[-1][0]
    window = [row for row in series if row[0] >= last_t - WINDOW_HOURS]
    if len(window) < 2:
        window = series[-2:] if len(series) >= 2 else series

    # Fit log players and log earners against time (per day) over the window.
    times = [row[0] / 24.0 for row in window]
    log_n = [math.log(max(row[1], 1)) for row in window]
    g_n, sd_n, _ = least_squares(times, log_n)
    g_n = min(max(g_n, 0.0), MAX_GROWTH_PLAYERS)

    earn_points = [(row[0] / 24.0, row[2] * row[1] / 100.0) for row in window
                 if row[2] is not None and row[2] > 0 and row[1] > 0]
    if len(earn_points) >= 2:
        e_times = [t for t, _ in earn_points]
        log_e = [math.log(max(e, 0.5)) for _, e in earn_points]
        g_e, sd_e, _ = least_squares(e_times, log_e)
    else:
        g_e, sd_e = 0.0, 0.0
    g_e = min(max(g_e, 0.0), MAX_GROWTH_EARNERS)

    if last_earners <= 0:
        # Nobody has earned it yet in the window: median at the floor.
        q95 = ZERO_EARNER_Q95.get(tier, 1.0)
        return [0.01, 0.01, 0.01, round4(q95 / 4.0), round4(q95)]

    eff_days = damped_days(horizon_hours)
    n_target = last_players * math.exp(g_n * eff_days)
    e_target = last_earners * math.exp(g_e * eff_days)
    model_median = clamp_rate(100.0 * e_target / max(n_target, 1.0))
    # Move only part of the way from the last rate toward the model's
    # projection: the launch surge decays faster than the recent trend.
    centre = logit(last_rate) + TREND_SHRINK * (logit(model_median) - logit(last_rate))
    median = clamp_rate(logistic(centre))

    # Spread: residual scatter of the two fits, grown with the horizon,
    # over a floor, and never below the random-walk allowance for the
    # horizon. The scatter is per day, so it scales with sqrt(days).
    resid = math.sqrt(sd_n * sd_n + sd_e * sd_e)
    sigma = math.sqrt(SIGMA_FLOOR ** 2 + (resid ** 2) * max(horizon_days, 0.0))
    if len(window) < 3:
        # Too little movement data to trust the scatter: use the floor plus
        # a fixed allowance for an unseen refresh.
        sigma = max(sigma, 0.6)
    sigma = min(max(sigma, SIGMA_FLOOR), SIGMA_CAP)

    quantiles = [round4(clamp_rate(logistic(centre + z * sigma))) for z in Z]
    return enforce_order(quantiles)


def enforce_order(quantiles):
    """Quantiles must be non-decreasing after rounding."""
    out = []
    prev = 0.01
    for q in quantiles:
        q = max(q, prev)
        out.append(q)
        prev = q
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        # No reads at all: cannot forecast from nothing; emit the floor
        # with a wide upside so the case is still valid.
        rows = []
        for trophy in case["trophies"]:
            q95 = ZERO_EARNER_Q95.get(trophy.get("tier", "bronze"), 1.0)
            rows.append({"id": trophy["id"],
                         "q": [0.01, 0.01, 0.01, round4(q95 / 4.0), round4(q95)]})
        return {"trophies": rows}

    horizon_hours = case["horizon_minutes"] / 60.0
    distinct = distinct_reads(history)
    frozen = len(distinct) < FROZEN_MIN_READS

    # Hours since the first read, per distinct read.
    t0 = parse_time(distinct[0]["at"])
    trophies = case["trophies"]
    # Per-trophy series built from the distinct reads.
    series_by_trophy = []
    for i in range(len(trophies)):
        series = []
        for read in distinct:
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            players = read.get("players_tracked") or 0
            series.append(( (parse_time(read["at"]) - t0).total_seconds() / 3600.0,
                            players, rate ))
        series_by_trophy.append(series)

    rows = []
    for trophy, series in zip(trophies, series_by_trophy):
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(trophy, series, horizon_hours, frozen)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
