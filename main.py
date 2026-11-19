"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Two-regime model, written from the kit's RULES.txt and the practice cases.

The house refreshes Exophase's counts on its own schedule, so a read shows the
rates as of Exophase's last refresh before it.  When every read in the history
is the same page, the rates have not moved yet and the next refresh can move
them a lot: that is the Stale regime, and the forecast shifts the median down
on the logit scale (a bigger player base dilutes every rate) with a wide
spread.  When distinct pages exist, the rates are actually moving and a
per-trophy trend can be fitted: that is the Live regime.

  Stale:  median logit = logit(last rate) - 1.30, except a trophy at 100%
          (rate >= 99.5%), which gets no shift.
          spread = sqrt(0.50^2 + rw^2), or rw alone for the 100% trophy,
          where rw = 0.5 * sqrt(horizon in days) is the random-walk spread.

  Live:   per trophy, least-squares fit of logit(rate) against
          ln(1 + hours since release) over the last 24 hours of distinct
          reads.  Median = 25% last observed rate + 75% fitted line at the
          target.  spread = sqrt(residual variance + rw^2) * 1.5,
          clamped to [0.15, 3.0].  Falls back to Stale when the fit
          degenerates (fewer than 2 points, or no variance in time).

Quantiles: q_tau = 100 / (1 + exp(-(median logit + z_tau * spread))),
rounded to 4 decimals, half up.  Python 3.13 standard library only,
deterministic, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Stale-regime constants.
STALE_SHIFT = -1.30        # logit shift for refresh dilution
STALE_BASE_SPREAD = 0.50   # base logit spread before the random walk
FULL_RATE = 99.5           # a trophy at or above this is treated as 100%

# Live-regime constants.
LIVE_BLEND = 0.75          # weight of the fitted line in the median
LIVE_SPREAD_SCALE = 1.5
LIVE_SPREAD_MIN = 0.15
LIVE_SPREAD_MAX = 3.0

# Random-walk spread per sqrt(day), in logit units.
RW_SIGMA = 0.5


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float):
    """Round to 4 decimals, half up (as the kit's starter does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def rate_at(read: dict, i: int):
    """The i-th trophy's rate in a read, or None when the read lacks it."""
    rates = read.get("rates") or ()
    if i >= len(rates):
        return None
    return rates[i]


def distinct_reads(history):
    """Collapse consecutive identical reads into distinct pages.

    Two reads are identical when they list the same rates (and the same
    players-tracked count); only the first read of each run is kept.
    """
    pages = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if pages and pages[-1][0] == key:
            continue
        pages.append((key, read))
    return [read for _, read in pages]


def random_walk_spread(horizon_minutes: float) -> float:
    return RW_SIGMA * math.sqrt(horizon_minutes / 1440.0)


def stale_forecast(last_rate: float, rw: float):
    """Return (median logit, spread) for the Stale regime."""
    if last_rate >= FULL_RATE:
        return logit(last_rate), rw
    median = logit(last_rate) + STALE_SHIFT
    spread = math.sqrt(STALE_BASE_SPREAD ** 2 + rw ** 2)
    return median, spread


def live_fit(points):
    """Least-squares fit of y = a + b*x over (x, y) points.

    Returns (a, b, residual_variance) or None when the fit degenerates.
    """
    n = len(points)
    if n < 2:
        return None
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    if sxx <= 0.0:
        return None
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    resid_sq = sum((p[1] - (intercept + slope * p[0])) ** 2 for p in points)
    var = resid_sq / max(n - 2, 1)
    return intercept, slope, var


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    horizon = float(case["horizon_minutes"])
    rw = random_walk_spread(horizon)
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")

    pages = distinct_reads(case["history"])
    live = len(pages) >= 3
    window_start = checkpoint - timedelta(hours=24)
    target_x = math.log1p((target - release).total_seconds() / 3600.0)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        # Last observed rate for this trophy (most recent non-null read).
        last_rate = None
        for read in reversed(case["history"]):
            rate = rate_at(read, i)
            if rate is not None:
                last_rate = rate
                break
        if last_rate is None:
            last_rate = 0.0

        median = None
        spread = None
        if live:
            points = []
            for read in pages:
                at = parse_time(read["at"])
                if at < window_start:
                    continue
                rate = rate_at(read, i)
                if rate is None:
                    continue
                x = math.log1p((at - release).total_seconds() / 3600.0)
                points.append((x, logit(rate)))
            fit = live_fit(points)
            if fit is not None:
                intercept, slope, var = fit
                fitted = intercept + slope * target_x
                median = (1.0 - LIVE_BLEND) * logit(last_rate) + LIVE_BLEND * fitted
                spread = math.sqrt(var + rw ** 2) * LIVE_SPREAD_SCALE
                spread = min(LIVE_SPREAD_MAX, max(LIVE_SPREAD_MIN, spread))
        if median is None:
            median, spread = stale_forecast(last_rate, rw)

        quantiles = []
        for z in Z:
            q = round4(inv_logit(median + z * spread))
            q = min(100, max(0, q))
            quantiles.append(q)
        # Rounding can only move a value by <0.0001, but keep the row
        # non-decreasing regardless.
        for j in range(1, len(quantiles)):
            if quantiles[j] < quantiles[j - 1]:
                quantiles[j] = quantiles[j - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
