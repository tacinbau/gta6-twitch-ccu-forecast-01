"""GTA VI Twitch launch-day forecaster, round 2 (make.fun gta6-twitch-ccu-forecast-01).

Method, in brief.  Everything is forecast as a log ratio to the hour-1
print: for each later hour h we predict log(viewers_h / viewers_1), then
multiply back by the hour-1 print.  Working in log ratios makes the
forecast scale-free -- a launch twice as big gets (roughly) the same
shape -- and matches the log-scale pinball loss the contest scores.

Two candidate forecasts are built per hour from the past launches
("comparables") supplied in the case, then blended:

* Candidate A -- size-weighted kernel.  Each past launch contributes
  its own log ratio for hour h, weighted by a Gaussian kernel on how
  close its hour-1 print was to this launch's hour-1 print (bandwidth
  1.0 in log space).  The five quantiles are the weighted quantiles of
  those log ratios.  Big launches therefore borrow mostly from other
  big launches, small ones from small ones.

* Candidate B -- two-feature local linear.  A weighted ridge
  regression of the log ratio on two predictors: the log gap between
  the comparable's hour-1 print and this launch's, and the gap in
  "ramp24" -- the log growth of hourly viewers over the 24 hours
  before release day, which measures how much buzz was already
  building.  Weights are Gaussian in the size gap (bandwidth 1.5);
  the ridge penalty (0.3) steadies the two slopes.  The target's
  ramp24 is clipped into the range the comparables span, so a launch
  whose pre-release ramp is more extreme than any past launch cannot
  push the linear ramp term into wild extrapolation.  The size slope
  is then shrunk toward the median size slope across all hours (prior
  weight 2.0), because a single hour's slope is noisy while the
  typical size effect is shared.  The quantiles are the refitted
  intercept plus the weighted quantiles of the residuals, so the
  spread comes from how far past launches actually deviated from the
  regression, not from an assumed normal shape.

The blend centres on the average of the two candidates' medians and
keeps candidate B's spread around that centre: the regression's
calibrated spread is the more trustworthy of the two, while averaging
the centres hedges between the kernel's local honesty and the
regression's ability to extrapolate beyond the biggest comparable --
the regime a GTA VI-sized launch lives in.

When too few comparables cover an hour (under 3 for A, under 8 for B,
or a regression that cannot be solved), that candidate falls back to
persistence: median log ratio 0 with a spread widening like a random
walk, 0.15 * sqrt(h - 1) times the standard normal quantiles.

Comparables are kept only when their release-day series is usable:
hour 1 present and at least 20 of the 24 hours covered, and their
unlock note must not say "not a launch" (that entry is the target
game's own pre-release history, not a launch).

Python 3.13, standard library only, fully deterministic, no network.
"""

import bisect
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves this file's directory OFF sys.path; put it back
# first, so a module shipped beside this file would still import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Candidate A: Gaussian kernel bandwidth on the log hour-1 gap.
KERNEL_BW_A = 1.0
# Candidate B: Gaussian kernel bandwidth on the log hour-1 gap.
KERNEL_BW_B = 1.5
# Candidate B: ridge penalty added to each slope's normal-equation diagonal.
RIDGE = 0.3
# Candidate B: prior weight when shrinking the size slope to the pooled median.
SHRINK_PRIOR = 2.0
# Minimum comparables covering an hour for each candidate to be used.
MIN_COVER_A = 3
MIN_COVER_B = 8
# Persistence fallback: log-spread per sqrt hour.
PERSIST_SIGMA = 0.15

MAX_VIEWERS = 1_000_000_000


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up.

    A whole number comes back as an int, so json.dumps writes 1240
    where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(text: str) -> float:
    """ISO 8601 UTC timestamp string -> epoch seconds (float)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def median(values):
    """Median of a non-empty list; even counts average the two middles."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def weighted_quantile(pairs, level):
    """Smallest value whose cumulative weight reaches level * total weight.

    pairs is a list of (value, weight) with positive weights; values are
    sorted ascending before accumulating.
    """
    ordered = sorted(pairs, key=lambda p: p[0])
    total = sum(w for _, w in ordered)
    threshold = level * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def persistence_quantiles(h):
    """Fallback log-ratio quantiles: median 0, random-walk spread."""
    spread = PERSIST_SIGMA * math.sqrt(h - 1)
    return [z * spread for z in Z]


def build_series(day_start, hourly):
    """Release-day series: hour (1..24) -> viewers, from raw hourly points."""
    series = {}
    for point in hourly or []:
        try:
            ts = parse_ts(point.get("from"))
        except (ValueError, TypeError, AttributeError):
            continue
        delta = ts - day_start
        if 0 <= delta < 24 * 3600:
            viewers = point.get("viewers")
            if isinstance(viewers, bool) or not isinstance(viewers, (int, float)):
                continue
            if viewers <= 0:
                continue
            series[int(delta // 3600) + 1] = float(viewers)
    return series


def sorted_points(hourly):
    """(timestamps, viewers) sorted by time, for latest-at-or-before lookups."""
    pts = []
    for point in hourly or []:
        try:
            ts = parse_ts(point.get("from"))
        except (ValueError, TypeError, AttributeError):
            continue
        pts.append((ts, point.get("viewers")))
    pts.sort(key=lambda p: p[0])
    return [p[0] for p in pts], [p[1] for p in pts]


def viewers_at(times, viewers, when):
    """Viewers of the latest point at or before `when`; None if unusable."""
    idx = bisect.bisect_right(times, when) - 1
    if idx < 0:
        return None
    value = viewers[idx]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value)


def ramp24(times, viewers, day_start):
    """log(viewers 1h before day_start / viewers 24h before day_start)."""
    late = viewers_at(times, viewers, day_start - 3600)
    early = viewers_at(times, viewers, day_start - 24 * 3600)
    if late is None or early is None:
        return None
    return math.log(late / early)


def solve_3x3(matrix, rhs):
    """Gaussian elimination with partial pivoting; None if near-singular."""
    n = 3
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = aug[i][n] - sum(aug[i][j] * out[j] for j in range(i + 1, n))
        out[i] = acc / aug[i][i]
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    day_start = parse_ts(case["day_start"])

    # --- Target launch's pre-release ramp, from its own history. ---
    hist_times, hist_viewers = sorted_points(case.get("history_hourly") or [])

    # --- Keep only usable comparables, with their series and ramp. ---
    kept = []
    ramps_seen = []
    for comp in case.get("comparables") or []:
        unlock = comp.get("unlock") or ""
        if "not a launch" in str(unlock).lower():
            continue
        try:
            comp_start = parse_ts(comp.get("day_start"))
        except (ValueError, TypeError, AttributeError):
            continue
        series = build_series(comp_start, comp.get("hourly") or [])
        if 1 not in series or len(series) < 20:
            continue
        times, viewers = sorted_points(comp.get("hourly") or [])
        r24 = ramp24(times, viewers, comp_start)
        kept.append(
            {
                "series": series,
                "log_p1": math.log(series[1]),
                "ramp": r24,
            }
        )
        if r24 is not None:
            ramps_seen.append(r24)
    median_ramp = median(ramps_seen) if ramps_seen else 0.0
    for comp in kept:
        if comp["ramp"] is None:
            comp["ramp"] = median_ramp
    target_ramp = ramp24(hist_times, hist_viewers, day_start)
    if target_ramp is None:
        target_ramp = median_ramp
    # Clip the target's ramp into the comparables' observed range, so an
    # extreme pre-release ramp cannot extrapolate the linear ramp term
    # beyond anything a past launch showed. (A median-substituted ramp
    # already lies inside the range, so it is unaffected.)
    if ramps_seen:
        target_ramp = min(max(target_ramp, min(ramps_seen)), max(ramps_seen))

    # --- Per-hour data: (x1 size gap, x2 ramp gap, y log ratio, wA, wB). ---
    hours_data = {}
    for row in case["forecast"]:
        h = row["hour"]
        rows = []
        for comp in kept:
            if h not in comp["series"]:
                continue
            x1 = comp["log_p1"] - log_p1
            x2 = comp["ramp"] - target_ramp
            y = math.log(comp["series"][h] / comp["series"][1])
            w_a = math.exp(-0.5 * (x1 / KERNEL_BW_A) ** 2)
            w_b = math.exp(-0.5 * (x1 / KERNEL_BW_B) ** 2)
            rows.append((x1, x2, y, w_a, w_b))
        hours_data[h] = rows

    # --- Candidate B, first pass: per-hour ridge fits, for the pooled slope. ---
    fits = {}
    slopes_b1 = []
    for h, rows in hours_data.items():
        if len(rows) < MIN_COVER_B:
            continue
        # Normal equations for [1, x1, x2] with ridge on the slopes.
        ata = [[0.0] * 3 for _ in range(3)]
        atb = [0.0] * 3
        for x1, x2, y, _w_a, w_b in rows:
            x = (1.0, x1, x2)
            for i in range(3):
                atb[i] += w_b * x[i] * y
                for j in range(3):
                    ata[i][j] += w_b * x[i] * x[j]
        ata[1][1] += RIDGE
        ata[2][2] += RIDGE
        beta = solve_3x3(ata, atb)
        if beta is None:
            continue
        fits[h] = beta
        slopes_b1.append(beta[1])
    pooled_slope = median(slopes_b1) if slopes_b1 else 0.0

    # --- Per-hour candidates, blend, and final quantiles. ---
    out_hours = []
    for row in case["forecast"]:
        h = row["hour"]
        rows = hours_data[h]

        # Candidate A: kernel-weighted quantiles of the log ratios.
        if len(rows) >= MIN_COVER_A:
            cand_a = [
                weighted_quantile([(y, w_a) for _x1, _x2, y, w_a, _w_b in rows], lv)
                for lv in LEVELS
            ]
        else:
            cand_a = persistence_quantiles(h)

        # Candidate B: shrunk local-linear fit plus residual quantiles.
        if h in fits:
            beta1, beta2 = fits[h][1], fits[h][2]
            total_w = sum(w_b for *_rest, w_b in rows)
            shrunk_b1 = (total_w * beta1 + SHRINK_PRIOR * pooled_slope) / (
                total_w + SHRINK_PRIOR
            )
            intercept = sum(
                w_b * (y - shrunk_b1 * x1 - beta2 * x2)
                for x1, x2, y, _w_a, w_b in rows
            ) / total_w
            residuals = [
                (y - (intercept + shrunk_b1 * x1 + beta2 * x2), w_b)
                for x1, x2, y, _w_a, w_b in rows
            ]
            cand_b = [
                intercept + weighted_quantile(residuals, lv) for lv in LEVELS
            ]
        else:
            cand_b = persistence_quantiles(h)

        # Blend: average the medians, keep candidate B's spread.
        blended_median = 0.5 * cand_a[2] + 0.5 * cand_b[2]
        blended = [blended_median + (qb - cand_b[2]) for qb in cand_b]

        quantiles = []
        prev = None
        for log_ratio in blended:
            value = round2(p1 * math.exp(log_ratio))
            value = max(1, min(MAX_VIEWERS, value))
            if prev is not None and value < prev:
                value = prev
            quantiles.append(value)
            prev = value
        out_hours.append({"hour": h, "q": quantiles})

    return {"hours": out_hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
