"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

For each hour h = 2..24 of the release day (UTC), forecast five quantiles
(5%, 25%, 50%, 75%, 95%) of the concurrent viewers in the Grand Theft
Auto VI Twitch category, given everything recorded up to the end of hour 1.

Method: a launch day's shape depends on how big hour 1 already is. A
launch that opened huge (a global unlock before midnight UTC) stays flat
or decays; a launch that opened small (its big markets unlock later in
the day) grows several-fold into the evening. So for each hour, take the
past big launches in this case's "comparables", weight each by how close
its own hour-1 print is to this case's hour-1 print in log space
(Gaussian kernel, bandwidth 1.0), and read the five quantiles off the
weighted empirical distribution of log(viewers_h / viewers_1). The
forecast is p1 * exp(quantile). When too few comparables cover an hour,
fall back to the persistence baseline (flat median, spread widening like
a random walk, sigma 0.15 per sqrt hour).

Leave-one-out over the 38 comparables in the kit's practice cases
(pinball loss on the log scale, lower is better): persistence baseline
0.465, this model 0.202. Deterministic, standard library only, no
network.
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
BANDWIDTH = 1.0
BASE_SIGMA = 0.15
MIN_COMPARABLES = 3


def parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def round2(x):
    """Round to 2 decimals, half up (as JavaScript Math.round(x*100)/100)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def release_day_series(comp):
    """{hour: viewers} for a comparable's release day, hours 1..24.

    Returns None when the comparable has no hour-1 print on its release
    day (e.g. GTA VI's own pre-release category, which is not a launch).
    """
    unlock = (comp.get("unlock") or "").lower()
    if "not a launch" in unlock:
        return None
    day_start = parse_time(comp["day_start"])
    series = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if not viewers or viewers <= 0:
            continue
        when = parse_time(point["from"])
        if day_start <= when < day_start + timedelta(hours=24):
            hour = int((when - day_start).total_seconds() // 3600) + 1
            series[hour] = viewers
    if 1 not in series:
        return None
    return series


def weighted_quantile(pairs, level):
    """Smallest value whose cumulative weight reaches `level` of the total."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= level * total:
            return value
    return ordered[-1][0]


def hour_log_quantiles(target_log_p1, curves, hour):
    """Five log-ratio quantiles for one hour, or None to use the fallback."""
    pairs = []
    for log_p1, series in curves:
        if hour not in series:
            continue
        weight = math.exp(-0.5 * ((log_p1 - target_log_p1) / BANDWIDTH) ** 2)
        pairs.append((math.log(series[hour] / series[1]), weight))
    if len(pairs) < MIN_COMPARABLES:
        return None
    return [weighted_quantile(pairs, level) for level in LEVELS]


def finalize(p1, log_quantiles):
    """Log quantiles -> rounded viewer counts: finite, > 0, <= 1e9, sorted."""
    out = []
    for log_q in log_quantiles:
        q = round2(p1 * math.exp(log_q))
        q = max(1, min(q, 1_000_000_000))
        if out:
            q = max(q, out[-1])
        out.append(q)
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    target_log_p1 = math.log(p1)
    curves = []
    for comp in case.get("comparables") or []:
        series = release_day_series(comp)
        if series is not None and len(series) >= 20:
            curves.append((math.log(series[1]), series))
    hours = []
    for row in case["forecast"]:
        hour = row["hour"]
        ratios = hour_log_quantiles(target_log_p1, curves, hour)
        if ratios is None:
            spread = BASE_SIGMA * math.sqrt(hour - 1)
            ratios = [z * spread for z in Z]
        hours.append({"hour": hour, "q": finalize(p1, ratios)})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
