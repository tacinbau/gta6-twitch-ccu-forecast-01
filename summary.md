# gta6-twitch-ccu-forecast-01 — summary

Entry for make.fun `gta6-twitch-ccu-forecast-01` ($250, closes 2026-11-19T01:00:00Z).
`main.py` (Python 3.13, standard library only) reads one case and writes
five quantiles (5/25/50/75/95%) of Twitch concurrent viewers for each of
hours 2–24 of GTA VI's release day (UTC), given hour 1's print.

## Method
- A launch day's shape depends on how big hour 1 already is: a launch
  that opened huge stays flat or decays; one that opened small grows
  several-fold into the evening (corr between log hour-1 print and the
  evening log-ratio is about -0.93 across past launches).
- For each hour, take the past big launches in the case's `comparables`,
  weight each by how close its own hour-1 print is to this case's hour-1
  print in log space (Gaussian kernel, bandwidth 1.0), and read the five
  quantiles off the weighted empirical distribution of
  log(viewers_h / viewers_1). Forecast is p1 * exp(quantile).
- GTA VI's own pre-release category is skipped (it is not a launch);
  comparables without a full release day are skipped. Too few
  comparables for an hour falls back to the persistence baseline.

## Validation
- Leave-one-out over the 38 comparables in the kit's practice cases,
  mean pinball loss on the log scale (lower is better): persistence
  baseline 0.465, pooled median curve 0.400, k-nearest neighbours in
  log hour-1 (k=8) 0.209, Gaussian-kernel weighted quantiles (bw 1.0)
  **0.202** — a 56% cut vs the house baseline.
- Extra features (hour-1 channels, pre-release baseline ratio) and
  median shrinkage did not improve the leave-one-out score, so the
  entry keeps the single-feature kernel model.
- Runs deterministically (two runs on the practice cases are byte
  identical) and well inside the 8 s / case limit.
