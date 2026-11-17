# gta6-twitch-ccu-forecast-01 — summary

Entry for make.fun `gta6-twitch-ccu-forecast-01` ($250, closes 2026-11-19T01:00:00Z).
`main.py` (Python 3.13, standard library only) reads one case and writes
five quantiles (5/25/50/75/95%) of Twitch concurrent viewers for each of
hours 2–24 of GTA VI's release day (UTC), given hour 1's print. Round 2.

## Method
- Everything is forecast as a log ratio to the hour-1 print,
  log(viewers_h / viewers_1); the forecast is p1 * exp(quantile).
  A launch day's shape depends on how big hour 1 already is, and on
  how fast viewers were already ramping before release day.
- Candidate A (kernel): for each hour, the past launches' log ratios,
  weighted by a Gaussian kernel on the log hour-1 gap (bandwidth 1.0);
  the five quantiles are the weighted quantiles. Needs ≥ 3 comparables
  covering the hour.
- Candidate B (two-feature local linear): a weighted ridge regression
  (penalty 0.3 on the slopes) of the log ratio on the log hour-1 gap
  and the ramp24 gap — ramp24 is the log growth of hourly viewers over
  the 24 hours before release day. Weights are Gaussian in the size
  gap (bandwidth 1.5). The size slope is shrunk toward the median size
  slope across all hours (prior weight 2.0); the quantiles are the
  refitted intercept plus the weighted quantiles of the residuals.
  The target's ramp24 is clipped into the comparables' observed range
  so an extreme pre-release ramp cannot extrapolate the linear term
  too far. Needs ≥ 8 comparables covering the hour and a solvable fit.
- Blend: the median is the average of the two candidates' medians;
  the spread is candidate B's spread around that centre. Either
  candidate falls back to persistence (median log ratio 0, spread
  0.15 * sqrt(h − 1) times the normal quantiles) when it lacks coverage.
- GTA VI's own pre-release category is skipped ("not a launch");
  comparables without hour 1 or with fewer than 20 of 24 release-day
  hours are skipped. Missing ramp24 values take the median ramp24.

## Validation
- Leave-one-out over the comparables, mean pinball loss on the log
  scale (lower is better): persistence baseline 0.465; round-1 kernel
  model 0.202; this round-2 blend **0.194** (0.1934 unrounded).
- On big launches (hour-1 print ≥ 20,000 — the regime of the practice
  cases and of GTA VI): 0.121 vs 0.129 for round 1.
- On the one-sided extrapolation test (train only on smaller launches,
  predict bigger ones — the regime of forecasting a launch bigger than
  any comparable): 0.136 vs 0.159 for round 1.
- Round 1 scored 0.243 on the server practice leaderboard (rank 2547
  of 3337); round 2 is expected to improve on that modestly.
- Runs deterministically (two runs on the practice cases are byte
  identical) and well inside the 8 s / case limit.
