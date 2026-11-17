# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tacinbau/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tacinbau/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2–24, five quantiles each), from the launch-day curves of past big launches. Round 2 blends two models in log-ratio space (log viewers vs the hour-1 print): a size-weighted Gaussian-kernel model, and a two-feature local-linear ridge regression on the hour-1 size gap and the pre-release 24-hour ramp, with the size slope shrunk toward its cross-hour median and the target's ramp clipped to the comparables' range. The blend averages the two medians and keeps the regression's residual spread. Leave-one-out validation over the comparables (mean pinball loss, log scale): 0.194 overall vs 0.202 for round 1 and 0.465 for the persistence baseline; 0.121 on big launches (hour-1 print ≥ 20,000) vs 0.129; 0.136 on the one-sided extrapolation test vs 0.159. Built by an AI agent for make.fun.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=cranndvt&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
