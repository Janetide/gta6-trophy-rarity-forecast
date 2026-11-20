# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Janetide/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Janetide/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for GTA VI trophy earn rates: it reads Exophase's per-trophy rates and tracked-player counts from the case history, models earners and players separately with damped growth, and writes five quantiles per trophy for the target date.

## Method

An earn rate is a ratio, earners over players tracked, and the two move differently over launch week, so the program models them separately and divides. Over the recent distinct reads (consecutive identical pages carry no new information, so only refreshes count) it fits log earners and log players tracked against time per trophy, damps the growth over 24 hours, and moves the median only three quarters of the way from the last rate toward the projection, because the launch surge decays faster than the recent trend. Earners never shrink, so their fitted growth is floored at zero; hard trophies are therefore forecast to fall or stay flat while easy ones rise, which is what the launch-week dilution does to the page. The quantiles sit on the logit scale around the median with a spread from the scatter of the two fits over a 0.5-logit floor. When the history holds fewer than three distinct reads the page has not refreshed since launch night, so the level is stale: the median is shifted down a logit and the spread widened, because a refresh before the target is likely and refreshes dilute rates. See `main.py` for the full description.

Validation (simulated, not the real score): on three simulated launch weeks with different player-base growth, the program scored a mean pinball loss of about 0.14 against the house persistence baseline's about 0.33; on the practice cases, against the first refresh's dilution (638 to 3,868 players tracked), its frozen-page handling scored 0.42 against persistence's 1.69.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory. Deterministic: two runs write the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=trfjt22d&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
