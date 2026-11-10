# talib-runner

Runs [TA-Lib](https://ta-lib.org/) technical analysis via GitHub Actions — commands supplied through the `COMMANDS` repository variable.

TA-Lib is the standard open-source technical-analysis library: 150+ indicator and candlestick-pattern functions (RSI, MACD, EMA, ADX, Bollinger Bands, CDLDOJI, CDLENGULFING, …) implemented in C with a Python wrapper.

## Pattern

This repo follows the one-public-repo-per-tool pattern proven by `vectorbt-runner` and `finbert-runner`:

1. Edit the repository Actions variable **`COMMANDS`** (Settings → Secrets and variables → Actions → Variables, or via the GitHub API at `POST /repos/moos-labs/talib-runner/actions/variables`).
2. Dispatch the workflow manually (Actions tab → *Run analysis* → *Run workflow*, or via the API: `POST /repos/moos-labs/talib-runner/actions/workflows/run.yml/dispatches` with `{"ref":"main"}`).
3. Download the **`results`** artifact from the completed run — it contains everything the commands wrote under `results/`.

The workflow (`.github/workflows/run.yml`) triggers on `workflow_dispatch` only. It checks out the repo, sets up Python 3.11, runs `pip install -r requirements.txt`, executes `${{ vars.COMMANDS }}` verbatim, and uploads `results/` as an artifact (`if: always()`, so partial output survives a failing command).

Current `COMMANDS` value (pilot):

```
mkdir -p results
python scripts/ta.py 2>&1 | tee results/run.log
```

## Pilot: `scripts/ta.py`

Generic script — the ticker is a CLI argument (default `POUL.CA`), nothing else is hardcoded.

```
python scripts/ta.py [TICKER]
```

- Price data comes from Yahoo's chart API called **directly** with `requests` — never the `yfinance` package, which crashes on Africa/Cairo DST timestamps. Endpoint: `https://query1.finance.yahoo.com/v8/finance/chart/<ticker>?period1=<epoch>&period2=<epoch>&interval=1d`
- For `POUL.CA`, rows before 2019-01-01 are dropped (Yahoo's pre-2019 segment is corrupted: stale flat prices on a ×0.4115 scale).
- Indicators on closes: RSI(14), MACD(12,26,9), EMA(20), EMA(50), ADX(14).
- Candlestick patterns over the full series: CDLDOJI, CDLENGULFING, CDLHAMMER, CDLSHOOTINGSTAR, CDLMORNINGSTAR.

Outputs in the `results` artifact:

- `ta_latest.json` — last date, last close, latest value of each indicator (exact floats), and for each pattern the most recent date it fired (non-zero) in the last 60 sessions, or `"none in last 60 sessions"`.
- `indicators.csv` — the last 250 rows of OHLC + indicators + pattern columns.
- `run.log` — the script's stdout/stderr.

## Requirements

`ta-lib`, `pandas`, `numpy`, `requests` (see `requirements.txt`). Modern `ta-lib` ships prebuilt wheels, so `pip install` should not need to compile the underlying C library.
