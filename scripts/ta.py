#!/usr/bin/env python3
"""TA-Lib technical analysis for a single ticker.

Price data comes from Yahoo's chart API called DIRECTLY with requests —
never the yfinance package (it crashes on Africa/Cairo DST timestamps).

Usage:
    python scripts/ta.py [TICKER]

Default ticker is the pilot ticker POUL.CA. Nothing else is hardcoded:
any Yahoo ticker can be passed as the first CLI argument.

Outputs (under results/):
    ta_latest.json   - last date/close, latest indicator values, and the
                       most recent date each candlestick pattern fired
                       (non-zero) in the last 60 sessions
    indicators.csv   - last 250 rows of OHLC + indicators + patterns

For POUL.CA specifically, all rows before 2019-01-01 are dropped: that
segment of Yahoo's series is corrupted (stale flat prices on a
x0.4115 scale).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
import talib

PATTERNS = {
    "CDLDOJI": talib.CDLDOJI,
    "CDLENGULFING": talib.CDLENGULFING,
    "CDLHAMMER": talib.CDLHAMMER,
    "CDLSHOOTINGSTAR": talib.CDLSHOOTINGSTAR,
    "CDLMORNINGSTAR": talib.CDLMORNINGSTAR,
}


def fetch_ohlc(ticker: str) -> pd.DataFrame:
    period1 = 0
    period2 = int(time.time())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    result = payload["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert("UTC")
        .normalize()
        .tz_localize(None),
    )
    df.index.name = "date"
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", nargs="?", default="POUL.CA",
                        help="Yahoo ticker (default: POUL.CA)")
    args = parser.parse_args()
    ticker = args.ticker

    df = fetch_ohlc(ticker)
    rows_fetched = len(df)

    # POUL.CA: drop the corrupted pre-2019 segment.
    if ticker.upper() == "POUL.CA":
        df = df[df.index >= "2019-01-01"]

    if df.empty:
        print(f"No usable rows for {ticker}", file=sys.stderr)
        return 1

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    df["rsi_14"] = talib.RSI(c, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(
        c, fastperiod=12, slowperiod=26, signalperiod=9
    )
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["ema_20"] = talib.EMA(c, timeperiod=20)
    df["ema_50"] = talib.EMA(c, timeperiod=50)
    df["adx_14"] = talib.ADX(h, l, c, timeperiod=14)

    for name, fn in PATTERNS.items():
        df[name] = fn(o, h, l, c)

    def latest(col):
        v = df[col].iloc[-1]
        return None if pd.isna(v) else float(v)

    last60 = df.tail(60)
    pattern_latest = {}
    for name in PATTERNS:
        fired = last60[last60[name] != 0]
        if fired.empty:
            pattern_latest[name] = "none in last 60 sessions"
        else:
            pattern_latest[name] = fired.index[-1].strftime("%Y-%m-%d")

    out = {
        "ticker": ticker,
        "rows_fetched": int(rows_fetched),
        "rows_used": int(len(df)),
        "date_range": {
            "first": df.index[0].strftime("%Y-%m-%d"),
            "last": df.index[-1].strftime("%Y-%m-%d"),
        },
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
        "last_close": float(df["close"].iloc[-1]),
        "indicators": {
            "RSI_14": latest("rsi_14"),
            "MACD": latest("macd"),
            "MACD_signal": latest("macd_signal"),
            "MACD_hist": latest("macd_hist"),
            "EMA_20": latest("ema_20"),
            "EMA_50": latest("ema_50"),
            "ADX_14": latest("adx_14"),
        },
        "patterns_last_60_sessions": pattern_latest,
    }

    os.makedirs("results", exist_ok=True)
    with open("results/ta_latest.json", "w") as f:
        json.dump(out, f, indent=2)
    df.tail(250).to_csv("results/indicators.csv", index_label="date")

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
