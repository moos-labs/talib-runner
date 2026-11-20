#!/usr/bin/env python3
"""Tests for --as-of (borsagator #436, walk-forward backtest).

--as-of YYYY-MM-DD: fetch full history, drop every row with date >
as-of, fail loudly if fewer than MIN_ROWS (60) sessions remain. The
truncation must happen inside the runner, before any indicator call,
so rows after the cutoff never reach TA-Lib. The snapshot's last_date
must equal the as-of date (when it is a session in the series).

Run: python3 tests/test_asof_436.py   (stdlib + pandas + numpy only,
no network -- fetch_ohlc is stubbed for the main() tests; the
truncation logic under test does not depend on indicator math, so
when the real TA-Lib C library is unavailable a deterministic stub
module stands in for it).

Failing-before probe: point TA_UNDER_TEST at the pre-change script
and every test below fails (the helper does not exist there).
"""
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
TA_PATH = os.environ.get("TA_UNDER_TEST", REPO / "scripts" / "ta.py")

try:
    import talib  # noqa: F401  -- real library when available (CI)
except ImportError:
    # Deterministic stand-in: indicator values are irrelevant to the
    # --as-of truncation under test; only shapes and callability matter.
    stub = types.ModuleType("talib")

    def _arr(x, val=50.0):
        return np.full(len(np.asarray(x, dtype=float)), val, dtype=float)

    stub.RSI = lambda c, timeperiod=14: _arr(c)
    stub.MACD = lambda c, fastperiod=12, slowperiod=26, signalperiod=9: (
        _arr(c, 0.0), _arr(c, 0.0), _arr(c, 0.0))
    stub.EMA = lambda c, timeperiod=20: np.asarray(c, dtype=float)
    stub.ADX = lambda h, l, c, timeperiod=14: _arr(c, 25.0)
    for _name in ("CDLDOJI", "CDLENGULFING", "CDLHAMMER",
                  "CDLSHOOTINGSTAR", "CDLMORNINGSTAR"):
        setattr(stub, _name,
                lambda o, h, l, c: np.zeros(len(np.asarray(c)), dtype=int))
    sys.modules["talib"] = stub

spec = importlib.util.spec_from_file_location("ta_asof", TA_PATH)
ta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ta)

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}: {name}"
          + (f" -- {detail}" if detail and not ok else ""))


def _frame(n=200, start="2024-01-01"):
    """n business-day sessions; close rises so the as-of close is
    distinguishable from any later close."""
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "open": [10.0 + 0.1 * i for i in range(n)],
            "high": [10.5 + 0.1 * i for i in range(n)],
            "low": [9.9 + 0.1 * i for i in range(n)],
            "close": [10.2 + 0.1 * i for i in range(n)],
            "volume": [100 + i for i in range(n)],
        },
        index=idx,
    )


AS_OF = "2024-06-28"  # a Friday session in the frame

# 1. apply_asof drops every row after the cutoff, keeps the cutoff row.
df = _frame()
out = ta.apply_asof(df, AS_OF, min_rows=20)
n_expected = int((df.index <= AS_OF).sum())
check("apply_asof drops rows after as-of",
      len(out) == n_expected
      and out.index[-1] == pd.Timestamp(AS_OF),
      f"len={len(out)} expected={n_expected} last={out.index[-1]}")
i_asof = df.index.get_loc(pd.Timestamp(AS_OF))
check("apply_asof keeps the as-of row itself",
      out["close"].iloc[-1] == 10.2 + 0.1 * i_asof)

# 2. A cutoff on a non-session date keeps the last session on/before it.
out2 = ta.apply_asof(df, "2024-06-29", min_rows=20)  # a Saturday
check("non-session as-of keeps last session on/before it",
      out2.index[-1] == pd.Timestamp(AS_OF))

# 3. Too-short remainder fails loudly instead of snapshotting a sliver.
try:
    ta.apply_asof(_frame(70), "2024-01-15", min_rows=20)
    check("too-short remainder raises RuntimeError", False)
except RuntimeError as e:
    check("too-short remainder raises RuntimeError",
          "only 11 sessions" in str(e) and "2024-01-15" in str(e), str(e))

# 4. The default floor is the design's 60-session warm-up (§1.3).
try:
    ta.apply_asof(_frame(70), "2024-03-15")  # 55 sessions <= cutoff
    check("default MIN_ROWS=60 enforced", False)
except RuntimeError as e:
    check("default MIN_ROWS=60 enforced",
          "< 60 minimum rows" in str(e), str(e))
check("MIN_ROWS constant is 60", ta.MIN_ROWS == 60)

# 5. main() wiring: stubbed fetch; the snapshot must end at as-of and
# indicators.csv must contain no later row.
tmp = Path(tempfile.mkdtemp(prefix="talib436-main-"))
os.chdir(tmp)


def fake_fetch(ticker):
    return _frame()


ta.fetch_ohlc = fake_fetch
old_argv = sys.argv
sys.argv = ["ta.py", "TEST", "--as-of", AS_OF]
try:
    rc = ta.main()
    check("main() exits 0 with --as-of", rc == 0, f"rc={rc}")
    snap = json.loads((tmp / "results" / "ta_latest.json").read_text())
    check("snapshot last_date equals as-of",
          snap["last_date"] == AS_OF, snap["last_date"])
    check("snapshot records as_of",
          snap["as_of"] == AS_OF, snap.get("as_of"))
    check("snapshot last_close is the as-of close",
          snap["last_close"] == 10.2 + 0.1 * i_asof,
          str(snap["last_close"]))
    check("snapshot rows_used equals truncated count",
          snap["rows_used"] == n_expected,
          f"rows_used={snap['rows_used']} expected={n_expected}")
    ind_csv = pd.read_csv(tmp / "results" / "indicators.csv",
                          index_col=0, parse_dates=True)
    check("indicators.csv contains no row after as-of",
          ind_csv.index.max() == pd.Timestamp(AS_OF),
          str(ind_csv.index.max()))
finally:
    sys.argv = old_argv

# 6. main() fails loudly when the truncated series is too short.
tmp2 = Path(tempfile.mkdtemp(prefix="talib436-short-"))
os.chdir(tmp2)


def fake_fetch_short(ticker):
    return _frame(70)


ta.fetch_ohlc = fake_fetch_short
sys.argv = ["ta.py", "TEST", "--as-of", "2024-02-01"]
try:
    ta.main()
    check("main(): too-short remainder raises", False)
except RuntimeError as e:
    check("main(): too-short remainder raises",
          "only 24 sessions" in str(e), str(e))
finally:
    sys.argv = old_argv

failed = [c for c in CHECKS if not c[1]]
print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks pass")
sys.exit(1 if failed else 0)
