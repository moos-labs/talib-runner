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

# 7. Shared history floor (borsagator #446 / INS-REVIEW-025): every
# ticker is floored at 2019-01-01, not just POUL.CA. A frame spanning
# 2018-01-01..2020-04-17 (600 business days: 261 pre-2019, 339 from
# 2019-01-01 on) must come out floored for COMI.CA and identically for
# POUL.CA; the floor runs before --as-of, so the as-of count is taken
# on the floored frame; a frame entirely before 2019 fails closed.
FLOOR_FRAME_N = 600
FLOOR_FRAME_START = "2018-01-01"
FLOOR_FIRST = "2019-01-01"
FLOOR_ROWS = 339  # business days 2019-01-01..2020-04-17
FLOOR_LAST = "2020-04-17"


def fake_fetch_floor(ticker):
    return _frame(FLOOR_FRAME_N, FLOOR_FRAME_START)


tmp3 = Path(tempfile.mkdtemp(prefix="talib436-floor-"))
os.chdir(tmp3)
ta.fetch_ohlc = fake_fetch_floor
old_argv = sys.argv
sys.argv = ["ta.py", "COMI.CA"]
try:
    rc = ta.main()
    check("main() exits 0 for COMI.CA with pre-2019 rows", rc == 0, f"rc={rc}")
    snap = json.loads((tmp3 / "results" / "ta_latest.json").read_text())
    check("snapshot records drop_before",
          snap.get("drop_before") == FLOOR_FIRST, snap.get("drop_before"))
    check("COMI.CA floored: date_range.first is 2019-01-01",
          snap["date_range"]["first"] == FLOOR_FIRST,
          snap["date_range"]["first"])
    check("COMI.CA floored: rows_used excludes pre-2019 rows",
          snap["rows_used"] == FLOOR_ROWS,
          f"rows_used={snap['rows_used']} expected={FLOOR_ROWS}")
    check("COMI.CA floored: rows_fetched still counts the full frame",
          snap["rows_fetched"] == FLOOR_FRAME_N,
          f"rows_fetched={snap['rows_fetched']}")
    check("COMI.CA floored: last_date is the frame end",
          snap["last_date"] == FLOOR_LAST, snap["last_date"])
    check("COMI.CA floored: last_close is the frame-end close",
          snap["last_close"] == 10.2 + 0.1 * (FLOOR_FRAME_N - 1),
          str(snap["last_close"]))
    ind_csv = pd.read_csv(tmp3 / "results" / "indicators.csv",
                          index_col=0, parse_dates=True)
    check("indicators.csv contains no row before 2019-01-01",
          ind_csv.index.min() >= pd.Timestamp(FLOOR_FIRST),
          str(ind_csv.index.min()))
finally:
    sys.argv = old_argv

# 7b. POUL.CA keeps its current behaviour on the same frame.
tmp4 = Path(tempfile.mkdtemp(prefix="talib436-floor-poul-"))
os.chdir(tmp4)
ta.fetch_ohlc = fake_fetch_floor
sys.argv = ["ta.py", "POUL.CA"]
try:
    rc = ta.main()
    check("main() exits 0 for POUL.CA with pre-2019 rows", rc == 0, f"rc={rc}")
    snap = json.loads((tmp4 / "results" / "ta_latest.json").read_text())
    check("POUL.CA unchanged: date_range.first is 2019-01-01",
          snap["date_range"]["first"] == FLOOR_FIRST,
          snap["date_range"]["first"])
    check("POUL.CA unchanged: rows_used equals the floored count",
          snap["rows_used"] == FLOOR_ROWS,
          f"rows_used={snap['rows_used']} expected={FLOOR_ROWS}")
    check("POUL.CA unchanged: drop_before recorded",
          snap.get("drop_before") == FLOOR_FIRST, snap.get("drop_before"))
finally:
    sys.argv = old_argv

# 7c. Floor runs before --as-of: the as-of count is taken on the floored
# frame (129 sessions 2019-01-01..2019-06-28, first still 2019-01-01).
tmp5 = Path(tempfile.mkdtemp(prefix="talib436-floor-asof-"))
os.chdir(tmp5)
ta.fetch_ohlc = fake_fetch_floor
sys.argv = ["ta.py", "COMI.CA", "--as-of", "2019-06-28"]
try:
    rc = ta.main()
    check("main() exits 0 for COMI.CA with floor + --as-of", rc == 0, f"rc={rc}")
    snap = json.loads((tmp5 / "results" / "ta_latest.json").read_text())
    check("floor + as-of: last_date equals as-of",
          snap["last_date"] == "2019-06-28", snap["last_date"])
    check("floor + as-of: date_range.first is still 2019-01-01",
          snap["date_range"]["first"] == FLOOR_FIRST,
          snap["date_range"]["first"])
    check("floor + as-of: rows_used counts floored sessions only",
          snap["rows_used"] == 129,
          f"rows_used={snap['rows_used']} expected=129")
    check("floor + as-of: last_close is the as-of close",
          snap["last_close"] == 10.2 + 0.1 * 389,
          str(snap["last_close"]))
finally:
    sys.argv = old_argv

# 7d. A frame entirely before 2019 floors to zero rows and fails closed.
tmp6 = Path(tempfile.mkdtemp(prefix="talib436-floor-empty-"))
os.chdir(tmp6)


def fake_fetch_pre2019(ticker):
    return _frame(200, "2018-01-01")


ta.fetch_ohlc = fake_fetch_pre2019
sys.argv = ["ta.py", "COMI.CA"]
try:
    rc = ta.main()
    check("main() exits 1 when the floor leaves no rows", rc == 1, f"rc={rc}")
    check("no fresh snapshot is written on the clean refusal",
          not (tmp6 / "results" / "ta_latest.json").exists(),
          "ta_latest.json written despite refusal")
finally:
    sys.argv = old_argv

# 7f. Order-sensitive MIN_ROWS (INS-REVIEW-027 §3): the minimum-row
# count is taken on the FLOORED frame. A frame with 600 total sessions
# but only 59 floored sessions on/before the cutoff must refuse, even
# though the unfloored count (600) is far above 60; with one more
# floored session (60) it must pass. This catches an implementation
# that checks warm-up before flooring.
_floor_frame = _frame(FLOOR_FRAME_N, FLOOR_FRAME_START)  # 600 sessions,
# 261 pre-2019 + 339 floored; the unfloored total (600) is far above 60
_floored_idx = _floor_frame.index[_floor_frame.index >= FLOOR_FIRST]
_asof_59 = str(_floored_idx[58].date())
_asof_60 = str(_floored_idx[59].date())
tmp7 = Path(tempfile.mkdtemp(prefix="talib436-floor-minrows-"))
os.chdir(tmp7)
ta.fetch_ohlc = fake_fetch_floor
try:
    sys.argv = ["ta.py", "COMI.CA", "--as-of", _asof_59]
    try:
        rc = ta.main()
        check("59 floored sessions refuse (post-floor count, not total)",
              False, f"main() returned rc={rc} instead of raising")
    except RuntimeError as e:
        check("59 floored sessions refuse (post-floor count, not total)",
              "only 59 sessions" in str(e) and "< 60 minimum rows" in str(e),
              str(e))
    check("no snapshot on the 59-session refusal",
          not (tmp7 / "results" / "ta_latest.json").exists(),
          "ta_latest.json written despite refusal")
    sys.argv = ["ta.py", "COMI.CA", "--as-of", _asof_60]
    rc = ta.main()
    check("60 floored sessions pass", rc == 0, f"rc={rc}")
    snap = json.loads((tmp7 / "results" / "ta_latest.json").read_text())
    check("60 floored sessions: rows_used is the floored count",
          snap["rows_used"] == 60, f"rows_used={snap['rows_used']}")
finally:
    sys.argv = old_argv

# 7g. Inclusive floor boundary (INS-REVIEW-027 §3): a supplied
# 2019-01-01 row is retained, a 2018-12-31 row is dropped. The frame
# has 300 sessions from 2018-12-01 (21 pre-2019, 279 floored), so
# rows_used == 279 pins the boundary exactly; indicators.csv holds
# only the last 250 floored rows, so its check is >=, not ==.
_boundary = _frame(300, "2018-12-01")
_boundary_idx = _boundary.index
check("boundary frame contains 2018-12-31",
      pd.Timestamp("2018-12-31") in _boundary_idx)
check("boundary frame contains 2019-01-01",
      pd.Timestamp("2019-01-01") in _boundary_idx)
tmp8 = Path(tempfile.mkdtemp(prefix="talib436-floor-boundary-"))
os.chdir(tmp8)


def fake_fetch_boundary(ticker):
    return _boundary


ta.fetch_ohlc = fake_fetch_boundary
sys.argv = ["ta.py", "COMI.CA"]
try:
    rc = ta.main()
    check("boundary frame exits 0", rc == 0, f"rc={rc}")
    snap = json.loads((tmp8 / "results" / "ta_latest.json").read_text())
    check("2019-01-01 row retained (date_range.first)",
          snap["date_range"]["first"] == FLOOR_FIRST,
          snap["date_range"]["first"])
    check("boundary frame: rows_used is the exact floored count",
          snap["rows_used"] == 279,
          f"rows_used={snap['rows_used']} expected=279")
    ind_csv = pd.read_csv(tmp8 / "results" / "indicators.csv",
                          index_col=0, parse_dates=True)
    check("2018-12-31 row dropped (indicators.csv has no pre-2019 row)",
          ind_csv.index.min() >= pd.Timestamp(FLOOR_FIRST),
          str(ind_csv.index.min()))
finally:
    sys.argv = old_argv

# 7h. Cutoff before the floor (INS-REVIEW-027 §2, second empty path):
# fetched rows span 2019+, but --as-of precedes the floor. The guard
# passes (the floored frame is non-empty); apply_asof then raises for
# zero sessions, with the minimum-row wording the harness classifies
# as skip. No snapshot is written.
tmp9 = Path(tempfile.mkdtemp(prefix="talib436-floor-asof-before-"))
os.chdir(tmp9)
ta.fetch_ohlc = fake_fetch_floor
sys.argv = ["ta.py", "COMI.CA", "--as-of", "2018-06-30"]
try:
    try:
        rc = ta.main()
        check("cutoff before floor refuses (zero sessions on/before it)",
              False, f"main() returned rc={rc} instead of raising")
    except RuntimeError as e:
        check("cutoff before floor refuses (zero sessions on/before it)",
              "only 0 sessions" in str(e)
              and "< 60 minimum rows" in str(e), str(e))
    check("no snapshot on the cutoff-before-floor refusal",
          not (tmp9 / "results" / "ta_latest.json").exists(),
          "ta_latest.json written despite refusal")
finally:
    sys.argv = old_argv

# 7e. The floor constant is the shared 2019-01-01 value.
check("DROP_BEFORE constant is 2019-01-01", ta.DROP_BEFORE == "2019-01-01")

failed = [c for c in CHECKS if not c[1]]
print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks pass")
sys.exit(1 if failed else 0)
