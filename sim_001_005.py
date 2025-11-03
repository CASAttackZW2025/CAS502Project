#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified script: historical → stages_df → (optional) rebuild +
discrete-event simulation → simulated durations →
historical vs simulated duration comparison.

This merges your “old” historical pipeline and your newer simulation script,
and adds a bridge at the end of the simulation so you can compare
historical WC/stage durations to the simulated WC/stage durations
in a single run.

Output of the comparison:
    analysis/duration_compare/duration_compare.csv
"""

from __future__ import annotations

# ============================================================
# IMPORTS
# ============================================================
import os
import re
import math
import json
import time
import logging
import warnings
from pathlib import Path
from collections import defaultdict
from itertools import combinations, product
from typing import Dict, Any, List, Tuple, Optional

from datetime import date, datetime, timedelta
import datetime as dt

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import networkx as nx
import simpy

from scipy import stats, special  # for historical part – duration fitting etc.

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# GLOBAL COLORMAPS / CONSTANTS
# ============================================================
cmap = mpl.colormaps.get("bwr")
tab20 = mpl.colormaps.get("tab20")

DATE_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d",
    "%m-%d-%Y", "%m-%d-%y", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M"
]

# for sim
_EXCEL_EPOCH = dt.date(1899, 12, 30)
MAX_REWORK_ROUNDS = 5
PAST_LOOKBACK_DAYS = 365
HOURS_PER_DAY_DEFAULT = 13.0
REMOVE_WEEKENDS_DEFAULT = True
SIM_END_DATE = dt.date(2027, 1, 1)
DURATION_TUNING_FILENAME_JSON = "duration_tuning.json"
DURATION_TUNING_FILENAME_CSV = "duration_tuning.csv"

# ============================================================
# BASIC HELPERS (shared)
# ============================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_one_datetime(x):
    if pd.isna(x):
        return pd.NaT
    # Excel serial days
    if isinstance(x, (int, float)) and np.isfinite(x):
        if x > 60:  # 1900 leap bug pivot
            return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(x), unit="D")
        return pd.NaT
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return pd.NaT
        for fmt in DATE_FORMATS:
            try:
                return pd.to_datetime(datetime.strptime(s, fmt))
            except Exception:
                pass
    # Last resort
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT


def parse_datetime_series(series: pd.Series) -> pd.Series:
    return series.apply(_parse_one_datetime)


def canonical_wc(s: str) -> str:
    """Normalize names like WC06eE01_Blah to WC06."""
    if not isinstance(s, str):
        return str(s)
    s = s.strip()
    m = re.match(r"(WC\d+)", s, re.I)
    if m:
        return m.group(1).upper()
    return s.strip()


def find_block_columns(all_cols: List[str], first_col: str, last_col: str) -> List[str]:
    if first_col not in all_cols or last_col not in all_cols:
        raise ValueError(f"Block endpoints not found: {first_col} .. {last_col}")
    i = all_cols.index(first_col)
    j = all_cols.index(last_col)
    return all_cols[i:j + 1] if i <= j else all_cols[j:i + 1]


def _distinct_colors(n: int, seed: int = 7) -> List[tuple]:
    """Stable palette: tab20 first, then HSV overflow."""
    base = list(tab20.colors)
    if n <= len(base):
        return base[:n]
    rng = np.random.default_rng(seed)
    hsv = np.column_stack(
        [rng.random(n), 0.65 + 0.35 * rng.random(n), 0.65 + 0.35 * rng.random(n)]
    )
    rgb = mcolors.hsv_to_rgb(hsv)
    return [tuple(rgb[i]) for i in range(n)]


def _rgba_hex_list(colors: List[tuple]) -> List[str]:
    return [mcolors.to_hex(c, keep_alpha=False) for c in colors]


# ============================================================
# HOLIDAYS & DATE UTIL (historical pipeline part)
# ============================================================
def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    cnt = 0
    while True:
        if d.weekday() == weekday:
            cnt += 1
            if cnt == n:
                return d
        d += timedelta(days=1)


def get_holidays(years: List[int]) -> set:
    h = set()
    for y in years:
        # Memorial Day (last Mon May)
        h.add(last_weekday_of_month(y, 5, 0))
        # Labor Day (1st Mon Sep)
        h.add(nth_weekday_of_month(y, 9, 0, 1))
        # July 4
        h.add(date(y, 7, 4))
        # Thanksgiving (4th Thu) + day after
        th = nth_weekday_of_month(y, 11, 3, 4)
        h.add(th)
        h.add(th + timedelta(days=1))
        # Dec 24 .. Jan 1 (inclusive)
        d = date(y, 12, 24)
        while d <= date(y + 1, 1, 1):
            h.add(d)
            d += timedelta(days=1)
    return h


# ============================================================
# SAVE HELPER (for historical plots)
# ============================================================
def _save_via_monkeypatched_show(plot_func, out_path: str, *args, **kwargs):
    """
    Call a plotting function `plot_func(*args, **kwargs)` that normally ends with plt.show(),
    but intercept plt.show() to save a PNG and close instead.
    """
    ensure_dir(os.path.dirname(out_path))
    _orig_show = plt.show
    try:
        def _save_and_close():
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close('all')
        plt.show = _save_and_close
        plot_func(*args, **kwargs)
    finally:
        plt.show = _orig_show


# ============================================================
# READER (historical pipeline)
# ============================================================
def read_tabular(input_file: str, sheet: int | str | None = None) -> pd.DataFrame:
    """
    Read Excel (.xls/.xlsx) or CSV (.csv) into a DataFrame.
    """
    lower = input_file.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(input_file)
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(input_file, sheet_name=sheet)
    else:
        raise ValueError(f"Unsupported file type for: {input_file}")


# ============================================================
# HISTORICAL PIPELINE CORE
# (this is your big produce_all_outputs(), trimmed only by comments)
# ============================================================
def build_wip_for_subset(
    df_subset: pd.DataFrame,
    end_block: List[str],
    date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    wip = pd.DataFrame(0, index=date_index, columns=end_block)
    for _, r in df_subset.iterrows():
        s = r.get("Start", pd.NaT)
        e = r.get("End", pd.NaT)
        stg = r.get("StageName", None)
        if stg not in end_block:
            continue
        if pd.isna(s) and pd.isna(e):
            continue
        if pd.isna(s):
            s = pd.Timestamp.min
        if pd.isna(e):
            continue
        day_start = pd.to_datetime(s).normalize()
        day_end = pd.to_datetime(e).normalize()
        for d in pd.date_range(day_start, day_end, freq="D"):
            start_of_day = pd.Timestamp(d)
            end_of_day = start_of_day + pd.Timedelta(days=1)
            if (pd.to_datetime(s) < end_of_day) and (pd.to_datetime(e) > start_of_day):
                wip.at[start_of_day, stg] += 1
    return wip


def recover_end_block_from_stages(stages_df: pd.DataFrame) -> List[str]:
    if "StageName" not in stages_df.columns or "DeclaredPos" not in stages_df.columns:
        raise ValueError("stages_df missing StageName / DeclaredPos; cannot recover end_block.")
    order = (
        stages_df[["StageName", "DeclaredPos"]]
        .dropna()
        .groupby("StageName")["DeclaredPos"]
        .median()
        .sort_values()
    )
    return list(order.index)


def plot_gantt_for_product(
    stages_df: pd.DataFrame,
    product: str,
    outfile: str,
    end_block: List[str],
) -> None:
    sdf = stages_df[stages_df["Product"] == product].copy()
    if sdf.empty:
        return
    jobs = sorted(sdf["SourceRow"].unique())
    ymap = {j: i for i, j in enumerate(jobs)}
    fig, ax = plt.subplots(figsize=(12, max(4, len(jobs) * 0.15)))
    colors = list(tab20.colors)

    for _, r in sdf.iterrows():
        s = r["Start"]
        e = r["End"]
        st = r["StageName"]
        if pd.isna(s) or pd.isna(e):
            continue
        y = ymap[r["SourceRow"]]
        try:
            stage_idx = end_block.index(st)
        except ValueError:
            stage_idx = 0
        ax.barh(
            y + stage_idx / (len(end_block) + 1),
            (pd.to_datetime(e) - pd.to_datetime(s)).total_seconds() / 86400.0,
            left=pd.to_datetime(s),
            height=0.6 / len(end_block),
            color=colors[stage_idx % len(colors)],
            edgecolor="k",
        )

    ax.set_yticks(list(ymap.values()))
    ax.set_yticklabels(list(ymap.keys()))
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.title(f"Gantt - product {product}")
    plt.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def save_gantts_from_history(stages_df: pd.DataFrame, out_folder: str = "analysis", end_block: List[str] | None = None) -> None:
    ensure_dir(out_folder)
    csv_dir_local = os.path.join(out_folder, "csv")
    ensure_dir(csv_dir_local)
    try:
        stages_df.to_csv(os.path.join(csv_dir_local, "stages_long_from_history.csv"), index=False)
    except Exception:
        pass

    products = stages_df["Product"].fillna("UNKNOWN").unique().tolist()
    for p in products:
        prod_folder = os.path.join(out_folder, "gantts", str(p).replace(" ", "_"))
        ensure_dir(prod_folder)
        outfile = os.path.join(prod_folder, f"gantt_{str(p).replace(' ', '_')}.png")

        dfp = stages_df[stages_df["Product"] == p].copy()
        if (
            not pd.api.types.is_datetime64_any_dtype(dfp["Start"])
            and dfp["Start"].dropna().apply(lambda x: isinstance(x, (int, float))).all()
        ):
            base = pd.Timestamp("1970-01-01")
            dfp["Start"] = dfp["Start"].apply(lambda h: base + pd.Timedelta(hours=float(h)) if pd.notna(h) else pd.NaT)
            dfp["End"] = dfp["End"].apply(lambda h: base + pd.Timedelta(hours=float(h)) if pd.notna(h) else pd.NaT)

        try:
            plot_gantt_for_product(dfp, p, outfile, end_block or sorted(dfp["StageName"].unique()))
            print("Saved Gantt for product", p, "->", outfile)
        except Exception as e:
            print("Failed to create Gantt for product", p, ":", e)


def produce_all_outputs(
    input_file: str = "jobs.csv",
    sheet: int | str = 0,
    product_col: str = "Type",
    first_end_col: str = "Frame",
    last_end_col: str = "Cure Times.ShipCureDate",
    out_prefix: str = "analysis",
    use_now_for_null: bool = True,
) -> pd.DataFrame:
    # 1) read data and detect block
    df = read_tabular(input_file, sheet=sheet)
    cols = list(df.columns)
    if product_col not in cols:
        raise ValueError(f"Missing product column: {product_col}")
    end_block = find_block_columns(cols, first_end_col, last_end_col)

    # 2) parse end columns
    for c in end_block:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    if use_now_for_null:
        df[end_block[-1]] = df[end_block[-1]].fillna(pd.Timestamp.now())

    # 3) build canonical stages_df (Start = previous End)
    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        prod = row.get(product_col, "UNKNOWN")
        ends = [pd.to_datetime(row[c], errors="coerce") for c in end_block]
        starts = [pd.NaT] + ends[:-1]
        for pos, (stg, s, e) in enumerate(zip(end_block, starts, ends), start=1):
            rows.append(
                {
                    "SourceRow": idx,
                    "Product": prod if pd.notna(prod) else "UNKNOWN",
                    "StageName": stg,
                    "DeclaredPos": pos,
                    "Start": s,
                    "End": e,
                }
            )
    stages_df = pd.DataFrame(rows)
    stages_df["DurationHours"] = (stages_df["End"] - stages_df["Start"]).dt.total_seconds() / 3600.0
    stages_df["DurationDays"] = stages_df["DurationHours"] / 24.0

    # 4) build date_index and holidays
    all_dates = (
        pd.to_datetime(stages_df["End"].dropna()).dt.date.tolist()
        + pd.to_datetime(stages_df["Start"].dropna()).dt.date.tolist()
    )
    if not all_dates:
        raise SystemExit("No start or end dates detected.")
    date_min = min(all_dates)
    date_max = max(all_dates)
    date_index = pd.date_range(start=date_min - timedelta(days=1), end=date_max + timedelta(days=1), freq="D")
    years = sorted({d.year for d in pd.to_datetime(date_index).date})
    _ = get_holidays(years)  # reserved for future exclusions

    # make output dirs
    ensure_dir(out_prefix)
    ensure_dir(os.path.join(out_prefix, "csv"))
    ensure_dir(os.path.join(out_prefix, "gantts"))
    ensure_dir(os.path.join(out_prefix, "wip_by_product"))
    ensure_dir(os.path.join(out_prefix, "wip_combined"))
    ensure_dir(os.path.join(out_prefix, "throughput_by_product"))
    ensure_dir(os.path.join(out_prefix, "per_stage_histograms"))

    # 5) Save canonical stages CSV
    stages_df.to_csv(os.path.join(out_prefix, "csv", "stages_long_safeorder.csv"), index=False)

    # 6) Per-product GANTTs
    save_gantts_from_history(stages_df, out_folder=out_prefix, end_block=end_block)

    # 7) Per-product WIP and per-stage plots
    products = stages_df["Product"].fillna("UNKNOWN").unique().tolist()
    for p in products:
        subset = stages_df[stages_df["Product"] == p].copy()
        if subset.empty:
            continue

        wip = build_wip_for_subset(subset, end_block=end_block, date_index=date_index)
        prod_folder = os.path.join(out_prefix, "wip_by_product", str(p).replace(" ", "_"))
        ensure_dir(prod_folder)
        wip.reset_index().rename(columns={"index": "Date"}).to_csv(
            os.path.join(prod_folder, f"wip_{str(p).replace(' ', '_')}.csv"), index=False
        )

        for st in end_block:
            series = wip.get(st, None)
            if series is None or series.sum() == 0:
                continue
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.step(series.index, series.values, where="mid", linewidth=1.5)
            ax.set_title(f"WIP (step) - {p} - {st}")
            ax.set_ylabel("Items in process")
            ax.set_xlabel("Date")
            fig.autofmt_xdate()
            plt.tight_layout()
            fname = os.path.join(prod_folder, f"wip_{str(p).replace(' ', '_')}_{st.replace(' ', '_')}.png")
            fig.savefig(fname, dpi=150)
            plt.close(fig)

    # 8) Combined WIP – STEP-FILL
    wip_combined = build_wip_for_subset(stages_df, end_block=end_block, date_index=date_index)
    wip_combined = wip_combined.loc[(wip_combined.sum(axis=1) > 0)]
    wip_combined.reset_index().rename(columns={"index": "Date"}).to_csv(
        os.path.join(out_prefix, "wip_combined", "wip_combined.csv"), index=False
    )

    stage_names = end_block
    colors = _distinct_colors(len(stage_names))
    fig, ax = plt.subplots(figsize=(12, 6))
    x = wip_combined.index
    cum = np.zeros(len(x), dtype=float)
    for i, st in enumerate(stage_names):
        y = wip_combined[st].astype(float).values
        ax.step(x, cum + y, where='post', linewidth=1.2, color=colors[i])
        ax.fill_between(
            x,
            cum,
            cum + y,
            step='post',
            facecolor=colors[i],
            edgecolor='black',
            linewidth=0.5,
            alpha=0.95,
            label=st,
        )
        cum += y
    ax.set_ylabel("Items in process (WIP)")
    ax.set_xlabel("Date")
    ax.set_title("Stacked WIP by Stage over Time (step)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(os.path.join(out_prefix, "stacked_wip.png"), dpi=150)
    plt.close(fig)

    # 9) Per-stage histograms
    for st in end_block:
        series = stages_df.loc[stages_df["StageName"] == st, "DurationHours"].dropna()
        if series.empty:
            continue
        fname = os.path.join(out_prefix, "per_stage_histograms", f"hist_{st.replace(' ', '_')}.png")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(series, bins=40, alpha=0.7)
        ax.set_title(f"Duration histogram (hours) - {st}")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Count")
        fig.tight_layout()
        fig.savefig(fname, dpi=120)
        plt.close(fig)

    # 10) Throughput (combined + per-product) – date-based
    finished_stage = end_block[-1]
    fin_all = stages_df[(stages_df["StageName"] == finished_stage) & (stages_df["End"].notna())].copy()
    if not fin_all.empty:
        fin_all["EndDate"] = pd.to_datetime(fin_all["End"]).dt.normalize()
        daily_all = fin_all.groupby("EndDate").size().reindex(pd.date_range(date_min, date_max, freq="D"), fill_value=0)
        daily_all_df = daily_all.reset_index()
        daily_all_df.columns = ["Date", "Completions"]
        daily_all_df.to_csv(os.path.join(out_prefix, "csv", "throughput_combined.csv"), index=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.step(daily_all.index, daily_all.values, where="mid", linewidth=1.5)
        ax.set_title("Completions/day - Combined")
        ax.set_ylabel("Completed units")
        ax.set_xlabel("Date")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(out_prefix, "throughput_by_product", "throughput_combined.png"), dpi=150)
        plt.close(fig)

    # 11) simple power-law learning curve
    finished = stages_df[stages_df["StageName"] == finished_stage].dropna(subset=["End"]).copy()
    if not finished.empty:
        earliest_start = stages_df.groupby("SourceRow")["Start"].min().rename("JobStart")
        finished = finished.merge(earliest_start, left_on="SourceRow", right_index=True, how="left")
        finished["ProdTimeHours"] = ((finished["End"] - finished["JobStart"]).dt.total_seconds() / 3600.0)
        finished = finished[finished["ProdTimeHours"] > 0].sort_values("End").reset_index(drop=True)
        finished["UnitIndex"] = np.arange(1, len(finished) + 1)

        x = finished["UnitIndex"].values
        y = finished["ProdTimeHours"].values
        mask = (x > 0) & (y > 0)
        if mask.sum() > 3:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log(x[mask]), np.log(y[mask])
            )
            b = -slope
            T1 = np.exp(intercept)
            finished["Fitted"] = T1 * (finished["UnitIndex"] ** (-b))
            finished.to_csv(os.path.join(out_prefix, "csv", "learning_curve_units.csv"), index=False)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(np.log(finished["UnitIndex"]), np.log(finished["ProdTimeHours"]), s=8, label="obs")
            ax.plot(np.log(finished["UnitIndex"]), np.log(finished["Fitted"]), label="fitted")
            ax.set_xlabel("log(UnitIndex)")
            ax.set_ylabel("log(ProdTimeHours)")
            ax.legend()
            ax.set_title(f"Learning curve log-log (b={b:.3f}, R2={r_value ** 2:.3f})")
            fig.tight_layout()
            fig.savefig(os.path.join(out_prefix, "csv", "learning_curve_loglog.png"), dpi=150)
            plt.close(fig)

    print("All outputs produced under:", out_prefix)
    return stages_df


# ============================================================
# SIMULATION-SIDE HELPERS (file reading, dates)
# ============================================================
def _read_csv_relaxed(filename: str) -> pd.DataFrame:
    """Look in CWD, then /mnt/data."""
    if os.path.exists(filename):
        return pd.read_csv(filename)
    alt = os.path.join("/mnt/data", filename)
    if os.path.exists(alt):
        return pd.read_csv(alt)
    raise FileNotFoundError(f"{filename} not found in CWD or /mnt/data")


def _maybe_excel_ordinal(x):
    try:
        v = float(x)
    except Exception:
        return None
    if 20000 <= v <= 80000:
        return _EXCEL_EPOCH + dt.timedelta(days=int(v))
    return None


def coerce_date_series(col: pd.Series) -> pd.Series:
    """
    Parse messy date columns without noisy warnings.
    """
    col = col.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=col.index)

    common_formats = ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"]
    remaining = pd.Series(True, index=col.index)

    # explicit formats
    for fmt in common_formats:
        try:
            parsed = pd.to_datetime(col[remaining], format=fmt, errors="coerce")
            hit = parsed.notna()
            out.loc[remaining & hit] = parsed[hit]
            remaining = remaining & ~hit
        except Exception:
            pass

    # "Wed 2/4/26"
    if remaining.any():
        parsed = pd.to_datetime(col[remaining], format="%a %m/%d/%y", errors="coerce")
        hit = parsed.notna()
        out.loc[remaining & hit] = parsed[hit]
        remaining = remaining & ~hit

    # excel ordinals
    if remaining.any():
        ords = col[remaining].apply(_maybe_excel_ordinal)
        parsed = pd.to_datetime(ords, errors="coerce")
        hit = parsed.notna()
        out.loc[remaining & hit] = parsed[hit]
        remaining = remaining & ~hit

    # fallback
    if remaining.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(col[remaining], errors="coerce")
        out.loc[remaining] = parsed

    return out


# ============================================================
# CALENDAR FOR SIM
# ============================================================
class WorkingCalendar:
    def __init__(self, start_date: dt.date, remove_weekends: bool, holidays: List[dt.date], hours_per_day: float):
        self.start = start_date
        self.remove_weekends = remove_weekends
        self.holidays = set(holidays)
        self.hpd = float(hours_per_day)

    def to_hours(self, d: dt.datetime | dt.date) -> float:
        """Convert a real date to 'simulation hours since calendar start'."""
        if isinstance(d, dt.datetime):
            d = d.date()
        if d < self.start:
            return 0.0
        hours = 0.0
        cur = self.start
        while cur < d:
            if (not self.remove_weekends or cur.weekday() < 5) and cur not in self.holidays:
                hours += self.hpd
            cur += dt.timedelta(days=1)
        return hours


# ============================================================
# SIM – LOAD SUPPORTING TABLES
# ============================================================
def load_resource_caps(df: pd.DataFrame) -> Dict[str, int]:
    """resource.csv → WC → capacity (default 1)."""
    caps = {}
    if df is None or df.empty:
        return caps
    name_col = None
    qty_cols = []
    for c in df.columns:
        if re.search(r"(work.?center|wc|resource)", str(c), re.I):
            name_col = c
        if re.search(r"(qty|capacity|units|stations|no\.?of)", str(c), re.I):
            qty_cols.append(c)
    if name_col is None:
        name_col = df.columns[0]
    for _, r in df.iterrows():
        wc_raw = str(r.get(name_col, "")).strip()
        wc = canonical_wc(wc_raw)
        total = 0
        for qc in qty_cols:
            try:
                v = float(r.get(qc, np.nan))
                if np.isfinite(v) and v > 0:
                    total += int(round(v))
            except Exception:
                pass
        caps[wc] = max(1, total) if total > 0 else 1
    # known singletons
    caps.setdefault("WC17", 1)
    return caps


def load_pto_days(df: pd.DataFrame) -> List[dt.date]:
    if df is None or df.empty:
        return []
    dates = []
    for c in df.columns:
        if re.search(r"(date|day)", str(c), re.I):
            s = coerce_date_series(df[c]).dt.date
            dates.extend([d for d in s.dropna().tolist()])
    return sorted(set(dates))


# ============================================================
# PRODUCT / ROUTES (SIM)
# ============================================================
def _extract_number(text: str) -> Optional[float]:
    if not isinstance(text, str):
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _to_hours_from_row(row: pd.Series) -> float:
    for col in ["DurationHours", "DurationHours ", "durationhours"]:
        if col in row and pd.notna(row[col]):
            try:
                v = float(row[col])
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass
    if "Nominal" in row and pd.notna(row["Nominal"]):
        try:
            v = float(row["Nominal"])
            if np.isfinite(v) and v > 0:
                return v / 60.0
        except Exception:
            v = _extract_number(str(row["Nominal"]))
            if v and v > 0:
                return v / 60.0
    if "ProcessingTime" in row and pd.notna(row["ProcessingTime"]):
        pt = row["ProcessingTime"]
        try:
            v = float(pt)
            if np.isfinite(v) and v > 0:
                if v >= 1000:
                    return v / 60.0
                return v
        except Exception:
            v = _extract_number(str(pt))
            if v and v > 0:
                if v >= 1000:
                    return v / 60.0
                return v
    return 0.0


def _to_wait_hours_from_row(row: pd.Series) -> float:
    wait = 0.0

    def _num(x):
        if pd.isna(x):
            return None
        try:
            return float(x)
        except Exception:
            return _extract_number(str(x))

    if "WaitTimeDurationHours" in row:
        w = _num(row["WaitTimeDurationHours"])
        if w and w > 0:
            if w >= 1000:
                wait += w / 60.0
            else:
                wait += w
    if "WaitTime" in row:
        w = _num(row["WaitTime"])
        if w and w > 0:
            if w >= 1000:
                wait += w / 60.0
            else:
                wait += w
    if "NDayCure" in row:
        d = _num(row["NDayCure"])
        if d and d > 0:
            wait += d * 24.0
    if "MHrCureDrying" in row:
        h = _num(row["MHrCureDrying"])
        if h and h > 0:
            wait += h
    return float(wait)


def build_routes_from_product_primary(df: pd.DataFrame, caps: Dict[str, int]) -> Dict[str, Dict[str, Any]]:
    """Build per-part precedence graphs from product.csv."""
    if df is None or df.empty:
        return {}

    part_col = "Part"
    seq_col = "ID" if "ID" in df.columns else ("Sequence" if "Sequence" in df.columns else df.columns[0])
    wc_col = "Workcenter" if "Workcenter" in df.columns else "WorkCenter"
    desc_col = "OPDESC" if "OPDESC" in df.columns else None

    routes: Dict[str, Dict[str, Any]] = {}
    for part, sub in df.groupby(part_col):
        sub = sub.copy()
        sub["__seq__"] = pd.to_numeric(sub[seq_col], errors="coerce")
        sub = sub.sort_values("__seq__", kind="mergesort")

        G = nx.DiGraph()
        proc_times = {
            "Start_Job": {"processing_time": 0.0, "capacity": 1, "stochastic": False},
            "Units_Delivered": {"processing_time": 0.0, "capacity": 1, "stochastic": False},
        }
        prev_node = "Start_Job"

        for _, row in sub.iterrows():
            seq = int(row["__seq__"]) if pd.notna(row["__seq__"]) else _
            wc_raw = str(row.get(wc_col, f"{part}_STEP_{seq}")).strip()
            op_node = f"{wc_raw}_{seq}"
            desc = str(row.get(desc_col, "")).strip() if desc_col else ""

            dur_hours = _to_hours_from_row(row)
            cap = caps.get(canonical_wc(wc_raw), 1)

            proc_times[op_node] = {
                "processing_time": dur_hours,
                "capacity": cap,
                "stochastic": bool(dur_hours > 0),
                "desc": desc,
            }
            if not G.has_node(op_node):
                G.add_node(op_node)

            G.add_edge(prev_node, op_node)
            prev_node = op_node

            wait_hours = _to_wait_hours_from_row(row)
            if wait_hours > 0.0:
                wait_node = f"{wc_raw}_{seq}_WAIT"
                proc_times[wait_node] = {
                    "processing_time": wait_hours,
                    "capacity": 999999,
                    "stochastic": False,
                    "desc": f"WAIT/CURE after {desc}" if desc else "WAIT/CURE",
                }
                G.add_node(wait_node)
                G.add_edge(op_node, wait_node)
                prev_node = wait_node

        G.add_edge(prev_node, "Units_Delivered")

        routes[f"Path:{part}"] = {
            "processing_times": proc_times,
            "graph": G,
        }

    return routes


# ============================================================
# DURATION TUNING LOADER (SIM)
# ============================================================
def load_duration_tuning() -> Dict[str, float]:
    """Look for duration_tuning.json/csv → {processor_name: multiplier}."""
    if os.path.exists(DURATION_TUNING_FILENAME_JSON):
        with open(DURATION_TUNING_FILENAME_JSON, "r") as f:
            data = json.load(f)
        return {str(k): float(v) for k, v in data.items()}

    if os.path.exists(DURATION_TUNING_FILENAME_CSV):
        df = pd.read_csv(DURATION_TUNING_FILENAME_CSV)
        name_col = next((c for c in df.columns if "proc" in c.lower() or "name" in c.lower()), df.columns[0])
        mult_col = next((c for c in df.columns if "mult" in c.lower() or "factor" in c.lower()), df.columns[1])
        out = {}
        for _, r in df.iterrows():
            pname = str(r[name_col]).strip()
            try:
                mult = float(r[mult_col])
            except Exception:
                continue
            out[pname] = mult
        return out

    return {}


# ============================================================
# JOBS + SCHEDULE → ARRIVALS (SIM)
# ============================================================
def load_jobs_as_df(path: str = "jobs.csv") -> Optional[pd.DataFrame]:
    try:
        return _read_csv_relaxed(path)
    except FileNotFoundError:
        return None


def load_schedule_as_df(path: str = "schedule.csv") -> Optional[pd.DataFrame]:
    try:
        return _read_csv_relaxed(path)
    except FileNotFoundError:
        return None


def infer_date_col(df: pd.DataFrame, prefer_future: bool = False) -> str:
    for pat in ["Finish", "finish", "Complete", "complete", "Date", "date", "P&S Finish", "P&S Prev Week Finish"]:
        for c in df.columns:
            if pat.lower() in c.lower():
                return c
    return df.columns[1]


def infer_flight_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "program" in c.lower() and "flight" in c.lower():
            return c
        if "flight" in c.lower():
            return c
        if "program" in c.lower():
            return c
    return df.columns[0]


def flight_to_part(name: str) -> str:
    name = str(name)
    if "tanzanite" in name.lower():
        return "SPM_Tanzanite"
    if "sapphire" in name.lower():
        return "SPM_Sapphire"
    return name.strip()


def load_parts_map(parts_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    if parts_df is None or parts_df.empty:
        return {}
    key_col = None
    val_col = None
    for c in parts_df.columns:
        if "flight" in c.lower() or "program" in c.lower() or "event" in c.lower():
            key_col = c
        if "part" in c.lower() or "product" in c.lower() or "config" in c.lower():
            val_col = c
    if key_col is None:
        return {}
    if val_col is None:
        val_col = key_col
    m = {}
    for _, r in parts_df.iterrows():
        key = str(r.get(key_col, "")).strip()
        val = str(r.get(val_col, "")).strip()
        if not val:
            val = flight_to_part(key)
        if key:
            m[key] = val
    return m


def jobs_to_arrivals(jobs_df: pd.DataFrame, parts_map: Dict[str, str], today: dt.date) -> List[Dict[str, Any]]:
    date_col = infer_date_col(jobs_df)
    flight_col = infer_flight_col(jobs_df)
    jobs_df = jobs_df.copy()
    jobs_df["__date__"] = coerce_date_series(jobs_df[date_col])
    lower_bound = today - dt.timedelta(days=PAST_LOOKBACK_DAYS)
    jobs_df = jobs_df[(jobs_df["__date__"].notna()) & (jobs_df["__date__"].dt.date >= lower_bound) & (jobs_df["__date__"].dt.date <= today)]
    arrivals = []
    for _, r in jobs_df.iterrows():
        flight = str(r[flight_col]).strip()
        part = parts_map.get(flight, flight_to_part(flight))
        when = pd.Timestamp(r["__date__"]).to_pydatetime().replace(hour=8, minute=0, second=0, microsecond=0)
        arrivals.append({"part": part, "when": when, "qty": 1, "source": "history"})
    return arrivals


def schedule_to_arrivals(schedule_df: pd.DataFrame, jobs_df: Optional[pd.DataFrame], parts_map: Dict[str, str], today: dt.date) -> List[Dict[str, Any]]:
    flight_col = infer_flight_col(schedule_df)
    date_col = infer_date_col(schedule_df, prefer_future=True)
    schedule_df = schedule_df.copy()
    schedule_df["__date__"] = coerce_date_series(schedule_df[date_col])

    # infer qty per flight from jobs history
    jobs_counts = {}
    if jobs_df is not None and not jobs_df.empty:
        jflight = infer_flight_col(jobs_df)
        jdate = infer_date_col(jobs_df)
        jdf = jobs_df.copy()
        jdf["__date__"] = coerce_date_series(jdf[jdate])
        jdf = jdf[jdf["__date__"].notna()]
        jobs_counts = jdf.groupby(jflight).size().to_dict()

    arrivals = []
    for _, r in schedule_df.iterrows():
        flight = str(r[flight_col]).strip()
        when_ts = r["__date__"]
        if pd.isna(when_ts):
            continue
        when_dt = pd.Timestamp(when_ts).to_pydatetime().replace(hour=8, minute=0, second=0, microsecond=0)
        if when_dt.date() <= today:
            continue
        if when_dt.date() >= SIM_END_DATE:
            continue
        qty = int(jobs_counts.get(flight, 1))
        part = parts_map.get(flight, flight_to_part(flight))
        arrivals.append({"part": part, "when": when_dt, "qty": qty, "source": "future"})
    return arrivals


# ============================================================
# ENTITY SPECS (SIM)
# ============================================================
def build_entity_specs(arrivals: List[Dict[str, Any]], calendar: WorkingCalendar, default_priority: int = 10) -> List[Dict[str, Any]]:
    specs = []
    eid = 1
    for a in arrivals:
        part = a["part"]
        when = a["when"]
        qty = int(a.get("qty", 1))
        for _ in range(qty):
            at = float(calendar.to_hours(when))
            specs.append(
                {
                    "entity_id": eid,
                    "arrival_time": at,
                    "path": f"Path:{part}",
                    "p_level": 1,
                    "priority": default_priority,
                }
            )
            eid += 1
    specs.sort(key=lambda d: d["arrival_time"])
    return specs


def specs_to_arrival_pattern(entity_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(int)
    meta = {}
    for s in entity_specs:
        key = (s["arrival_time"], s["path"], s["p_level"], s["priority"])
        grouped[key] += 1
        meta[key] = s
    pattern = []
    for key, cnt in sorted(grouped.items(), key=lambda kv: kv[0]):
        at, path, plevel, prio = key
        pattern.append(
            {
                "arrival_time": float(at),
                "num_entities": int(cnt),
                "entity_paths": [path] * int(cnt),
                "p_levels": [plevel] * int(cnt),
                "priorities": [prio] * int(cnt),
            }
        )
    return pattern


# ============================================================
# STOCHASTIC SAMPLING (SIM)
# ============================================================
def sample_pert(a: float, m: float, b: float, lamb: float = 4.0) -> float:
    if not (a <= m <= b):
        return m
    alpha = 1 + lamb * (m - a) / (b - a)
    beta = 1 + lamb * (b - m) / (b - a)
    x = np.random.beta(alpha, beta)
    return a + x * (b - a)


def sample_stochastic_duration(t0: float) -> float:
    if t0 <= 0:
        return 0.0
    r = np.random.rand()
    lo = 0.9 * t0
    hi = 1.3 * t0
    if r < 0.45:
        return sample_pert(lo, t0, hi)
    elif r < 0.9:
        return float(np.random.triangular(lo, t0, hi))
    else:
        tail = t0 * (1.0 + np.random.pareto(2.5))
        return float(min(3.0 * t0, tail))


# ============================================================
# PROCESS SIMULATION CLASS
# ============================================================
class ProcessSimulation:
    def __init__(
        self,
        initial_entities,
        unit_arrival_pattern,
        paths_info,
        exclude_processors=None,
        n_entities=100,
        default_p_level=1,
        default_entity_path="Path:SPM_Tanzanite",
        default_priority=10,
        top_n=10,
        duration_scalars: Optional[Dict[str, float]] = None,
    ):
        self.env = simpy.Environment()
        self.initial_entities = initial_entities
        self.unit_arrival_pattern = unit_arrival_pattern
        self.paths_info = paths_info
        self.exclude_processors = exclude_processors or []
        self.n_entities = n_entities
        self.default_p_level = default_p_level
        self.default_entity_path = default_entity_path
        self.default_priority = default_priority
        self.top_n = top_n

        # apply user tuning
        if duration_scalars:
            self.apply_duration_scalars(duration_scalars)

        # composed graph of all routes
        self.G_all = self.compose_graphs()
        self.processor_capacities, self.processor_to_sub, self.sub_to_proc = self.set_processor_capacities()
        self.processor_resources = {p: simpy.PriorityResource(self.env, capacity=1) for p in self.processor_capacities}
        for p, cap in self.processor_capacities.items():
            if cap > 1:
                for s in self.processor_to_sub[p]:
                    self.processor_resources[s] = simpy.PriorityResource(self.env, capacity=1)
            else:
                self.processor_resources[p] = simpy.PriorityResource(self.env, capacity=1)

        self.entity_attributes = {}
        self.entity_processor_history = defaultdict(lambda: defaultdict(list))
        self.start_times = defaultdict(dict)
        self.finish_times = defaultdict(dict)
        self.busy_periods = defaultdict(list)
        self.entity_id_counter = 1
        self.total_entities_generated = 0
        self.results = None

        # new logs
        self.wait_times = defaultdict(list)          # proc -> list of wait durations
        self.queue_snapshots = defaultdict(list)     # proc -> list of (time, q_len)
        self.entity_cycle = {}                       # eid -> {"arrival_h": ..., "finish_h": ...}
        self.rework_counts = defaultdict(int)        # optional rework stats
        self.proc_durations_log = defaultdict(list)  # proc -> [(time, duration)]

    # ------------------------------------------------------------------
    def apply_duration_scalars(self, scalars: Dict[str, float]):
        for _, pdata in self.paths_info.items():
            for node, info in pdata["processing_times"].items():
                if node in scalars:
                    try:
                        factor = float(scalars[node])
                        info["processing_time"] = float(info.get("processing_time", 0.0)) * factor
                    except Exception:
                        pass

    def compose_graphs(self):
        G_all = nx.DiGraph()
        for _, pdata in self.paths_info.items():
            G_all = nx.compose(G_all, pdata["graph"])
        return G_all

    def set_processor_capacities(self):
        proc_caps = {}
        proc_to_sub = {}
        sub_to_proc = {}
        for _, pdata in self.paths_info.items():
            for proc, info in pdata["processing_times"].items():
                cap = int(info.get("capacity", 1))
                proc_caps[proc] = cap
                if cap > 1:
                    subs = [f"{proc}_{i+1}" for i in range(cap)]
                    proc_to_sub[proc] = subs
                    for s in subs:
                        sub_to_proc[s] = proc
                        proc_caps[s] = 1
                else:
                    proc_to_sub[proc] = [proc]
                    sub_to_proc[proc] = proc
        return proc_caps, proc_to_sub, sub_to_proc

    def get_processing_time(self, proc_info):
        base = proc_info.get("processing_time", 0.0)
        stoch = proc_info.get("stochastic", True)
        if not stoch:
            return base
        return sample_stochastic_duration(base)

    # ------------------------------------------------------------------
    # arrivals
    # ------------------------------------------------------------------
    def generate_entities(self):
        # explicit
        for info in self.initial_entities:
            at = info.get("arrival_time", 0.0)
            num = info["num_entities"]
            if self.total_entities_generated + num > self.n_entities:
                num = self.n_entities - self.total_entities_generated
            for _ in range(num):
                eid = self.entity_id_counter
                self.entity_id_counter += 1
                self.total_entities_generated += 1
                self.entity_attributes[eid] = {
                    "arrival_time": at,
                    "path": info["entity_paths"][0],
                    "p_level": info["p_levels"][0],
                    "priority": info["priorities"][0],
                }
                self.env.process(self.entity_arrival(eid, at))
            if self.total_entities_generated >= self.n_entities:
                break

        # repeat pattern if needed
        if self.total_entities_generated < self.n_entities:
            rep = 0
            max_at = max(i["arrival_time"] for i in self.unit_arrival_pattern)
            while self.total_entities_generated < self.n_entities:
                for info in self.unit_arrival_pattern:
                    at = info["arrival_time"] + rep * (max_at + 1)
                    num = info["num_entities"]
                    if self.total_entities_generated + num > self.n_entities:
                        num = self.n_entities - self.total_entities_generated
                    for j in range(num):
                        eid = self.entity_id_counter
                        self.entity_id_counter += 1
                        self.total_entities_generated += 1
                        self.entity_attributes[eid] = {
                            "arrival_time": at,
                            "path": info["entity_paths"][j % len(info["entity_paths"])],
                            "p_level": info["p_levels"][j % len(info["p_levels"])],
                            "priority": info["priorities"][j % len(info["priorities"])],
                        }
                        self.env.process(self.entity_arrival(eid, at))
                    if self.total_entities_generated >= self.n_entities:
                        break
                rep += 1

    def entity_arrival(self, eid, at):
        yield self.env.timeout(at)
        self.entity_cycle[eid] = {"arrival_h": self.env.now}
        ent = self.entity_attributes[eid]
        path = ent["path"]
        if path not in self.paths_info:
            logger.error(f"Entity {eid} path {path} missing in paths_info")
            return
        G = self.paths_info[path]["graph"]
        firsts = [n for n in G.nodes if G.in_degree(n) == 0]
        for f in firsts:
            if f in self.paths_info[path]["processing_times"]:
                self.env.process(self.process_at(eid, f))

    # ------------------------------------------------------------------
    # processing
    # ------------------------------------------------------------------
    def process_at(self, eid: int, proc: str):
        ent = self.entity_attributes[eid]
        path = ent["path"]
        if proc not in self.paths_info[path]["processing_times"]:
            return
        proc_info = self.paths_info[path]["processing_times"][proc]
        subprocs = self.processor_to_sub.get(proc, [proc])

        def _best_subproc():
            best = None
            best_len = 1e9
            for s in subprocs:
                qlen = len(self.processor_resources[s].queue)
                if qlen < best_len:
                    best_len = qlen
                    best = s
            return best

        sp = _best_subproc()
        res = self.processor_resources[sp]

        # snapshot queue length before requesting
        self.queue_snapshots[sp].append((self.env.now, len(res.queue)))

        with res.request(priority=ent["priority"]) as req:
            req_time = self.env.now
            yield req
            wait_dur = self.env.now - req_time
            self.wait_times[sp].append(wait_dur)

            start_t = self.env.now
            self.start_times[sp][eid] = start_t
            orig = self.sub_to_proc.get(sp, sp)
            self.entity_processor_history[eid][orig].append({"start_time": start_t})

            ptime = self.get_processing_time(proc_info)
            # log realized duration
            self.proc_durations_log[orig].append((self.env.now, ptime))

            yield self.env.timeout(ptime)

            finish_t = self.env.now
            self.finish_times[sp][eid] = finish_t
            self.entity_processor_history[eid][orig][-1]["end_time"] = finish_t
            self.busy_periods[sp].append((start_t, finish_t))

            # final delivery
            if orig == "Units_Delivered":
                if eid in self.entity_cycle:
                    self.entity_cycle[eid]["finish_h"] = self.env.now
                else:
                    self.entity_cycle[eid] = {"arrival_h": 0.0, "finish_h": self.env.now}

            # schedule successors
            self.schedule_successors(eid, orig)

    def schedule_successors(self, eid: int, proc: str):
        ent = self.entity_attributes[eid]
        path = ent["path"]
        for succ in self.G_all.successors(proc):
            if succ in self.paths_info[path]["processing_times"]:
                preds = list(self.G_all.predecessors(succ))
                all_done = True
                for pr in preds:
                    subps = self.processor_to_sub.get(pr, [pr])
                    done_here = any(eid in self.finish_times.get(sp, {}) for sp in subps)
                    if not done_here:
                        all_done = False
                        break
                if all_done:
                    self.env.process(self.process_at(eid, succ))

    # ------------------------------------------------------------------
    # analysis helpers
    # ------------------------------------------------------------------
    def get_idle_periods(self, busy_periods, total_time):
        idle = []
        prev = 0.0
        for s, f in sorted(busy_periods):
            if s > prev:
                idle.append((prev, s))
            prev = max(prev, f)
        if prev < total_time:
            idle.append((prev, total_time))
        return idle

    def calculate_idle_busy(self, total_time):
        idle_times = defaultdict(float)
        busy_times = defaultdict(float)
        for proc, periods in self.busy_periods.items():
            periods = sorted(periods)
            idle = self.get_idle_periods(periods, total_time)
            idle_times[proc] = sum(e - s for s, e in idle)
            busy_times[proc] = sum(e - s for s, e in periods)
        return idle_times, busy_times

    def intersect_intervals(self, a, b):
        out = []
        i = j = 0
        while i < len(a) and j < len(b):
            s1, e1 = a[i]
            s2, e2 = b[j]
            s = max(s1, s2)
            e = min(e1, e2)
            if s < e:
                out.append((s, e))
            if e1 < e2:
                i += 1
            else:
                j += 1
        return out

    def compute_joint_times(self, all_procs, k, total_time):
        joint = []
        procs = [p for p in all_procs.keys() if p not in self.exclude_processors]
        for size in range(1, k + 1):
            for combo in combinations(procs, size):
                for st_combo in product(["busy", "idle"], repeat=size):
                    pstate = dict(zip(combo, st_combo))
                    intervals = [(0, total_time)]
                    for p in combo:
                        bps = all_procs[p]["busy_periods"]
                        if pstate[p] == "busy":
                            periods = bps
                        else:
                            periods = self.get_idle_periods(bps, total_time)
                        intervals = self.intersect_intervals(intervals, periods)
                        if not intervals:
                            break
                    total_int = sum(e - s for s, e in intervals)
                    if total_int > 0:
                        joint.append({"processors": pstate, "total_time": total_int, "intervals": intervals})
        return joint

    def find_top_idle_combos(self, joint_times, k=2, top_n=10):
        joint_times = sorted(joint_times, key=lambda x: x["total_time"], reverse=True)
        return joint_times[:top_n]

    def run(self):
        self.generate_entities()
        self.env.run()

        total_time = 0.0
        for proc, times in self.finish_times.items():
            if times:
                total_time = max(total_time, max(times.values()))
        idle_times, busy_times = self.calculate_idle_busy(total_time)
        all_procs = {p: {"busy_periods": self.busy_periods[p]} for p in self.busy_periods}
        joint_times = self.compute_joint_times(all_procs, k=2, total_time=total_time)
        top_idle = self.find_top_idle_combos(joint_times, k=2, top_n=self.top_n)

        self.results = {
            "start_times": self.start_times,
            "finish_times": self.finish_times,
            "idle_times": idle_times,
            "busy_times": busy_times,
            "entity_processor_history": self.entity_processor_history,
            "paths_info": self.paths_info,
            "G_all": self.G_all,
            "busy_periods": self.busy_periods,
            "total_processing_time": total_time,
            "joint_times": joint_times,
            "top_idle_combinations": top_idle,
        }
        return self.results

    # -----------------------------------------------------------
    # reporting helpers — WC-based
    # -----------------------------------------------------------
    def build_workcenter_report(self):
        """
        Aggregate busy/idle/jobs per workcenter (WCxx)
        """
        if not self.results:
            return {}
        wc_busy = defaultdict(float)
        wc_idle = defaultdict(float)
        wc_jobs = defaultdict(int)

        idle = self.results["idle_times"]
        busy = self.results["busy_times"]

        # accumulate
        for proc, b in busy.items():
            wc = canonical_wc(proc)
            if not wc.startswith("WC"):
                continue
            wc_busy[wc] += b
            wc_idle[wc] += idle.get(proc, 0.0)

        # jobs: walk entity history
        eph = self.results["entity_processor_history"]
        for eid, hist in eph.items():
            for proc, visits in hist.items():
                wc = canonical_wc(proc)
                if not wc.startswith("WC"):
                    continue
                wc_jobs[wc] += len(visits)

        rep = {}
        for wc in wc_busy.keys():
            b = wc_busy[wc]
            i = wc_idle.get(wc, 0.0)
            total = b + i
            util = b / total if total > 0 else 0.0
            rep[wc] = {
                "busy_hours": b,
                "idle_hours": i,
                "utilization": util,
                "jobs": wc_jobs.get(wc, 0),
            }
        return rep

    def throughput_by_month(self, calendar: WorkingCalendar):
        eph = self.results["entity_processor_history"]
        counts = defaultdict(int)
        for eid, ph in eph.items():
            if "Units_Delivered" in ph and ph["Units_Delivered"]:
                ctime = ph["Units_Delivered"][-1]["end_time"]
                days = ctime / calendar.hpd
                d = calendar.start + dt.timedelta(days=days)
                counts[(d.year, d.month)] += 1
        return dict(counts)

    def find_bottlenecks(self, top=10):
        rep = self.build_workcenter_report()
        return sorted(rep.items(), key=lambda kv: kv[1]["utilization"], reverse=True)[:top]

    def plot_processor_schedule_clean(self):
        """Simple schedule figure from sim results."""
        if not self.results:
            print("run() first")
            return

        eph = self.results["entity_processor_history"]
        procs = set()
        for eid in eph:
            procs.update(eph[eid].keys())
        procs = [p for p in procs if p not in ("Start_Job", "Units_Delivered")]
        if not procs:
            print("no processor history to plot")
            return

        proc_start = {}
        for p in procs:
            earliest = float("inf")
            for eid in eph:
                if p in eph[eid]:
                    for v in eph[eid][p]:
                        s = v["start_time"]
                        earliest = min(earliest, s)
            proc_start[p] = earliest if earliest < float("inf") else 0.0
        procs = sorted(procs, key=lambda x: proc_start.get(x, 0.0))

        fig, ax = plt.subplots(figsize=(14, max(4, len(procs) * 0.4)))

        yticks = []
        ylabels = []
        y = 0

        for idx, proc in enumerate(procs):
            if idx % 2 == 0:
                ax.axhspan(y - 0.45, y + 0.45, facecolor="0.95", zorder=0)

            visits = []
            for eid in eph:
                if proc in eph[eid]:
                    for v in eph[eid][proc]:
                        s = v["start_time"]
                        f = v.get("end_time", s)
                        visits.append((s, f))
            visits.sort(key=lambda t: t[0])

            for (s, f) in visits:
                ax.broken_barh([(s, f - s)], (y - 0.3, 0.6), facecolors="tab:blue", edgecolors="none")

            yticks.append(y)
            ylabels.append(proc)
            y += 1

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("time (hours)")
        ax.set_title("Processor Schedule (clean)")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()


# ============================================================
# SIM-SIDE ANALYSIS FUNCTIONS (THROUGHPUT, UTIL, ETC.)
# ============================================================
def build_time_series(results, calendar: WorkingCalendar, hpd: float = None):
    if hpd is None:
        hpd = calendar.hpd

    eph = results["entity_processor_history"]
    deliveries = []
    for eid, ph in eph.items():
        if "Units_Delivered" in ph and ph["Units_Delivered"]:
            end_h = ph["Units_Delivered"][-1]["end_time"]
            days = end_h / hpd
            cal_date = calendar.start + dt.timedelta(days=days)
            deliveries.append({"entity_id": eid, "date": cal_date.date()})
    deliveries_df = pd.DataFrame(deliveries)
    if deliveries_df.empty:
        return deliveries_df, pd.DataFrame(), pd.DataFrame(), {}

    deliveries_df["date_ts"] = pd.to_datetime(deliveries_df["date"])
    monthly_df = deliveries_df.groupby(deliveries_df["date_ts"].dt.to_period("M")).size().reset_index(name="delivered")
    monthly_df["month_ts"] = monthly_df["date_ts"].dt.to_timestamp()
    weekly_df = deliveries_df.groupby(deliveries_df["date_ts"].dt.to_period("W-MON")).size().reset_index(name="delivered")
    weekly_df["week_ts"] = weekly_df["date_ts"].dt.start_time

    util_weekly_wc: Dict[str, pd.DataFrame] = {}
    total_time = results["total_processing_time"]
    last_day = calendar.start + dt.timedelta(days=total_time / hpd)
    week_starts = []
    cur = calendar.start
    while cur <= last_day:
        week_starts.append(cur)
        cur += dt.timedelta(days=7)

    wc_periods = defaultdict(list)
    for proc, periods in results["busy_periods"].items():
        wc = canonical_wc(proc)
        if not wc.startswith("WC"):
            continue
        wc_periods[wc].extend(periods)

    for wc, periods in wc_periods.items():
        rows = []
        for ws in week_starts:
            we = ws + dt.timedelta(days=7)
            ws_h = calendar.to_hours(ws)
            we_h = calendar.to_hours(we)
            busy = 0.0
            for (s, f) in periods:
                overlap = max(0.0, min(f, we_h) - max(s, ws_h))
                busy += overlap
            span = we_h - ws_h
            rows.append({
                "week_start": ws,
                "busy_hours": busy,
                "util": busy / span if span > 0 else 0.0
            })
        util_weekly_wc[wc] = pd.DataFrame(rows)

    return deliveries_df, monthly_df, weekly_df, util_weekly_wc


def build_hist_monthly_from_arrivals(arrivals: List[Dict[str, Any]]):
    if not arrivals:
        return pd.DataFrame()
    rows = []
    for a in arrivals:
        if a["source"] == "history":
            rows.append({"date": a["when"].date(), "qty": a.get("qty", 1)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date_ts"] = pd.to_datetime(df["date"])
    hist_monthly_df = df.groupby(df["date_ts"].dt.to_period("M"))["qty"].sum().reset_index(name="hist")
    hist_monthly_df["month_ts"] = hist_monthly_df["date_ts"].dt.to_timestamp()
    return hist_monthly_df


def plot_throughput_monthly(monthly_df, hist_monthly_df=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(monthly_df["month_ts"], monthly_df["delivered"], where="mid", label="Simulated")
    if hist_monthly_df is not None and not hist_monthly_df.empty:
        ax.step(hist_monthly_df["month_ts"], hist_monthly_df["hist"], where="mid", label="Historical")
    ax.set_title("Monthly Throughput (Historical vs Simulated)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Units")
    ax.legend()
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def analyze_cycle_times(sim: ProcessSimulation, calendar: WorkingCalendar):
    rows = []
    for eid, rec in sim.entity_cycle.items():
        arr_h = rec.get("arrival_h")
        fin_h = rec.get("finish_h")
        if arr_h is not None and fin_h is not None:
            rows.append(fin_h - arr_h)
    if not rows:
        print("No cycle times recorded.")
        return
    ct = pd.Series(rows)
    print("Cycle time (h): mean={:.2f}, p50={:.2f}, p90={:.2f}, p95={:.2f}, max={:.2f}".format(
        ct.mean(), ct.quantile(0.5), ct.quantile(0.9), ct.quantile(0.95), ct.max()
    ))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ct, bins=30)
    ax.set_title("Cycle Time Distribution (hours)")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()


def analyze_queues(sim: ProcessSimulation):
    stats_rows = []
    for proc, waits in sim.wait_times.items():
        if not waits:
            continue
        arr = np.array(waits)
        stats_rows.append({
            "processor": proc,
            "mean_wait_h": arr.mean(),
            "p90_wait_h": np.percentile(arr, 90),
            "n_waits": len(arr),
        })
    dfw = pd.DataFrame(stats_rows).sort_values("mean_wait_h", ascending=False)
    print("\n=== Queue / Wait Analysis (top 15) ===")
    if not dfw.empty:
        print(dfw.head(15).to_string(index=False))
    else:
        print("No waits recorded.")

    if not dfw.empty:
        worst = dfw.iloc[0]["processor"]
        snaps = sim.queue_snapshots.get(worst, [])
        if snaps:
            t = [s[0] for s in snaps]
            q = [s[1] for s in snaps]
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.step(t, q, where="post")
            ax.set_title(f"Queue length over time – {worst}")
            ax.set_xlabel("Sim hours")
            ax.set_ylabel("Queue length")
            plt.tight_layout()
            plt.show()


def build_system_wip(sim: ProcessSimulation):
    times = set()
    for _, rec in sim.entity_cycle.items():
        if "arrival_h" in rec:
            times.add(rec["arrival_h"])
        if "finish_h" in rec:
            times.add(rec["finish_h"])
    times = sorted(times)
    rows = []
    for t in times:
        wip = 0
        for _, rec in sim.entity_cycle.items():
            arr = rec.get("arrival_h", 1e9)
            fin = rec.get("finish_h", 1e9)
            if arr <= t < fin:
                wip += 1
        rows.append({"time_h": t, "wip": wip})
    df = pd.DataFrame(rows)
    if not df.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.step(df["time_h"], df["wip"], where="post")
        ax.set_title("System WIP over time")
        ax.set_xlabel("Sim hours")
        ax.set_ylabel("WIP")
        plt.tight_layout()
        plt.show()
    return df


def plot_utilization_vs_availability(results, calendar: WorkingCalendar):
    idle = results["idle_times"]
    busy = results["busy_times"]
    wc_busy = defaultdict(float)
    wc_idle = defaultdict(float)

    for proc, b in busy.items():
        wc = canonical_wc(proc)
        if not wc.startswith("WC"):
            continue
        wc_busy[wc] += b
        wc_idle[wc] += idle.get(proc, 0.0)

    data = []
    for wc in wc_busy:
        b = wc_busy[wc]
        i = wc_idle.get(wc, 0.0)
        total = b + i
        if total <= 0:
            continue
        util = b / total
        data.append((wc, util, b, total))

    df = pd.DataFrame(data, columns=["wc", "util", "busy_h", "avail_h"]).sort_values("util", ascending=False)
    top = df.head(15)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["wc"], top["util"])
    ax.invert_yaxis()
    ax.set_xlabel("Utilization")
    ax.set_title("Top 15 workcenters by utilization")
    plt.tight_layout()
    plt.show()


def summarize_scenarios(base_results, scenario_results, calendar: WorkingCalendar):
    rows = []
    base_del, _, _, _ = build_time_series(base_results, calendar)
    rows.append({
        "scenario": "BASE",
        "units": len(base_del),
        "makespan_h": base_results["total_processing_time"],
    })
    for sname, sres in scenario_results.items():
        s_del, _, _, _ = build_time_series(sres, calendar)
        rows.append({
            "scenario": sname,
            "units": len(s_del),
            "makespan_h": sres["total_processing_time"],
        })
    df = pd.DataFrame(rows)
    print("\n=== Scenario comparison ===")
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(df["scenario"], df["units"])
    ax.set_title("Units delivered per scenario")
    ax.set_ylabel("Units")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(df["scenario"], df["makespan_h"])
    ax.set_title("Makespan (hours) per scenario")
    ax.set_ylabel("Hours")
    plt.tight_layout()
    plt.show()


def plot_wc_util_over_time(util_weekly_wc: Dict[str, pd.DataFrame], max_wcs: int = 5):
    if not util_weekly_wc:
        return
    shown = 0
    for wc, df in util_weekly_wc.items():
        if shown >= max_wcs:
            break
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.step(df["week_start"], df["util"], where="post")
        ax.set_title(f"Weekly utilization – {wc}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Utilization")
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
        shown += 1


# ============================================================
# SCENARIOS / WHAT-IF (SIM)
# ============================================================
def apply_capacity_deltas(paths_info, deltas: Dict[str, int]):
    for _, pdata in paths_info.items():
        for node, info in pdata["processing_times"].items():
            if node in deltas:
                base_cap = int(info.get("capacity", 1))
                info["capacity"] = max(1, base_cap + int(deltas[node]))


def run_capacity_scenarios(base_paths_info, entity_specs, calendar, scenarios, exclude_processors=None, duration_scalars=None):
    unit_arrival_pattern = specs_to_arrival_pattern(entity_specs)
    total_entities = int(sum(r["num_entities"] for r in unit_arrival_pattern))
    results = {}
    for name, deltas in scenarios.items():
        paths_copy = {}
        for path, pdata in base_paths_info.items():
            pt = {n: dict(info) for n, info in pdata["processing_times"].items()}
            G = nx.DiGraph(pdata["graph"])
            paths_copy[path] = {"processing_times": pt, "graph": G}

        apply_capacity_deltas(paths_copy, deltas)

        sim = ProcessSimulation(
            initial_entities=[],
            unit_arrival_pattern=unit_arrival_pattern,
            paths_info=paths_copy,
            exclude_processors=exclude_processors or ["Start_Job", "Units_Delivered"],
            n_entities=total_entities,
            top_n=10,
            duration_scalars=duration_scalars,
        )
        res = sim.run()
        results[name] = res
        logger.info(f"Scenario {name}: horizon={res['total_processing_time']:.2f}h")
    return results


# ============================================================
# NEW: DURATION COMPARISON HELPERS (this is the glue you wanted)
# ============================================================
def build_historical_wc_durations_from_stages(stages_df: pd.DataFrame) -> dict[str, list[float]]:
    """
    Take the long historical stages_df (with StageName, Start, End, DurationHours)
    and roll it up to WC-level durations, using canonical_wc(stage_name).
    """
    if stages_df is None or stages_df.empty:
        return {}

    if "DurationHours" not in stages_df.columns:
        stages_df = stages_df.copy()
        stages_df["DurationHours"] = (
            pd.to_datetime(stages_df["End"], errors="coerce")
            - pd.to_datetime(stages_df["Start"], errors="coerce")
        ).dt.total_seconds() / 3600.0

    wc_buckets: dict[str, list[float]] = {}
    for _, r in stages_df.iterrows():
        stg = r.get("StageName")
        dur = r.get("DurationHours")
        if pd.isna(stg) or pd.isna(dur):
            continue
        wc = canonical_wc(str(stg))
        if not wc.upper().startswith("WC"):
            wc = str(stg)
        wc_buckets.setdefault(wc, []).append(float(dur))

    for k in list(wc_buckets.keys()):
        vals = [v for v in wc_buckets[k] if np.isfinite(v) and v > 0]
        if vals:
            wc_buckets[k] = vals
        else:
            del wc_buckets[k]
    return wc_buckets


def build_simulated_wc_durations_from_sim(sim: ProcessSimulation) -> dict[str, list[float]]:
    """
    Use what the DES actually ran.
    """
    wc_buckets: dict[str, list[float]] = {}

    # 1) the dedicated log
    for proc, items in getattr(sim, "proc_durations_log", {}).items():
        wc = canonical_wc(proc)
        if not wc.upper().startswith("WC"):
            wc = proc
        for _, dur in items:
            if dur is None:
                continue
            if np.isfinite(dur) and dur > 0:
                wc_buckets.setdefault(wc, []).append(float(dur))

    # 2) fallback from visit history
    eph = getattr(sim, "entity_processor_history", {})
    for eid, procs in eph.items():
        for proc, visits in procs.items():
            wc = canonical_wc(proc)
            if not wc.upper().startswith("WC"):
                wc = proc
            for v in visits:
                s = v.get("start_time")
                e = v.get("end_time")
                if s is None or e is None:
                    continue
                dur = float(e) - float(s)
                if dur > 0:
                    wc_buckets.setdefault(wc, []).append(dur)

    for k in list(wc_buckets.keys()):
        vals = [v for v in wc_buckets[k] if np.isfinite(v) and v > 0]
        if vals:
            wc_buckets[k] = vals
        else:
            del wc_buckets[k]
    return wc_buckets


def compare_historical_vs_simulated_durations(
    hist_wc_durs: dict[str, list[float]],
    sim_wc_durs: dict[str, list[float]],
    out_dir: str = "analysis/duration_compare",
    make_plots: bool = False,
) -> pd.DataFrame:
    """
    Line up WC-by-WC (or stage-by-stage) and make a tidy comparison table.
    Saves CSV; optional plots.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    all_keys = set(hist_wc_durs.keys()) | set(sim_wc_durs.keys())
    for wc in sorted(all_keys):
        h = np.array(hist_wc_durs.get(wc, []), float)
        s = np.array(sim_wc_durs.get(wc, []), float)

        def _stats(arr):
            if arr.size == 0:
                return {
                    "n": 0,
                    "mean": np.nan,
                    "p50": np.nan,
                    "p90": np.nan,
                    "p95": np.nan,
                }
            return {
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p95": float(np.percentile(arr, 95)),
            }

        hs = _stats(h)
        ss = _stats(s)
        rows.append(
            {
                "WC_or_Stage": wc,
                "hist_n": hs["n"],
                "hist_mean_h": hs["mean"],
                "hist_p50_h": hs["p50"],
                "hist_p90_h": hs["p90"],
                "hist_p95_h": hs["p95"],
                "sim_n": ss["n"],
                "sim_mean_h": ss["mean"],
                "sim_p50_h": ss["p50"],
                "sim_p90_h": ss["p90"],
                "sim_p95_h": ss["p95"],
            }
        )

        if make_plots and hs["n"] > 0 and ss["n"] > 0:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(h, bins=30, alpha=0.5, label="historical")
            ax.hist(s, bins=30, alpha=0.5, label="simulated")
            ax.set_title(f"Duration dist – {wc}")
            ax.set_xlabel("hours")
            ax.set_ylabel("count")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"dur_{wc.replace('/', '_')}.png"), dpi=130)
            plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "duration_compare.csv"), index=False)
    return df


# ============================================================
# COMPARISON: SIM VS HISTORY – simple monthly counts
# ============================================================
def compare_sim_vs_history(results, historical_arrivals, calendar: WorkingCalendar):
    hist_counts = defaultdict(int)
    for a in historical_arrivals:
        d = a["when"].date()
        hist_counts[(d.year, d.month)] += a.get("qty", 1)

    eph = results["entity_processor_history"]
    sim_counts = defaultdict(int)
    for eid, ph in eph.items():
        if "Units_Delivered" in ph and ph["Units_Delivered"]:
            ctime = ph["Units_Delivered"][-1]["end_time"]
            days = ctime / calendar.hpd
            d = calendar.start + dt.timedelta(days=days)
            d = d.date()
            sim_counts[(d.year, d.month)] += 1

    print("\n=== Historical vs Simulated deliveries (by month) ===")
    months = sorted(set(list(hist_counts.keys()) + list(sim_counts.keys())))
    for ym in months:
        hy = hist_counts.get(ym, 0)
        sy = sim_counts.get(ym, 0)
        print(f"{ym[0]}-{ym[1]:02d}: hist={hy} sim={sy}")


# ============================================================
# SYSTEM MODEL (MAIN SIM PIPELINE)
# ============================================================
def system_model():
    today = dt.date.today()

    # 1. load CSVs
    product_df = _read_csv_relaxed("product.csv")
    resource_df = _read_csv_relaxed("resource.csv")
    schedule_df = load_schedule_as_df("schedule.csv")
    try:
        parts_df = _read_csv_relaxed("parts.csv")
    except FileNotFoundError:
        parts_df = None
    try:
        pto_df = _read_csv_relaxed("pto.csv")
    except FileNotFoundError:
        pto_df = None
    jobs_df = load_jobs_as_df("jobs.csv")

    # duration tuning
    duration_scalars = load_duration_tuning()
    if duration_scalars:
        logger.info("Loaded duration tuning for %d processors", len(duration_scalars))

    # 2. capacities
    caps = load_resource_caps(resource_df)

    # 3. routes
    paths_info = build_routes_from_product_primary(product_df, caps)

    # 4. parts map
    parts_map = load_parts_map(parts_df)

    # 5. arrivals
    arrivals_hist = []
    if jobs_df is not None and not jobs_df.empty:
        arrivals_hist = jobs_to_arrivals(jobs_df, parts_map, today)
    arrivals_fut = []
    if schedule_df is not None and not schedule_df.empty:
        arrivals_fut = schedule_to_arrivals(schedule_df, jobs_df, parts_map, today)

    all_arrivals = arrivals_hist + arrivals_fut
    if not all_arrivals:
        raise ValueError("No arrivals from jobs.csv or schedule.csv — cannot run sim.")

    # 6. calendar
    start_date = min(a["when"].date() for a in all_arrivals)
    holidays = load_pto_days(pto_df) if pto_df is not None else []
    calendar = WorkingCalendar(start_date, REMOVE_WEEKENDS_DEFAULT, holidays, HOURS_PER_DAY_DEFAULT)

    # 7. ensure path exists for all arrivals
    for a in all_arrivals:
        pname = f"Path:{a['part']}"
        if pname not in paths_info:
            G = nx.DiGraph()
            G.add_node("Start_Job")
            G.add_node("Units_Delivered")
            G.add_edge("Start_Job", "Units_Delivered")
            paths_info[pname] = {
                "processing_times": {
                    "Start_Job": {"processing_time": 0.0, "capacity": 1, "stochastic": False},
                    "Units_Delivered": {"processing_time": 0.0, "capacity": 1, "stochastic": False},
                },
                "graph": G,
            }

    # 8. specs
    entity_specs = build_entity_specs(all_arrivals, calendar, default_priority=10)
    unit_arrival_pattern = specs_to_arrival_pattern(entity_specs)
    total_entities = int(sum(r["num_entities"] for r in unit_arrival_pattern))

    # 9. run base sim
    sim = ProcessSimulation(
        initial_entities=[],
        unit_arrival_pattern=unit_arrival_pattern,
        paths_info=paths_info,
        exclude_processors=["Start_Job", "Units_Delivered"],
        n_entities=total_entities,
        top_n=10,
        duration_scalars=duration_scalars,
    )
    results = sim.run()

    # 10. compare to history
    if arrivals_hist:
        compare_sim_vs_history(results, arrivals_hist, calendar)

    # workcenter report + bottlenecks
    wc_report = sim.build_workcenter_report()
    bottlenecks = sim.find_bottlenecks(top=10)
    monthly_tp = sim.throughput_by_month(calendar)

    print("\nTop 10 busiest WORKCENTERS (bottleneck-style):")
    for wc, info in bottlenecks:
        print(f"{wc:12s} util={info['utilization']:.2f} busy={info['busy_hours']:.1f}h jobs={info['jobs']}")

    print("\nThroughput by month (simulated):")
    for (y, m), cnt in sorted(monthly_tp.items()):
        print(f"{y}-{m:02d}: {cnt}")

    # 11. capacity scenarios
    scenario_defs = {
        "CIC+1": {"WC10_15_21a_CICRwkTbl_19": +1},
        "BAKEOUT+1": {"WC17E02_BakeOutOven_27": +1},
        "BAKEOUT-1": {"WC17E02_BakeOutOven_27": -1},
    }
    scenario_results = run_capacity_scenarios(
        paths_info,
        entity_specs,
        calendar,
        scenario_defs,
        exclude_processors=["Start_Job", "Units_Delivered"],
        duration_scalars=duration_scalars,
    )

    # 12. summary
    horizon = results["total_processing_time"]
    idle_sum = float(sum(results["idle_times"].values()))
    busy_sum = float(sum(results["busy_times"].values()))
    print("\n=== DES SUMMARY (product-driven) ===")
    print(f"Calendar start date : {calendar.start}")
    print(f"Entities simulated  : {total_entities}")
    print(f"Processors (nodes)  : {len(results['G_all'].nodes())}")
    print(f"Edges (precedence)  : {len(results['G_all'].edges())}")
    print(f"Horizon (hours)     : {horizon:.2f}")
    print(f"Total busy hours    : {busy_sum:.2f}")
    print(f"Total idle hours    : {idle_sum:.2f}")

    print("\nScenario horizon comparison:")
    for sname, sres in scenario_results.items():
        print(f"  {sname}: {sres['total_processing_time']:.2f}h")

    # 13. advanced analysis / plots
    deliveries_df, monthly_df, weekly_df, wc_util_weekly = build_time_series(results, calendar)
    hist_monthly_df = build_hist_monthly_from_arrivals(arrivals_hist)
    if not monthly_df.empty:
        plot_throughput_monthly(monthly_df, hist_monthly_df)
    analyze_cycle_times(sim, calendar)
    analyze_queues(sim)
    build_system_wip(sim)
    plot_utilization_vs_availability(results, calendar)
    summarize_scenarios(results, scenario_results, calendar)
    plot_wc_util_over_time(wc_util_weekly, max_wcs=5)

    # 14. clean scheduling figure
    try:
        sim.plot_processor_schedule_clean()
    except Exception as e:
        print(f"Plotting skipped: {e}")

    # 15. artifacts / exports that DES uses
    os.makedirs("artifacts", exist_ok=True)

    def serialize_paths_info_for_json(paths_info: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        out = {}
        for path, pdata in paths_info.items():
            G = pdata["graph"]
            pt = pdata["processing_times"]
            out[path] = {
                "edges": list(map(list, G.edges())),
                "nodes": {n: pt.get(n, {}) for n in G.nodes()},
            }
        return out

    with open("artifacts/paths_info.json", "w") as f:
        json.dump(serialize_paths_info_for_json(paths_info), f, indent=2)
    pd.DataFrame(entity_specs).to_csv("artifacts/entity_specs.csv", index=False)
    pd.DataFrame(unit_arrival_pattern).to_csv("artifacts/unit_arrival_pattern.csv", index=False)

    # 16. *** NEW PART *** – duration comparison
    # try to use existing historical stages if available; otherwise rebuild
    stages_df = None
    hist_csv_path = os.path.join("analysis", "csv", "stages_long_safeorder.csv")
    if os.path.exists(hist_csv_path):
        try:
            stages_df = pd.read_csv(hist_csv_path, parse_dates=["Start", "End"])
        except Exception:
            stages_df = None

    if stages_df is None:
        # try to rebuild from jobs.csv using the historical pipeline
        if os.path.exists("jobs.csv"):
            try:
                stages_df = produce_all_outputs(
                    input_file="jobs.csv",
                    sheet=0,
                    product_col="Type",                  # adjust if your column name differs
                    first_end_col="Frame",
                    last_end_col="Cure Times.ShipCureDate",
                    out_prefix="analysis",
                    use_now_for_null=True,
                )
            except Exception as e:
                print("could not rebuild historical stages_df for duration compare:", e)

    if stages_df is not None:
        hist_wc = build_historical_wc_durations_from_stages(stages_df)
        sim_wc = build_simulated_wc_durations_from_sim(sim)
        df_compare = compare_historical_vs_simulated_durations(
            hist_wc,
            sim_wc,
            out_dir="analysis/duration_compare",
            make_plots=False,
        )
        print("\n=== saved historical vs simulated duration compare → analysis/duration_compare/duration_compare.csv ===")
        print(df_compare.head(25).to_string(index=False))
    else:
        print("\n(no historical stages_df available, skipped duration compare)")

    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # run the unified model
    system_model()
