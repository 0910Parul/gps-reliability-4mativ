"""
4MATIV — GPS Reliability & Route Execution Dashboard
=====================================================
Three-page interactive Dash dashboard powered by Plotly.

Usage:
    python gps_dashboard.py
Then open: http://127.0.0.1:8050

Requirements:
    pip install dash plotly pandas numpy scikit-learn

Data files expected (same folder as this script, or update DATA_DIR below):
    trips.csv, trip_positions.csv, vendors.csv, stop_locations.csv,
    route_stops.csv, geofencing_config.json
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 0 — IMPORTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output

warnings.filterwarnings("ignore")

# ── Data path ──────────────────────────────────────────────────────────────────
DATA_DIR = Path(".")          # ← update if CSVs are in a different folder

# ── Layer 1: Temporal Coverage ────────────────────────────────────────────────
COVERAGE_HARD_GATE  = 0.20
GAP_MULTIPLIER      = 3

# ── Layer 2: Geofence Termination ─────────────────────────────────────────────
GPS_FAIL_TRIGGERS = {"no_gps_data", "old_last_point"}

# ── Layer 3: Spatial Consistency ──────────────────────────────────────────────
FREEZE_DURATION_SEC = 120
STOP_DWELL_RADIUS_M = 80
FREEZE_TRIP_RATIO   = 0.10
SPEED_THRESHOLD_MPH = 80

# ── Layer 4: Cross-Trip Consistency ───────────────────────────────────────────
PEER_MIN       = 5
IQR_MULTIPLIER = 1.5

# ── GPS Reliability score bands ───────────────────────────────────────────────
BAND_L1 = (0,  45)
BAND_L2 = (45, 55)
BAND_L3 = (55, 75)
BAND_L4 = (75, 90)
BAND_OK = (90, 100)

# ── Execution Performance ─────────────────────────────────────────────────────
ONTIME_THRESHOLD_MIN = 10
W_COMPLETION         = 0.40
W_ONTIME             = 0.40
W_SEQUENCE           = 0.20

# ── Color palette — white/light theme matching 4MATIV slides ──────────────────
TIER_COLORS = {
    "high":         "#27AE60",   # green
    "medium":       "#8BC34A",   # lime green
    "low":          "#F39C12",   # amber
    "insufficient": "#E67E22",   # orange
    "no_gps":       "#E74C3C",   # red
}
TIER_ORDER  = ["no_gps", "insufficient", "low", "medium", "high"]
TIER_LABELS = {"high":"High", "medium":"Medium", "low":"Low",
               "insufficient":"Insufficient", "no_gps":"No GPS"}

# 4MATIV brand colors (extracted from logo image)
BRAND_BLUE   = "#3D9BD4"
BRAND_ORANGE = "#F5A623"
BRAND_TEAL   = "#4ECDC4"

# Page / chart colors — white theme
BG      = "#FFFFFF"
SURFACE = "#F8F9FA"
BORDER  = "#E2E6EA"
TEXT    = "#1C2331"
MUTED   = "#6C757D"
ACCENT  = BRAND_ORANGE

WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ── 4MATIV SVG logo (inline, no external file needed) ─────────────────────────
# Recreates the logo: blue left square + orange right square with white lightning bolt + bold text
LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 170 48" width="136" height="38">
  <rect x="0"  y="2" width="40" height="40" rx="6" fill="#3D9BD4"/>
  <rect x="20" y="2" width="40" height="40" rx="6" fill="#F5A623"/>
  <polygon points="30,6 18,26 29,26 15,44 42,22 30,22" fill="white"/>
  <text x="68" y="33" font-family="Arial Black,sans-serif" font-size="21"
        font-weight="900" fill="#1C2331" letter-spacing="-0.3">4MATIV</text>
</svg>"""


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading CSV files …")
    trips          = pd.read_csv(DATA_DIR / "trips.csv")
    trip_positions = pd.read_csv(DATA_DIR / "trip_positions.csv")
    vendors        = pd.read_csv(DATA_DIR / "vendors.csv")
    stop_locations = pd.read_csv(DATA_DIR / "stop_locations.csv")
    route_stops    = pd.read_csv(DATA_DIR / "route_stops.csv")

    with open(DATA_DIR / "geofencing_config.json") as f:
        geo_cfg = json.load(f)
    complete_threshold_min = geo_cfg["complete_threshold_minutes"]

    for col in ["started_at", "ended_at", "pickup_time", "dropoff_time", "date"]:
        if col in trips.columns:
            trips[col] = pd.to_datetime(trips[col], errors="coerce", utc=True)

    trip_positions["timestamp"] = pd.to_numeric(trip_positions["timestamp"], errors="coerce")

    trips = trips.merge(
        vendors[["vendor_id", "vendor_label", "polling_interval_s"]].drop_duplicates(),
        on="vendor_id", how="left"
    )
    trips["polling_interval_s"] = trips["polling_interval_s"].fillna(30)

    print(f"  trips          : {len(trips):,} rows")
    print(f"  trip_positions : {len(trip_positions):,} rows")
    print(f"  vendors        : {len(vendors):,} rows")
    print(f"  stop_locations : {len(stop_locations):,} rows")
    print(f"  route_stops    : {len(route_stops):,} rows")
    return trips, trip_positions, vendors, stop_locations, route_stops, complete_threshold_min


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — PING-LEVEL FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1;  dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_ping_features(trip_positions, trips, stop_locations):
    print("Building ping-level features …")
    from sklearn.neighbors import BallTree

    tp = trip_positions.sort_values(["trip_id", "timestamp"]).copy()
    tp["prev_ts"] = tp.groupby("trip_id")["timestamp"].shift(1)
    tp["gap_s"]   = tp["timestamp"] - tp["prev_ts"]

    trip_polling = trips[["trip_id","polling_interval_s"]].drop_duplicates("trip_id")
    tp = tp.merge(trip_polling, on="trip_id", how="left")
    tp["polling_interval_s"] = tp["polling_interval_s"].fillna(30)
    tp["gap_is_large"] = (tp["gap_s"] > GAP_MULTIPLIER * tp["polling_interval_s"]).fillna(False)

    tp["prev_lat"] = tp.groupby("trip_id")["lat"].shift(1)
    tp["prev_lng"] = tp.groupby("trip_id")["lng"].shift(1)
    valid_move = (tp["prev_lat"].notna() & tp["prev_lng"].notna() &
                  tp["gap_s"].notna() & (tp["gap_s"] > 0))
    tp["dist_m"] = np.nan;  tp["speed_mps"] = np.nan
    tp.loc[valid_move, "dist_m"]    = haversine_m(
        tp.loc[valid_move,"prev_lat"], tp.loc[valid_move,"prev_lng"],
        tp.loc[valid_move,"lat"],      tp.loc[valid_move,"lng"])
    tp.loc[valid_move, "speed_mps"] = tp.loc[valid_move,"dist_m"] / tp.loc[valid_move,"gap_s"]
    tp["point_impossible_jump"] = (tp["speed_mps"] > SPEED_THRESHOLD_MPH/2.237).fillna(False).astype(int)

    stops_rad = np.radians(stop_locations[["lat","lng"]].values)
    pings_rad = np.radians(tp[["lat","lng"]].values)
    tree = BallTree(stops_rad, metric="haversine")
    dist_rad, _ = tree.query(pings_rad, k=1)
    tp["near_stop"] = (dist_rad[:,0] * 6_371_000 < STOP_DWELL_RADIUS_M).astype(int)

    tp["is_coord_identical"] = (
        (tp["lat"] == tp["prev_lat"]) & (tp["lng"] == tp["prev_lng"])
    ).fillna(False)
    tp["freeze_episode"] = (
        tp["is_coord_identical"] != tp.groupby("trip_id")["is_coord_identical"].shift(1)
    ).fillna(True).groupby(tp["trip_id"]).cumsum()

    freeze_dur = (
        tp[tp["is_coord_identical"]]
        .groupby(["trip_id","freeze_episode"])["timestamp"]
        .agg(freeze_start="min", freeze_end="max").reset_index()
    )
    freeze_dur["freeze_duration_s"] = freeze_dur["freeze_end"] - freeze_dur["freeze_start"]
    tp = tp.merge(freeze_dur[["trip_id","freeze_episode","freeze_duration_s"]],
                  on=["trip_id","freeze_episode"], how="left")
    tp["freeze_duration_s"] = tp["freeze_duration_s"].fillna(0)
    tp["point_frozen"] = (
        tp["is_coord_identical"] &
        (tp["freeze_duration_s"] > FREEZE_DURATION_SEC) &
        (tp["near_stop"] == 0)
    ).astype(int)
    tp.drop(columns=["is_coord_identical","freeze_episode","freeze_duration_s"], inplace=True)
    print(f"  Large gap pings: {tp['gap_is_large'].sum():,}  |  "
          f"Impossible jumps: {tp['point_impossible_jump'].sum():,}  |  "
          f"Frozen pings: {tp['point_frozen'].sum():,}")
    return tp


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — AGGREGATE PINGS TO TRIP LEVEL
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_trip_features(trips, trip_positions):
    print("Aggregating to trip level …")
    trip_metrics = trip_positions.groupby("trip_id").agg(
        gps_points       = ("timestamp",             "size"),
        max_gap_s        = ("gap_s",                 "max"),
        large_gap_count  = ("gap_is_large",          "sum"),
        frozen_pings     = ("point_frozen",          "sum"),
        impossible_jumps = ("point_impossible_jump", "sum"),
    ).reset_index()

    tf = trips.copy()
    tf = tf.merge(trip_metrics, on="trip_id", how="left")
    for c in ["gps_points","max_gap_s","large_gap_count","frozen_pings","impossible_jumps"]:
        tf[c] = tf[c].fillna(0)

    gps_dur = (tf["ended_at"] - tf["started_at"]).dt.total_seconds()
    tf["valid_duration"]  = gps_dur.notna() & (gps_dur > 0)
    tf["trip_duration_s"] = gps_dur.where(tf["valid_duration"])
    tf["expected_pings"]  = (tf["trip_duration_s"] / tf["polling_interval_s"]).round(0)
    tf.loc[~tf["valid_duration"], "expected_pings"] = np.nan

    tf["coverage_rate"] = (
        tf["trip_positions_count"] / tf["expected_pings"].replace(0, np.nan)
    ).clip(upper=1.0)
    tf.loc[~tf["valid_duration"], "coverage_rate"] = np.nan
    tf["frozen_ratio"] = tf["frozen_pings"] / tf["gps_points"].replace(0, np.nan)
    return tf


# ═══════════════════════════════════════════════════════════════════════════════
# 4 — PARSE GEOFENCE STOP DATA
# ═══════════════════════════════════════════════════════════════════════════════

def parse_geofence_stops(trips, complete_threshold_min):
    print("Parsing route_stops_info …")

    def parse_rsi(rsi_str):
        try:
            info = json.loads(rsi_str) if rsi_str and rsi_str != "{}" else {}
        except Exception:
            info = {}
        empty = {"planned_stops":0,"completed_stops":0,"stop_completion_rate":np.nan,
                 "completed_diffs":[],"completed_times":[],"floor_hit_count":0}
        if not info: return empty
        planned   = len(info)
        completed = sum(1 for s in info.values() if s.get("completed") is True)
        diffs = [(k, s["completed_diff"], s.get("completed_time"))
                 for k, s in info.items()
                 if s.get("completed") is True and s.get("completed_diff") is not None]
        return {
            "planned_stops":        planned,
            "completed_stops":      completed,
            "stop_completion_rate": completed/planned if planned > 0 else np.nan,
            "completed_diffs":      [d for _,d,_ in diffs],
            "completed_times":      [(k,t) for k,_,t in diffs if t is not None],
            "floor_hit_count":      sum(1 for _,d,_ in diffs if d == complete_threshold_min),
        }

    geo_df = pd.DataFrame(trips["route_stops_info"].apply(parse_rsi).tolist())
    geo_df["trip_id"] = trips["trip_id"].values
    return geo_df


# ═══════════════════════════════════════════════════════════════════════════════
# 5 — CROSS-TRIP CONSISTENCY (Layer 4 prep)
# ═══════════════════════════════════════════════════════════════════════════════

def build_layer4(trip_features):
    print("Computing Layer 4 cross-trip bounds …")
    route_peer = ["route_id","trip_type"]
    trip_features["route_peer_n"] = trip_features.groupby(route_peer)["trip_id"].transform("size")

    def iqr_bounds(s):
        s = s.dropna()
        if len(s) < PEER_MIN: return pd.Series({"lower":np.nan,"upper":np.nan})
        q1,q3 = s.quantile(0.25), s.quantile(0.75); iqr = q3-q1
        return pd.Series({"lower":q1-IQR_MULTIPLIER*iqr,"upper":q3+IQR_MULTIPLIER*iqr})

    def add_bounds(df, group_cols, metric, prefix):
        applied = df.groupby(group_cols)[metric].apply(iqr_bounds)
        bounds  = applied.reset_index() if isinstance(applied, pd.DataFrame) else applied.unstack().reset_index()
        return df.merge(bounds.rename(columns={"lower":f"{prefix}_lower","upper":f"{prefix}_upper"}),
                        on=group_cols, how="left")

    for metric, pfx in [("coverage_rate","cov"),("max_gap_s","gap"),("large_gap_count","lgap")]:
        trip_features = add_bounds(trip_features, route_peer, metric, pfx)

    trip_features["has_peer"]             = trip_features["cov_lower"].notna()
    trip_features["ct_flag_low_coverage"] = (trip_features["coverage_rate"] < trip_features["cov_lower"]).fillna(False).astype(int)
    trip_features["ct_flag_high_gap"]     = (trip_features["max_gap_s"]     > trip_features["gap_upper"]).fillna(False).astype(int)
    trip_features["ct_flag_many_gaps"]    = (trip_features["large_gap_count"] > trip_features["lgap_upper"]).fillna(False).astype(int)
    trip_features["layer4_fired"] = np.where(
        ~trip_features["has_peer"], np.nan,
        (trip_features[["ct_flag_low_coverage","ct_flag_high_gap","ct_flag_many_gaps"]].sum(axis=1) >= 2).astype(float)
    )
    trip_features["peer_source"] = np.where(trip_features["has_peer"],"route","none")
    return trip_features


# ═══════════════════════════════════════════════════════════════════════════════
# 6 — GPS RELIABILITY SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_gps_reliability_score(row):
    if row["gps_points"] == 0:
        return 0, "no_gps", 1, "zero_pings"
    cov = row["coverage_rate"]
    if pd.isna(cov) or cov < COVERAGE_HARD_GATE:
        score = 5 if pd.isna(cov) else round(BAND_L1[0]+(cov/COVERAGE_HARD_GATE)*(BAND_L1[1]-BAND_L1[0]),1)
        return score, "insufficient", 1, f"low_coverage({cov:.2f})"
    if row["end_trigger"] in GPS_FAIL_TRIGGERS:
        penalty = 5 if cov < 0.60 else 0
        return round(BAND_L2[0]+(BAND_L2[1]-BAND_L2[0])/2-penalty,1), "low", 2, f"gps_terminated({row['end_trigger']})"
    frozen_flag = (not pd.isna(row["frozen_ratio"])) and (row["frozen_ratio"] > FREEZE_TRIP_RATIO)
    speed_flag  = row["impossible_jumps"] >= 1
    if frozen_flag or speed_flag:
        both   = frozen_flag and speed_flag
        base   = BAND_L3[0] + (BAND_L3[1]-BAND_L3[0]) * 0.3
        bonus  = (BAND_L3[1]-BAND_L3[0]) * 0.3 if not both else 0
        reason = "frozen+impossible_speed" if both else ("frozen" if frozen_flag else "impossible_speed")
        return round(base+bonus,1), "medium", 3, reason
    if row["layer4_fired"] == 1:
        n_ct  = int(row["ct_flag_low_coverage"])+int(row["ct_flag_high_gap"])+int(row["ct_flag_many_gaps"])
        score = round(BAND_L4[1]-((n_ct-2)/1)*(BAND_L4[1]-BAND_L4[0])*0.6, 1)
        return score, "medium", 4, f"cross_trip_outlier({n_ct}_flags)"
    cq = min((cov-COVERAGE_HARD_GATE)/(1.0-COVERAGE_HARD_GATE), 1.0)
    return round(BAND_OK[0]+cq*(BAND_OK[1]-BAND_OK[0]),1), "high", None, "all_passed"


def apply_gps_scoring(trip_features):
    print("Scoring GPS reliability …")
    results = trip_features.apply(compute_gps_reliability_score, axis=1, result_type="expand")
    results.columns = ["gps_reliability_score","gps_reliability_tier","triggered_layer","trigger_reason"]
    trip_features = pd.concat([trip_features, results], axis=1)
    trip_features = trip_features.loc[:,~trip_features.columns.duplicated()].copy()
    for tier in TIER_ORDER:
        n = (trip_features["gps_reliability_tier"]==tier).sum()
        print(f"  {tier:<15}: {n:6,}  ({n/len(trip_features):.1%})")
    return trip_features


# ═══════════════════════════════════════════════════════════════════════════════
# 7 — EXECUTION PERFORMANCE SCORE
# ═══════════════════════════════════════════════════════════════════════════════

def build_stop_order_map(route_stops):
    route_stops["planned_order"] = route_stops.groupby(["route_id","trip_type"]).cumcount()
    return route_stops.set_index("route_stop_id")["planned_order"].to_dict()


def compute_stop_sequence_score(completed_times, stop_order_map):
    if not completed_times or len(completed_times) < 2: return np.nan
    try:    sorted_by_time = sorted(completed_times, key=lambda x: x[1])
    except: return np.nan
    positions = [stop_order_map.get(int(k), np.nan) for k,_ in sorted_by_time]
    known = [p for p in positions if not pd.isna(p)]
    if len(known) < 2: return np.nan
    asc  = sum(1 for i in range(len(known)-1) if known[i]<known[i+1])
    desc = sum(1 for i in range(len(known)-1) if known[i]>known[i+1])
    return max(asc,desc) / (len(known)-1)


def apply_execution_scoring(trip_features, stop_order_map):
    print("Scoring execution performance …")

    def score_row(row):
        null = (np.nan,np.nan,np.nan,np.nan)
        if row["gps_reliability_tier"] == "no_gps":        return *null, "no_gps", ""
        if row["gps_reliability_tier"] == "insufficient":  return *null, "insufficient_gps", ""
        if row.get("end_trigger") == "no_students":        return *null, "no_students", ""
        planned = row["planned_stops"]
        if pd.isna(planned) or planned == 0:               return *null, "no_stop_data", ""
        notes = []
        completion = row["stop_completion_rate"]
        if pd.isna(completion): completion=0.0; notes.append("completion=0")
        diffs = row["completed_diffs"]
        if diffs and len(diffs)>0:
            ontime_rate = sum(1 for d in diffs if abs(d)<=ONTIME_THRESHOLD_MIN)/len(diffs)
        else:
            ontime_rate=np.nan; notes.append("ontime=missing")
        seq_score = compute_stop_sequence_score(row["completed_times"], stop_order_map)
        if pd.isna(seq_score): notes.append("sequence=missing")
        components = [(completion,W_COMPLETION),(ontime_rate,W_ONTIME),(seq_score,W_SEQUENCE)]
        num = sum(v*w for v,w in components if not pd.isna(v))
        den = sum(w   for v,w in components if not pd.isna(v))
        if den==0: return *null, "no_stop_data", "all components missing"
        return round((num/den)*100,1), completion, ontime_rate, seq_score, \
               ("partial" if notes else "scored"), "; ".join(notes)

    exec_results = trip_features.apply(score_row, axis=1, result_type="expand")
    exec_results.columns = ["execution_score","exec_completion_rate","ontime_rate",
                            "sequence_score","execution_status","execution_note"]
    trip_features = pd.concat([trip_features, exec_results], axis=1)
    trip_features = trip_features.loc[:,~trip_features.columns.duplicated()].copy()
    for status in ["scored","partial","no_gps","insufficient_gps","no_students","no_stop_data"]:
        n = (trip_features["execution_status"]==status).sum()
        print(f"  {status:<20}: {n:6,}  ({n/len(trip_features):.1%})")
    return trip_features


# ═══════════════════════════════════════════════════════════════════════════════
# 8 — VENDOR AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_vendor_summary(trip_features, vendors):
    print("Building vendor summary …")
    vendor_map = vendors[["vendor_id","vendor_label","gps_provider"]].drop_duplicates()
    _drop = [c for c in ["vendor_label","gps_provider"] if c in trip_features.columns]
    tf_v  = trip_features.drop(columns=_drop)
    return (
        tf_v.merge(vendor_map, on="vendor_id", how="left")
        .groupby(["vendor_id","vendor_label","gps_provider"], as_index=False)
        .agg(
            total_trips         = ("trip_id",               "nunique"),
            avg_gps_reliability = ("gps_reliability_score", "mean"),
            pct_high            = ("gps_reliability_tier",  lambda x: (x=="high").mean()),
            pct_medium          = ("gps_reliability_tier",  lambda x: (x=="medium").mean()),
            pct_low             = ("gps_reliability_tier",  lambda x: (x=="low").mean()),
            pct_insufficient    = ("gps_reliability_tier",  lambda x: (x=="insufficient").mean()),
            pct_no_gps          = ("gps_reliability_tier",  lambda x: (x=="no_gps").mean()),
            avg_execution_score = ("execution_score",       "mean"),
            avg_completion      = ("exec_completion_rate",  "mean"),
            avg_ontime          = ("ontime_rate",           "mean"),
            avg_sequence        = ("sequence_score",        "mean"),
        )
        .sort_values("avg_gps_reliability", ascending=False)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9 — MONTHLY + WEEKDAY AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_time_data(trip_features, vendors):
    print("Building time aggregations …")
    tf = trip_features.copy()
    tf["date_dt"] = pd.to_datetime(tf["date"], utc=True).dt.tz_localize(None)
    tf["month"]   = tf["date_dt"].dt.to_period("M").astype(str)
    tf["weekday"] = tf["date_dt"].dt.day_name()

    vendor_map = vendors[["vendor_id","gps_provider"]].drop_duplicates()
    _drop = [c for c in ["gps_provider"] if c in tf.columns]
    tf = tf.drop(columns=_drop).merge(vendor_map, on="vendor_id", how="left")

    monthly_tier  = tf.groupby(["month","gps_reliability_tier"]).size().reset_index(name="count")
    weekday_tier  = tf.groupby(["weekday","gps_reliability_tier"]).size().reset_index(name="count")

    sc = tf[tf["execution_status"].isin(["scored","partial"])].copy()
    monthly_exec  = sc.groupby(["month","gps_provider"])["execution_score"].mean().reset_index()
    monthly_exec  = pd.concat([monthly_exec,
        sc.groupby("month")["execution_score"].mean().reset_index().assign(gps_provider="all")],
        ignore_index=True)
    weekday_exec  = sc.groupby(["weekday","gps_provider"])["execution_score"].mean().reset_index()
    weekday_exec  = pd.concat([weekday_exec,
        sc.groupby("weekday")["execution_score"].mean().reset_index().assign(gps_provider="all")],
        ignore_index=True)

    return monthly_tier, monthly_exec, weekday_tier, weekday_exec


# ═══════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("4MATIV — GPS Reliability & Execution Dashboard")
print("=" * 60)

trips, trip_positions, vendors, stop_locations, route_stops, complete_threshold_min = load_data()
trip_positions = build_ping_features(trip_positions, trips, stop_locations)
trip_features  = aggregate_trip_features(trips, trip_positions)

geo_df = parse_geofence_stops(trips, complete_threshold_min)
trip_features = trip_features.merge(
    geo_df[["trip_id","planned_stops","completed_stops","stop_completion_rate",
            "completed_diffs","completed_times","floor_hit_count"]],
    on="trip_id", how="left")

trip_features  = build_layer4(trip_features)
trip_features  = apply_gps_scoring(trip_features)
stop_order_map = build_stop_order_map(route_stops)
trip_features  = apply_execution_scoring(trip_features, stop_order_map)
vendor_summary = build_vendor_summary(trip_features, vendors)
monthly_tier, monthly_exec, weekday_tier, weekday_exec = build_time_data(trip_features, vendors)

MONTHS_SORTED   = sorted(monthly_tier["month"].unique())
PROVIDERS_AVAIL = sorted(vendor_summary["gps_provider"].dropna().unique())

print("\nPipeline complete. Starting dashboard …\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY LAYOUT HELPER
# ═══════════════════════════════════════════════════════════════════════════════

PLOTLY_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="Inter, sans-serif", color=TEXT, size=12),
    margin=dict(l=16, r=16, t=44, b=16),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11, color=MUTED)),
)

def apply_layout(fig, **kwargs):
    fig.update_layout(**{**PLOTLY_BASE, **kwargs})
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_provider_tier_bar():
    prov_totals = (
        vendor_summary
        .groupby("gps_provider")[["total_trips","pct_high","pct_medium","pct_low","pct_insufficient","pct_no_gps"]]
        .apply(lambda g: pd.Series({
            t: (g[f"pct_{t}"]*g["total_trips"]).sum()/g["total_trips"].sum()
            for t in ["high","medium","low","insufficient","no_gps"]
        })).reset_index()
    )
    prov_totals = prov_totals.sort_values("high", ascending=True)
    fig = go.Figure()
    for tier in ["high","medium","low","insufficient","no_gps"]:
        fig.add_trace(go.Bar(
            y=prov_totals["gps_provider"].str.upper(),
            x=(prov_totals[tier]*100).round(1),
            name=TIER_LABELS[tier],
            orientation="h",
            marker_color=TIER_COLORS[tier],
            hovertemplate="%{y} — "+TIER_LABELS[tier]+": %{x:.1f}%<extra></extra>",
        ))
    apply_layout(fig,
        title=dict(text="GPS Reliability by Provider (%)", font=dict(size=13,color=TEXT)),
        barmode="stack",
        xaxis=dict(range=[0,100], ticksuffix="%", gridcolor=BORDER),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.38, xanchor="center", x=0.5),
    )
    return fig


def fig_monthly_tier_trend():
    pivot = monthly_tier.pivot_table(
        index="month", columns="gps_reliability_tier", values="count", fill_value=0
    ).reindex(MONTHS_SORTED)
    fig = go.Figure()
    for tier in ["high","medium","low","insufficient","no_gps"]:
        if tier in pivot.columns:
            fig.add_trace(go.Bar(
                x=pivot.index.tolist(), y=pivot[tier].tolist(),
                name=TIER_LABELS[tier], marker_color=TIER_COLORS[tier],
                hovertemplate="%{x} — "+TIER_LABELS[tier]+": %{y:,}<extra></extra>",
            ))
    apply_layout(fig,
        title=dict(text="GPS Reliability Trend Over Time — Monthly Tier Breakdown", font=dict(size=13,color=TEXT)),
        barmode="stack",
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor=BORDER, title="Trips"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.38, xanchor="center", x=0.5),
    )
    return fig


def fig_weekday_tier():
    present = [d for d in WEEKDAY_ORDER if d in weekday_tier["weekday"].values]
    pivot = weekday_tier.pivot_table(
        index="weekday", columns="gps_reliability_tier", values="count", fill_value=0
    ).reindex(present)
    fig = go.Figure()
    for tier in ["high","medium","low","insufficient","no_gps"]:
        if tier in pivot.columns:
            fig.add_trace(go.Bar(
                x=[d[:3] for d in pivot.index.tolist()],
                y=pivot[tier].tolist(),
                name=TIER_LABELS[tier],
                marker_color=TIER_COLORS[tier],
                showlegend=False,
                hovertemplate="%{x} — "+TIER_LABELS[tier]+": %{y:,}<extra></extra>",
            ))
    apply_layout(fig,
        title=dict(text="GPS Reliability by Day of Week", font=dict(size=13,color=TEXT)),
        barmode="stack",
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor=BORDER, title="Trips"),
    )
    return fig


def fig_exec_trend(provider_filter):
    key  = "all" if provider_filter == "all" else provider_filter
    data = monthly_exec[monthly_exec["gps_provider"]==key].sort_values("month")
    label = "All Providers" if provider_filter=="all" else provider_filter.upper()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["month"], y=data["execution_score"].round(1),
        mode="lines+markers", name=label,
        line=dict(color=BRAND_ORANGE, width=2.5),
        marker=dict(size=7, color=BRAND_ORANGE),
        fill="tozeroy", fillcolor="rgba(245,166,35,0.07)",
        hovertemplate="%{x}: %{y:.1f}<extra></extra>",
    ))
    apply_layout(fig,
        title=dict(text="Execution Score Trend Over Time", font=dict(size=13,color=TEXT)),
        yaxis=dict(range=[60,100], gridcolor=BORDER, title="Avg Score"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


def fig_weekday_exec(provider_filter):
    key     = "all" if provider_filter == "all" else provider_filter
    data    = weekday_exec[weekday_exec["gps_provider"]==key].copy()
    present = [d for d in WEEKDAY_ORDER if d in data["weekday"].values]
    data    = data.set_index("weekday").reindex(present).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d[:3] for d in data["weekday"]],
        y=data["execution_score"].round(1),
        mode="lines+markers",
        line=dict(color=BRAND_TEAL, width=2.5),
        marker=dict(size=7, color=BRAND_TEAL),
        fill="tozeroy", fillcolor="rgba(78,205,196,0.07)",
        hovertemplate="%{x}: %{y:.1f}<extra></extra>",
    ))
    apply_layout(fig,
        title=dict(text="Execution Score by Day of Week", font=dict(size=13,color=TEXT)),
        yaxis=dict(range=[60,100], gridcolor=BORDER, title="Avg Score"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


def fig_vendor_grouped_bar(provider_filter):
    vs = (vendor_summary if provider_filter=="all"
          else vendor_summary[vendor_summary["gps_provider"]==provider_filter]).copy()
    vs = vs[vs["total_trips"]>=3].sort_values("avg_gps_reliability", ascending=False)
    fig = go.Figure()
    for col, label, color in [
        ("avg_completion", "Stop Completion", TIER_COLORS["high"]),
        ("avg_ontime",     "On-Time Rate",    BRAND_BLUE),
        ("avg_sequence",   "Sequence Score",  BRAND_ORANGE),
    ]:
        fig.add_trace(go.Bar(
            name=label,
            x=vs["vendor_label"],
            y=(vs[col]*100).round(1),
            marker_color=color,
            opacity=0.85,
            hovertemplate=f"%{{x}} — {label}: %{{y:.1f}}%<extra></extra>",
        ))
    apply_layout(fig,
        title=dict(text="Vendor Execution — Completion · On-Time · Sequence", font=dict(size=13,color=TEXT)),
        barmode="group",
        yaxis=dict(range=[0,110], gridcolor=BORDER, ticksuffix="%"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )
    return fig


def fig_scatter(gps_thresh, exec_thresh):
    def dot_color(g, e):
        if g < gps_thresh:  return TIER_COLORS["no_gps"]
        if e < exec_thresh: return TIER_COLORS["low"]
        return TIER_COLORS["high"]

    vs = vendor_summary[vendor_summary["avg_execution_score"].notna()].copy()
    vs["color"]       = vs.apply(lambda r: dot_color(r["avg_gps_reliability"], r["avg_execution_score"]), axis=1)
    vs["bubble_size"] = np.sqrt(vs["total_trips"]) * 1.8
    fig = go.Figure()
    for color, label in [
        (TIER_COLORS["high"],   "Meets both thresholds"),
        (TIER_COLORS["low"],    "GPS ok, execution below threshold"),
        (TIER_COLORS["no_gps"],"GPS below threshold"),
    ]:
        sub = vs[vs["color"]==color]
        if sub.empty: continue
        fig.add_trace(go.Scatter(
            x=sub["avg_gps_reliability"].round(1),
            y=sub["avg_execution_score"].round(1),
            mode="markers+text",
            name=label,
            text=sub["vendor_label"],
            textposition="top center",
            textfont=dict(size=9, color=MUTED),
            marker=dict(size=sub["bubble_size"], color=color, opacity=0.8,
                        line=dict(width=1.5, color=color)),
            customdata=sub[["vendor_label","gps_provider","total_trips"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "GPS: %{x:.1f} · Exec: %{y:.1f}<br>"
                "Trips: %{customdata[2]:,}<extra></extra>"
            ),
        ))
    fig.add_vline(x=gps_thresh,  line_dash="dash", line_color=ACCENT, opacity=0.6,
                  annotation_text=f"GPS ≥ {gps_thresh}", annotation_font_color=ACCENT)
    fig.add_hline(y=exec_thresh, line_dash="dash", line_color=ACCENT, opacity=0.6,
                  annotation_text=f"Exec ≥ {exec_thresh}", annotation_font_color=ACCENT)
    apply_layout(fig,
        title=dict(text="Vendor Risk — GPS Reliability vs Execution Score", font=dict(size=13,color=TEXT)),
        xaxis=dict(range=[30,105], title="GPS Reliability Score", gridcolor=BORDER),
        yaxis=dict(range=[30,108], title="Execution Score", gridcolor=BORDER),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
    )
    return fig


def fig_ranking_bar(gps_thresh):
    vs = vendor_summary.sort_values("avg_gps_reliability", ascending=True).copy()
    vs["color"] = vs["avg_gps_reliability"].apply(
        lambda g: TIER_COLORS["high"] if g>=gps_thresh else TIER_COLORS["no_gps"]
    )
    fig = go.Figure(go.Bar(
        y=vs["vendor_label"],
        x=vs["avg_gps_reliability"].round(1),
        orientation="h",
        marker_color=vs["color"], marker_opacity=0.85,
        hovertemplate="<b>%{y}</b><br>GPS Reliability: %{x:.1f}<extra></extra>",
    ))
    fig.add_vline(x=gps_thresh, line_dash="dash", line_color=ACCENT, opacity=0.7,
                  annotation_text=f"Threshold: {gps_thresh}", annotation_font_color=ACCENT)
    apply_layout(fig,
        title=dict(text="Vendor Ranking — GPS Reliability Score", font=dict(size=13,color=TEXT)),
        xaxis=dict(range=[0,105], gridcolor=BORDER),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# KPI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def most_problematic_provider():
    scores = vendor_summary.groupby("gps_provider").apply(
        lambda g: np.average(g["avg_gps_reliability"], weights=g["total_trips"])
    )
    return scores.idxmin(), round(scores.min(), 1)


def exec_kpis(provider_filter):
    vs = (vendor_summary if provider_filter=="all"
          else vendor_summary[vendor_summary["gps_provider"]==provider_filter]).copy()
    vs = vs[vs["avg_execution_score"].notna()]
    if vs.empty: return "—","—","—"
    w = vs["total_trips"]
    return (
        round(np.average(vs["avg_execution_score"], weights=w), 1),
        f"{round(np.average(vs['avg_completion'], weights=w)*100,1)}%",
        f"{round(np.average(vs['avg_ontime'], weights=w)*100,1)}%",
    )


WORST_PROV, WORST_SCORE = most_problematic_provider()
TOTAL_TRIPS   = len(trip_features)
TIER_COUNTS   = trip_features["gps_reliability_tier"].value_counts()
PCT_HIGH      = round(TIER_COUNTS.get("high",0)   / TOTAL_TRIPS * 100, 1)
PCT_NO_GPS    = round(TIER_COUNTS.get("no_gps",0) / TOTAL_TRIPS * 100, 1)
PCT_SCOREABLE = round(trip_features["execution_status"].isin(["scored","partial"]).sum() / TOTAL_TRIPS * 100, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# DASH COMPONENT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_card(label, value, sub="", color=TEXT, highlight=False):
    border = f"1.5px solid {TIER_COLORS['no_gps']}" if highlight else f"1px solid {BORDER}"
    bg     = "rgba(231,76,60,0.05)" if highlight else SURFACE
    return html.Div([
        html.Div(label, style={"fontSize":"11px","fontWeight":"600","letterSpacing":"0.07em",
                               "textTransform":"uppercase","color":MUTED,"marginBottom":"8px"}),
        html.Div(value, style={"fontFamily":"'IBM Plex Mono',monospace","fontSize":"26px",
                               "fontWeight":"700","color":color,"lineHeight":"1"}),
        html.Div(sub,   style={"fontSize":"12px","color":MUTED,"marginTop":"6px"}),
    ], style={"background":bg,"border":border,"borderRadius":"10px",
              "padding":"16px 20px","flex":"1","minWidth":"180px",
              "boxShadow":"0 1px 3px rgba(0,0,0,0.05)"})


def card(children, extra_style=None):
    s = {"background":SURFACE,"border":f"1px solid {BORDER}","borderRadius":"10px",
         "padding":"20px 24px","boxShadow":"0 1px 3px rgba(0,0,0,0.05)"}
    if extra_style: s.update(extra_style)
    return html.Div(children, style=s)


def section_label(text):
    return html.Div(text, style={
        "fontSize":"11px","fontWeight":"600","textTransform":"uppercase",
        "letterSpacing":"0.07em","color":MUTED,"marginBottom":"10px",
    })


# ── Vendor summary table (Page 1) ─────────────────────────────────────────────

def vendor_table():
    vs = vendor_summary.copy()
    vs["avg_gps_reliability"] = vs["avg_gps_reliability"].round(1)

    TH = {"background":SURFACE,"color":MUTED,"fontSize":"11px","fontWeight":"600",
          "textTransform":"uppercase","letterSpacing":"0.06em","padding":"10px 14px",
          "borderBottom":f"2px solid {BORDER}","textAlign":"left","whiteSpace":"nowrap"}
    TD = {"padding":"9px 14px","borderBottom":f"1px solid {BORDER}","fontSize":"13px",
          "color":TEXT,"verticalAlign":"middle"}

    def badge(score):
        if score >= 75:   c,bg = TIER_COLORS["high"],         "rgba(39,174,96,0.1)"
        elif score >= 55: c,bg = TIER_COLORS["medium"],       "rgba(139,195,74,0.1)"
        elif score >= 45: c,bg = TIER_COLORS["low"],          "rgba(243,156,18,0.1)"
        else:             c,bg = TIER_COLORS["no_gps"],       "rgba(231,76,60,0.1)"
        return html.Span(str(score), style={"background":bg,"color":c,"borderRadius":"4px",
                                            "padding":"2px 9px","fontWeight":"600","fontSize":"12px"})

    rows = []
    for _, r in vs.iterrows():
        exec_val = f"{r['avg_execution_score']:.1f}" if pd.notna(r["avg_execution_score"]) else "—"
        pct_high_str   = f"{r['pct_high']*100:.1f}%"
        pct_no_gps_str = f"{r['pct_no_gps']*100:.1f}%"
        rows.append(html.Tr([
            html.Td(r["vendor_label"],           style={**TD,"fontWeight":"600"}),
            html.Td(r["gps_provider"].upper(),   style={**TD,"color":BRAND_BLUE,"fontWeight":"500"}),
            html.Td(f"{r['total_trips']:,}",     style={**TD,"textAlign":"right","fontFamily":"'IBM Plex Mono',monospace"}),
            html.Td(badge(r["avg_gps_reliability"]), style={**TD,"textAlign":"center"}),
            html.Td(pct_high_str,    style={**TD,"textAlign":"center","color":TIER_COLORS["high"],"fontWeight":"600"}),
            html.Td(pct_no_gps_str,  style={**TD,"textAlign":"center","color":TIER_COLORS["no_gps"],"fontWeight":"600"}),
            html.Td(exec_val,        style={**TD,"textAlign":"center","fontFamily":"'IBM Plex Mono',monospace"}),
        ]))

    return html.Div(html.Table([
        html.Thead(html.Tr([
            html.Th("Vendor",          style=TH),
            html.Th("GPS Provider",    style=TH),
            html.Th("Trips",           style={**TH,"textAlign":"right"}),
            html.Th("GPS Reliability", style={**TH,"textAlign":"center"}),
            html.Th("% High",          style={**TH,"textAlign":"center"}),
            html.Th("% No GPS",        style={**TH,"textAlign":"center"}),
            html.Th("Exec Score",      style={**TH,"textAlign":"center"}),
        ])),
        html.Tbody(rows),
    ], style={"width":"100%","borderCollapse":"collapse"}), style={"overflowX":"auto"})


# ═══════════════════════════════════════════════════════════════════════════════
# DASH APP + LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

NAV_H = "56px"
NAV_STYLE = {
    "display":"flex","alignItems":"center",
    "background":BG,"borderBottom":f"1px solid {BORDER}",
    "padding":f"0 28px","height":NAV_H,
    "position":"sticky","top":"0","zIndex":"100",
    "boxShadow":"0 1px 4px rgba(0,0,0,0.06)",
}
TAB_BASE = {
    "padding":"0 20px","height":NAV_H,"display":"flex","alignItems":"center",
    "fontSize":"13px","fontWeight":"500","color":MUTED,"cursor":"pointer",
    "borderBottom":"2.5px solid transparent","whiteSpace":"nowrap",
    "background":"transparent","border":"none",
    "borderTop":"none","borderLeft":"none","borderRight":"none",
}
TAB_ACTIVE = {**TAB_BASE, "color":TEXT, "borderBottom":f"2.5px solid {ACCENT}", "fontWeight":"600"}
PAGE_STYLE = {"padding":"28px 32px 56px","background":BG,"minHeight":f"calc(100vh - {NAV_H})"}

app = dash.Dash(__name__, title="4MATIV GPS Dashboard")
app.layout = html.Div(
    style={"background":BG,"minHeight":"100vh","fontFamily":"Inter, sans-serif","color":TEXT},
    children=[
        html.Link(rel="stylesheet",
                  href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600;700&display=swap"),

        # ── Navigation ──────────────────────────────────────────────────────
        html.Nav(style=NAV_STYLE, children=[
            html.Div(
                dangerouslySetInnerHTML={"__html": LOGO_SVG},
                style={"marginRight":"32px","display":"flex","alignItems":"center","flexShrink":"0"},
            ),
            dcc.Tabs(id="tabs", value="p1",
                     style={"border":"none","flex":"1","height":NAV_H},
                     children=[
                dcc.Tab(label="Can we trust the GPS?", value="p1",
                        style={**TAB_BASE}, selected_style={**TAB_ACTIVE}),
                dcc.Tab(label="Route Execution",       value="p2",
                        style={**TAB_BASE}, selected_style={**TAB_ACTIVE}),
                dcc.Tab(label="Vendor Risk",           value="p3",
                        style={**TAB_BASE}, selected_style={**TAB_ACTIVE}),
            ]),
        ]),

        html.Div(id="page-content"),
    ]
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1
# ═══════════════════════════════════════════════════════════════════════════════

def layout_page1():
    return html.Div(style=PAGE_STYLE, children=[

        html.H2("GPS Data Reliability",
                style={"fontSize":"22px","fontWeight":"700","marginBottom":"4px","color":TEXT}),
        html.P(
            f"Before asking 'did the route execute?' we first need to know 'can we even answer that?' "
            f"— {TOTAL_TRIPS:,} trips · Sep 2025 – Feb 2026",
            style={"fontSize":"13px","color":MUTED,"marginBottom":"24px"}),

        # KPI row
        html.Div(style={"display":"flex","gap":"14px","marginBottom":"22px","flexWrap":"wrap"}, children=[
            kpi_card("⚠ Most Problematic Provider", WORST_PROV.upper(),
                     f"Avg GPS reliability score: {WORST_SCORE}",
                     TIER_COLORS["no_gps"], highlight=True),
            kpi_card("High Reliability Trips", f"{PCT_HIGH}%",
                     f"{TIER_COUNTS.get('high',0):,} of {TOTAL_TRIPS:,} trips",
                     TIER_COLORS["high"]),
            kpi_card("No GPS Trips", f"{PCT_NO_GPS}%",
                     f"{TIER_COUNTS.get('no_gps',0):,} trips with zero positions",
                     TIER_COLORS["no_gps"]),
            kpi_card("Scoreable for Execution", f"{PCT_SCOREABLE}%",
                     "Trips with medium/high GPS reliability",
                     BRAND_ORANGE),
        ]),

        # Provider bar + Monthly trend
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"18px","marginBottom":"18px"},
                 children=[
            card([
                section_label("GPS Reliability Tier Breakdown by Provider"),
                dcc.Graph(figure=fig_provider_tier_bar(),
                          config={"displayModeBar":False}, style={"height":"280px"}),
            ]),
            card([
                section_label("GPS Reliability Trend Over Time — Monthly Tier Breakdown"),
                dcc.Graph(figure=fig_monthly_tier_trend(),
                          config={"displayModeBar":False}, style={"height":"280px"}),
            ]),
        ]),

        # Weekday GPS breakdown
        card([
            section_label("GPS Reliability by Day of Week"),
            dcc.Graph(figure=fig_weekday_tier(),
                      config={"displayModeBar":False}, style={"height":"220px"}),
        ], {"marginBottom":"18px"}),

        # Vendor table
        card([
            section_label("Vendor Summary — GPS Reliability & Execution"),
            vendor_table(),
        ]),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2
# ═══════════════════════════════════════════════════════════════════════════════

def layout_page2():
    avg_exec, avg_comp, avg_ot = exec_kpis("all")
    return html.Div(style=PAGE_STYLE, children=[

        html.H2("Route Execution Performance",
                style={"fontSize":"22px","fontWeight":"700","marginBottom":"4px","color":TEXT}),
        html.P("Only trips with sufficient GPS data (medium or high reliability) are scored.",
               style={"fontSize":"13px","color":MUTED,"marginBottom":"20px"}),

        # Controls
        html.Div(style={"display":"flex","alignItems":"flex-end","gap":"20px","marginBottom":"18px"}, children=[
            html.Div([
                html.Div("GPS Provider Filter",
                         style={"fontSize":"11px","fontWeight":"600","textTransform":"uppercase",
                                "letterSpacing":"0.07em","color":MUTED,"marginBottom":"5px"}),
                dcc.Dropdown(
                    id="provider-filter",
                    options=[{"label":"All Providers","value":"all"}] +
                            [{"label":p.upper(),"value":p} for p in PROVIDERS_AVAIL],
                    value="all", clearable=False,
                    style={"color":TEXT,"border":f"1px solid {BORDER}","borderRadius":"6px",
                           "fontSize":"13px","width":"220px","background":SURFACE},
                ),
            ]),
        ]),

        # KPI row
        html.Div(id="exec-kpi-row",
                 style={"display":"flex","gap":"14px","marginBottom":"18px","flexWrap":"wrap"},
                 children=[
            kpi_card("Avg Execution Score",  str(avg_exec), "Across scoreable trips",    TIER_COLORS["high"]),
            kpi_card("Avg Stop Completion",  str(avg_comp), "Completed ÷ planned stops", TIER_COLORS["medium"]),
            kpi_card("Avg On-Time Rate",     str(avg_ot),   "|completed_diff| ≤ 10 min", BRAND_BLUE),
        ]),

        # Vendor grouped bar (full width)
        card([
            dcc.Graph(id="fig-vendor-grouped", figure=fig_vendor_grouped_bar("all"),
                      config={"displayModeBar":False}, style={"height":"340px"}),
        ], {"marginBottom":"18px"}),

        # Monthly + Weekday execution trends side by side
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"18px"}, children=[
            card([
                dcc.Graph(id="fig-exec-trend", figure=fig_exec_trend("all"),
                          config={"displayModeBar":False}, style={"height":"260px"}),
            ]),
            card([
                dcc.Graph(id="fig-exec-weekday", figure=fig_weekday_exec("all"),
                          config={"displayModeBar":False}, style={"height":"260px"}),
            ]),
        ]),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3
# ═══════════════════════════════════════════════════════════════════════════════

def layout_page3():
    legend_dot = lambda color, label: html.Div(
        style={"display":"flex","alignItems":"center","gap":"7px","fontSize":"12px","color":MUTED},
        children=[html.Div(style={"width":"10px","height":"10px","borderRadius":"2px","background":color}), label]
    )
    return html.Div(style=PAGE_STYLE, children=[

        html.H2("Vendor Risk Quadrant",
                style={"fontSize":"22px","fontWeight":"700","marginBottom":"4px","color":TEXT}),
        html.P("Drag the sliders to update thresholds — charts refresh instantly.",
               style={"fontSize":"13px","color":MUTED,"marginBottom":"20px"}),

        # Sliders
        html.Div(style={"display":"flex","gap":"48px","marginBottom":"18px","flexWrap":"wrap"}, children=[
            html.Div([
                html.Div("GPS Reliability Score Threshold",
                         style={"fontSize":"11px","fontWeight":"600","textTransform":"uppercase",
                                "letterSpacing":"0.07em","color":MUTED,"marginBottom":"6px"}),
                dcc.Slider(id="gps-threshold", min=0, max=100, step=1, value=60,
                           marks={0:"0",25:"25",50:"50",75:"75",100:"100"},
                           tooltip={"placement":"bottom","always_visible":True},
                           updatemode="drag"),
            ], style={"minWidth":"280px"}),
            html.Div([
                html.Div("Execution Score Threshold",
                         style={"fontSize":"11px","fontWeight":"600","textTransform":"uppercase",
                                "letterSpacing":"0.07em","color":MUTED,"marginBottom":"6px"}),
                dcc.Slider(id="exec-threshold", min=0, max=100, step=1, value=75,
                           marks={0:"0",25:"25",50:"50",75:"75",100:"100"},
                           tooltip={"placement":"bottom","always_visible":True},
                           updatemode="drag"),
            ], style={"minWidth":"280px"}),
        ]),

        # Legend
        html.Div(style={"display":"flex","gap":"20px","marginBottom":"18px","flexWrap":"wrap"}, children=[
            legend_dot(TIER_COLORS["high"],   "Meets both thresholds"),
            legend_dot(TIER_COLORS["low"],    "GPS ok, execution below threshold"),
            legend_dot(TIER_COLORS["no_gps"],"GPS below threshold"),
        ]),

        # Charts
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"18px"}, children=[
            card([dcc.Graph(id="fig-scatter", figure=fig_scatter(60,75),
                            config={"displayModeBar":False}, style={"height":"420px"})]),
            card([dcc.Graph(id="fig-ranking", figure=fig_ranking_bar(60),
                            config={"displayModeBar":False}, style={"height":"420px"})]),
        ]),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(Output("page-content","children"), Input("tabs","value"))
def render_page(tab):
    if tab == "p1": return layout_page1()
    if tab == "p2": return layout_page2()
    if tab == "p3": return layout_page3()


@app.callback(
    Output("exec-kpi-row",      "children"),
    Output("fig-vendor-grouped","figure"),
    Output("fig-exec-trend",    "figure"),
    Output("fig-exec-weekday",  "figure"),
    Input("provider-filter",    "value"),
    prevent_initial_call=True,
)
def update_page2(provider):
    avg_exec, avg_comp, avg_ot = exec_kpis(provider)
    kpis = [
        kpi_card("Avg Execution Score",  str(avg_exec), "Across scoreable trips",    TIER_COLORS["high"]),
        kpi_card("Avg Stop Completion",  str(avg_comp), "Completed ÷ planned stops", TIER_COLORS["medium"]),
        kpi_card("Avg On-Time Rate",     str(avg_ot),   "|completed_diff| ≤ 10 min", BRAND_BLUE),
    ]
    return kpis, fig_vendor_grouped_bar(provider), fig_exec_trend(provider), fig_weekday_exec(provider)


@app.callback(
    Output("fig-scatter","figure"),
    Output("fig-ranking","figure"),
    Input("gps-threshold",  "value"),
    Input("exec-threshold", "value"),
    prevent_initial_call=True,
)
def update_page3(gps_thresh, exec_thresh):
    return fig_scatter(gps_thresh, exec_thresh), fig_ranking_bar(gps_thresh)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run_server(mode="inline", port=8050)
