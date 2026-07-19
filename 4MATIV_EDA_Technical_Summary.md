# 4MATIV EDA Live Case — Technical Summary
*Comprehensive summary of data cleaning, EDA techniques, libraries, and insights*

---

## 1. Project Overview

**Client:** 4MATIV Technologies  
**University:** University of Minnesota — Carlson School of Management  
**Course:** Professor Teng's Exploratory Data Analysis  
**Core Question:** *"Can we validate the incoming GPS data to evaluate if the route is being executed as planned?"*  
**Desired Output:** A Python-based analysis allowing 4MATIV to monitor GPS unit health via data visualization, anomaly alerts, and route assignment validation with confidence metrics.

---

## 2. Data Sources

| File | Rows | Description |
|------|------|-------------|
| `trips.csv` | 14,859 | One row per trip per day per direction |
| `trip_positions.csv` | 2,351,477 | Raw GPS pings (lat, lng, timestamp) |
| `routes.csv` | ~200 | Planned routes |
| `route_stops.csv` | ~1,936 | Stops per route per direction |
| `stop_locations.csv` | ~500 | Physical stop lat/lng and addresses |
| `stop_times.csv` | ~260K | Scheduled times per stop per date |
| `vendors.csv` | 16 | Vendor labels, GPS providers, polling intervals |
| `vehicles.csv` | ~100 | Vehicle assignments |
| `tracking_devices.csv` | ~100 | GPS device assignments |
| `schools.csv` | 2 | School metadata |
| `school_times.csv` | ~50 | School schedule times |
| `geofencing_config.json` | — | Geofence radii and polling intervals |

**Final Cleaned Files:**
- `trip_stop_w_status.csv` — 102,636 rows, 29 columns (stop-level master table)
- `trip_stop_w_confidence.csv` — 102,636 rows, 33 columns (with confidence score)
- `trip_scores.csv` — 14,859 rows, 33 columns (trip-level GPS reliability + execution scores)
- `vendor_summary.csv` — 16 rows, 15 columns (vendor-level scorecard)

---

## 3. Libraries Used

```python
# Core
import pandas as pd
import numpy as np
import json
import os

# Geospatial
from sklearn.neighbors import BallTree   # Stop proximity exclusion for freeze detection

# Visualization
import matplotlib.pyplot as plt
import plotly.express as px              # Dashboard visualizations
import plotly.graph_objects as go

# Dashboard
import streamlit as st                   # Interactive dashboard

# Distance calculation
# Haversine formula — implemented manually using numpy
```

---

## 4. Data Cleaning

### 4.1 Table Joins / Relational Merges

Three relational chains were established:

**Chain 1 — Planned Route Data:**
```python
routes → route_stops → stop_locations & stop_times
# Merged on: route_id, route_stop_id, stop_location_id
# Exploded stop_times by date (one row per stop per occurrence date)
```

**Chain 2 — Vendor/Device Assignment:**
```python
vendor_schedules → vehicles → tracking_devices
# Merged on: vendor_id, vehicle_id
```

**Chain 3 — Actual Trip Data:**
```python
trips → trip_positions → route_stop_info (JSON parsed)
# Merged on: trip_id
```

**Final Master Table:**
```python
planned_df = routes + route_stops + stop_locations + stop_times  # planned.csv
actual_df = trips + GPS health metrics                            # actual.csv
final = planned_df merged with actual_df on [trip_id, route_stop_id, date]
# → trip_stop_w_status.csv
```

### 4.2 Datetime Conversions

```python
# Convert string timestamps to datetime
df['planned_datetime_local'] = pd.to_datetime(df['planned_datetime_local'], errors='coerce')
df['arrival'] = pd.to_datetime(df['arrival'], errors='coerce', utc=True)
trips["started_at"] = pd.to_datetime(trips["started_at"], errors="coerce", utc=True)
trips["ended_at"]   = pd.to_datetime(trips["ended_at"],   errors="coerce", utc=True)
```

### 4.3 NULL Handling

| Column | Nulls | Treatment |
|--------|-------|-----------|
| `tracking_device_id` | 2,377 | Flagged as `no_device_assigned` anomaly |
| `gps_first_time` / `gps_last_time` | 4,059 | Trips with zero GPS data — flagged as `no_gps` |
| `started_at` / `ended_at` | 4,059 / 3,675 | Flagged as trip never started/ended |
| `arrival` (stop level) | 19,898 | Unvisited stops — kept as `visited = False` |
| `delay_min` | 19,898 | Unvisited stops — treated as not on time |

**Key principle:** Nulls were NOT dropped — they represent real operational failures and are the anomalies being detected.

### 4.4 GPS Health Flag Classification

```python
# Classified per trip based on trip_positions
if gps_points_observed == 0:
    flag = 'no_gps'
elif gps_points_observed < 5:
    flag = 'very_sparse'
elif max_gap_sec > 300:          # 5+ minute gap
    flag = 'severe_gap'
elif large_gap_count >= 3:       # 3+ gaps > 120 seconds
    flag = 'intermittent'
else:
    flag = 'healthy'
```

### 4.5 Arrival Status Classification

```python
bins   = [-np.inf, -5, -2, 2, 5, np.inf]
labels = ['very_early', 'early', 'on_time', 'late', 'very_late']
df['arrival_status'] = pd.cut(df['delay_min'], bins=bins, labels=labels)
# on_time window: delay_min between -5 and +5 minutes
```

### 4.6 Vendor Merge

```python
vendors = pd.read_csv('vendors.csv')
df = pd.merge(df, vendors[['vendor_id', 'gps_provider', 'polling_interval_s']],
              on='vendor_id', how='left')
```

---

## 5. GPS Reliability Framework

A 4-component framework was designed to calculate a per-trip GPS reliability score (0–100):

| Component | Weight | Source |
|-----------|--------|--------|
| Temporal Coverage | 30% | `trips` + `trip_positions` |
| Spatial Consistency | 20% | `trip_positions` (lat/lng) |
| Geofence Validation | 30% | `route_stops_info` JSON |
| Cross-Trip Consistency | 20% | `trips` (same route/vendor) |

### 5.1 Temporal Coverage

```python
# Coverage rate = actual pings / expected pings
expected_pings = trip_duration_sec / polling_interval_s
coverage_rate  = gps_points_observed / expected_pings

# Max gap = largest timestamp gap within a trip
trip_positions = trip_positions.sort_values(['trip_id', 'timestamp_utc'])
trip_positions['gap_sec'] = trip_positions.groupby('trip_id')['timestamp_utc'] \
                                          .diff().dt.total_seconds()
max_gap_sec = trip_positions.groupby('trip_id')['gap_sec'].max()
```

### 5.2 Spatial Consistency

#### Frozen GPS Detection

```python
# Step 1: Sort by trip and time
trip_positions = trip_positions.sort_values(['trip_id', 'timestamp_utc'])

# Step 2: Get previous ping location
trip_positions['prev_lat'] = trip_positions.groupby('trip_id')['lat'].shift(1)
trip_positions['prev_lng'] = trip_positions.groupby('trip_id')['lng'].shift(1)

# Step 3: Flag frozen pings (exact coordinate match)
trip_positions['is_frozen'] = (
    (trip_positions['lat'] == trip_positions['prev_lat']) &
    (trip_positions['lng'] == trip_positions['prev_lng'])
)

# Step 4: Calculate freeze episode duration
trip_positions['frozen_episode'] = (
    trip_positions['is_frozen'] != trip_positions['is_frozen'].shift()
).cumsum()

episode_duration = trip_positions[trip_positions['is_frozen']].groupby(
    ['trip_id', 'frozen_episode']
)['timestamp_utc'].agg(freeze_start='min', freeze_end='max')

episode_duration['freeze_duration_sec'] = (
    pd.to_datetime(episode_duration['freeze_end']) -
    pd.to_datetime(episode_duration['freeze_start'])
).dt.total_seconds()

# Step 5: Frozen ratio — only abnormal freezes (> 120 seconds)
# Threshold: 120 seconds = normal bus stop duration
# Anything longer = genuine GPS failure
frozen_summary['frozen_ratio'] = (
    frozen_summary['abnormal_frozen_pings'] /  # freeze_duration_sec > 120
    frozen_summary['total_pings']
).round(4)
```

**Threshold justification:** 120 seconds (2 minutes) — normal bus stops at red lights or pickup points last up to 2 minutes. Beyond this = genuine GPS failure.

#### Impossible Speed Detection

```python
# Haversine formula for great-circle distance
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

trip_positions['distance_m'] = haversine_m(
    trip_positions['prev_lat'], trip_positions['prev_lng'],
    trip_positions['lat'],      trip_positions['lng']
).round(2)

trip_positions['time_diff_sec'] = (
    trip_positions.groupby('trip_id')['timestamp_utc']
    .diff().dt.total_seconds()
)

trip_positions['speed_mph'] = (
    trip_positions['distance_m'] / trip_positions['time_diff_sec'] * 2.237
).round(2)

# Flag impossible speeds (school bus cannot exceed 80 mph)
IMPOSSIBLE_SPEED_MPH = 80
trip_positions['impossible_speed'] = trip_positions['speed_mph'] > IMPOSSIBLE_SPEED_MPH
```

**Threshold justification:** 80 mph — physical maximum speed of a school bus. Source: 4MATIV GPS reliability framework.

### 5.3 Geofence Validation

```python
# From route_stops_info JSON field in trips.csv
# stop_radius_m = 80m (from geofencing_config.json)
# points_to_complete = 2 (bus needs 2 pings within 80m to mark stop as visited)

# Stop completion rate
completion_rate = completed_stops / planned_stops

# Timing deviation
completed_diff = actual_arrival - scheduled_arrival

# End trigger flag — trip ended due to GPS loss
trip['gps_terminated'] = trip['end_trigger'].isin(['no_gps_data', 'old_last_point'])
```

### 5.4 Cross-Trip Consistency

```python
# Same route: is coverage rate an outlier that day?
# Using IQR-based outlier detection per route group
route_groups = trip_features.groupby(['route_id', 'peer_source'])['coverage_rate']
Q1 = route_groups.transform('quantile', 0.25)
Q3 = route_groups.transform('quantile', 0.75)
IQR = Q3 - Q1
trip_features['ct_flag_low_coverage'] = trip_features['coverage_rate'] < (Q1 - 1.5 * IQR)
```

---

## 6. Execution Score

Calculated only for trips with sufficient GPS reliability.

**Formula:**
```
execution_score = (completion_component × 0.40) +
                  (ontime_component    × 0.40) +
                  (sequence_component  × 0.20)
```

### 6.1 Components

```python
# Stop Completion Rate (40%)
exec_completion_rate = completed_stops / planned_stops

# On-Time Rate (40%) — ±10 minute window
ontime_rate = (stops where |delay_min| <= 10) / total_planned_stops

# Sequence Score (20%)
# Fraction of consecutive stop pairs visited in correct order
# Takes max of ascending vs descending to handle to_school vs from_school
sequence_score = max(ascending_matches, descending_matches) / total_pairs
```

### 6.2 Stop-Level Adjusted Completion Rate

```python
# A stop is truly successful only if:
# 1. visited = True AND
# 2. delay_min between -5 and +5 minutes

trip_adjusted = df.groupby('trip_id').agg(
    total_stops=('route_stop_id', 'count'),
    successful_stops=('delay_min', lambda x: (
        (df.loc[x.index, 'visited'] == True) &
        (x.between(-5, 5))
    ).sum())
).reset_index()

trip_adjusted['adjusted_completion_rate'] = (
    trip_adjusted['successful_stops'] / trip_adjusted['total_stops']
).round(4)
# Median adjusted completion rate = 0.2857 (±5 min window)
# Median adjusted completion rate = 0.5000 (±10 min window)
```

---

## 7. Confidence Score

```
confidence_score = Route Execution Score × GPS Reliability Multiplier
```

| GPS Health Flag | Multiplier |
|----------------|------------|
| healthy | 1.0 |
| intermittent | 0.8 |
| severe_gap | 0.5 |
| very_sparse | 0.3 |
| no_gps | NULL |

**Known limitation:** GPS multiplier is at trip level not stop level.

---

## 8. Key EDA Findings

### 8.1 GPS Data Quality

| Metric | Value |
|--------|-------|
| Total GPS pings | 2,351,477 |
| Frozen pings (all) | 209,423 (9%) |
| Abnormal frozen pings (>120s) | ~134,906 (5.7%) |
| Impossible speed pings | 8,734 (0.37%) |
| Trips with no GPS | 4,059 (27.3%) |
| Trips with high reliability | 8,413 (56.6%) |

### 8.2 Frozen GPS by Provider

| GPS Provider | Total Pings | Frozen Pings | Overall Frozen Rate |
|-------------|-------------|--------------|-------------------|
| iCabbi | 87,446 | 13,075 | **14.95%** |
| OSG | 938,275 | 85,959 | **9.16%** |
| Samsara | 1,236,698 | 35,188 | 2.85% |
| Synovia | 72,992 | 675 | 0.92% |
| Zonar | 16,066 | 9 | 0.06% |

### 8.3 Impossible Speed by Provider

| GPS Provider | Overall Impossible Rate |
|-------------|------------------------|
| Zonar | **0.78%** |
| Synovia | 0.24% |
| Samsara | 0.11% |
| OSG | 0.10% |
| iCabbi | 0.09% |

**Key insight:** Zonar has almost no freezing but highest impossible speed — GPS jumps to wrong locations rather than freezing. iCabbi and OSG suffer primarily from freezing.

### 8.4 Confidence Score Distribution

| Score Range | Trips | % |
|-------------|-------|---|
| 0–20 | 2,682 | 27.7% |
| 20–40 | 4,619 | 47.7% |
| 40–60 | 1,506 | 15.6% |
| 60–80 | 672 | 6.9% |
| 80–100 | 207 | 2.1% |

**Average confidence score = 32.7/100** — only 2% of trips scored above 80.

### 8.5 Confidence Score by GPS Provider

| GPS Provider | Avg Confidence Score | % High (>80) | % Low (<40) |
|-------------|---------------------|--------------|-------------|
| Zonar | 9.68 | 0.81% | 99.19% |
| Synovia | 22.84 | 0.00% | 87.82% |
| iCabbi | 30.93 | 1.40% | 73.12% |
| Samsara | 32.87 | 1.41% | 69.11% |
| OSG | 35.17 | 3.91% | 64.63% |

### 8.6 Stop Completion Analysis

| Metric | Value |
|--------|-------|
| Total stop-level records | 102,636 |
| Visited stops | ~82,738 (80.6%) |
| Missed stops (arrival = NULL) | 19,898 (19.4%) |
| Median adjusted completion (±5 min) | 28.57% |
| Median adjusted completion (±10 min) | 50.00% |

### 8.7 On-Time Performance

| Status | Count | % |
|--------|-------|---|
| on_time | 22,892 | 27.6% |
| very_late | 22,000 | 26.5% |
| very_early | 13,577 | 16.4% |
| late | 14,318 | 17.3% |
| early | 9,951 | 12.0% |

**Max delay observed: 139 minutes** — indicative of wrong device assignment or GPS data corruption.

---

## 9. Geofencing Configuration

```json
{
  "school_radius_m": 300,
  "stop_radius_m": 80,
  "looping_stop_radius_m": 200,
  "eta_radius_m": 1000,
  "points_to_complete": 2,
  "points_to_end": 3,
  "complete_threshold_minutes": -20,
  "school_stop_threshold_minutes": 20,
  "missed_stops_allowance": 3,
  "provider_polling_intervals": {
    "samsara": 20,
    "osg": 20,
    "synovia": 30,
    "zonar": 60,
    "icabbi": 60
  }
}
```

---

## 10. Dashboard Design

**Tool:** Streamlit + Plotly

### Page 1 — "Can we trust our GPS data?"
- 1 KPI card: Most problematic GPS provider
- 1 Horizontal bar: GPS reliability tier breakdown (5 tiers, color coded)
- 1 Stacked bar over time: Monthly tier distribution trend

### Page 2 — "How well are routes executing?"
- Dropdown: GPS Provider filter (All / OSG / Samsara / iCabbi / Synovia / Zonar)
- 3 KPI cards: Avg execution score, avg stop completion, avg on-time rate
- Grouped bar chart: Stop completion vs on-time rate by vendor
- Stacked bar over time: Execution score trend

### Page 3 — "Which vendors need immediate attention?"
- 2 Sliders: GPS Reliability threshold (0–100), Execution Score threshold (0–100)
- Scatter plot: GPS reliability vs execution score per vendor (color coded by threshold)
- Horizontal bar chart: Vendor ranking by GPS reliability

---

## 11. Key Anomaly Types Defined

| Anomaly Type | Source Component | Definition |
|-------------|-----------------|------------|
| `no_gps` | Temporal | Zero GPS pings for entire trip |
| `large_gap` | Temporal | Any gap > 300 seconds |
| `low_coverage` | Temporal | Actual pings << expected pings |
| `frozen_gps` | Spatial | Freeze episode > 120 seconds |
| `impossible_speed` | Spatial | Speed > 80 mph between pings |
| `school_stop_missed` | Geofence | is_school = True AND visited = False |
| `gps_loss_termination` | Geofence | end_trigger = no_gps_data or old_last_point |
| `outlier_trip` | Cross-trip | Coverage rate is IQR outlier vs same route peers |

---

## 12. Design Decisions & Assumptions

| Decision | Rationale |
|----------|-----------|
| Freeze threshold = 120 seconds | Normal bus stops last ≤ 2 minutes |
| Impossible speed = 80 mph | Physical maximum speed of a school bus |
| On-time window = ±5 min (stop level) | Based on arrival_status classification in cleaned data |
| On-time window = ±10 min (execution score) | Per 4MATIV operational standard |
| GPS multiplier at trip level | Stop-level GPS reliability requires raw trip_positions join — accepted as known limitation |
| Median over mean for frozen ratio | Extreme outliers inflate mean unfairly |
| Overall frozen rate over per-trip average | Raw totals more honest than averaged ratios |
| No PCA | Project is exploratory/unsupervised, only 4 components, interpretability critical |
| NULL for no_gps trips | Cannot validate = unknown, not failure |

---

*Generated from EDA Live Case conversation — May 2026*
