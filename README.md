# 🛰️ GPS Reliability & Route Execution Analytics — 4MATIV Live Case

**Can we trust incoming GPS data to confirm that school-bus routes are executed as planned?**
An end-to-end analytics project that turns 2.3M raw GPS pings into a trip-level **Confidence Score** and a 3-page decision dashboard — built for a real client, **4MATIV Technologies**, as part of the University of Minnesota (Carlson School of Management) MSBA program.

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/pandas-EDA-150458" />
  <img src="https://img.shields.io/badge/scikit--learn-BallTree-orange" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red" />
  <img src="https://img.shields.io/badge/Plotly-Viz-3f4f75" />
</p>

---

## 📌 Business Problem

4MATIV coordinates school transportation across many third-party vendors, each using different GPS providers and polling intervals. Before they can trust route-execution reporting, they first need to know **whether the GPS data itself is reliable**. Bad GPS looks the same as a bad route — so the business question was:

> *"Validate the incoming GPS data to evaluate whether a route is actually being executed as planned, and give us a way to monitor GPS unit health with confidence metrics and anomaly alerts."*

**Why it matters:** unreliable GPS = unverifiable service, disputed vendor performance, and safety blind spots. A trustworthy score lets 4MATIV hold vendors accountable and flag failing devices proactively.

---

## 🎯 What I Delivered

- A reproducible **Python data pipeline** joining 12 relational tables into a stop-level master table (102,636 rows) and a trip-level scorecard (14,859 trips).
- A **GPS Reliability Framework** (0–100) built from 4 weighted components.
- A **Route Execution Score** and a combined **Confidence Score** = execution × GPS-reliability multiplier.
- A **3-page Streamlit + Plotly dashboard** answering the client's three real questions.
- A vendor-level scorecard ranking GPS providers so 4MATIV knows exactly **which vendors need attention**.

---

## 📊 Data

| Source | Rows | What it is |
|---|---|---|
| `trip_positions` | 2,351,477 | Raw GPS pings (lat, lng, timestamp) |
| `trips` | 14,859 | One row per trip/day/direction |
| `route_stops` / `stop_times` | ~1,936 / ~260K | Planned stops and schedules |
| `vendors` | 16 | Vendor labels, GPS providers, polling intervals |
| **→ `trip_scores.csv`** | 14,859 | **Final trip-level reliability + execution + confidence** |
| **→ `vendor_summary.csv`** | 16 | **Final vendor scorecard** |

*Nulls were treated as signal, not noise — a missing GPS timestamp is a real operational failure and one of the anomalies we detect.*

---

## 🧠 Approach

**GPS Reliability Score (0–100)** — four weighted components:

| Component | Weight | Idea |
|---|---|---|
| Temporal Coverage | 30% | Are we getting pings as often as we should? (actual vs expected pings, max gap) |
| Spatial Consistency | 20% | Frozen-GPS detection (>120s freeze) + impossible-speed detection (>80 mph) via Haversine |
| Geofence Validation | 30% | Did the bus actually hit each stop within an 80m radius? |
| Cross-Trip Consistency | 20% | Is this trip an IQR outlier vs other trips on the same route? |

**Execution Score** = 40% stop completion + 40% on-time (±10 min) + 20% stop-sequence correctness.

**Confidence Score** = Execution Score × GPS-reliability multiplier (healthy 1.0 → no_gps NULL), so a great-looking route on bad GPS is correctly discounted.

Full methodology, thresholds, and rationale: [`4MATIV_EDA_Technical_Summary.md`](4MATIV_EDA_Technical_Summary.md).

---

## 🔑 Key Findings

- **Only 2% of trips scored above 80** on confidence; the average was **32.7/100** — GPS trust was far lower than assumed.
- **27.3% of trips had no usable GPS data at all.**
- Providers fail differently:
  - **iCabbi (15%) and OSG (9%)** suffer mainly from **frozen GPS**.
  - **Zonar** rarely freezes but has the **highest impossible-speed rate** — it *jumps* to wrong locations instead.
- **On-time performance:** only **27.6%** of stops were on time; max observed delay was **139 minutes** (a sign of wrong device assignment / corrupted data).

| GPS Provider | Avg Confidence | % Low (<40) |
|---|---|---|
| OSG | 35.2 | 64.6% |
| Samsara | 32.9 | 69.1% |
| iCabbi | 30.9 | 73.1% |
| Synovia | 22.8 | 87.8% |
| Zonar | 9.7 | 99.2% |

---

## 📈 Dashboard

Built in **Streamlit + Plotly**, structured around the client's three questions:

1. **"Can we trust our GPS data?"** — provider risk KPI, reliability-tier breakdown, monthly trend.
2. **"How well are routes executing?"** — execution/completion/on-time KPIs, vendor comparison.
3. **"Which vendors need immediate attention?"** — interactive thresholds + a reliability-vs-execution scatter to triage vendors.

Run it locally:

```bash
pip install -r requirements.txt
streamlit run gps_dashboard.py
```

---

## 🗂️ Repository Structure

```
gps-reliability-4mativ/
├── README.md
├── GPS_Reliability_Execution_v6.ipynb   # full pipeline: cleaning → scoring
├── gps_dashboard.py                      # Streamlit + Plotly app
├── 4MATIV_EDA_Technical_Summary.md       # methodology & thresholds
├── 4Mativ_Presentation_G7.pptx           # final client presentation
├── Diagnosing_GPS_Data_Reliability.pdf   # written report
└── requirements.txt
```

---

## 🛠️ Skills Demonstrated

`Data Cleaning` · `Relational Joins` · `Feature Engineering` · `Geospatial Analysis (Haversine, BallTree)` · `Anomaly Detection` · `KPI/Score Design` · `Data Visualization` · `Dashboarding (Streamlit/Plotly)` · `Stakeholder Communication`

---

## 👤 About

**Team:** Group 7, Carlson MSBA — Exploratory Data Analysis live case (Spring 2026).
**My contribution:** _[Add 1–2 lines on what you specifically owned — e.g., built the GPS reliability scoring pipeline and the vendor scorecard / designed the Streamlit dashboard.]_

Built by **Parul Chaudhary** · [LinkedIn](#) · [Email](mailto:parul.jaswant@gmail.com)

> *Client data used with permission. Analysis and framework presented for portfolio purposes.*
