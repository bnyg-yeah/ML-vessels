import pandas as pd
import numpy as np
from datetime import time
from sklearn.neighbors import NearestNeighbors

# ============================================================
#  LOAD DATA
# ============================================================

df = pd.read_csv("ais-2025-06-01-with-anomalies.csv", low_memory=False)

if "anomaly" not in df.columns:
    raise ValueError("Missing 'anomaly' column. Run Isolation Forest first.")

# Convert timestamp
df["base_date_time"] = pd.to_datetime(df["base_date_time"], errors="coerce")

# Drop rows without valid positions
df = df.dropna(subset=["latitude", "longitude"])


# ============================================================
# 1. VESSEL RISK SCORING
# ============================================================

print("\n Calculating vessel risk scores...")

risk_table = (
    df.groupby("mmsi")
      .agg(
          total_points=("anomaly", "count"),
          anomalies=("anomaly", lambda x: (x == -1).sum())
      )
)

risk_table["risk_score"] = risk_table["anomalies"] / risk_table["total_points"]
risk_table = risk_table.sort_values("risk_score", ascending=False)


# ============================================================
# 2. BEHAVIORAL ANOMALY SUMMARY
# ============================================================

print("Summarizing anomaly behavior features...")

behavior_summary = (
    df[df["anomaly"] == -1]
    .groupby("mmsi")
    .agg(
        avg_delta_sog=("delta_sog", "mean"),
        avg_delta_heading=("delta_heading", "mean"),
        avg_delta_cog=("delta_cog", "mean"),
        avg_delta_time=("delta_time", "mean"),
        avg_speed=("sog", "mean"),
        avg_heading=("heading", "mean"),
    )
)


# ============================================================
# 3. TIME-OF-DAY ANOMALY PATTERNS
# ============================================================

print("Detecting nighttime anomalies...")

def is_night(ts):
    if pd.isna(ts):
        return False
    return ts.time() < time(6,0) or ts.time() > time(20,0)

df["is_night"] = df["base_date_time"].apply(is_night)

night_summary = (
    df[df["anomaly"] == -1]
    .groupby("mmsi")
    .agg(
        night_anomalies=("is_night", "sum"),
        total_anomalies=("anomaly", lambda x: (x == -1).sum())
    )
)

night_summary["night_ratio"] = night_summary["night_anomalies"] / night_summary["total_anomalies"]


# ============================================================
# 4. ROUTE DEVIATION CHECK (via bounding box outlier)
# ============================================================

print("Checking for large route deviations...")

route_deviation = (
    df.groupby("mmsi")
      .agg(
          lat_std=("latitude", "std"),
          lon_std=("longitude", "std")
      )
)

route_deviation["spread_score"] = route_deviation["lat_std"] + route_deviation["lon_std"]


# ============================================================
# 5. PROXIMITY / RENDEZVOUS DETECTION
# ============================================================

print("Detecting possible rendezvous events...")

# Only anomaly points for rendezvous analysis
anom = df[df["anomaly"] == -1][["mmsi", "latitude", "longitude"]].copy()

coords = np.radians(anom[["latitude", "longitude"]])
knn = NearestNeighbors(radius=0.0005, metric="haversine")  # ~50m
knn.fit(coords)

# Count neighbors within radius
neighbors = knn.radius_neighbors(coords, return_distance=False)

rendezvous_count = [
    len(nbrs) - 1  # subtract self
    for nbrs in neighbors
]

anom["rendezvous_score"] = rendezvous_count

rendezvous_summary = (
    anom.groupby("mmsi")["rendezvous_score"].sum()
)


# ============================================================
# 6. AIS SPOOFING SIGNALS
# ============================================================

print("Checking for AIS spoofing indicators...")

spoofing_df = df[df["anomaly"] == -1].groupby("mmsi").agg(
    max_speed_jump=("delta_sog", "max"),
    max_heading_jump=("delta_heading", "max"),
    position_jumps=("latitude", lambda x: x.diff().abs().gt(0.2).sum()+ 
                                       x.diff().abs().gt(0.2).sum()),
)

spoofing_df["spoofing_score"] = (
    spoofing_df["max_speed_jump"] +
    spoofing_df["max_heading_jump"] +
    spoofing_df["position_jumps"]
)


# ============================================================
# 7. FINAL SUSPICIOUS VESSEL REPORT
# ============================================================

print("\nBuilding final suspicious vessel report...\n")

final_report = (
    risk_table
    .join(behavior_summary, how="left")
    .join(night_summary, how="left")
    .join(route_deviation, how="left")
    .join(rendezvous_summary.rename("rendezvous_score"), how="left")
    .join(spoofing_df["spoofing_score"], how="left")
)

# Fill missing progress for vessels without anomalies
final_report = final_report.fillna(0)

# Add final suspicion score
final_report["suspicion_score"] = (
    final_report["risk_score"] * 3 +
    final_report["night_ratio"] * 2 +
    final_report["spread_score"] * 1 +
    final_report["rendezvous_score"] * 2 +
    final_report["spoofing_score"] * 3
)


# Rank suspicious vessels
final_report = final_report.sort_values("suspicion_score", ascending=False)

final_report.to_csv("suspicious_vessel_report.csv")

print("Suspicious vessel analysis complete.")
print("➡ Output saved to: suspicious_vessel_report.csv\n")

print("Top 10 most suspicious vessels:")
print(final_report.head(10))
