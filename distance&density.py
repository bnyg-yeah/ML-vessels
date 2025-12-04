import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

# ============================================================
# LOAD THE CSV WITH ANOMALY LABELS
# ============================================================

df = pd.read_csv("ais-2025-06-01-with-anomalies.csv", low_memory=False)

# Ensure the required columns exist
required_cols = ["latitude", "longitude", "anomaly"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Drop rows with missing coordinates
df = df.dropna(subset=["latitude", "longitude"])

# Convert anomaly labels to readable form
df["is_anomaly"] = df["anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

# ============================================================
# 1. DENSITY PLOT FOR SOME KEY FEATURES
# ============================================================

features = ["sog", "cog", "heading", "delta_sog", "delta_heading", "delta_cog"]
existing_features = [f for f in features if f in df.columns]

if not existing_features:
    raise ValueError("No numeric movement features available for density plotting.")

plt.figure(figsize=(16, 10))
for i, feat in enumerate(existing_features[:6], 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(
        data=df,
        x=feat,
        hue="is_anomaly",
        fill=True,
        common_norm=False,
        alpha=0.5
    )
    plt.title(f"Density of {feat} (Anomaly vs Normal)")
    plt.xlabel(feat)
    plt.ylabel("Density")

plt.tight_layout()
plt.show()


# ============================================================
# 2. DISTANCE-TO-NEAREST-NEIGHBOR ANALYSIS
# ============================================================

print("Calculating nearest neighbor distances...")

coords = df[["latitude", "longitude"]].values
coords_rad = np.radians(coords)

# Fit Nearest Neighbors
nbrs = NearestNeighbors(n_neighbors=2, metric="haversine")
nbrs.fit(coords_rad)

distances, indices = nbrs.kneighbors(coords_rad)
# distances[:,1] gives nearest non-self distance
df["nn_distance_km"] = distances[:, 1] * 6371  # convert haversine rad → km

# Plot distribution
plt.figure(figsize=(12, 6))
sns.kdeplot(
    data=df,
    x="nn_distance_km",
    hue="is_anomaly",
    fill=True,
    common_norm=False,
    alpha=0.5
)
plt.title("Nearest Neighbor Distance Distribution (km)\nAnomaly vs Normal")
plt.xlabel("Distance to Nearest Neighbor (km)")
plt.ylabel("Density")
plt.xlim(0, df["nn_distance_km"].quantile(0.99))  # trim long tail for visibility
plt.show()


