import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
df = pd.read_csv("input.csv", low_memory=False)

# Ensure the columns exist
expected_cols = [
    'mmsi', 'base_date_time', 'longitude', 'latitude', 'sog', 'cog', 'heading',
    'vessel_name', 'imo', 'call_sign', 'vessel_type', 'status', 'length',
    'width', 'draft', 'cargo', 'transceiver'
]

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    print("Missing columns:", missing)
    raise SystemExit()


# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────

# Convert timestamp to datetime
df["base_date_time"] = pd.to_datetime(df["base_date_time"], errors="coerce")

# Sort by vessel and time (important for speed/heading deltas)
df = df.sort_values(["mmsi", "base_date_time"])

# Numeric conversion (non-numeric become NaN)
num_cols = ['longitude', 'latitude', 'sog', 'cog', 'heading',
            'length', 'width', 'draft']

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=num_cols + ["base_date_time"])


# ─────────────────────────────────────────────
# 3. Feature Engineering (based on literature)
#    Context from your papers:
#    Ford 2018 → speed anomalies, course changes
#    Radhakrishnan 2023 → abrupt turns, speed deviations
#    Rodríguez 2024 → behavioral changes & movement irregularities
# ─────────────────────────────────────────────

# Speed change (|ΔSOG|)
df["delta_sog"] = df.groupby("mmsi")["sog"].diff().abs()

# Heading change (|Δheading|)
df["delta_heading"] = df.groupby("mmsi")["heading"].diff().abs()

# Course change (|ΔCOG|)
df["delta_cog"] = df.groupby("mmsi")["cog"].diff().abs()

# Time delta (in seconds)
df["delta_time"] = df.groupby("mmsi")["base_date_time"].diff().dt.total_seconds()

# Replace NaNs created by diff()
df.fillna(0, inplace=True)


# ─────────────────────────────────────────────
# 4. Select Features for Isolation Forest
#    These match what the AIS anomaly papers use:
#    - speed, heading, course, lat/lon (Ford 2018)
#    - change features (Rodriguez 2024)
#    - vessel metadata (length, width, draft)
# ─────────────────────────────────────────────

feature_cols = [
    "longitude", "latitude",
    "sog", "cog", "heading",
    "delta_sog", "delta_heading", "delta_cog", "delta_time",
    "length", "width", "draft"
]

X = df[feature_cols].copy()


# ─────────────────────────────────────────────
# 5. Scale numerical features
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ─────────────────────────────────────────────
# 6. Train Isolation Forest
#    contamination = expected % anomalies
#    Papers suggest 1–5% anomaly rate for AIS data
# ─────────────────────────────────────────────

model = IsolationForest(
    n_estimators=200,
    contamination=0.02,  # assume ~2% anomalies
    random_state=42,
    n_jobs=-1
)

model.fit(X_scaled)

# Predictions
df["anomaly_score"] = model.decision_function(X_scaled)
df["anomaly"] = model.predict(X_scaled)   # -1 = anomaly, +1 = normal


# ─────────────────────────────────────────────
# 7. Save results
# ─────────────────────────────────────────────
df.to_csv("ais-2025-06-01-with-anomalies.csv", index=False)

print("Isolation Forest complete.")
print(df["anomaly"].value_counts())
print("Output saved to ais-2025-06-01-with-anomalies.csv")
