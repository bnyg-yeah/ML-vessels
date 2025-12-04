import argparse
from datetime import datetime
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


SAMPLE_SIZE = 60000
CHUNKSIZE = 200000

EPS = 0.4
MIN_SAMPLES = 30

RANDOM_STATE = 42


def haversine_nm(lat1, lon1, lat2, lon2):
    R_nm = 3440.065
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R_nm * c


def build_behavior_sample(path):
    rng = np.random.default_rng(RANDOM_STATE)

    last_sog = {}
    last_lat = {}
    last_lon = {}
    last_time = {}

    rows = []

    usecols = ["mmsi", "longitude", "latitude", "sog", "cog", "base_date_time"]

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNKSIZE):
        if "base_date_time_parsed" not in chunk.columns:
            chunk["base_date_time_parsed"] = pd.to_datetime(chunk["base_date_time"], errors="coerce")

        for row in chunk.itertuples(index=False):
            mmsi = getattr(row, "mmsi")
            sog_raw = getattr(row, "sog")
            lon_raw = getattr(row, "longitude")
            lat_raw = getattr(row, "latitude")
            t_raw = getattr(row, "base_date_time_parsed")
            cog_raw = getattr(row, "cog")

            try:
                sog_now = float(sog_raw)
            except (TypeError, ValueError):
                sog_now = None

            try:
                lon_now = float(lon_raw)
                lat_now = float(lat_raw)
            except (TypeError, ValueError):
                lon_now = None
                lat_now = None

            if pd.isna(t_raw):
                time_now = None
            else:
                if isinstance(t_raw, datetime):
                    time_now = t_raw
                else:
                    time_now = None

            prev_sog = last_sog.get(mmsi)
            prev_lon = last_lon.get(mmsi)
            prev_lat = last_lat.get(mmsi)
            prev_time = last_time.get(mmsi)

            if (
                sog_now is not None
                and prev_sog is not None
                and lon_now is not None
                and lat_now is not None
                and prev_lon is not None
                and prev_lat is not None
                and time_now is not None
                and prev_time is not None
            ):
                delta_sog = sog_now - prev_sog
                distance_nm = haversine_nm(prev_lat, prev_lon, lat_now, lon_now)

                delta_seconds = (time_now - prev_time).total_seconds()
                if delta_seconds > 0.0:
                    time_gap_hours = delta_seconds / 3600.0
                else:
                    time_gap_hours = 0.0

                if time_gap_hours > 0.0:
                    implied_speed = distance_nm / time_gap_hours
                else:
                    implied_speed = 0.0

                rows.append(
                    {
                        "mmsi": mmsi,
                        "longitude": lon_now,
                        "latitude": lat_now,
                        "sog_now": sog_now,
                        "sog_prev": prev_sog,
                        "delta_sog": delta_sog,
                        "distance_nm": distance_nm,
                        "time_gap_hours": time_gap_hours,
                        "implied_speed_knots": implied_speed,
                        "cog": cog_raw,
                        "base_date_time": getattr(row, "base_date_time"),
                    }
                )

            if sog_now is not None:
                last_sog[mmsi] = sog_now
            if lon_now is not None and lat_now is not None:
                last_lon[mmsi] = lon_now
                last_lat[mmsi] = lat_now
            if time_now is not None:
                last_time[mmsi] = time_now

    if not rows:
        raise ValueError("No behavioral rows constructed. Check input data.")

    return pd.DataFrame(rows)



def run_dbscan_on_behavior(df):
    feature_cols = [
        "sog_now",
        "sog_prev",
        "delta_sog",
        "distance_nm",
        "time_gap_hours",
        "implied_speed_knots",
    ]

    feature_matrix = df[feature_cols].to_numpy()

    finite_mask = np.isfinite(feature_matrix).all(axis=1)
    df_clean = df.loc[finite_mask].copy()
    feature_matrix_clean = feature_matrix[finite_mask]

    if feature_matrix_clean.shape[0] == 0:
        raise ValueError("No finite behavioral rows left after NaN filtering.")

    print(f"Total behavioral rows: {df.shape[0]}")
    print(f"Rows kept after finite filter: {df_clean.shape[0]}")
    print(f"Rows dropped: {df.shape[0] - df_clean.shape[0]}")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix_clean)

    model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, n_jobs=1)
    labels = model.fit_predict(scaled)
    df_clean["dbscan_behavior_label"] = labels

    plt.figure(figsize=(8, 6))
    plt.scatter(df_clean["sog_now"], df_clean["sog_prev"], c=labels, s=5)
    plt.xlabel("sog_now (knots)")
    plt.ylabel("sog_prev (knots)")
    plt.title("DBSCAN Clusters on Behavior Features")
    plt.tight_layout()
    plt.savefig("dbscan_feature_space.png")
    plt.show()



    plt.figure(figsize=(8, 6))
    plt.scatter(df_clean["longitude"], df_clean["latitude"], c=labels, s=5)
    plt.xlabel("longitude (degrees)")
    plt.ylabel("latitude (degrees)")
    plt.title("Clustered Vessel Positions")
    plt.tight_layout()
    plt.savefig("dbscan_geographic.png")
    plt.show()


    return df_clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="dbscan_behavior_output.csv")
    args = parser.parse_args()

    behavior_df = build_behavior_sample(args.input_csv)
    result_df = run_dbscan_on_behavior(behavior_df)
    result_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
