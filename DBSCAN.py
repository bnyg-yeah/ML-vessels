#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Clustering configuration
DEFAULT_SAMPLE_SIZE = 200000
DEFAULT_CHUNKSIZE = 200000
DEFAULT_EPS = 0.5
DEFAULT_MIN_SAMPLES = 25
DEFAULT_RANDOM_STATE = 42

# AIS Data Quality Thresholds
MAX_VALID_SOG = 102.3
MAX_REALISTIC_SPEED = 50.0
MAX_TIME_GAP_HOURS = 24.0
MIN_TIME_GAP_SECONDS = 1.0
SPOOFING_SPEED_RATIO = 2.0
POSITION_JUMP_NM = 50.0

# Geographic Calculations
def haversine_nm_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized Haversine calculation returning nautical miles."""
    R_nm = 3440.065
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R_nm * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# Data Loading and Preprocessing
def load_and_preprocess(path, chunksize, sample_size, random_state):
    """Load AIS data with quality filtering and feature engineering."""
    
    print(f"Loading data from: {path}")
    usecols = ["mmsi", "longitude", "latitude", "sog", "cog", "base_date_time"]
    
    all_chunks = []
    for chunk_num, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize)):
        chunk["base_date_time_parsed"] = pd.to_datetime(chunk["base_date_time"], errors="coerce")
        for col in ["sog", "cog", "latitude", "longitude"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        
        # Quality filters
        valid = (
            chunk["sog"].notna() & chunk["latitude"].notna() & 
            chunk["longitude"].notna() & chunk["base_date_time_parsed"].notna() &
            (chunk["sog"] >= 0) & (chunk["sog"] <= MAX_VALID_SOG) &
            (chunk["latitude"].abs() > 0.1) & (chunk["longitude"].abs() > 0.1) &
            (chunk["latitude"] >= -90) & (chunk["latitude"] <= 90) &
            (chunk["longitude"] >= -180) & (chunk["longitude"] <= 180)
        )
        all_chunks.append(chunk.loc[valid].copy())
        print(f"  Chunk {chunk_num + 1}: {len(chunk):,} rows -> {valid.sum():,} valid")
        
        if sum(len(c) for c in all_chunks) >= sample_size * 2:
            break
    
    if not all_chunks:
        raise ValueError("No valid data found.")
    
    df = pd.concat(all_chunks, ignore_index=True)
    print(f"Total valid rows: {len(df):,}")
    
    # Sort by vessel and time (critical!)
    df = df.sort_values(["mmsi", "base_date_time_parsed"]).reset_index(drop=True)
    
    # Compute previous values per vessel
    grp = df.groupby("mmsi")
    df["sog_prev"] = grp["sog"].shift(1)
    df["cog_prev"] = grp["cog"].shift(1)
    df["lat_prev"] = grp["latitude"].shift(1)
    df["lon_prev"] = grp["longitude"].shift(1)
    df["time_prev"] = grp["base_date_time_parsed"].shift(1)
    
    # Drop first observation per vessel (no previous)
    df = df.dropna(subset=["sog_prev", "lat_prev", "lon_prev", "time_prev"])
    
    # Time and distance
    df["time_delta_sec"] = (df["base_date_time_parsed"] - df["time_prev"]).dt.total_seconds()
    df = df[df["time_delta_sec"] >= MIN_TIME_GAP_SECONDS].copy()
    df["time_gap_hours"] = df["time_delta_sec"] / 3600.0
    
    df["distance_nm"] = haversine_nm_vectorized(
        df["lat_prev"], df["lon_prev"], df["latitude"], df["longitude"]
    )
    
    # Implied speed from position change
    df["implied_speed"] = np.where(
        df["time_gap_hours"] > 0,
        df["distance_nm"] / df["time_gap_hours"],
        0.0
    )
    
    # Speed change
    df["delta_sog"] = df["sog"] - df["sog_prev"]
    
    # Course change (handle 360° wraparound)
    cog_diff = np.abs(df["cog"] - df["cog_prev"])
    df["delta_cog"] = np.minimum(cog_diff, 360 - cog_diff)
    
    # Speed discrepancy (key spoofing indicator)
    df["speed_discrepancy"] = np.abs(df["sog"] - df["implied_speed"])
    
    # Random sample if needed
    if len(df) > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(df), size=sample_size, replace=False)
        df = df.iloc[idx].copy()
    
    print(f"Final sample: {len(df):,} rows")
    return df


# Anomaly Detection
def compute_anomaly_flags(df, max_time_gap, spoofing_ratio, max_speed, jump_threshold):
    """Flag suspicious behaviors."""
    
    # Dark activity: AIS turned off for extended period
    df["flag_dark"] = df["time_gap_hours"] > max_time_gap
    
    # Spoofing: reported speed doesn't match calculated speed from positions
    # If vessel reports 20 knots but positions show 5 knots (or vice versa), suspicious
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(df["sog"] > 0.5, df["implied_speed"] / df["sog"], 1.0)
    df["flag_spoofing"] = ((ratio > spoofing_ratio) | (ratio < 1/spoofing_ratio)) & (df["sog"] > 1.0)
    
    # Position jump: teleported unrealistically far
    df["flag_jump"] = df["distance_nm"] > jump_threshold
    
    # Speed anomaly: implied speed exceeds any real vessel capability
    df["flag_speed_anomaly"] = df["implied_speed"] > max_speed
    
    # Combined: any anomaly
    df["is_anomaly"] = df["flag_dark"] | df["flag_spoofing"] | df["flag_jump"] | df["flag_speed_anomaly"]
    
    print(f"\nAnomaly flags:")
    print(f"  Dark activity:  {df['flag_dark'].sum():,}")
    print(f"  Spoofing:       {df['flag_spoofing'].sum():,}")
    print(f"  Position jump:  {df['flag_jump'].sum():,}")
    print(f"  Speed anomaly:  {df['flag_speed_anomaly'].sum():,}")
    print(f"  Any anomaly:    {df['is_anomaly'].sum():,} ({100*df['is_anomaly'].mean():.1f}%)")
    
    return df


def run_dbscan(df, eps, min_samples):
    """
    DBSCAN clusters behavioral patterns. 
    Noise points (label=-1) are outliers - potential anomalies.
    Small clusters may also indicate unusual behavior.
    """
    
    print(f"\nRunning DBSCAN (eps={eps}, min_samples={min_samples})...")
    
    features = ["sog", "delta_sog", "delta_cog", "time_gap_hours", "implied_speed", "speed_discrepancy"]
    X = df[features].to_numpy()
    
    # Remove non-finite rows
    finite_mask = np.isfinite(X).all(axis=1)
    df_clean = df.loc[finite_mask].copy()
    X_clean = X[finite_mask]
    
    if len(X_clean) == 0:
        raise ValueError("No valid rows for clustering.")
    
    # Standardize and cluster
    X_scaled = StandardScaler().fit_transform(X_clean)
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(X_scaled)
    df_clean["cluster"] = labels
    
    # DBSCAN noise points (label=-1) are statistical outliers
    df_clean["dbscan_outlier"] = labels == -1
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Clusters: {n_clusters}, Noise/outliers: {n_noise:,} ({100*n_noise/len(labels):.1f}%)")
    
    # Combine: flag as suspicious if rule-based anomaly OR dbscan outlier
    df_clean["suspicious"] = df_clean["is_anomaly"] | df_clean["dbscan_outlier"]
    print(f"  Total suspicious records: {df_clean['suspicious'].sum():,}")
    
    return df_clean


# Visualization 
def create_visualizations(df, output_dir):
    """Generate 3 focused, interpretable graphs."""
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving visualizations to: {output_dir}")
    
    # Colors
    COLOR_NORMAL = "#2E86AB"      # Blue
    COLOR_SUSPICIOUS = "#E94F37"  # Red
    
    # Graph 1: Reported SOG vs Implied Speed (Spoofing Detection)
    # Points should cluster near the diagonal. Off-diagonal = suspicious.
    fig, ax = plt.subplots(figsize=(10, 8))
    
    normal = df[~df["suspicious"]]
    suspicious = df[df["suspicious"]]
    
    ax.scatter(normal["sog"], normal["implied_speed"], 
               c=COLOR_NORMAL, s=5, alpha=0.3, label=f"Normal ({len(normal):,})")
    ax.scatter(suspicious["sog"], suspicious["implied_speed"], 
               c=COLOR_SUSPICIOUS, s=15, alpha=0.7, label=f"Suspicious ({len(suspicious):,})")
    
    # Reference line: if no spoofing, reported SOG ≈ implied speed
    max_val = min(60, df["sog"].quantile(0.99) * 1.2)
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, label="Expected (no spoofing)")
    
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, min(100, df["implied_speed"].quantile(0.99) * 1.2))
    ax.set_xlabel("Reported Speed (SOG, knots)", fontsize=12)
    ax.set_ylabel("Implied Speed from Position (knots)", fontsize=12)
    ax.set_title("Spoofing Detection: Reported vs Actual Speed\n"
                 "Points far from diagonal indicate GPS/AIS manipulation", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_spoofing_detection.png"), dpi=150)
    plt.close()
    
    # Graph 2: Geographic Map of Suspicious Activity
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(normal["longitude"], normal["latitude"],
               c=COLOR_NORMAL, s=3, alpha=0.2, label=f"Normal ({len(normal):,})")
    ax.scatter(suspicious["longitude"], suspicious["latitude"],
               c=COLOR_SUSPICIOUS, s=20, alpha=0.8, label=f"Suspicious ({len(suspicious):,})")
    
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title("Geographic Distribution of Suspicious AIS Activity", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_geographic_anomalies.png"), dpi=150)
    plt.close()
    
    # Graph 3: Anomaly Breakdown by Type
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ["Spoofing", "Dark Activity", "Position Jump", "Speed Anomaly", "DBSCAN Outlier"]
    counts = [
        df["flag_spoofing"].sum(),
        df["flag_dark"].sum(),
        df["flag_jump"].sum(),
        df["flag_speed_anomaly"].sum(),
        df["dbscan_outlier"].sum()
    ]
    
    colors = ["#E94F37", "#3C5488", "#F39B7F", "#00A087", "#808080"]
    bars = ax.barh(categories, counts, color=colors, edgecolor="white", height=0.6)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                f"{count:,}", va="center", fontsize=11)
    
    ax.set_xlabel("Number of Flagged Records", fontsize=12)
    ax.set_title("Anomaly Breakdown by Detection Method", fontsize=13)
    ax.set_xlim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_anomaly_breakdown.png"), dpi=150)
    plt.close()
    
    print("  Saved: 1_spoofing_detection.png, 2_geographic_anomalies.png, 3_anomaly_breakdown.png")


# Output Generation
def generate_suspicious_vessels_csv(df, output_dir):
    """Create CSV listing suspicious vessels ranked by suspicion level."""
    
    # Aggregate by vessel
    vessel_stats = df.groupby("mmsi").agg(
        total_records=("mmsi", "count"),
        suspicious_records=("suspicious", "sum"),
        spoofing_flags=("flag_spoofing", "sum"),
        dark_flags=("flag_dark", "sum"),
        jump_flags=("flag_jump", "sum"),
        speed_anomaly_flags=("flag_speed_anomaly", "sum"),
        dbscan_outliers=("dbscan_outlier", "sum"),
    ).reset_index()
    
    # Calculate suspicion rate
    vessel_stats["suspicion_rate"] = vessel_stats["suspicious_records"] / vessel_stats["total_records"]
    
    # Filter to only suspicious vessels and sort
    suspicious_vessels = vessel_stats[vessel_stats["suspicious_records"] > 0].copy()
    suspicious_vessels = suspicious_vessels.sort_values(
        ["suspicious_records", "suspicion_rate"], ascending=[False, False]
    )
    
    # Save
    output_path = os.path.join(output_dir, "suspicious_vessels.csv")
    suspicious_vessels.to_csv(output_path, index=False)
    
    print(f"\nSuspicious vessels: {len(suspicious_vessels):,} out of {df['mmsi'].nunique():,} total")
    print(f"Saved to: {output_path}")
    
    # Print top 10
    if len(suspicious_vessels) > 0:
        print("\nTop 10 most suspicious vessels:")
        print("-" * 80)
        for _, row in suspicious_vessels.head(10).iterrows():
            print(f"  MMSI {int(row['mmsi']):>12}: {int(row['suspicious_records']):>5} suspicious / "
                  f"{int(row['total_records']):>5} total ({100*row['suspicion_rate']:.1f}%) | "
                  f"spoof={int(row['spoofing_flags'])}, dark={int(row['dark_flags'])}, "
                  f"jump={int(row['jump_flags'])}")
    
    return suspicious_vessels


def generate_summary(df, output_dir):
    """Generate text summary."""
    
    path = os.path.join(output_dir, "analysis_summary.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("AIS ANOMALY DETECTION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Records analyzed: {len(df):,}\n")
        f.write(f"Unique vessels: {df['mmsi'].nunique():,}\n\n")
        
        f.write("ANOMALY COUNTS:\n")
        f.write(f"  Spoofing:       {df['flag_spoofing'].sum():,}\n")
        f.write(f"  Dark activity:  {df['flag_dark'].sum():,}\n")
        f.write(f"  Position jump:  {df['flag_jump'].sum():,}\n")
        f.write(f"  Speed anomaly:  {df['flag_speed_anomaly'].sum():,}\n")
        f.write(f"  DBSCAN outlier: {df['dbscan_outlier'].sum():,}\n")
        f.write(f"  Total suspicious: {df['suspicious'].sum():,} ({100*df['suspicious'].mean():.1f}%)\n")
    
    print(f"Summary saved to: {path}")


# Main
def main():
    parser = argparse.ArgumentParser(
        description="AIS Anomaly Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_csv", help="Input AIS CSV file")
    parser.add_argument("--output-dir", "-d", default="ais_output", help="Output directory")
    parser.add_argument("--eps", "-e", type=float, default=DEFAULT_EPS, help="DBSCAN eps")
    parser.add_argument("--min-samples", "-m", type=int, default=DEFAULT_MIN_SAMPLES, help="DBSCAN min_samples")
    parser.add_argument("--sample-size", "-s", type=int, default=DEFAULT_SAMPLE_SIZE, help="Max records to analyze")
    parser.add_argument("--chunksize", "-c", type=int, default=DEFAULT_CHUNKSIZE, help="CSV chunk size")
    parser.add_argument("--random-state", "-r", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed")
    parser.add_argument("--max-time-gap", type=float, default=MAX_TIME_GAP_HOURS, help="Dark activity threshold (hours)")
    parser.add_argument("--spoofing-ratio", type=float, default=SPOOFING_SPEED_RATIO, help="Spoofing speed ratio threshold")
    parser.add_argument("--max-speed", type=float, default=MAX_REALISTIC_SPEED, help="Max realistic speed (knots)")
    parser.add_argument("--jump-threshold", type=float, default=POSITION_JUMP_NM, help="Position jump threshold (nm)")
    parser.add_argument("--no-graphs", action="store_true", help="Skip graph generation")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AIS ANOMALY DETECTION")
    print("=" * 60)
    
    if not os.path.exists(args.input_csv):
        print(f"ERROR: File not found: {args.input_csv}")
        sys.exit(1)
    
    # Process
    df = load_and_preprocess(args.input_csv, args.chunksize, args.sample_size, args.random_state)
    df = compute_anomaly_flags(df, args.max_time_gap, args.spoofing_ratio, args.max_speed, args.jump_threshold)
    df = run_dbscan(df, args.eps, args.min_samples)
    
    # Outputs
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.no_graphs:
        create_visualizations(df, args.output_dir)
    
    generate_suspicious_vessels_csv(df, args.output_dir)
    generate_summary(df, args.output_dir)
    
    # Save full results
    full_output = os.path.join(args.output_dir, "full_results.csv")
    df.to_csv(full_output, index=False)
    print(f"\nFull results saved to: {full_output}")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()