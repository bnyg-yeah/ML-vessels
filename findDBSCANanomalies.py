import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("--output", default="dbscan_anomaly_list.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    anomalies = df[df["dbscan_behavior_label"] == -1]
    anomalies.to_csv(args.output, index=False)
    print(anomalies.to_string(index=False))

if __name__ == "__main__":
    main()
