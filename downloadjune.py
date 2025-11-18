import os
import requests
from datetime import datetime, timedelta

def download_ais_files(start_date, end_date, folder_name):
    """
    Downloads AIS .csv.zst files from NOAA for dates in range.
    Creates a folder in the current directory automatically.
    """

    # Create folder in current working directory
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, folder_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading files into: {output_dir}")

    current = start_date
    while current <= end_date:
        fname = f"ais-{current.strftime('%Y-%m-%d')}.csv.zst"
        url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{current.year}/{fname}"
        target_path = os.path.join(output_dir, fname)

        print(f"Downloading {url} → {target_path}")

        try:
            resp = requests.get(url, stream=True)
            resp.raise_for_status()

            with open(target_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"  → Success")

        except requests.HTTPError as e:
            print(f"  → HTTP Error: {e}")
        except Exception as e:
            print(f"  → Error: {e}")

        current += timedelta(days=1)


if __name__ == "__main__":
    start = datetime(2025, 6, 1)
    end   = datetime(2025, 6, 30)

    download_ais_files(start, end, folder_name="ais_2025_06")