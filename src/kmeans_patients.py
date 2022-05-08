import csv
from typing import List

from src.kmeans import DataPoint, KMeans

_AGE_HEADERS = ["10歳未満", "10代", "20代", "30代", "40代", "50代", "60代", "70代", "80代", "90代", "100歳以上"]

_DATE_HEADERS: List[str] = [f"2022-04-{str(i).zfill(2)}" for i in range(1, 31)]


if __name__ == "__main__":
    with open("data/patients.csv") as f:
        reader = csv.DictReader(f)
        rows = [{"age": row["患者_年代"], "date": row["発症_年月日"]} for row in reader]
    filtered_rows = [row for row in rows if row["age"] in _AGE_HEADERS and row["date"] in _DATE_HEADERS]
    points = [DataPoint([_AGE_HEADERS.index(row["age"]), _DATE_HEADERS.index(row["date"])]) for row in filtered_rows]
    k_means = KMeans(10, points)
    clusters = k_means.run()
    for index, cluster in enumerate(clusters):
        print(f"Cluster {index}: {len(cluster.points)}")
