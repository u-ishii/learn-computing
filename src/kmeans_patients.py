import csv
from typing import List

import matplotlib.pyplot as plt

from src.kmeans import DataPoint, KMeans

_DATE_HEADER = "発症_年月日"
_AGE_HEADER = "患者_年代"
_DATE_OPTIONS: List[str] = [f"2022-04-{str(i).zfill(2)}" for i in range(1, 31)]
_AGE_OPTIONS = ["10歳未満", "10代", "20代", "30代", "40代", "50代", "60代", "70代", "80代", "90代", "100歳以上"]


if __name__ == "__main__":
    with open("data/patients.csv") as f:
        reader = csv.DictReader(f)
        points = [
            DataPoint([_DATE_OPTIONS.index(row[_DATE_HEADER]), _AGE_OPTIONS.index(row[_AGE_HEADER])])
            for row in reader
            if row[_DATE_HEADER] in _DATE_OPTIONS and row[_AGE_HEADER] in _AGE_OPTIONS
        ]
    plt.scatter([p.dimensions[0] for p in points], [p.dimensions[1] for p in points])
    plt.show()
    # k_means = KMeans(10, points)
    # clusters = k_means.run()
    # for index, cluster in enumerate(clusters):
    #     print(f"Cluster {index}: {len(cluster.points)}")
