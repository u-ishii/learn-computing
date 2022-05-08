import csv

from src.kmeans import DataPoint, KMeans

if __name__ == "__main__":
    with open("data/patients.csv") as f:
        reader = csv.DictReader(f)
        rows = [{"age": row["患者_年代"], "date": row["発症_年月日"]} for row in reader]
    print({row["age"] for row in rows})
    print({row["date"] for row in rows})
    # k_means = KMeans(2, [DataPoint([2, 1, 1]), DataPoint([2, 2, 5]), DataPoint([3, 1.5, 2.5])])
    # clusters = k_means.run()
    # for index, cluster in enumerate(clusters):
    #     print(f"Cluster {index}: {cluster.points}")
