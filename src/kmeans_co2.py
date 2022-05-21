import csv
import statistics
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.animation as anm
import matplotlib.pyplot as plt

from src.kmeans import Cluster, DataPoint, KMeans


@dataclass
class CsvRow:
    country_name: str
    country_code: str
    amount: float


class Co2Point(DataPoint):
    def __init__(self, country_code: str, country_name: str, gdp_amount: float, co2_amount: float):
        super().__init__([gdp_amount, co2_amount])
        self.country_code: str = country_code
        self.country_name: str = country_name
        self.gdp_amount: float = gdp_amount
        self.co2_amount: float = co2_amount


def _read_gdp_rows() -> List[CsvRow]:
    return _read_csv_rows("gdp")


def _read_co2_rows() -> List[CsvRow]:
    return _read_csv_rows("gdp-growth")


def _read_csv_rows(name: str) -> List[CsvRow]:
    with open(f"data/{name}.csv", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [
            CsvRow(row["Country Name"], row["Country Code"], float(row["2018"]))
            for row in reader
            if len(row["2020"]) > 0 and row["Country Code"] != "WLD"
        ]


def _merge_csv_rows(gdp_rows: List[CsvRow], co2_rows: List[CsvRow]) -> List[Tuple[CsvRow, CsvRow]]:
    key_gdp_rows = {row.country_code: row for row in gdp_rows}
    key_co2_rows = {row.country_code: row for row in co2_rows}
    country_codes = list(set(key_gdp_rows.keys()) & set(key_co2_rows.keys()))
    country_codes.sort()
    return [(key_gdp_rows[code], key_co2_rows[code]) for code in country_codes]


def _scatter_points(clusters: List[Cluster[Co2Point]]):
    gdp_amounts, co2_amounts, cluster_indices = zip(
        *[
            (point.gdp_amount, point.co2_amount, i)
            for i, cluster in enumerate(clusters)
            for point in cluster.points
            if len(cluster.points) > 0
        ]
    )
    return plt.scatter(x=gdp_amounts, y=co2_amounts, c=cluster_indices)


def _scatter_centroids(clusters: List[Cluster[Co2Point]], **kwargs):
    gdp_positions, co2_positions, cluster_indices = zip(
        *[
            (
                statistics.mean([point.gdp_amount for point in cluster.points]),
                statistics.mean([point.co2_amount for point in cluster.points]),
                i,
            )
            for i, cluster in enumerate(clusters)
            if len(cluster.points) > 0
        ]
    )
    return plt.scatter(x=gdp_positions, y=co2_positions, c=cluster_indices, **kwargs)


if __name__ == "__main__":
    gdp_rows = _read_gdp_rows()
    co2_rows = _read_co2_rows()
    merged_rows = _merge_csv_rows(gdp_rows, co2_rows)
    co2_points = [
        Co2Point(gdp_row.country_code, gdp_row.country_name, gdp_row.amount, co2_row.amount)
        for gdp_row, co2_row in merged_rows
    ]
    k_means = KMeans(7, co2_points)
    cluster_histories = k_means.run()
    for index, cluster in enumerate(cluster_histories[-1]):
        print(f"Cluster {index}: {len(cluster.points)}")
    figure = plt.figure()
    animation = anm.ArtistAnimation(
        figure,
        [
            [
                _scatter_points(clusters),
                _scatter_centroids(clusters, marker="*", edgecolors="black"),
            ]
            for clusters in cluster_histories[1:]
        ],
        interval=1000,
        repeat=False,
    )
    plt.show()
