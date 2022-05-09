import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt

from src.kmeans import Cluster, DataPoint, KMeans

_DATE_HEADER = "発症_年月日"
_AGE_HEADER = "患者_年代"
_DATE_OPTIONS: List[str] = [f"2022-04-{str(i).zfill(2)}" for i in range(1, 31)]
_AGE_OPTIONS = ["10歳未満", "10代", "20代", "30代", "40代", "50代", "60代", "70代", "80代", "90代", "100歳以上"]


@dataclass
class PatientRow:
    date: int
    age: int


@dataclass
class GroupedPatientRow(PatientRow):
    amount: int


@dataclass
class ScatterParams:
    x: Sequence[int]
    y: Sequence[int]
    s: Sequence[int]
    c: Sequence[int]


class PatientPoint(DataPoint):
    def __init__(self, row: GroupedPatientRow):
        super().__init__([row.date, row.age, row.amount])
        self.row: GroupedPatientRow = row


def _read_patient_rows() -> List[PatientRow]:
    with open("data/patients.csv") as f:
        reader = csv.DictReader(f)
        return [
            PatientRow(_DATE_OPTIONS.index(row[_DATE_HEADER]), _AGE_OPTIONS.index(row[_AGE_HEADER]))
            for row in reader
            if row[_DATE_HEADER] in _DATE_OPTIONS and row[_AGE_HEADER] in _AGE_OPTIONS
        ]


def _group_patient_rows(rows: List[PatientRow]) -> List[GroupedPatientRow]:
    counted: Dict[Tuple[int, int], int] = defaultdict(lambda: 0)
    for row in rows:
        counted[row.date, row.age] += 1
    return [GroupedPatientRow(date, age, counted[date, age]) for date, age in counted.keys()]


def _convert_as_scatter_params(clusters: List[Cluster[PatientPoint]]) -> ScatterParams:
    dates, ages, amounts, cluster_indices = zip(
        *[
            (point.row.date, point.row.age, point.row.amount, i)
            for i, cluster in enumerate(clusters)
            for point in cluster.points
        ]
    )
    return ScatterParams(x=dates, y=ages, s=amounts, c=cluster_indices)


if __name__ == "__main__":
    patient_rows = _read_patient_rows()
    grouped_patient_rows = _group_patient_rows(patient_rows)
    patient_points = [PatientPoint(row) for row in grouped_patient_rows]
    k_means = KMeans(5, patient_points)
    clusters = k_means.run()
    plt.scatter(**_convert_as_scatter_params(clusters).__dict__)
    plt.show()
    # for index, cluster in enumerate(clusters):
    #     print(f"Cluster {index}: {len(cluster.points)}")
