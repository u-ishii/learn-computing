import csv
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

from src.kmeans import DataPoint, KMeans

_DATE_HEADER = "発症_年月日"
_AGE_HEADER = "患者_年代"
_DATE_OPTIONS: List[str] = [f"2022-04-{str(i).zfill(2)}" for i in range(1, 31)]
_AGE_OPTIONS = ["10歳未満", "10代", "20代", "30代", "40代", "50代", "60代", "70代", "80代", "90代", "100歳以上"]

PatientRow = Tuple[int, int]
GroupedPatientRow = Tuple[int, int, int]


def _read_patient_rows() -> List[PatientRow]:
    with open("data/patients.csv") as f:
        reader = csv.DictReader(f)
        return [
            (_DATE_OPTIONS.index(row[_DATE_HEADER]), _AGE_OPTIONS.index(row[_AGE_HEADER]))
            for row in reader
            if row[_DATE_HEADER] in _DATE_OPTIONS and row[_AGE_HEADER] in _AGE_OPTIONS
        ]


def _group_patient_rows(rows: List[PatientRow]) -> List[GroupedPatientRow]:
    counted: Dict[Tuple[int, int], int] = defaultdict(lambda: 0)
    for row in rows:
        counted[row[0], row[1]] += 1
    return [(date, age, counted[date, age]) for date, age in counted.keys()]


def _convert_as_scatter_params(rows: List[GroupedPatientRow]) -> Dict[str, Any]:
    dates, ages, amounts = zip(*rows)
    return {"x": dates, "y": ages, "s": amounts}


if __name__ == "__main__":
    patient_rows = _read_patient_rows()
    grouped_patient_rows = _group_patient_rows(patient_rows)
    plt.scatter(**_convert_as_scatter_params(grouped_patient_rows))
    plt.show()
    # k_means = KMeans(10, points)
    # clusters = k_means.run()
    # for index, cluster in enumerate(clusters):
    #     print(f"Cluster {index}: {len(cluster.points)}")
