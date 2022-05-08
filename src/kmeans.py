import functools
import math
import random
import statistics
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, List, Sequence, Tuple, TypeVar


def calc_z_scores(original: Sequence[float]) -> List[float]:
    avg = statistics.mean(original)
    std = statistics.pstdev(original)
    if std == 0:
        return [0] * len(original)
    return [(x - avg) / std for x in original]


class DataPoint:
    def __init__(self, original: Iterable[float]):
        self._originals: Tuple[float, ...] = tuple(original)
        self.dimensions: Tuple[float, ...] = tuple(original)

    @property
    def num_dimensions(self) -> int:
        return len(self.dimensions)

    def distance(self, other: "DataPoint") -> float:
        combined: Iterator[Tuple[float, float]] = zip(self.dimensions, other.dimensions)
        differences = [(x - y) ** 2 for x, y in combined]
        return math.sqrt(sum(differences))

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, DataPoint)
        return self.dimensions == other.dimensions

    def __repr__(self) -> str:
        return self._originals.__repr__()


P = TypeVar("P", bound=DataPoint)


@dataclass()
class Cluster(Generic[P]):
    points: List[P]
    centroid: DataPoint


class KMeans(Generic[P]):
    points: List[P]

    def __init__(self, k: int, points: List[P]):
        assert k > 1
        self._points = points
        self._normalize_z_score()
        self._clusters: List[Cluster[P]] = []
        for _ in range(k):
            rand_point = self._generate_random_point()
            cluster: Cluster[P] = Cluster([], rand_point)
            self._clusters.append(cluster)

    @property
    def _centroids(self) -> List[DataPoint]:
        return [x.centroid for x in self._clusters]

    def _slice_dimension(self, dimension: int) -> List[float]:
        return [x.dimensions[dimension] for x in self._points]

    def _normalize_z_score(self) -> None:
        z_scores: List[List[float]] = [[] for _ in range(len(self._points))]
        for dimension in range(self._points[0].num_dimensions):
            dimension_slice = self._slice_dimension(dimension)
            for index, z_score in enumerate(calc_z_scores(dimension_slice)):
                z_scores[index].append(z_score)
        for i in range(len(self._points)):
            self._points[i].dimensions = tuple(z_scores[i])

    def _generate_random_point(self) -> DataPoint:
        rand_dimensions: List[float] = []
        for dimension in range(self._points[0].num_dimensions):
            values = self._slice_dimension(dimension)
            rand_value = random.uniform(min(values), max(values))
            rand_dimensions.append(rand_value)
        return DataPoint(rand_dimensions)

    def _assign_clusters(self) -> None:
        for point in self._points:
            closest = min(self._centroids, key=functools.partial(DataPoint.distance, point))
            i = self._centroids.index(closest)
            self._clusters[i].points.append(point)

    def _generate_centroids(self) -> None:
        for cluster in self._clusters:
            if len(cluster.points) == 0:
                continue
            means: List[float] = []
            for dimension in range(cluster.points[0].num_dimensions):
                dimension_slice = [p.dimensions[dimension] for p in cluster.points]
                means.append(statistics.mean(dimension_slice))
            cluster.centroid = DataPoint(means)

    def run(self, max_iterations: int = 100) -> List[Cluster]:
        for iteration in range(max_iterations):
            for cluster in self._clusters:
                cluster.points.clear()
            self._assign_clusters()
            old_centroids: List[DataPoint] = deepcopy(self._centroids)
            self._generate_centroids()
            if old_centroids == self._centroids:
                print(f"Converged after {iteration} iterations")
                return self._clusters
        return self._clusters


if __name__ == "__main__":
    k_means = KMeans(2, [DataPoint([2, 1, 1]), DataPoint([2, 2, 5]), DataPoint([3, 1.5, 2.5])])
    clusters = k_means.run()
    for index, cluster in enumerate(clusters):
        print(f"Cluster {index}: {cluster.points}")
