import functools
import math
import random
import statistics
from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar


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
        return isinstance(other, DataPoint) and self.dimensions == other.dimensions

    def __repr__(self) -> str:
        return self._originals.__repr__()


P = TypeVar("P", bound=DataPoint)


@dataclass(frozen=True)
class Cluster(Generic[P]):
    points: Tuple[P, ...]
    centroid: Optional[DataPoint]


class KMeans(Generic[P]):
    points: Tuple[P, ...]

    def __init__(self, k: int, points: List[P]):
        assert k > 1
        self._points = points
        self._normalize_z_score()
        initial_clusters: List[Cluster[P]] = [Cluster((), self._generate_random_point()) for _ in range(k)]
        self._cluster_histories: List[List[Cluster[P]]] = [initial_clusters]

    @property
    def _clusters(self) -> List[Cluster[P]]:
        return self._cluster_histories[-1]

    @property
    def _centroids(self) -> List[Optional[DataPoint]]:
        return [x.centroid for x in self._clusters]

    @property
    def _not_empty_centroids(self) -> List[DataPoint]:
        return [x for x in self._centroids if x is not None]

    @property
    def _is_stable(self) -> bool:
        old_centroids = [x.centroid for x in self._cluster_histories[-2]]
        return self._centroids == old_centroids

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

    def _generate_next_cluster_points(self) -> List[List[P]]:
        next_cluster_points: List[List[P]] = [[] for _ in range(len(self._clusters))]
        for point in self._points:
            closest = min(self._not_empty_centroids, key=functools.partial(DataPoint.distance, point))
            i = self._centroids.index(closest)
            next_cluster_points[i].append(point)
        return next_cluster_points

    def _generate_next_centroid(self, next_points: List[P]) -> Optional[DataPoint]:
        if len(next_points) == 0:
            return None
        means: List[float] = []
        for dimension in range(next_points[0].num_dimensions):
            dimension_slice = [p.dimensions[dimension] for p in next_points]
            means.append(statistics.mean(dimension_slice))
        return DataPoint(means)

    def _generate_next_centroids(self, next_cluster_points: List[List[P]]) -> List[Optional[DataPoint]]:
        return [self._generate_next_centroid(next_points) for next_points in next_cluster_points]

    def run(self, max_iterations: int = 100) -> List[List[Cluster]]:
        for iteration in range(max_iterations):
            next_cluster_points = self._generate_next_cluster_points()
            next_centroids = self._generate_next_centroids(next_cluster_points)
            self._cluster_histories.append(
                [Cluster(tuple(points), centroid) for points, centroid in zip(next_cluster_points, next_centroids)]
            )
            if self._is_stable:
                print(f"Converged after {iteration} iterations")
                break
        return self._cluster_histories


def main() -> None:
    k_means = KMeans(2, [DataPoint([2, 1, 1]), DataPoint([2, 2, 5]), DataPoint([3, 1.5, 2.5])])
    clusters = k_means.run()[-1]
    for index, cluster in enumerate(clusters):
        print(f"Cluster {index}: {cluster.points}")


if __name__ == "__main__":
    main()
