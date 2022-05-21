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

    def __init__(self, points: List[P]):
        self._points = points
        self._normalize_z_score()

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

    def _generate_next_cluster_points(self, k: int, centroids: Sequence[Optional[DataPoint]]) -> List[List[P]]:
        not_empty_centroids = [x for x in centroids if x is not None]
        next_cluster_points: List[List[P]] = [[] for _ in range(k)]
        for point in self._points:
            closest = min(not_empty_centroids, key=functools.partial(DataPoint.distance, point))
            i = centroids.index(closest)
            next_cluster_points[i].append(point)
        return next_cluster_points

    def _generate_next_centroid(self, next_points: Sequence[P]) -> Optional[DataPoint]:
        if len(next_points) == 0:
            return None
        means: List[float] = []
        for dimension in range(next_points[0].num_dimensions):
            dimension_slice = [p.dimensions[dimension] for p in next_points]
            means.append(statistics.mean(dimension_slice))
        return DataPoint(means)

    def _generate_next_centroids(self, next_cluster_points: Sequence[Sequence[P]]) -> List[Optional[DataPoint]]:
        return [self._generate_next_centroid(next_points) for next_points in next_cluster_points]

    def run(self, k: int, max_iterations: int = 100) -> List[List[Cluster]]:
        initial_centroids: List[DataPoint] = [self._generate_random_point() for _ in range(k)]
        initial_cluster_points: List[List[P]] = self._generate_next_cluster_points(k, initial_centroids)
        initial_clusters: List[Cluster[P]] = [
            Cluster(tuple(points), centroid) for centroid, points in zip(initial_centroids, initial_cluster_points)
        ]
        cluster_histories = [initial_clusters]
        for iteration in range(max_iterations):
            next_centroids = self._generate_next_centroids([cluster.points for cluster in cluster_histories[-1]])
            next_cluster_points = self._generate_next_cluster_points(k, next_centroids)
            cluster_histories.append(
                [Cluster(tuple(points), centroid) for points, centroid in zip(next_cluster_points, next_centroids)]
            )
            is_stable = [x.centroid for x in cluster_histories[-1]] == [x.centroid for x in cluster_histories[-2]]
            if is_stable:
                print(f"Converged after {iteration} iterations")
                break
        return cluster_histories


def main() -> None:
    k_means = KMeans([DataPoint([2, 1, 1]), DataPoint([2, 2, 5]), DataPoint([3, 1.5, 2.5])])
    clusters = k_means.run(2)[-1]
    for index, cluster in enumerate(clusters):
        print(f"Cluster {index}: {cluster.points}")


if __name__ == "__main__":
    main()
