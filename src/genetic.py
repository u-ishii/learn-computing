import heapq
import random
from abc import ABC, abstractmethod
from enum import Enum
from statistics import mean
from typing import Callable, Generic, List, Tuple, Type, TypeVar

T = TypeVar("T", bound="Chromosome")


class Chromosome(ABC):
    @abstractmethod
    def fitness(self) -> float:
        ...

    @classmethod
    @abstractmethod
    def random_instance(cls: Type[T]) -> T:
        ...

    @abstractmethod
    def crossover(self: T, other: T) -> Tuple[T, T]:
        ...

    @abstractmethod
    def mutate(self: T) -> T:
        ...


class SelectionType(Enum):
    ROULETTE = 1
    TOURNAMENT = 2


C = TypeVar("C", bound=Chromosome)


class GeneticAlgorithm(Generic[C]):
    def __init__(
        self,
        initial_population: List[C],
        threshold: float,
        max_generations: int = 100,
        mutation_chance: float = 0.01,
        crossover_chance: float = 0.7,
        selection_type: SelectionType = SelectionType.TOURNAMENT,
    ):
        self._population: List[C] = initial_population
        self._threshold: float = threshold
        self._max_generations: int = max_generations
        self._mutation_chance: float = mutation_chance
        self._crossover_chance: float = crossover_chance
        self._selection_type: SelectionType = selection_type
        self._fitness_key: Callable[[C], float] = type(self._population[0]).fitness

    def _pick_roulette(self) -> Tuple[C, C]:
        wheel = [x.fitness() for x in self._population]
        chosen = random.choices(self._population, weights=wheel, k=2)
        return chosen[0], chosen[1]

    def _pick_tournament(self) -> Tuple[C, C]:
        participants = random.choices(self._population, k=(len(self._population) // 2))
        chosen = heapq.nlargest(2, participants, key=self._fitness_key)
        return chosen[0], chosen[1]

    def _pick_parents(self) -> Tuple[C, C]:
        if self._selection_type == SelectionType.ROULETTE:
            return self._pick_roulette()
        else:
            return self._pick_tournament()

    def _reproduce_and_replace(self) -> None:
        new_population: List[C] = []
        while len(new_population) < len(self._population):
            if random.random() < self._crossover_chance:
                parents = self._pick_parents()
                new_population.extend(parents[0].crossover(parents[1]))
            elif random.random() < self._mutation_chance:
                new_population.append(self._pick_parents()[0].mutate())
            else:
                new_population.append(self._pick_parents()[0])
        self._population = new_population[: len(self._population)]

    def run(self) -> C:
        best: C = max(self._population, key=self._fitness_key)
        for generation in range(self._max_generations):
            if best.fitness() >= self._threshold:
                return best
            if generation % 100 == 0:
                print(f"{generation}\t{best.fitness()}\t{mean(map(self._fitness_key, self._population))}")
            self._reproduce_and_replace()
            highest: C = max(self._population, key=self._fitness_key)
            if highest.fitness() > best.fitness():
                best = highest
        return best
