import functools
import random
from typing import List, Tuple

from src.genetic import Chromosome, GeneticAlgorithm
from src.sudoku import Sudoku, convert_sudoku_text

_SUDOKU_INDICES = list(range(81))
_SUDOKU_CANDIDATES = list(range(1, 10))


def main() -> None:
    initial_population: List[SudokuChromosome] = [SudokuChromosome.random_instance() for _ in range(20)]
    algorithm: GeneticAlgorithm[SudokuChromosome] = GeneticAlgorithm(
        initial_population=initial_population,
        threshold=9 * 3 * 9,
        mutation_chance=0.5,
        crossover_chance=0,
        max_generatios=100,
    )
    result = algorithm.run()
    print(result)


@functools.lru_cache()
def _get_row_ranges() -> List[List[int]]:
    return [list(range(i * 9, (i + 1) * 9)) for i in range(9)]


@functools.lru_cache()
def _get_col_ranges() -> List[List[int]]:
    return [list(range(i, 81, 9)) for i in range(9)]


@functools.lru_cache()
def _get_constraint_ranges() -> List[List[int]]:
    return [
        *_get_row_ranges(),
        *_get_col_ranges(),
    ]


class SudokuChromosome(Chromosome):
    def __init__(self, values: Sudoku) -> None:
        self.values: Sudoku = values

    def fitness(self) -> float:
        self._validate_values()
        result = 0
        for constarint_range in _get_constraint_ranges():
            block = [self.values[i] for i in constarint_range]
            result += len(set(block))
        return result

    @classmethod
    def random_instance(cls) -> "SudokuChromosome":
        random_values: Sudoku = {}
        for i in range(9):
            candidates = [*_SUDOKU_CANDIDATES]
            random.shuffle(candidates)
            new_values = dict(zip(range(i * 9, (i + 1) * 9), candidates))
            random_values = {**random_values, **new_values}
        return SudokuChromosome(random_values)

    def crossover(self: "SudokuChromosome", other: "SudokuChromosome") -> Tuple["SudokuChromosome", "SudokuChromosome"]:
        child_values1: Sudoku = {**self.values}
        child_values2: Sudoku = {**other.values}
        for i in _SUDOKU_INDICES:
            if random.choice((True, False)):
                child_values1[i], child_values2[i] = child_values2[i], child_values1[i]
        return SudokuChromosome(child_values1), SudokuChromosome(child_values2)

    def mutate(self) -> None:
        row_range = random.choice(_get_row_ranges())
        i1, i2 = random.sample(row_range, k=2)
        self.values[i1], self.values[i2] = self.values[i2], self.values[i1]

    def _validate_values(self) -> None:
        for row_range in _get_row_ranges():
            assert len({self.values[i] for i in row_range}) == 9

    def __str__(self) -> str:
        return f"{convert_sudoku_text(self.values)}\nFitness {self.fitness()}"


main()
