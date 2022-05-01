import functools
import random
from typing import List, Tuple

from src.genetic import Chromosome, GeneticAlgorithm
from src.sudoku import Sudoku, convert_sudoku_text


def main() -> None:
    initial_population: List[SudokuChromosome] = [SudokuChromosome.random_instance() for _ in range(20)]
    algorithm: GeneticAlgorithm[SudokuChromosome] = GeneticAlgorithm(
        initial_population=initial_population,
        threshold=9 * 3,
        mutation_chance=0.5,
        max_generatios=10000,
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
def _get_block_ranges() -> List[List[int]]:
    base = [0, 1, 2, 9, 10, 11, 18, 19, 20]
    return [[v + (j * 3) + (i * 27) for v in base] for i in range(3) for j in range(3)]


@functools.lru_cache()
def _get_constraint_ranges() -> List[List[int]]:
    return [
        *_get_row_ranges(),
        *_get_col_ranges(),
        *_get_block_ranges(),
    ]


class SudokuChromosome(Chromosome):
    def __init__(self, values: Sudoku) -> None:
        self.values: Sudoku = values

    def fitness(self) -> float:
        assert self._validate_values()
        result = 0
        for constarint_range in _get_constraint_ranges():
            block = [self.values[i] for i in constarint_range]
            if len(block) == len(set(block)):
                result += 1
        return result

    @classmethod
    def random_instance(cls) -> "SudokuChromosome":
        random_values: Sudoku = {}
        for i in range(9):
            candidates = list(range(1, 10))
            random.shuffle(candidates)
            new_values = dict(zip(range(i * 9, (i + 1) * 9), candidates))
            random_values = {**random_values, **new_values}
        return SudokuChromosome(random_values)

    def crossover(self, other: "SudokuChromosome") -> Tuple["SudokuChromosome", "SudokuChromosome"]:
        child_values1: Sudoku = {**self.values}
        child_values2: Sudoku = {**other.values}
        for row_range in _get_row_ranges():
            if random.choice((True, False)):
                for i in row_range:
                    child_values1[i], child_values2[i] = child_values2[i], child_values1[i]
        return SudokuChromosome(child_values1), SudokuChromosome(child_values2)

    def mutate(self) -> None:
        row_range = random.choice(_get_row_ranges())
        i1, i2 = random.sample(row_range, k=2)
        self.values[i1], self.values[i2] = self.values[i2], self.values[i1]

    def _validate_values(self) -> bool:
        for row_range in _get_row_ranges():
            if len({self.values[i] for i in row_range}) != 9:
                return False
        return True

    def __str__(self) -> str:
        return f"{convert_sudoku_text(self.values)}\nFitness: {self.fitness()}"


main()
