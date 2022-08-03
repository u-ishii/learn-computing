import functools
import random
from typing import List, Tuple

from src.genetic import Chromosome, GeneticAlgorithm
from src.sudoku import Sudoku, convert_sudoku_text


def main() -> None:
    initial_population: List[SudokuChromosome] = [SudokuChromosome.random_instance() for _ in range(20)]
    algorithm: GeneticAlgorithm[SudokuChromosome] = GeneticAlgorithm(
        initial_population=initial_population,
        threshold=9 * 9 * 3,
        mutation_chance=0.05,
        max_generations=10000,
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
        assert len(values) == 81
        self.values: Sudoku = values

    def fitness(self) -> float:
        result = 0
        for constraint_range in _get_constraint_ranges():
            block = [self.values[i] for i in constraint_range]
            result += len(set(block))
        return result

    @classmethod
    def random_instance(cls) -> "SudokuChromosome":
        candidates = [i for _ in range(9) for i in range(1, 10)]
        random.shuffle(candidates)
        values: Sudoku = dict(zip(range(81), candidates))
        return SudokuChromosome(values)

    def crossover(self, other: "SudokuChromosome") -> Tuple["SudokuChromosome", "SudokuChromosome"]:
        child_values1: Sudoku = {**self.values}
        child_values2: Sudoku = {**other.values}
        for row_col_range in random.choice((_get_row_ranges(), _get_col_ranges())):
            if random.choice((True, False)):
                for i in row_col_range:
                    child_values1[i], child_values2[i] = child_values2[i], child_values1[i]
        # fmt: off
        assert sum(child_values1.values()) + sum(child_values2.values()) == sum(self.values.values()) + sum(other.values.values())  # noqa: E501
        return SudokuChromosome(child_values1), SudokuChromosome(child_values2)

    def mutate(self) -> "SudokuChromosome":
        mutated_values: Sudoku = {**self.values}
        for i in random.sample(range(81), k=3):
            mutated_values[i] = random.randint(1, 9)
        return SudokuChromosome(mutated_values)

    def __str__(self) -> str:
        return f"{convert_sudoku_text(self.values)}\nFitness: {self.fitness()}"


main()
