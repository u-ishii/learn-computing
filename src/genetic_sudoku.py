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
        max_generations=100000,
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
        return SudokuChromosome(candidates)

    def crossover(self, other: "SudokuChromosome") -> Tuple["SudokuChromosome", "SudokuChromosome"]:
        left_partition, right_partition = sorted(random.sample(range(81), k=2))
        assert left_partition < right_partition
        child_values1: Sudoku = [
            *self.values[:left_partition],
            *other.values[left_partition:right_partition],
            *self.values[right_partition:],
        ]
        child_values2: Sudoku = [
            *other.values[:left_partition],
            *self.values[left_partition:right_partition],
            *other.values[right_partition:],
        ]
        assert sum(child_values1) + sum(child_values2) == sum(self.values) + sum(other.values)
        return SudokuChromosome(child_values1), SudokuChromosome(child_values2)

    def mutate(self) -> "SudokuChromosome":
        candidates = list(range(1, 10))
        times = {i: 0 for i in candidates}
        for value in self.values:
            times[value] += 1
        new_value = sorted(candidates, key=lambda x: times[x])[0]
        assert times[new_value] <= 9
        mutated_values: Sudoku = [*self.values]
        mutated_values[random.choice(range(81))] = new_value
        return SudokuChromosome(mutated_values)

    def __str__(self) -> str:
        return f"{convert_sudoku_text(self.values)}\nFitness: {self.fitness()}"


main()
