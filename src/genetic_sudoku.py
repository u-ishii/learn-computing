import functools
import random
from typing import List, Tuple

from src.genetic import Chromosome, GeneticAlgorithm
from src.sudoku import Sudoku, convert_sudoku_text

_SUDOKU_INDICES = [i for i in range(81)]


def main() -> None:
    initial_population: List[SudokuChromosome] = [SudokuChromosome.random_instance() for _ in range(20)]
    algorithm: GeneticAlgorithm[SudokuChromosome] = GeneticAlgorithm(
        initial_population=initial_population,
        threshold=9 * 3 * 9,
        mutation_chance=0.5,
        max_generatios=10000,
    )
    result = algorithm.run()
    print(result)


@functools.lru_cache()
def _get_constraint_ranges() -> List[List[int]]:
    return [
        *[list(range(i * 9, (i + 1) * 9)) for i in range(9)],
        *[list(range(i, 81, 9)) for i in range(9)],
    ]


class SudokuChromosome(Chromosome):
    def __init__(self, values: Sudoku) -> None:
        self.values: Sudoku = values

    def fitness(self) -> float:
        result = 0
        for constarint_range in _get_constraint_ranges():
            block = [self.values[i] for i in constarint_range]
            result += len(set(block))
        return result

    @classmethod
    def random_instance(cls) -> "SudokuChromosome":
        values = [(i % 9) + 1 for i in _SUDOKU_INDICES]
        random.shuffle(values)
        return SudokuChromosome(dict(zip(_SUDOKU_INDICES, values)))

    def crossover(self: "SudokuChromosome", other: "SudokuChromosome") -> Tuple["SudokuChromosome", "SudokuChromosome"]:
        child_values1: Sudoku = {**self.values}
        child_values2: Sudoku = {**other.values}
        for i in _SUDOKU_INDICES:
            if random.choice((True, False)):
                child_values1[i], child_values2[i] = child_values2[i], child_values1[i]
        return SudokuChromosome(child_values1), SudokuChromosome(child_values2)

    def mutate(self) -> None:
        index = random.choice(_SUDOKU_INDICES)
        candidates = set(range(1, 10))
        candidates.remove(self.values[index])
        self.values[index] = random.choice(list(candidates))

    def __str__(self) -> str:
        return f"{convert_sudoku_text(self.values)}\nFitness {self.fitness()}"


main()
