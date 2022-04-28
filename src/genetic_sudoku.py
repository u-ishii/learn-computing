import random
from typing import Tuple

from src.genetic import Chromosome, GeneticAlgorithm
from src.sudoku import Sudoku, convert_sudoku_text

_SUDOKU_INDICES = [i for i in range(81)]


def main() -> None:
    print(SudokuChromosome.random_instance())
    # initial_population: List[Equation] = [Equation.random_instance() for _ in range(20)]
    # algorithm: GeneticAlgorithm[Equation] = GeneticAlgorithm(
    #     initial_population=initial_population, threshold=13.0, mutation_chance=0.1
    # )
    # result = algorithm.run()
    # print(result)


class SudokuChromosome(Chromosome):
    def __init__(self, values: Sudoku) -> None:
        self.values: Sudoku = values

    def fitness(self) -> float:
        return 1

    @classmethod
    def random_instance(cls) -> "SudokuChromosome":
        random_values: Sudoku = {}
        for i in range(9):
            candidates = list(range(1, 10))
            random.shuffle(candidates)
            new_values = dict(zip(range(i * 9, (i + 1) * 9), candidates))
            random_values = {**random_values, **new_values}
        return SudokuChromosome(random_values)

    def crossover(self: "SudokuChromosome", other: "SudokuChromosome") -> Tuple["SudokuChromosome", "SudokuChromosome"]:
        child_values1: Sudoku = {}
        child_values2: Sudoku = {}
        for i in _SUDOKU_INDICES:
            if i % 2 == 0:
                child_values1[i] = self.values[i]
                child_values2[i] = other.values[i]
            else:
                child_values1[i] = other.values[i]
                child_values2[i] = self.values[i]
        return SudokuChromosome(child_values1), SudokuChromosome(child_values2)

    def mutate(self) -> None:
        index = random.choice(_SUDOKU_INDICES)
        candidates = set(range(1, 10))
        candidates.remove(self.values[index])
        self.values[index] = random.choice(list(candidates))

    def __str__(self) -> str:
        return f"{convert_sudoku_text(self.values)} Fitness {self.fitness()}"


main()
