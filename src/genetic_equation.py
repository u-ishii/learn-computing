import random
from copy import deepcopy
from typing import List, Tuple

from src.genetic import Chromosome, GeneticAlgorithm


def main() -> None:
    initial_population: List[Equation] = [Equation.random_instance() for _ in range(20)]
    algorithm: GeneticAlgorithm[Equation] = GeneticAlgorithm(
        initial_population=initial_population, threshold=13.0, mutation_chance=0.1
    )
    result = algorithm.run()
    print(result)


class Equation(Chromosome):
    def __init__(self, x: int, y: int) -> None:
        self.x: int = x
        self.y: int = y

    def fitness(self) -> float:
        return 6 * self.x - self.x * self.x + 4 * self.y - self.y * self.y

    @classmethod
    def random_instance(cls) -> "Equation":
        return Equation(random.randrange(100), random.randrange(100))

    def crossover(self: "Equation", other: "Equation") -> Tuple["Equation", "Equation"]:
        child1: Equation = deepcopy(self)
        child2: Equation = deepcopy(other)
        child1.y = other.y
        child2.y = self.y
        return child1, child2

    def mutate(self) -> None:
        if random.choice([True, False]):
            if random.choice([True, False]):
                self.x += 1
            else:
                self.x -= 1
        else:
            if random.choice([True, False]):
                self.y += 1
            else:
                self.y -= 1

    def __str__(self) -> str:
        return f"X: {self.x} Y: {self.y} Fitness {self.fitness()}"


main()
