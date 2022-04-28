import random
from typing import Dict, List

from src.csp import CSP, Constraint
from src.sudoku import print_sudoku_history


def main() -> None:
    indices: List[int] = [i for i in range(81)]
    sudoku = {index: random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 9) for index in indices}
    csp: CSP[int, int] = CSP(indices, sudoku)
    constraint = SudokuConstraint(indices)
    csp.add_constraint(constraint)
    solution = csp.backtracking_search()
    if solution is None:
        print("No solution found!")
    else:
        input("render?")
        print_sudoku_history(constraint.history)


class SudokuConstraint(Constraint[int, int]):
    def __init__(self, indices: List[int]) -> None:
        super().__init__(indices)
        self.history: List[Dict[int, int]] = []

    def satisfied(self, assignment: Dict[int, int]) -> bool:
        self.history.append(assignment)
        for i in range(9):
            row = [assignment[j] for j in assignment.keys() if j // 9 == i]
            if len(row) != len(set(row)):
                return False
            col = [assignment[j] for j in assignment.keys() if j % 9 == i]
            if len(col) != len(set(col)):
                return False
        for x in range(3):
            for y in range(3):
                block = [assignment[j] for j in assignment.keys() if j // 9 // 3 == x and j % 9 // 3 == y]
                if len(block) != len(set(block)):
                    return False
        return True


main()
