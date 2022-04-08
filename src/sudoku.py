from typing import Dict, List, Tuple

from src.csp import CSP, Constraint


def main() -> None:
    indices: List[int] = [i for i in range(81)]
    sudoku = {index: [1, 2, 3, 4, 5, 6, 7, 8, 9] for index in indices}
    csp: CSP[int, int] = CSP(indices, sudoku)
    csp.add_constraint(SudokuConstraint(indices))
    solution = csp.backtracking_search()
    if solution is None:
        print("No solution found!")
    else:
        for i in range(9):
            print("".join([str(solution[i * 9 + j]) for j in range(9)]))
        print(solution)


class SudokuConstraint(Constraint[int, int]):

    numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9}

    def __init__(self, indices: List[int]) -> None:
        super().__init__(indices)
        self.indices = indices

    def satisfied(self, assignment: Dict[int, int]) -> bool:
        row = [assignment[j] for j in assignment.keys() if j < 9]
        if len(row) != len(set(row)):
            return False
        return True


main()
