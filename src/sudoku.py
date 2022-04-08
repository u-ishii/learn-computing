from typing import Dict, List, Tuple

from src.csp import CSP, Constraint


def main() -> None:
    positions: List[Tuple[int, int]] = []
    for x in range(9):
        for y in range(9):
            positions.append((x, y))
    sudoku = {position: [1, 2, 3, 4, 5, 6, 7, 8, 9] for position in positions}
    csp: CSP[Tuple[int, int], int] = CSP(positions, sudoku)
    csp.add_constraint(SudokuConstraint(positions))
    solution = csp.backtracking_search()
    if solution is None:
        print("No solution found!")
    else:
        print(solution)


class SudokuConstraint(Constraint[Tuple[int, int], int]):

    numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9}

    def __init__(self, positions: List[Tuple[int, int]]) -> None:
        super().__init__(positions)
        self.positions = positions

    def satisfied(self, assignment: Dict[Tuple[int, int], int]) -> bool:
        if len(assignment) != 81:
            return False
        for i in range(9):
            if set([assignment[(i, j)] for j in range(9)]) != self.numbers:
                return False
        return True


main()
