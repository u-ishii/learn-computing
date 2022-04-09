from typing import Dict, List

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
        _render_sudoku(solution)


def _render_sudoku(sudoku: Dict[int, int]) -> None:
    for y in range(9):
        for x in range(9):
            print(chr(ord("０") + sudoku[y * 9 + x]), end="")
        print()


class SudokuConstraint(Constraint[int, int]):

    numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9}

    def __init__(self, indices: List[int]) -> None:
        super().__init__(indices)

    def satisfied(self, assignment: Dict[int, int]) -> bool:
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
