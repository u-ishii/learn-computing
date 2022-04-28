import random
import time
from typing import Dict, List

from src.csp import CSP, Constraint


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
        turn = 1
        for past in constraint.history:
            print(f"turn: {turn}\n{_convert_sudoku_text(past)}")
            turn += 1
            time.sleep(0.01)


def _convert_sudoku_text(sudoku: Dict[int, int]) -> str:
    lines: List[str] = []
    for y in range(9):
        if y != 0 and y % 3 == 0:
            lines.append("─" * 6 + "┼" + "─" * 6 + "┼" + "─" * 6)
        number_line = ""
        for x in range(9):
            if x != 0 and x % 3 == 0:
                number_line += "│"
            i = y * 9 + x
            cell = chr(ord("０") + sudoku[i]) if i in sudoku else "  "
            number_line += cell
        lines.append(number_line)
    return "\n".join(lines)


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
