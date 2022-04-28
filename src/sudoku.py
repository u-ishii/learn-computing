import time
from typing import Dict, List

Sudoku = Dict[int, int]


def print_sudoku_history(sudoku_history: List[Sudoku]) -> None:
    turn = 1
    for past in sudoku_history:
        print(f"turn: {turn}\n{convert_sudoku_text(past)}")
        turn += 1
        time.sleep(0.01)


def convert_sudoku_text(sudoku: Sudoku) -> str:
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
