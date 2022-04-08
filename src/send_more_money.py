from typing import Dict, List

from src.csp import CSP, Constraint


def main() -> None:
    letters: List[str] = ["S", "E", "N", "D", "M", "O", "R", "Y"]
    possible_digits: Dict[str, List[int]] = {}
    for letter in letters:
        possible_digits[letter] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    possible_digits["M"] = [1]
    csp: CSP[str, int] = CSP(letters, possible_digits)
    csp.add_constraint(SendMoreMoneyConstraint(letters))
    solution = csp.backtracking_search()
    if solution is None:
        print("No solution found!")
    else:
        print(solution)


class SendMoreMoneyConstraint(Constraint[str, int]):
    def __init__(self, letters: List[str]) -> None:
        super().__init__(letters)
        self.letters = letters

    def satisfied(self, assignment: Dict[str, int]) -> bool:
        if len(set(assignment.values())) < len(assignment):
            return False
        if len(assignment) == len(self.letters):
            send = assignment["S"] * 1000 + assignment["E"] * 100 + assignment["N"] * 10 + assignment["D"]
            more = assignment["M"] * 1000 + assignment["O"] * 100 + assignment["R"] * 10 + assignment["E"]
            money = (
                assignment["M"] * 10000
                + assignment["O"] * 1000
                + assignment["N"] * 100
                + assignment["E"] * 10
                + assignment["Y"]
            )
            return send + more == money
        return True


main()
