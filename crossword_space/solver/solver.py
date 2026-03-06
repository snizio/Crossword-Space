from tabulate import tabulate
import z3

class Solve:
    def __init__(self, crossword_grid, clues, prefer_top_rank=True):
        self.GRID = crossword_grid
        self.CLUES = clues
        self.candidate_size = max(len(cands) for cands in clues.values())
        self.prefer_top_rank = prefer_top_rank

    def get_matrix(self):
        self.start_positions = {
            (info["start"][0], info["start"][1], info["direction"]): clue
            for clue, info in self.GRID.items()
        }

        max_row = max(
            i["start"][0] + (i["length"] if i["direction"] == "D" else 1)
            for i in self.GRID.values()
        )
        max_col = max(
            i["start"][1] + (i["length"] if i["direction"] == "A" else 1)
            for i in self.GRID.values()
        )
        self.n = max(max_row, max_col)

        self.matrix = [[None for _ in range(self.n)] for _ in range(self.n)]
        for (x, y, direction), clue in self.start_positions.items():
            L = self.GRID[clue]["length"]
            for k in range(L):
                r = x + (k if direction == "D" else 0)
                c = y + (0 if direction == "D" else k)
                if self.matrix[r][c] is None:
                    self.matrix[r][c] = z3.Int(f"a_{r}_{c}")

    @staticmethod
    def to_codes(word: str):
        return [ord(ch) for ch in word]

    def build_optimize(self):
        self.get_matrix()
        opt = z3.Optimize()
        self.sel_vars = {}

        for (x, y, direction), clue in self.start_positions.items():
            candidates = self.CLUES[clue]
            sels = []

            for idx, cand in enumerate(candidates):
                s = z3.Bool(f"sel::{clue}::{idx}")
                self.sel_vars[(clue, idx)] = s
                sels.append(s)

                codes = self.to_codes(cand)  # assume already case-normalized
                r, c = x, y
                letter_eqs = []
                for code in codes:
                    letter_eqs.append(self.matrix[r][c] == code)
                    if direction == "D": r += 1
                    else: c += 1

                # If selected, enforce letters
                opt.add(z3.Implies(s, z3.And(letter_eqs)))

                # Soft: prefer selecting this candidate
                w = 1 if not self.prefer_top_rank else (self.candidate_size - idx)
                opt.add_soft(s, weight=str(max(1, w)))

            # Hard: at most one per clue
            if sels:
                opt.add(z3.PbLe([(b, 1) for b in sels], 1))
                # Soft: prefer selecting at least one for this clue
                opt.add_soft(z3.Or(sels), weight="1")

        return opt

    def apply(self):
        opt = self.build_optimize()
        res = opt.check()
        if res not in (z3.sat, z3.unknown):
            return None

        model = opt.model()
        filled = [[None for _ in range(self.n)] for _ in range(self.n)]

        # Place only words that were actually selected
        for (x, y, direction), clue in self.start_positions.items():
            for idx, cand in enumerate(self.CLUES[clue]):
                s = self.sel_vars[(clue, idx)]
                if z3.is_true(model.eval(s, model_completion=True)):  # <-- IMPORTANT
                    r, c = x, y
                    for ch in cand:
                        filled[r][c] = ch
                        if direction == "D": r += 1
                        else: c += 1
                    break
        return filled

    def solution(self):
        return self.apply()


def main_solving(crossword_grid, clues):
    """
    Function to solve the crossword puzzle given the grid and clues.
    Returns the filled grid (2D list with characters or None).
    """
    solver = Solve(crossword_grid=crossword_grid, clues=clues)
    solution = solver.solution()
    print("Solution:")
    if solution is None:
        print("No solution found")
        return None
    printable = [[("" if v is None else v) for v in row] for row in solution]
    print(tabulate(printable, tablefmt="grid"))
    return solution
