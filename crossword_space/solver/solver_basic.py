from ast import literal_eval
from tabulate import tabulate
# from file_path import *
import json
import z3
# from schema import CROSSWORD_GRID

"""
NOTE: No length-verification is required in Solve(),
      as all clues retrieved from CLUES_PATH are of required length
"""

# Pattern is (initial_X, initial_Y), direction(D or A), length
# "": {"start":(), "direction":"", "length": },
# GRID = CROSSWORD_GRID

class Solve():
	def __init__(self, crossword_grid, clues):
		"""Initializes the solver with the crossword grid and clues.
		"""
		self.GRID = crossword_grid
		self.clues = clues
		# print(crossword_grid, clues)

	# def fetch_clues(self):
	# 	"""Fetches the clues present in "clues.json"
	# 	"""
	# 	clues = dict()
	# 	with open(CLUES_PATH) as fp:
	# 		clues = json.load(fp)
	# 	return clues

	def get_matrix(self):
		clues = self.GRID
		start_positions = {
			(info["start"][0], info["start"][1], info["direction"]): clue
			for clue, info in clues.items()
		}

		# Compute max size
		max_row = max(
			info["start"][0] + (info["length"] if info["direction"] == "D" else 1)
			for info in clues.values()
		)
		max_col = max(
			info["start"][1] + (info["length"] if info["direction"] == "A" else 1)
			for info in clues.values()
		)
		max_val = max(max_row, max_col)

		matrix = [[None for _ in range(max_val)] for _ in range(max_val)]

		for (x, y, direction), clue in start_positions.items():
			length = clues[clue]["length"]
			for i in range(length):
				if direction == "A":
					cell = matrix[x][y + i]
					if isinstance(cell, z3.z3.ArithRef):
						matrix[x][y + i] = (
							z3.Int(f"alpha_{x}_{y + i}_1"),
							z3.Int(f"alpha_{x}_{y + i}_2"),
						)
					elif isinstance(cell, tuple):
						# already intersection – do nothing
						pass
					else:
						matrix[x][y + i] = z3.Int(f"alpha_{x}_{y + i}")
				else:  # "D"
					cell = matrix[x + i][y]
					if isinstance(cell, z3.z3.ArithRef):
						matrix[x + i][y] = (
							z3.Int(f"alpha_{x + i}_{y}_1"),
							z3.Int(f"alpha_{x + i}_{y}_2"),
						)
					elif isinstance(cell, tuple):
						pass
					else:
						matrix[x + i][y] = z3.Int(f"alpha_{x + i}_{y}")
						
		return matrix, start_positions, max_val


	def convert_clues_code(self):
		"""converts all the clues (in string format) to a list of their ascii characters value
		   example: "mumbai" -> [109, 117, 109, 98, 97, 105]
		"""
		clues = self.clues
		clues_ord = dict()

		for clue in clues:
			for guess in clues[clue]:
				try:
					clues_ord[clue].append([ord(ch) for ch in guess.lower()])
				except:
					clues_ord[clue] = [[ord(ch) for ch in guess.lower()]]

		return clues_ord

	def make_guess_constraint(self):
		clues = self.convert_clues_code()
		matrix, start_positions, max_val = self.get_matrix()
		# print("constraint", matrix, start_positions, max_val)

		clue_constraint = list()

		for (x_index, y_index, direction), clue in start_positions.items():
			pos_info = self.GRID[clue]
			guesses_ord = clues[clue]
			all_guesses_constraint = []

			x_i, y_i = x_index, y_index

			for guess_ord in guesses_ord:
				if len(guess_ord) != pos_info["length"]:
					continue  # safety check
				guess_constraint = []
				x_i, y_i = x_index, y_index

				for ch in guess_ord:
					if not (0 <= x_i < len(matrix) and 0 <= y_i < len(matrix[0])):
						break  # skip guess if out-of-bounds

					cell = matrix[x_i][y_i]
					if isinstance(cell, tuple):
						matrix_val = cell[0] if direction == "D" else cell[1]
					else:
						matrix_val = cell

					guess_constraint.append(matrix_val == ch)

					if direction == "D":
						x_i += 1
					else:
						y_i += 1

				if len(guess_constraint) == len(guess_ord):
					all_guesses_constraint.append(z3.And(guess_constraint))

			clue_constraint.append(z3.Or(all_guesses_constraint))


		clues_constraint = z3.And(clue_constraint)

		return clues_constraint

	def common_position_constraint(self):
		clues = self.convert_clues_code()
		matrix, start_positions, max_val = self.get_matrix()

		equality_constraint = list()

		for x_index in range(max_val):
			for y_index in range(max_val):
				if isinstance(matrix[x_index][y_index], tuple):
					first = matrix[x_index][y_index][0]
					second = matrix[x_index][y_index][1]

					equality_constraint.append(z3.And(first == second))

		common_position_constraint = z3.And(equality_constraint)

		return common_position_constraint

	def apply_constraints(self):
		solver = z3.Solver()
		solver.add(self.make_guess_constraint())
		solver.add(self.common_position_constraint())
		solver.check()
		
		return solver.model()

	def solution(self):
		solved = self.apply_constraints()
		solved_keys = [index.name() for index in solved]
		matrix, start_positions, max_val = self.get_matrix()

		for x_index in range(max_val):
			for y_index in range(max_val):
				if isinstance(matrix[x_index][y_index], tuple):
					matrix_val = matrix[x_index][y_index][0]
				elif matrix[x_index][y_index] != None:
					matrix_val = matrix[x_index][y_index]
				else:
					continue

				no = solved[matrix_val]
				ch = chr(no.as_long())
				matrix[x_index][y_index] = ch

		return matrix
	
def main_solving(crossword_grid, clues):
	"""
	Function to solve the crossword puzzle given the grid and clues.
	"""
	solver = Solve(crossword_grid=crossword_grid, clues=clues)
	solution = solver.solution()
	print("Solution:")
	print(tabulate(solution, tablefmt="grid"))
	return solution

# if __name__ == '__main__':
# 	# solve = Solve()
# 	# print(tabulate(solve.solution()))
