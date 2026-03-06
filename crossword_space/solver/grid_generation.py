import argparse

from .crossword import Crossword
from .timer import Timer
from .fill_algorithm import *
from . import DATA_DIR
import numpy as np
import pandas as pd
import json
import sys

# def generate_mask_with_length_constraints(rows=15, cols=15, num_blocks=40, seed=None, min_len=2, max_len=11):
#     """
#     Generates a grid mask with black boxes placed randomly and then adjusted
#     to ensure word lengths fall within [min_len, max_len].
#     """
#     if seed is not None:
#         np.random.seed(seed)

#     # Initial random placement of black squares
#     mask = np.zeros((rows, cols), dtype=int)
#     empty_cells = [(i, j) for i in range(rows) for j in range(cols)]
#     np.random.shuffle(empty_cells)
#     for i, (r, c) in enumerate(empty_cells[:num_blocks]):
#         mask[r, c] = 1

#     def enforce_length_constraints(grid):
#         # Ensure each row has white spans only in [min_len, max_len]
#         for r in range(grid.shape[0]):
#             span = 0
#             for c in range(grid.shape[1] + 1):
#                 if c < grid.shape[1] and grid[r, c] == 0:
#                     span += 1
#                 else:
#                     if span < min_len:
#                         for k in range(c - span, c):
#                             grid[r, k] = 1  # force black
#                     elif span > max_len:
#                         # Insert black squares to split the span
#                         insert_positions = list(range(c - span + max_len, c, max_len))
#                         for pos in insert_positions:
#                             grid[r, pos] = 1
#                     span = 0

#         # Same for columns
#         for c in range(grid.shape[1]):
#             span = 0
#             for r in range(grid.shape[0] + 1):
#                 if r < grid.shape[0] and grid[r, c] == 0:
#                     span += 1
#                 else:
#                     if span < min_len:
#                         for k in range(r - span, r):
#                             grid[k, c] = 1
#                     elif span > max_len:
#                         insert_positions = list(range(r - span + max_len, r, max_len))
#                         for pos in insert_positions:
#                             grid[pos, c] = 1
#                     span = 0
#         return grid

#     mask = enforce_length_constraints(mask)
#     return mask

def generate_mask_with_length_constraints(rows=15, cols=15, num_blocks=40, seed=None, min_len=2, max_len=11):
    """
    Generates a symmetric crossword mask with black boxes placed randomly
    and adjusted to ensure word lengths fall within [min_len, max_len].
    Symmetry is 180° rotational (standard in crosswords).
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure num_blocks is even for symmetry
    if num_blocks % 2 != 0:
        num_blocks -= 1

    mask = np.zeros((rows, cols), dtype=int)
    empty_cells = [(i, j) for i in range(rows) for j in range(cols)
                   if (i < rows // 2 or (i == rows // 2 and j <= cols // 2))]  # only fill half for symmetry
    np.random.shuffle(empty_cells)

    placed = 0
    for r, c in empty_cells:
        if placed >= num_blocks // 2:
            break
        mask[r, c] = 1
        mirror_r, mirror_c = rows - 1 - r, cols - 1 - c
        mask[mirror_r, mirror_c] = 1
        placed += 1

    def enforce_length_constraints(grid):
        for r in range(grid.shape[0]):
            span = 0
            for c in range(grid.shape[1] + 1):
                if c < grid.shape[1] and grid[r, c] == 0:
                    span += 1
                else:
                    if span < min_len:
                        for k in range(c - span, c):
                            grid[r, k] = 1
                    elif span > max_len:
                        for pos in range(c - span + max_len, c, max_len):
                            grid[r, pos] = 1
                    span = 0

        for c in range(grid.shape[1]):
            span = 0
            for r in range(grid.shape[0] + 1):
                if r < grid.shape[0] and grid[r, c] == 0:
                    span += 1
                else:
                    if span < min_len:
                        for k in range(r - span, r):
                            grid[k, c] = 1
                    elif span > max_len:
                        for pos in range(r - span + max_len, r, max_len):
                            grid[pos, c] = 1
                    span = 0
        return grid

    # mask = enforce_length_constraints(mask)
    # mask eventual single white cells
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] == 0: # 0 is when it's white
                left = c - 1
                right = c + 1
                up = r - 1
                down = r + 1
                if (left < 0 or mask[r, left] == 1) and (right >= cols or mask[r, right] == 1) and (up < 0 or mask[up, c] == 1) and (down >= rows or mask[down, c] == 1):
                    mask[r, c] = 1
                    mirror_r, mirror_c = rows - 1 - r, cols - 1 - c
                    mask[mirror_r, mirror_c] = 1
    return mask


# --- CONFIGURATION ---
parser = argparse.ArgumentParser(description="Generate crossword grids from a dataset CSV.")
parser.add_argument("split", choices=["train", "val", "test"], help="Dataset split to use.")
parser.add_argument("--dataset-csv", type=str, required=True,
                    help="Path to the dataset CSV file (must contain an 'answer' column).")
args = parser.parse_args()

data_source = args.split
rows_cols_tuple = [(5, 5), (7, 7), (9, 9), (11, 11), (13, 13)]  # rows, cols
rows, cols = rows_cols_tuple[0]
num_blocks_list = [0.15, 0.16, 0.22, 0.27, 0.27]  # Percentage of blocks in the grid
if data_source == "train":
    number_of_crosswords = [300, 150, 25, 15, 10] # 500
elif data_source == "val":
    number_of_crosswords = [10, 10, 10, 10, 10] # 50
elif data_source == "test":
    number_of_crosswords = [10, 10, 10, 10, 10] # 50
min_word_len = 2
max_word_len = 12
seed = None

wordlist_file = 'test_word_list'

df = pd.read_csv(args.dataset_csv)
with open(DATA_DIR / "wordlists" / "all_words.txt", 'r') as f:
    allowed_words = f.read().splitlines()

allowed_words = set(allowed_words)
word_list = sorted(set(df["answer"].tolist()))
word_list = [w for w in word_list if w in allowed_words]

with open(DATA_DIR / "wordlists" / "test_word_list.txt", 'w') as f:
    for word in word_list:
        f.write(word + '\n')


for i in range(5): # 5 grid sizes

    rows, cols = rows_cols_tuple[i]  # Change this to select different grid sizes
    num_blocks = num_blocks_list[i] * rows * cols
    n = number_of_crosswords[i]  # Number of crosswords to generate for this grid size

    # Try until a valid crossword is constructed
    for i in range(n): # n crosswords per interval
        flag = False 
        while not flag:
            # Generate mask and convert to printable grid
            mask = generate_mask_with_length_constraints(rows, cols, num_blocks, seed, min_word_len, max_word_len)
            test_grid = np.where(mask == 1, ".", " ")

            result, used_words = IntelligentLookahead.construct(rows, cols, test_grid.tolist(), wordlist_file, data_source)
            if result:
                print("SUCCESS")
                print("Used words:", used_words)
                flag = True
            else:
                print(result, used_words)
                print("FAILED")

            Timer.outputAll()