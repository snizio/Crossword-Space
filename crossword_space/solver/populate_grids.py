import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from .solver_basic import *
from . import DATA_DIR
from z3.z3types import Z3Exception
import random
import json

# set seed for reproducibility
np.random.seed(42)

parser = argparse.ArgumentParser(description="Populate crossword grids with clues from a dataset CSV.")
parser.add_argument("split", choices=["train", "val", "test"], help="Dataset split to use.")
parser.add_argument("--dataset-csv", type=str, required=True,
                    help="Path to the dataset CSV file (must contain 'answer' and 'clue' columns).")
args = parser.parse_args()

split = args.split

test_df = pd.read_csv(args.dataset_csv)


with open(DATA_DIR / "crosswords_datasets" / f"{split}_grids_gold.txt", "r") as f:
    generated_crosses_temp = f.read().splitlines()

generated_crosswords = []

for line in generated_crosses_temp:
    if line.strip():  # Check if the line is not empty
        generated_cross = literal_eval(line)
        generated_crosswords.append(generated_cross)
print("Total generated crosswords:", len(generated_crosswords))


solved = []

for generated_cross in generated_crosswords:

    generated_cross = np.array(generated_cross)
    words = []

    # Extract horizontal words
    for row in range(generated_cross.shape[0]):
        word = ""
        start_col = 0
        for col in range(generated_cross.shape[1]):
            elem = generated_cross[row, col]
            if elem == ".":
                if len(word) > 1:
                    words.append((word, row, start_col, "A"))
                word = ""
                start_col = col + 1
            else:
                if word == "":
                    start_col = col
                word += elem
        if len(word) > 1:
            words.append((word, row, start_col, "A"))


    # Extract vertical words
    for col in range(generated_cross.shape[1]):
        word = ""
        start_row = 0
        for row in range(generated_cross.shape[0]):
            elem = generated_cross[row, col]
            if elem == ".":
                if len(word) > 1:
                    words.append((word, start_row, col, "D"))
                word = ""
                start_row = row + 1
            else:
                if word == "":
                    start_row = row
                word += elem
        if len(word) > 1:
            words.append((word, start_row, col, "D"))

    correct_words = [x[0] for x in words]

    clues = []
    used_defs = []
    for word, row, col, line_type in words:
        assert line_type in ["A", "D"]
        if line_type == "A":
            assert "".join(generated_cross[row, col:col+len(word)]) == word
        if line_type == "D":
            assert "".join(generated_cross[row:row+len(word), col]) == word
        definitions = test_df[test_df.answer == word.lower()].clue.tolist()
        definitions.extend(test_df[test_df.answer == word.upper()].clue.tolist())
        if definitions == []:
            print(word)
            raise
        random_definition = random.choice(definitions)
        clues.append({"target": word, "clue": random_definition, "row": row, "col": col, "direction": line_type, "length": len(word)})
        used_defs.append(random_definition)

    print(clues)

    if split != "test":
        with open(DATA_DIR / "crosswords_datasets" / f"{split}_cross_clues.json", "a") as f:
            f.write(json.dumps(clues) + "\n")
    else:
        with open(DATA_DIR / "crosswords_datasets" / f"{split}_cross_clues.json", "a") as f:
            clues_gold = []
            for clue in clues:
                clues_gold.append({k:v for k,v in clue.items() if k != "target"})
            f.write(json.dumps(clues_gold) + "\n")
        with open(DATA_DIR / "crosswords_datasets" / f"{split}_gold_cross_clues.json", "a") as f:
            f.write(json.dumps(clues) + "\n")