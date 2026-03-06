"""Levenshtein distance clue-to-clue baseline.

Ranks training clues by edit distance to the test clue, then returns
their associated answers.

Usage:
    python -m scripts.baselines.baseline_levenshtein \
        -tr datasets/dict_train.csv \
        -ts datasets/dict_test.csv \
        -i indices.json \
        -c crossword
"""

import argparse
import json

import pandas as pd
from Levenshtein import distance
from tqdm import tqdm


def parse_arg():
    parser = argparse.ArgumentParser(description="Levenshtein distance clue-to-clue baseline")
    parser.add_argument("-tr", "--train", type=str, help="Training CSV file")
    parser.add_argument("-ts", "--test", type=str, help="Test CSV file")
    parser.add_argument("-i", "--indices", type=str, help="JSON with sampling indices per config")
    parser.add_argument("-c", "--config", type=str, help='Evaluation config: "crossword", "dict", "neo"')
    return parser.parse_args()


def baseline_model(train_data, test_data, indices_file, config):
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)

    train_df = train_df[train_df["data_source"].isin(["crossword", "dict"])]
    test_df = test_df[test_df["data_source"] == config]

    with open(indices_file, "r") as f:
        sample_idx = json.load(f)[config]

    test_df = test_df.iloc[sample_idx]

    top_k_hits = {1: 0, 10: 0, 100: 0, 1000: 0}
    top_k_hits_star = {1: 0, 10: 0, 100: 0, 1000: 0}
    reciprocal_ranks = []
    reciprocal_ranks_star = []

    train_df = train_df.dropna(subset=["source", "target"]).copy()

    prediction_rows = []
    for _, test_row in tqdm(test_df.iterrows(), total=len(test_df)):
        test_source = test_row["source"]
        gold_target = test_row["target"]
        target_len = len(gold_target)

        distances = train_df["source"].apply(lambda x: distance(test_source, x))

        top_k_idx = distances.nsmallest(1000).index
        retrieved_targets = train_df.loc[top_k_idx, "target"].tolist()
        ranked_candidates = list(dict.fromkeys(retrieved_targets))

        if gold_target in ranked_candidates:
            rank = ranked_candidates.index(gold_target) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

        for k in top_k_hits:
            if gold_target in ranked_candidates[:k]:
                top_k_hits[k] += 1

        # Length-constrained
        filtered_train = train_df[train_df["target"].str.len() == target_len]
        ranked_candidates_star = []
        if not filtered_train.empty:
            filtered_distances = filtered_train["source"].apply(lambda x: distance(test_source, x))
            top_k_filtered_idx = filtered_distances.nsmallest(1000).index
            filtered_targets = filtered_train.loc[top_k_filtered_idx, "target"].tolist()
            ranked_candidates_star = list(dict.fromkeys(filtered_targets))

            for k in top_k_hits_star:
                if gold_target in ranked_candidates_star[:k]:
                    top_k_hits_star[k] += 1

            if gold_target in ranked_candidates_star:
                rr_star = 1 / (ranked_candidates_star.index(gold_target) + 1)
            else:
                rr_star = 0.0
            reciprocal_ranks_star.append(rr_star)
        else:
            reciprocal_ranks_star.append(0.0)

        prediction_rows.append({
            "source": test_source,
            "pred": ranked_candidates[:5],
            "pred_len": ranked_candidates_star[:5],
            "target": gold_target,
        })

    print("\n--- Regular ---")
    for k in top_k_hits:
        acc = top_k_hits[k] / len(test_df)
        print(f"Top-{k} accuracy: {acc:.4f}")
    print(f"MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}")

    print("\n--- Length-constrained ---")
    for k in top_k_hits_star:
        acc_star = top_k_hits_star[k] / len(test_df)
        print(f"Top-{k}* accuracy: {acc_star:.4f}")
    print(f"MRR*: {sum(reciprocal_ranks_star) / len(reciprocal_ranks_star):.4f}")

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(f"results/baseline_lev/predictions_{config}.csv", index=False)


if __name__ == "__main__":
    args = parse_arg()
    baseline_model(args.train, args.test, args.indices, args.config)
