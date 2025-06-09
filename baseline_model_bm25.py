import argparse

import json

from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pandas as pd

def parse_arg():
    parser = argparse.ArgumentParser(description='Code for reproducing the c2c model based on BM25')
    parser.add_argument('-tr', '--train', type=str, 
                        help='Training file')
    parser.add_argument('-ts', '--test', type=str, 
                        help='Test file')
    parser.add_argument('-i', '--indices', type=str, 
                        help='File with indexes for sampling the test set')
    parser.add_argument('-c', '--config', type=str,
                        help='Name of the configuration for the evaluation (values are "crossword", "dict", "neo")')

    return parser.parse_args()

def baseline_model(train_data, test_data, indices_file, config):
    train_df = pd.read_csv(train_data)
    test_df = pd.read_csv(test_data)

    # Filter training and test data sources
    train_df = train_df[train_df["data_source"].isin(["crossword", "dict"])]
    test_df = test_df[test_df["data_source"] == config]

    with open(indices_file, "r") as f:
        sample_idx = json.load(f)[config]
    
    # Sample 100 instances from the test set for evaluation (comment if running on full test set)
    test_df = test_df.iloc[sample_idx]

    # Drop NaN values in 'source' and 'target' columns
    train_df = train_df.dropna(subset=['source', 'target']).copy()
    test_df = test_df.dropna(subset=['source', 'target']).copy()

    # Tokenize all training sources
    train_df['tokenized_source'] = train_df['source'].apply(lambda x: x.split())
    bm25 = BM25Okapi(train_df['tokenized_source'].tolist())

    top_k_hits = {1: 0, 10: 0, 100: 0, 1000: 0}
    top_k_hits_star = {1: 0, 10: 0, 100: 0, 1000: 0}
    reciprocal_ranks = []
    reciprocal_ranks_star = []

    prediction_rows = []  # for saving outputs
    for _, test_row in tqdm(test_df.iterrows(), total=len(test_df)):
        test_source = test_row['source']
        gold_target = test_row['target']
        target_len = len(gold_target)
        tokenized_query = test_source.split()

        # -------- Regular BM25 --------
        doc_scores = bm25.get_scores(tokenized_query)
        score_series = pd.Series(doc_scores, index=train_df.index)
        top_k_idx = score_series.nlargest(1000).index
        retrieved_targets = train_df.loc[top_k_idx, 'target'].tolist()
        ranked_candidates = list(dict.fromkeys(retrieved_targets))

        for k in top_k_hits:
            if gold_target in ranked_candidates[:k]:
                top_k_hits[k] += 1
        if gold_target in ranked_candidates:
            reciprocal_ranks.append(1 / (ranked_candidates.index(gold_target) + 1))
        else:
            reciprocal_ranks.append(0.0)

        # -------- * variant (length-constrained BM25) --------
        filtered_train = train_df[train_df['target'].str.len() == target_len]
        if not filtered_train.empty:
            filtered_tokenized = filtered_train['tokenized_source'].tolist()
            bm25_filtered = BM25Okapi(filtered_tokenized)

            doc_scores_star = bm25_filtered.get_scores(tokenized_query)
            score_series_star = pd.Series(doc_scores_star, index=filtered_train.index)
            top_k_star_idx = score_series_star.nlargest(1000).index
            retrieved_targets_star = filtered_train.loc[top_k_star_idx, 'target'].tolist()
            ranked_candidates_star = list(dict.fromkeys(retrieved_targets_star))

            for k in top_k_hits_star:
                if gold_target in ranked_candidates_star[:k]:
                    top_k_hits_star[k] += 1
            if gold_target in ranked_candidates_star:
                reciprocal_ranks_star.append(1 / (ranked_candidates_star.index(gold_target) + 1))
            else:
                reciprocal_ranks_star.append(0.0)
        else:
            reciprocal_ranks_star.append(0.0)

        # Save top-5 predictions
        prediction_rows.append({
            "source": test_source,
            "pred": ranked_candidates[:5],
            "pred_len": ranked_candidates_star[:5],
            "target": gold_target
        })

    print("\n--- Regular BM25 ---")
    for k in top_k_hits:
        print(f"Top-{k} accuracy: {top_k_hits[k] / len(test_df):.4f}")
    print(f"MRR: {sum(reciprocal_ranks) / len(reciprocal_ranks):.4f}")

    print("\n--- Length-constrained BM25 (*) ---")
    for k in top_k_hits_star:
        print(f"Top-{k}* accuracy: {top_k_hits_star[k] / len(test_df):.4f}")
    print(f"MRR*: {sum(reciprocal_ranks_star) / len(reciprocal_ranks_star):.4f}")

    # Save predictions to DataFrame
    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv("output/predictions_bm25_" + config + ".csv", index=False)

if __name__ == "__main__":
    args = parse_arg()
    train_data = args.train
    test_data = args.test
    indices_file = args.indices
    config = args.config

    baseline_model(train_data, test_data, indices_file, config)
