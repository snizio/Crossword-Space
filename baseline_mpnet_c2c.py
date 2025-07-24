from transformers import AutoTokenizer, AutoModel
from crossword_utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
import json
import numpy as np
import torch
import pandas as pd
from collections import defaultdict

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

df_all = pd.read_csv("datasets/dict_train.csv")
clue_to_word_dict = {}
clue_lengths = {}
clues = []

for _, row in df_all.iterrows():
    clue = row['source']
    word = row['target']
    clue_to_word_dict[clue] = word
    clue_lengths[clue] = len(word)
    clues.append(clue)

train_data = CrosswordDataset("datasets/dict_train.csv", tokenizer)
dataset_loader = DataLoader(train_data, batch_size=256, shuffle=False)

clue_embeddings = []
with torch.no_grad():
    for batch in tqdm(dataset_loader, desc="Encoding clue bank"):
        clue_input_ids = batch['def_input_ids'].to(device)
        clue_attention_mask = batch['def_attention_mask'].to(device)
        model_output = model(input_ids=clue_input_ids, attention_mask=clue_attention_mask)
        sentence_embeddings = mean_pooling(model_output, clue_attention_mask)
        clue_embeddings.append(sentence_embeddings.cpu())

item_embeddings = torch.cat(clue_embeddings).numpy()
item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)

clue_to_idx = {clue: idx for idx, clue in enumerate(clues)}
embedding_dim = item_embeddings.shape[1]

global_index = faiss.IndexFlatIP(embedding_dim)
global_index.add(item_embeddings)

length_faiss_raw = defaultdict(lambda: ([], []))  # length → (clues[], vectors[])
for clue in clues:
    word_len = clue_lengths[clue]
    idx = clue_to_idx[clue]
    length_faiss_raw[word_len][0].append(clue)
    length_faiss_raw[word_len][1].append(item_embeddings[idx])

length_indices = {}
for word_len, (clue_list, vecs) in length_faiss_raw.items():
    matrix = np.stack(vecs)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    length_indices[word_len] = (clue_list, index)

for data_source in ["crossword", "dict", "onli", "neo"]:
    test_data = CrosswordDataset("datasets/dict_test.csv", tokenizer, data_source=[f"{data_source}"])
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    sources, targets = [], []
    full_predictions = [None] * len(test_data)
    filtered_predictions = [None] * len(test_data)

    k_clues = 1000
    eval_progress_bar = tqdm(total=len(test_loader), desc=f"Evaluating {data_source}...")

    with torch.no_grad():
        offset = 0
        for batch in test_loader:
            bsz = len(batch["def_texts"])
            def_input_ids = batch['def_input_ids'].to(device)
            def_attention_mask = batch['def_attention_mask'].to(device)
            def_texts = batch["def_texts"]
            ans_texts = batch["ans_texts"]

            sources.extend(def_texts)
            targets.extend(ans_texts)

            model_output = model(input_ids=def_input_ids, attention_mask=def_attention_mask)
            def_embeddings = mean_pooling(model_output, def_attention_mask)
            def_embeddings = def_embeddings.cpu().numpy()
            def_embeddings = def_embeddings / np.linalg.norm(def_embeddings, axis=1, keepdims=True)

            D_full, I_full = global_index.search(def_embeddings, k_clues)
            for i in range(bsz):
                retrieved_clues = [clues[idx] for idx in I_full[i]]
                predicted_words = [clue_to_word_dict[clue] for clue in retrieved_clues]
                full_predictions[offset + i] = predicted_words

            grouped_queries = defaultdict(list)
            idx_map = defaultdict(list)
            for i, vec in enumerate(def_embeddings):
                target_len = len(ans_texts[i])
                if target_len in length_indices:
                    grouped_queries[target_len].append(vec)
                    idx_map[target_len].append(offset + i)

            for target_len, vecs in grouped_queries.items():
                clue_list, faiss_index = length_indices[target_len]
                vecs_np = np.stack(vecs)
                D_len, I_len = faiss_index.search(vecs_np, k_clues)
                for j, row in enumerate(I_len):
                    original_idx = idx_map[target_len][j]
                    retrieved_clues = [clue_list[x] for x in row]
                    predicted_words = [clue_to_word_dict[clue] for clue in retrieved_clues]
                    filtered_predictions[original_idx] = predicted_words

            offset += bsz
            eval_progress_bar.update(1)


    df_preds = pd.DataFrame({
        "source": sources,
        "pred": full_predictions,
        "pred_len": filtered_predictions,
        "target": targets
    })

    df_preds.to_csv(f"baseline_results/predictions-{data_source}.csv", index=False)

    def compute_metrics(pred_column):
        acc1 = acc10 = acc100 = acc1000 = mrr = 0
        for preds, target in zip(df_preds[pred_column], df_preds["target"]):
            if target in preds:
                rank = preds.index(target) + 1
                mrr += 1 / rank
            if target == preds[0]: acc1 += 1
            if target in preds[:10]: acc10 += 1
            if target in preds[:100]: acc100 += 1
            if target in preds[:1000]: acc1000 += 1
        total = len(df_preds)
        suffix = '-len' if pred_column == 'pred_len' else ''
        return {
            f"acc@1{suffix}": acc1 / total,
            f"acc@10{suffix}": acc10 / total,
            f"acc@100{suffix}": acc100 / total,
            f"acc@1000{suffix}": acc1000 / total,
            f"MRR{suffix}": mrr / total
        }

    results_all = compute_metrics("pred")
    results_len = compute_metrics("pred_len")
    results_all.update(results_len)

    with open(f"baseline_results/results-{data_source}.json", "w") as f:
        json.dump(results_all, f, indent=2)
