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

all_words = CrosswordDataset("all_unique_words.csv", tokenizer)
dataset_loader = DataLoader(all_words, batch_size=256, shuffle=False)

embeddings = []
word_lengths = []

with torch.no_grad():
    for batch in tqdm(dataset_loader, desc="Encoding items"):
        ans_input_ids = batch['ans_input_ids'].to(device)
        ans_attention_mask = batch['ans_attention_mask'].to(device)
        ans_texts = batch["ans_texts"]
        word_lengths.extend([len(word) for word in ans_texts])

        model_output = model(input_ids=ans_input_ids, attention_mask=ans_attention_mask)
        sentence_embeddings = mean_pooling(model_output, ans_attention_mask)
        embeddings.append(sentence_embeddings.cpu())

item_embeddings = torch.cat(embeddings).numpy()
torch.save(item_embeddings, 'baseline_results/all_embeddings.pt')

item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)
embedding_dim = item_embeddings.shape[1]

unique_lengths = np.unique(word_lengths)
faiss_indices = {}
length_indices_map = {}

for l in unique_lengths:
    idxs = np.where(np.array(word_lengths) == l)[0]
    sub_index = faiss.IndexFlatIP(embedding_dim)
    sub_index.add(item_embeddings[idxs])
    faiss_indices[l] = sub_index
    length_indices_map[l] = idxs

# Full index
index = faiss.IndexFlatIP(embedding_dim)
index.add(item_embeddings)

with open("index_dict.json", "r") as f:
    index_dict = json.load(f)

for data_source in ["crossword", "dict", "onli", "neo"]:
    test_data = CrosswordDataset("datasets/dict_test.csv", tokenizer, data_source=[f"{data_source}"])
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    sources, targets = [], []
    full_predictions, filtered_predictions = [], []

    eval_progress_bar = tqdm(total=len(test_loader), desc=f"Evaluating {data_source}...")
    k = 1000

    with torch.no_grad():
        for batch in test_loader:
            def_input_ids = batch['def_input_ids'].to(device)
            def_attention_mask = batch['def_attention_mask'].to(device)
            def_texts = batch["def_texts"]
            ans_texts = batch["ans_texts"]
            sources.extend(def_texts)
            targets.extend(ans_texts)

            model_output = model(input_ids=def_input_ids, attention_mask=def_attention_mask)
            sentence_embeddings = mean_pooling(model_output, def_attention_mask)
            sentence_embeddings = sentence_embeddings.cpu().numpy()
            sentence_embeddings = sentence_embeddings / np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)

            # Full search
            D_full, I_full = index.search(sentence_embeddings, k)
            batch_full_predictions = [
                [index_dict[str(x)] for x in I_full[i]] for i in range(len(sentence_embeddings))
            ]
            full_predictions.extend(batch_full_predictions)

            # Group by length
            length_query_map = defaultdict(list)
            length_query_indices = defaultdict(list)

            for i, vec in enumerate(sentence_embeddings):
                target_len = len(ans_texts[i])
                if target_len in faiss_indices:
                    length_query_map[target_len].append(vec)
                    length_query_indices[target_len].append(i)

            batch_filtered_predictions = [None] * len(sentence_embeddings)

            for target_len, queries in length_query_map.items():
                queries_np = np.stack(queries)
                idxs = length_indices_map[target_len]
                sub_index = faiss_indices[target_len]
                D_len, I_len = sub_index.search(queries_np, k)
                for pos, row in enumerate(I_len):
                    i = length_query_indices[target_len][pos]
                    batch_filtered_predictions[i] = [index_dict[str(idxs[x])] for x in row]

            filtered_predictions.extend(batch_filtered_predictions)
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
        if pred_column == "pred_len":
            return {
                "acc@1-len": acc1 / total,
                "acc@10-len": acc10 / total,
                "acc@100-len": acc100 / total,
                "acc@1000-len": acc1000 / total,
                "MRR-len": mrr / total
            }
        else:
            return {
                "acc@1": acc1 / total,
                "acc@10": acc10 / total,
                "acc@100": acc100 / total,
                "acc@1000": acc1000 / total,
                "MRR": mrr / total
            }

    results_all = compute_metrics("pred")
    results_len = compute_metrics("pred_len")
    results_all.update(results_len)

    with open(f"baseline_results/results-{data_source}.json", "w") as f:
        json.dump(results_all, f, indent=2)
