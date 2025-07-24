import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_scheduler
from torch.utils.data import DataLoader
from crossword_utils import (CrosswordDataset, DualEncoderADE, 
                            DualEncoderSDE, symmetric_contrastive_loss,
                            info_nce_loss, symmetric_contrastive_loss_with_hard_negatives,
                            linear_decay_hard_negatives_fraction, UniqueLabelBatchSampler)
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import logging
import sys 
import faiss
import numpy as np
import json
import gc
from collections import defaultdict

# folder path, model url, device, sde/ade

checkpoint_folder_path = sys.argv[1]

logging.basicConfig(
    filename=f'{checkpoint_folder_path}training_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def evaluate_model(model, dataloader, device_def, device_ans):
    model.eval()
    val_loss = 0.0
    eval_progress_bar = tqdm(total=len(dataloader), desc="Evaluating...")
    with torch.no_grad():
        for batch in dataloader:
            def_input_ids = batch['def_input_ids'].to(device_def)
            def_attention_mask = batch['def_attention_mask'].to(device_def)
            ans_input_ids = batch['ans_input_ids'].to(device_ans)
            ans_attention_mask = batch['ans_attention_mask'].to(device_ans)

            def_proj, ans_proj = model(
                def_input_ids=def_input_ids,
                def_attention_mask=def_attention_mask,
                ans_input_ids=ans_input_ids,
                ans_attention_mask=ans_attention_mask
            )

            # loss = symmetric_contrastive_loss(def_proj, ans_proj, model.temperature)
            loss = symmetric_contrastive_loss(def_proj, ans_proj, model.temperature)
            val_loss += loss.item()
            eval_progress_bar.update(1)
            # logit scaling set as max 100
            dual_encoder.temperature.data = torch.clamp(dual_encoder.temperature.data, 0, 4.6052)

    avg_val_loss = val_loss / len(dataloader)
    return avg_val_loss
    

# model_name = "nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl"
model_name = sys.argv[2]

parallel = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


if parallel == "parallel":
    device_def = torch.device("cuda:0")
    device_ans = torch.device("cuda:1")
elif parallel == "not-parallel":
    device_def = torch.device(f"cuda")
    device_ans = torch.device(f"cuda")

projection_dim = 768  # Chosen embedding size for shared space

tied_conf = sys.argv[4]

if tied_conf == "ade": # we load two encoders

    if "it5" in model_name or "NeoIT5" in model_name:
        encoder_def = AutoModel.from_pretrained(model_name, trust_remote_code=True).encoder
        encoder_ans = AutoModel.from_pretrained(model_name, trust_remote_code=True).encoder
    else:
        encoder_def = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        encoder_ans = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    encoder_def.to(device_def)
    encoder_ans.to(device_ans)

    print(encoder_def.config_class)

    hidden_dim = encoder_def.config.hidden_size  # usually 768 for BERT base models

    dual_encoder = DualEncoderADE(encoder_def, encoder_ans, hidden_dim, projection_dim, device_def, device_ans)

elif tied_conf == "sde": # for siamese we load only one

    if "it5" in model_name or "NeoIT5" in model_name:
        encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True).encoder
    else:
        encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    encoder.to(device_def) # device def and deviceans must be the same for sde

    print(encoder.config_class)

    hidden_dim = encoder.config.hidden_size  # usually 768 for BERT base models
    dual_encoder = DualEncoderSDE(encoder, hidden_dim, projection_dim, device_def)


train_csv = 'datasets/dict_train.csv'
val_csv = 'datasets/dict_val.csv'
test_csv = 'datasets/dict_test.csv'

if "dict" in checkpoint_folder_path:
    # Initialize datasets, if dict we train with dict but we val and test only on crossword to have comparable results
    train_dataset = CrosswordDataset(train_csv, tokenizer, 64, 16, ["crossword", "neo", "dict"])
    val_dataset = CrosswordDataset(val_csv, tokenizer, 64, 16, ["crossword"])
    test_dataset = CrosswordDataset(test_csv, tokenizer, 64, 16, ["crossword"])
else:
    train_dataset = CrosswordDataset(train_csv, tokenizer, 64, 16, ["crossword"])
    val_dataset = CrosswordDataset(val_csv, tokenizer, 64, 16, ["crossword"])
    test_dataset = CrosswordDataset(test_csv, tokenizer, 64, 16, ["crossword"])

# Initialize dataloaders
batch_size = 256

train_sampler = UniqueLabelBatchSampler(train_dataset, batch_size=batch_size, drop_last=True)

train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# optimizer = torch.optim.Adam(dual_encoder.parameters(), lr=3e-5, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1) # optimizer

lr_dict = {
    "gsarti/it5-small": 5e-4,
    "gsarti/it5-base": 5e-4,
    "snizio/NeoIT5-base": 5e-4,
    "nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl": 2e-4,
    "DeepMount00/Italian-ModernBERT-base": 2e-5,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 2e-4,
    "EuroBERT/EuroBERT-210m": 5e-5
}

decay_dict = {
    "gsarti/it5-small": 1e-3,
    "gsarti/it5-base": 1e-3,
    "snizio/NeoIT5-base": 1e-3,
    "nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl": 1e-3,
    "DeepMount00/Italian-ModernBERT-base": 0.0,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 1e-3,
    "EuroBERT/EuroBERT-210m": 0.1
}   


if model_name == "EuroBERT/EuroBERT-210m":
    optimizer = torch.optim.Adam(dual_encoder.parameters(), lr=lr_dict[model_name], betas=(0.9, 0.95), eps=1e-5, weight_decay=decay_dict[model_name])
else:
    optimizer = torch.optim.AdamW(dual_encoder.parameters(), lr=lr_dict[model_name], weight_decay=decay_dict[model_name])

num_epochs = 6
num_training_steps = num_epochs * len(train_loader)
eval_steps = int(0.05 * num_training_steps)
lr_scheduler = get_scheduler(name = "linear", optimizer=optimizer, num_warmup_steps=0.05, num_training_steps=num_training_steps)

best_val_loss = float('inf')
saved_checkpoints = []

progress_bar = tqdm(total=num_training_steps, desc="Training")

global_step = 0

# hard_negative_fractions = 1.0

dual_encoder.train()

for epoch in range(num_epochs):

    train_loss = 0.0
    
    for step, batch in enumerate(train_loader):
        def_input_ids = batch['def_input_ids'].to(device_def)
        def_attention_mask = batch['def_attention_mask'].to(device_def)
        ans_input_ids = batch['ans_input_ids'].to(device_ans)
        ans_attention_mask = batch['ans_attention_mask'].to(device_ans)

        optimizer.zero_grad()

        def_proj, ans_proj = dual_encoder(
            def_input_ids=def_input_ids,
            def_attention_mask=def_attention_mask,
            ans_input_ids=ans_input_ids,
            ans_attention_mask=ans_attention_mask
        )

        # loss = symmetric_contrastive_loss(def_proj, ans_proj, dual_encoder.temperature)
        hard_negative_fractions = linear_decay_hard_negatives_fraction(global_step, num_training_steps, 0.8, 0.05)
        loss = symmetric_contrastive_loss_with_hard_negatives(def_proj, ans_proj, dual_encoder.temperature, hard_negative_fractions)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_postfix({'train_loss': loss.item()})
        global_step += 1        
        logging.info(f"Step: {global_step} Train Loss: {loss.item()}")
        writer.add_scalar("Loss/train", loss, global_step)
        writer.add_scalar("hard-neg-fraction", hard_negative_fractions, global_step)

        # Perform evaluation at specified steps
        if (step + 1) % eval_steps == 0:
            logging.info(f"Doing evaluation  for step {global_step}")
            avg_train_loss = train_loss / eval_steps
            avg_val_loss = evaluate_model(dual_encoder, val_loader, device_def, device_ans)
            writer.add_scalar("Loss/val", avg_val_loss, global_step)

            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{global_step}/{num_training_steps}], "
                         f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

            # Save the model if validation loss has improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = f"{checkpoint_folder_path}{global_step}.pth"
                torch.save({
                    'model_state_dict': dual_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'epoch': epoch,
                    'global_step': global_step
                }, checkpoint_path)
                logging.info(f"Model saved at epoch {epoch+1}, step {global_step}")
                saved_checkpoints.append(checkpoint_path)

                if len(saved_checkpoints) > 2:
                    oldest_checkpoint = saved_checkpoints.pop(0)
                    if os.path.exists(oldest_checkpoint):
                        os.remove(oldest_checkpoint)
                        logging.info(f"Deleted old checkpoint: {oldest_checkpoint}")

            train_loss = 0.0  # Reset training loss after evaluation
            dual_encoder.train()

progress_bar.close()
writer.flush()
writer.close()

# avg_test_loss = evaluate_model(dual_encoder, test_loader, device_def, device_ans)
# logging.info(f"Test Loss: {avg_test_loss:.4f}")

## EVALUATION

del dual_encoder
torch.cuda.empty_cache()
gc.collect()

# Load the best model
if tied_conf == "ade":
    dual_encoder = DualEncoderADE(encoder_def, encoder_ans, hidden_dim, projection_dim, device_def, device_ans)
elif tied_conf == "sde":
    dual_encoder = DualEncoderSDE(encoder, hidden_dim, projection_dim, device=device_def)
checkpoint = torch.load(checkpoint_path, weights_only=True)
dual_encoder.load_state_dict(checkpoint['model_state_dict'])

all_words = CrosswordDataset("all_unique_words.csv", tokenizer)

dual_encoder.eval()
batch_size = 256

dataset_loader = DataLoader(all_words, batch_size=batch_size, shuffle=False)

embeddings = []
word_lengths = []
with torch.no_grad():
    for batch in tqdm(dataset_loader, desc="Encoding items"):

        def_input_ids = batch['def_input_ids'].to(device_def)
        def_attention_mask = batch['def_attention_mask'].to(device_def)
        ans_input_ids = batch['ans_input_ids'].to(device_ans)
        ans_attention_mask = batch['ans_attention_mask'].to(device_ans)
        ans_texts = batch["ans_texts"]
        word_lengths.extend([len(word) for word in ans_texts])
        
        _, embedding_words_batch = dual_encoder(
            def_input_ids=def_input_ids,
            def_attention_mask=def_attention_mask,
            ans_input_ids=ans_input_ids, 
            ans_attention_mask=ans_attention_mask
        )
        embeddings.append(embedding_words_batch.cpu())

# Concatenate all embeddings and save
item_embeddings = torch.cat(embeddings).numpy()
torch.save(item_embeddings, f'{checkpoint_folder_path}all_embeddings.pt')

def calculate_mrr(predictions, targets):
    rr_sum = 0
    for pred, target in zip(predictions, targets):
        if target in pred:
            rank = pred.index(target) + 1
            rr_sum += 1 / rank
    return rr_sum / len(targets)

item_embeddings = item_embeddings / np.linalg.norm(item_embeddings, axis=1, keepdims=True)
embedding_dim = item_embeddings.shape[1]

length_indices_map = {}

unique_lengths = np.unique(word_lengths)
faiss_indices = {} # length indexes dict

for l in unique_lengths:
    idxs = np.where(word_lengths == l)[0]
    sub_index = faiss.IndexFlatIP(embedding_dim)
    sub_index.add(item_embeddings[idxs])
    faiss_indices[l] = sub_index
    length_indices_map[l] = idxs  # To map back to index_dict

index = faiss.IndexFlatIP(embedding_dim)
index.add(item_embeddings) # full index for all words

with open("index_dict.json", "r") as f:
    index_dict = json.load(f)

for data_source in ["crossword", "dict", "onli", "neo"]:

    test_data = CrosswordDataset("datasets/dict_test.csv", tokenizer, data_source=[f"{data_source}"])
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    eval_progress_bar = tqdm(total=len(test_loader), desc=f"Evaluating {data_source}...")

    k = 1000

    filtered_predictions = []
    full_predictions = []
    sources = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            def_input_ids = batch['def_input_ids'].to(device_def)
            def_attention_mask = batch['def_attention_mask'].to(device_def)
            ans_input_ids = batch['ans_input_ids'].to(device_ans)
            ans_attention_mask = batch['ans_attention_mask'].to(device_ans)
            def_texts = batch["def_texts"]
            ans_texts = batch["ans_texts"]
            sources.extend(def_texts)
            targets.extend(ans_texts)

            def_proj, _ = dual_encoder(
                def_input_ids=def_input_ids,
                def_attention_mask=def_attention_mask,
                ans_input_ids=ans_input_ids,
                ans_attention_mask=ans_attention_mask
            )

            def_proj = def_proj.cpu().numpy()
            def_proj = def_proj / np.linalg.norm(def_proj, axis=1, keepdims=True)

            # Full batched search
            D_full, I_full = index.search(def_proj, k)
            batch_full_predictions = [
                [index_dict[str(x)] for x in I_full[i]] for i in range(len(def_proj))]
            full_predictions.extend(batch_full_predictions)

            # Group queries by target length
            length_query_map = defaultdict(list)
            length_query_indices = defaultdict(list)

            for i, vec in enumerate(def_proj):
                target_len = len(ans_texts[i])
                if target_len in faiss_indices:
                    length_query_map[target_len].append(vec)
                    length_query_indices[target_len].append(i)

            # Preallocate filtered predictions
            batch_filtered_predictions = [None] * len(def_proj)

            # Perform batched search per length
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


    df_preds.to_csv(f"{checkpoint_folder_path}predictions-{data_source}.csv", index = False)

    def compute_metrics(pred_column):
        acc1 = acc10 = acc100 = acc1000 = mrr = 0
        for preds, target in zip(df_preds[pred_column], df_preds["target"]):
            if target in preds:
                rank = preds.index(target) + 1
                mrr += 1 / rank
            if target == preds[0]:
                acc1 += 1
            if target in preds[:10]:
                acc10 += 1
            if target in preds[:100]:
                acc100 += 1
            if target in preds[:1000]:
                acc1000 += 1
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

    with open(f"{checkpoint_folder_path}results-{data_source}.json", "w") as f:
        json.dump(results_all, f, indent=2)