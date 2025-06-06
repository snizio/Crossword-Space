from torch.utils.data import Dataset
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler
import random
from collections import defaultdict


class UniqueLabelBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last, generator: torch.Generator = None, seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.generator = generator
        self.seed = seed

    def __iter__(self):
        """
        Iterate over the remaining non-yielded indices. For each index, check if the sample values are already in the
        batch. If not, add the sample values to the batch keep going until the batch is full. If the batch is full, yield
        the batch indices and continue with the next batch.
        """
        if self.generator and self.seed:
            self.generator.manual_seed(self.seed + self.epoch)

        # We create a dictionary to None because we need a data structure that:
        # 1. Allows for cheap removal of elements
        # 2. Preserves the order of elements, i.e. remains random
        remaining_indices = dict.fromkeys(torch.randperm(len(self.dataset), generator=self.generator).tolist())

        while remaining_indices:
            batch_values = set()
            batch_indices = []
            for index in remaining_indices:
                sample_values = {
                    self.dataset[index]["ans_texts"]
                }
                if sample_values & batch_values:
                    continue

                batch_indices.append(index)
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                    break

                batch_values.update(sample_values)

            else:
                # NOTE: some indices might still have been ignored here
                if not self.drop_last:
                    yield batch_indices

            for index in batch_indices:
                del remaining_indices[index]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def linear_decay_hard_negatives_fraction(step, total_steps, v_start, v_end):
    step = min(step, total_steps) 
    return v_start - (step / total_steps) * (v_start - v_end)


def mean_pooling(token_embeddings, attention_mask):
    
    """Implementazione di sentence transformers"""

    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

    sum_mask = input_mask_expanded.sum(1)

    sum_mask = torch.clamp(sum_mask, min=1e-9)

    return sum_embeddings / sum_mask

def info_nce_loss(def_proj, ans_proj, temperature):
    # we l2 norm
    def_proj_norm = F.normalize(def_proj, dim=-1)
    ans_proj_norm = F.normalize(ans_proj, dim=-1)

    logits = torch.matmul(def_proj_norm, ans_proj_norm.T) / temperature # better to have learnable temperature
    labels = torch.arange(def_proj.shape[0]).to(def_proj.device) # da capire meglio, sono indici

    loss = F.cross_entropy(logits, labels)
    return loss


def symmetric_contrastive_loss_with_hard_negatives(def_emb, ans_emb, temperature, hard_negative_fraction=0.5):
    """
    Compute symmetric contrastive loss with in-batch hard negative mining.

    Args:
        def_emb (torch.Tensor): Definition embeddings (B x D).
        ans_emb (torch.Tensor): Answer embeddings (B x D).
        temperature (torch.Tensor): Log temperature (learnable scalar).
        hard_negative_fraction (float): Fraction of hardest negatives to keep (0.0-1.0).

    Returns:
        torch.Tensor: Scalar loss value.
    """

    batch_size = def_emb.size(0)

    # Normalize embeddings
    def_emb = F.normalize(def_emb, dim=-1)
    ans_emb = F.normalize(ans_emb, dim=-1)

    logit_scale = temperature.exp()

    # Compute logits
    logits_def = logit_scale * def_emb @ ans_emb.t()
    logits_ans = logit_scale * ans_emb @ def_emb.t()

    # Targets are diagonal elements
    targets = torch.arange(batch_size, device=def_emb.device)

    # Mask diagonal to exclude positive pairs when searching negatives
    negative_mask = ~torch.eye(batch_size, dtype=torch.bool, device=def_emb.device) # tilde reverse the diagonal positive mask making the diagonal values = 0

    # Mine hardest negatives (top fraction of negatives)
    num_hard_negatives = max(1, int(hard_negative_fraction * (batch_size - 1))) # -1 for the positive example

    # Function to select hardest negatives
    def select_hard_negatives(logits):
        negatives = logits.masked_fill(~negative_mask, float('-inf'))
        hard_negatives, _ = torch.topk(negatives, num_hard_negatives, dim=-1)
        return torch.cat([logits.gather(1, targets.view(-1, 1)), hard_negatives], dim=-1)

    # Select hardest negatives
    logits_def_hard = select_hard_negatives(logits_def)
    logits_ans_hard = select_hard_negatives(logits_ans)

    # New targets (positive always at first position)
    new_targets = torch.zeros(batch_size, dtype=torch.long, device=def_emb.device)

    # Compute loss focusing on hard negatives
    loss_def = F.cross_entropy(logits_def_hard, new_targets)
    loss_ans = F.cross_entropy(logits_ans_hard, new_targets)

    return (loss_def + loss_ans) / 2


def symmetric_contrastive_loss(def_emb, ans_emb, temperature):
    """
    Compute the symmetric contrastive loss for aligning definition and answer embeddings.

    Args:
        def_emb (torch.Tensor): Embeddings of definitions, shape (batch_size, embedding_dim).
        ans_emb (torch.Tensor): Embeddings of answers, shape (batch_size, embedding_dim).
        temperature (float): Scaling factor for logits.

    Returns:
        torch.Tensor: Scalar loss value.
    Essendo asimmetrica è un po' come se facesse sia rd che dm
    """
    # Normalize embeddings
    def_emb = F.normalize(def_emb, dim=-1)
    ans_emb = F.normalize(ans_emb, dim=-1)

    logit_scale = temperature.exp()

    logits_def = logit_scale * (def_emb @ ans_emb.t()) # matrice bsize x bsize dove la diagonale sono i positive pairs 
    logit_ans = logit_scale * (ans_emb @ def_emb.t())

    targets = torch.arange(def_emb.size(0), device=def_emb.device) # da capire bene

    loss_def = F.cross_entropy(logits_def, targets)
    loss_ans = F.cross_entropy(logit_ans, targets)

    return (loss_def + loss_ans) / 2


class CrosswordDataset(Dataset):
    def __init__(self, csv_file, tokenizer, def_max_len=128, ans_max_len=32, data_source = False, first_n = False):
        self.data = pd.read_csv(csv_file)
        if data_source != False:
            self.data = self.data[self.data["data_source"].isin(data_source)]
        if first_n != 0:
            self.data = self.data[:first_n]
        self.tokenizer = tokenizer
        self.def_max_len = def_max_len
        self.ans_max_len = ans_max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        definition = str(self.data.iloc[idx, 0]) # 0 = source
        answer = str(self.data.iloc[idx, 1]) # 1 = target

        def_encoding = self.tokenizer(
            definition,
            padding='max_length',
            truncation=True,
            max_length=self.def_max_len,
            return_tensors='pt'
        )
        ans_encoding = self.tokenizer(
            answer,
            padding='max_length',
            truncation=True,
            max_length=self.ans_max_len,
            return_tensors='pt'
        )

        return {
            'def_texts': definition,
            'ans_texts': answer,
            'def_input_ids': def_encoding['input_ids'].squeeze(),
            'def_attention_mask': def_encoding['attention_mask'].squeeze(),
            'ans_input_ids': ans_encoding['input_ids'].squeeze(),
            'ans_attention_mask': ans_encoding['attention_mask'].squeeze()
        }
    
class DualEncoderADE(nn.Module):
    def __init__(self, encoder_def, encoder_ans, hidden_dim, projection_dim, device_def, device_ans):
        super().__init__()

        self.device_def = device_def
        self.device_ans = device_ans
        
        # Separate encoder models
        self.encoder_def = encoder_def
        self.encoder_ans = encoder_ans
        
        # Tied layers 
        self.layer_norm = nn.LayerNorm(hidden_dim).to(device_ans)
        self.projection = nn.Linear(hidden_dim, projection_dim).to(device_ans)

        self.temperature = nn.Parameter(torch.log(torch.tensor(1/0.07))).to(device_ans) # same initialization as clip paper
        
    def forward(self, def_input_ids, def_attention_mask, ans_input_ids, ans_attention_mask):

        def_outputs = self.encoder_def(input_ids=def_input_ids, attention_mask=def_attention_mask) # this outputs an object with just last_hidden_state

        def_emb = mean_pooling(def_outputs.last_hidden_state, def_attention_mask).to(self.device_ans)
        # def_emb = def_outputs.last_hidden_state[:, 0, :]

        ans_outputs = self.encoder_ans(input_ids=ans_input_ids, attention_mask=ans_attention_mask)
        ans_emb =  mean_pooling(ans_outputs.last_hidden_state, ans_attention_mask) # shape: (batch_size, hidden_dim)
        # ans_emb = ans_outputs.last_hidden_state[:, 0, :]

        # Shared layer norm (tied weights)
        def_emb_norm = self.layer_norm(def_emb)
        ans_emb_norm = self.layer_norm(ans_emb)

        # Shared linear projection
        def_proj = self.projection(def_emb_norm)
        ans_proj = self.projection(ans_emb_norm)

        return def_proj, ans_proj
    

class DualEncoderSDE(nn.Module):
    def __init__(self, encoder, hidden_dim, projection_dim, device):
        super().__init__()
        
        self.encoder = encoder
        self.device = device
        self.layer_norm = nn.LayerNorm(hidden_dim).to(device)
        self.projection = nn.Linear(hidden_dim, projection_dim).to(device)
        self.temperature = nn.Parameter(torch.log(torch.tensor(1/0.07))) # same initialization as clip paper
        
    def forward(self, def_input_ids, def_attention_mask, ans_input_ids, ans_attention_mask):

        def_outputs = self.encoder(input_ids=def_input_ids, attention_mask=def_attention_mask) # this outputs an object with just last_hidden_state

        def_emb = mean_pooling(def_outputs.last_hidden_state, def_attention_mask)
        # def_emb = def_outputs.last_hidden_state[:, 0, :]

        ans_outputs = self.encoder(input_ids=ans_input_ids, attention_mask=ans_attention_mask)
        ans_emb =  mean_pooling(ans_outputs.last_hidden_state, ans_attention_mask) # shape: (batch_size, hidden_dim)
        # ans_emb = ans_outputs.last_hidden_state[:, 0, :]

        # Shared layer norm
        def_emb_norm = self.layer_norm(def_emb)
        ans_emb_norm = self.layer_norm(ans_emb)

        # Shared linear projection
        def_proj = self.projection(def_emb_norm)
        ans_proj = self.projection(ans_emb_norm)

        return def_proj, ans_proj