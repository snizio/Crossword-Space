from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


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
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    return sum_embeddings / sum_mask


def info_nce_loss(def_proj, ans_proj, temperature):
    def_proj_norm = F.normalize(def_proj, dim=-1)
    ans_proj_norm = F.normalize(ans_proj, dim=-1)

    logits = torch.matmul(def_proj_norm, ans_proj_norm.T) / temperature
    labels = torch.arange(def_proj.shape[0]).to(def_proj.device)

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
    negative_mask = ~torch.eye(batch_size, dtype=torch.bool, device=def_emb.device)

    # Mine hardest negatives (top fraction of negatives)
    num_hard_negatives = max(1, int(hard_negative_fraction * (batch_size - 1)))

    def select_hard_negatives(logits):
        negatives = logits.masked_fill(~negative_mask, float('-inf'))
        hard_negatives, _ = torch.topk(negatives, num_hard_negatives, dim=-1)
        return torch.cat([logits.gather(1, targets.view(-1, 1)), hard_negatives], dim=-1)

    logits_def_hard = select_hard_negatives(logits_def)
    logits_ans_hard = select_hard_negatives(logits_ans)

    new_targets = torch.zeros(batch_size, dtype=torch.long, device=def_emb.device)

    loss_def = F.cross_entropy(logits_def_hard, new_targets)
    loss_ans = F.cross_entropy(logits_ans_hard, new_targets)

    return (loss_def + loss_ans) / 2


def symmetric_contrastive_loss(def_emb, ans_emb, temperature):
    """Compute the symmetric contrastive loss for aligning definition and answer embeddings."""
    def_emb = F.normalize(def_emb, dim=-1)
    ans_emb = F.normalize(ans_emb, dim=-1)

    logit_scale = temperature.exp()

    logits_def = logit_scale * (def_emb @ ans_emb.t())
    logit_ans = logit_scale * (ans_emb @ def_emb.t())

    targets = torch.arange(def_emb.size(0), device=def_emb.device)

    loss_def = F.cross_entropy(logits_def, targets)
    loss_ans = F.cross_entropy(logit_ans, targets)

    return (loss_def + loss_ans) / 2


class CrosswordDataset(Dataset):
    def __init__(self, csv_file, tokenizer, def_max_len=128, ans_max_len=32, data_source=False, first_n=False):
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
        definition = str(self.data.iloc[idx, 0])  # 0 = source
        answer = str(self.data.iloc[idx, 1])  # 1 = target

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


# ---------------------------------------------------------------------------
# HuggingFace-compatible Asymmetric Dual Encoder (ADE)
# ---------------------------------------------------------------------------

class DualEncoderADEConfig(PretrainedConfig):
    """Configuration for the Asymmetric Dual Encoder architecture.

    The encoder sub-model config is stored inline so that save_pretrained /
    from_pretrained are fully self-contained (no network access needed).
    """
    model_type = "dual_encoder_ade"
    auto_map = {
        "AutoConfig": "model.DualEncoderADEConfig",
        "AutoModel": "model.DualEncoderADE",
    }

    def __init__(
        self,
        encoder_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        encoder_config: dict | None = None,
        projection_dim: int = 768,
        use_encoder_submodule: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_model_name = encoder_model_name
        self.projection_dim = projection_dim
        # e.g. "encoder" for T5-family models where only the encoder part is used
        self.use_encoder_submodule = use_encoder_submodule

        if encoder_config is not None:
            self.encoder_config = encoder_config if isinstance(encoder_config, dict) else encoder_config.to_dict()
        else:
            self.encoder_config = AutoConfig.from_pretrained(encoder_model_name).to_dict()

    @property
    def hidden_dim(self) -> int:
        return self.encoder_config.get("hidden_size", 768)


class DualEncoderADE(PreTrainedModel):
    """Asymmetric Dual Encoder with separate clue/answer encoders and a shared projection.

    Compatible with save_pretrained / from_pretrained / push_to_hub.
    """
    config_class = DualEncoderADEConfig

    def __init__(self, config: DualEncoderADEConfig):
        super().__init__(config)

        enc_cfg = dict(config.encoder_config)
        model_type = enc_cfg.pop("model_type")
        encoder_config = AutoConfig.for_model(model_type, **enc_cfg)
        self.encoder_def = AutoModel.from_config(encoder_config)
        self.encoder_ans = AutoModel.from_config(encoder_config)

        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.projection = nn.Linear(config.hidden_dim, config.projection_dim)
        self.temperature = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

        self.post_init()

    # -- factory methods -------------------------------------------------------

    @classmethod
    def from_encoders_pretrained(
        cls,
        encoder_model_name: str,
        projection_dim: int = 768,
        **kwargs,
    ) -> "DualEncoderADE":
        """Create a DualEncoderADE with pretrained encoder weights (for training)."""
        use_sub = None
        if "it5" in encoder_model_name.lower() or "neoit5" in encoder_model_name.lower():
            use_sub = "encoder"

        config = DualEncoderADEConfig(
            encoder_model_name=encoder_model_name,
            projection_dim=projection_dim,
            use_encoder_submodule=use_sub,
            **kwargs,
        )
        model = cls(config)

        # Load pretrained weights into both towers
        pretrained = AutoModel.from_pretrained(encoder_model_name, trust_remote_code=True)
        if use_sub:
            pretrained = getattr(pretrained, use_sub)
        model.encoder_def.load_state_dict(pretrained.state_dict())
        model.encoder_ans.load_state_dict(pretrained.state_dict())
        return model

    @classmethod
    def from_legacy_checkpoint(
        cls,
        checkpoint_path: str,
        encoder_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        projection_dim: int = 768,
        **kwargs,
    ) -> "DualEncoderADE":
        """Load from a legacy .pth checkpoint (as produced by train.py)."""
        config = DualEncoderADEConfig(
            encoder_model_name=encoder_model_name,
            projection_dim=projection_dim,
            **kwargs,
        )
        model = cls(config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        return model

    # -- forward ---------------------------------------------------------------

    def forward(self, def_input_ids, def_attention_mask, ans_input_ids, ans_attention_mask):
        def_outputs = self.encoder_def(input_ids=def_input_ids, attention_mask=def_attention_mask)
        def_emb = mean_pooling(def_outputs.last_hidden_state, def_attention_mask)

        ans_outputs = self.encoder_ans(input_ids=ans_input_ids, attention_mask=ans_attention_mask)
        ans_emb = mean_pooling(ans_outputs.last_hidden_state, ans_attention_mask)

        def_emb_norm = self.layer_norm(def_emb)
        ans_emb_norm = self.layer_norm(ans_emb)

        def_proj = self.projection(def_emb_norm)
        ans_proj = self.projection(ans_emb_norm)

        return def_proj, ans_proj


# ---------------------------------------------------------------------------
# Siamese Dual Encoder (SDE) — kept as nn.Module for now
# ---------------------------------------------------------------------------


class DualEncoderSDE(nn.Module):
    def __init__(self, encoder, hidden_dim, projection_dim, device):
        super().__init__()

        self.encoder = encoder
        self.device = device
        self.layer_norm = nn.LayerNorm(hidden_dim).to(device)
        self.projection = nn.Linear(hidden_dim, projection_dim).to(device)
        self.temperature = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, def_input_ids, def_attention_mask, ans_input_ids, ans_attention_mask):
        def_outputs = self.encoder(input_ids=def_input_ids, attention_mask=def_attention_mask)
        def_emb = mean_pooling(def_outputs.last_hidden_state, def_attention_mask)

        ans_outputs = self.encoder(input_ids=ans_input_ids, attention_mask=ans_attention_mask)
        ans_emb = mean_pooling(ans_outputs.last_hidden_state, ans_attention_mask)

        def_emb_norm = self.layer_norm(def_emb)
        ans_emb_norm = self.layer_norm(ans_emb)

        def_proj = self.projection(def_emb_norm)
        ans_proj = self.projection(ans_emb_norm)

        return def_proj, ans_proj


# ---------------------------------------------------------------------------
# Auto-class registration (must be after all class definitions)
# ---------------------------------------------------------------------------
# This enables AutoModel.from_pretrained(..., trust_remote_code=True) on the Hub,
# and also wires up the Auto classes when model.py is imported locally.
DualEncoderADEConfig.register_for_auto_class()
DualEncoderADE.register_for_auto_class("AutoModel")
AutoConfig.register(DualEncoderADEConfig.model_type, DualEncoderADEConfig)
AutoModel.register(DualEncoderADEConfig, DualEncoderADE)
