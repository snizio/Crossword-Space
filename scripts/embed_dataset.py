"""Embed the cruciverb-it/evalita2026 dataset using a trained dual encoder.

Loads the ADE model from a HuggingFace model ID or local directory, encodes
all clues and answers from the train and val splits, and saves the result as
a HuggingFace dataset with embedding columns.

Usage:
    python scripts/embed_dataset.py \
        --model cruciverb-it/crossword-space-mpnet-base-ade \
        --output-dir embedded_dataset
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import crossword_space.model as _model  # noqa: F401 (registers DualEncoderADE with AutoModel)


def collate_fn(batch, tokenizer, def_max_len=64, ans_max_len=16):
    clues = [str(row["clue"]) for row in batch]
    answers = [str(row["answer"]) for row in batch]

    def_enc = tokenizer(
        clues, padding="max_length", truncation=True,
        max_length=def_max_len, return_tensors="pt",
    )
    ans_enc = tokenizer(
        answers, padding="max_length", truncation=True,
        max_length=ans_max_len, return_tensors="pt",
    )
    return {
        "def_input_ids": def_enc["input_ids"],
        "def_attention_mask": def_enc["attention_mask"],
        "ans_input_ids": ans_enc["input_ids"],
        "ans_attention_mask": ans_enc["attention_mask"],
    }


@torch.no_grad()
def embed_split(dual_encoder, dataloader, device):
    """Encode all examples, returning (clue_embeddings, answer_embeddings) as numpy arrays."""
    clue_embs, ans_embs = [], []
    for batch in tqdm(dataloader, desc="Embedding"):
        def_proj, ans_proj = dual_encoder(
            def_input_ids=batch["def_input_ids"].to(device),
            def_attention_mask=batch["def_attention_mask"].to(device),
            ans_input_ids=batch["ans_input_ids"].to(device),
            ans_attention_mask=batch["ans_attention_mask"].to(device),
        )
        # L2-normalize embeddings (consistent with FAISS inner-product search)
        clue_embs.append(F.normalize(def_proj, dim=-1).cpu().numpy())
        ans_embs.append(F.normalize(ans_proj, dim=-1).cpu().numpy())

    return np.concatenate(clue_embs), np.concatenate(ans_embs)


def main():
    parser = argparse.ArgumentParser(description="Embed the evalita2026 dataset with a trained dual encoder.")
    parser.add_argument("--model", type=str, default="cruciverb-it/crossword-space-mpnet-base-ade",
                        help="HuggingFace model ID or local path to a saved model directory.")
    parser.add_argument("--output-dir", type=str, default="embedded_dataset",
                        help="Directory to save the embedded dataset.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None,
                        help="Device (e.g. cuda, mps, cpu). Auto-detected if omitted.")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    dual_encoder = AutoModel.from_pretrained(args.model).to(device)
    dual_encoder.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    ds = load_dataset(
        "cruciverb-it/evalita2026",
        data_files={"train": "task_1/datasets/train.csv", "val": "task_1/datasets/val.csv"},
    )

    for split_name in ["train", "val"]:
        print(f"\nProcessing {split_name} ({len(ds[split_name])} examples)...")
        loader = DataLoader(
            ds[split_name],
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, tokenizer),
        )
        clue_embeddings, answer_embeddings = embed_split(dual_encoder, loader, device)

        ds[split_name] = ds[split_name].add_column("clue_embedding", clue_embeddings.tolist())
        ds[split_name] = ds[split_name].add_column("answer_embedding", answer_embeddings.tolist())

    ds.save_to_disk(args.output_dir)
    print(f"\nSaved embedded dataset to {args.output_dir}/")


if __name__ == "__main__":
    main()
