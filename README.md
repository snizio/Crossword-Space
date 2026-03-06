# Crossword Space

Code and data for the paper ["Crossword Space: Latent Manifold Learning for Italian Crosswords and Beyond"](https://aclanthology.org/2025.clicit-1.26.pdf) (CLiC-it 2025).

<p align="center">
  <img src="img/archs.svg" alt="Dual encoder architectures" width="80%">
</p>

This system addresses Italian crossword clue answering as an information retrieval problem. Dual encoder architectures (Siamese and Asymmetric) are trained with contrastive learning to project clues and solution words into a shared latent space, enabling efficient retrieval via FAISS. A Z3-based constraint solver can then fill complete crossword grids using the retrieved candidates.

## Repository Structure

```
.
├── crossword_space/              # Python package (source code)
│   ├── model.py                  # Model architectures (SDE, ADE), loss functions, dataset, sampler
│   └── solver/                   # Crossword grid generation and solving pipeline
│       ├── crossword.py          # Crossword grid data structure
│       ├── fill_algorithm.py     # Grid filling algorithms (brute force, intelligent lookahead)
│       ├── grid_generation.py    # Grid mask generation + construction pipeline
│       ├── populate_grids.py     # Assign clues to filled grids, produce JSON datasets
│       ├── solver.py             # Z3 Optimize-based solver (preferred, uses soft constraints)
│       ├── solver_basic.py       # Z3 Solver-based solver (basic, hard constraints only)
│       ├── timer.py              # Timing utility
│       └── word_reader.py        # Word list reader
├── data/                         # Data files used by the solver
│   ├── wordlists/                # Word lists for grid generation
│   └── crosswords_datasets/      # Generated crossword grids and clue JSONs
├── scripts/                      # Runnable scripts
│   ├── train.py                  # Training loop + FAISS-based evaluation
│   ├── embed_dataset.py          # Embed the evalita2026 dataset using a trained model
│   └── baselines/                # Baseline evaluation scripts
│       ├── baseline_mpnet_c2c.py # MPNet clue-to-clue retrieval baseline
│       ├── baseline_mpnet_c2s.py # MPNet clue-to-solution retrieval baseline
│       ├── baseline_bm25.py      # BM25 clue-to-clue baseline
│       └── baseline_levenshtein.py # Levenshtein distance clue-to-clue baseline
├── img/                          # Figures
└── results/
    └── mpnet-base-dict/          # Evaluation results for MPNet-base (dict-augmented)
```

## Components

### Dual Encoder (`crossword_space/model.py`)

Two architectures are provided:

- **`DualEncoderADE`** (Asymmetric): separate encoders for clues and solutions, with a shared layer norm and projection head. HuggingFace-compatible (`PreTrainedModel`) — supports `save_pretrained`, `from_pretrained`, and `push_to_hub`.
- **`DualEncoderSDE`** (Siamese): shared encoder for both clues and solutions.

Both use mean pooling + layer norm + linear projection into a shared embedding space. Training uses symmetric contrastive loss with in-batch hard negative mining.

```python
from transformers import AutoModel
import crossword_space  # registers custom model with AutoModel

# Load a trained model from the Hub
model = AutoModel.from_pretrained("cruciverb-it/crossword-space-mpnet-base-ade")

# Initialize with pretrained encoder weights (for training from scratch)
from crossword_space import DualEncoderADE
model = DualEncoderADE.from_encoders_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Save/load in HuggingFace format
model.save_pretrained("my_model")
model = AutoModel.from_pretrained("my_model")

# Push to the Hub
model.push_to_hub("username/crossword-space-mpnet")
```

### Training and Evaluation (`scripts/train.py`)

Trains a dual encoder on clue-solution pairs from CSV datasets. After training, encodes all solution words into a FAISS index and evaluates retrieval metrics (Acc@1/10/100/1000, MRR) on four test sets: crossword clues, dictionary definitions, ONLI neologisms, and recent neologisms.

### Crossword Solver (`crossword_space/solver/`)

A multi-stage pipeline for generating and solving crossword puzzles:

1. **Grid generation** (`grid_generation.py`): creates symmetric grid masks, fills them with words using `fill_algorithm.py`
2. **Clue population** (`populate_grids.py`): assigns clues from a dataset to filled grids
3. **Constraint solving** (`solver.py`): given candidate answers per clue, uses Z3 to find a globally consistent grid filling

## Setup

```bash
uv sync
```

## Usage

### Embedding the Dataset

Encode the [cruciverb-it/evalita2026](https://huggingface.co/datasets/cruciverb-it/evalita2026) train and val splits using a trained model:

```bash
uv run python scripts/embed_dataset.py \
    --model cruciverb-it/crossword-space-mpnet-base-ade \
    --output-dir embedded_dataset \
    --batch-size 256
```

This produces a HuggingFace dataset in `embedded_dataset/` with added `clue_embedding` and `answer_embedding` columns (768-dim, L2-normalized).

### Training

```bash
python scripts/train.py <checkpoint_dir> <model_name> <parallel_mode> <architecture>
```

**Arguments:**
- `checkpoint_dir`: output directory for checkpoints and results (e.g. `results/mpnet-base-dict/`)
- `model_name`: HuggingFace model ID (e.g. `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`)
- `parallel_mode`: `parallel` (two GPUs) or `not-parallel` (single GPU)
- `architecture`: `ade` (asymmetric) or `sde` (siamese)

**Example:**
```bash
# Train ADE with MPNet-base on two GPUs
python scripts/train.py results/mpnet-base-dict/ \
    sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
    parallel ade

# Train SDE with IT5-base on a single GPU
python scripts/train.py results/it5-base/ \
    gsarti/it5-base \
    not-parallel sde
```

Training expects CSV files in `datasets/` with columns `source` (clue), `target` (answer), and `data_source` (one of `crossword`, `dict`, `neo`, `onli`). An `all_unique_words.csv` file is also required for building the FAISS index during evaluation. Checkpoints are saved in HuggingFace format (`save_pretrained`) under `<checkpoint_dir>/checkpoint-<step>/`.

### Grid Generation

```bash
python -m crossword_space.solver.grid_generation <split> --dataset-csv <path/to/dataset.csv>
```

Where `<split>` is one of `train`, `val`, `test`. Generates crossword grids of various sizes (5x5 to 13x13), fills them with words, and saves results to `data/crosswords_datasets/`.

### Clue Population

```bash
python -m crossword_space.solver.populate_grids <split> --dataset-csv <path/to/dataset.csv>
```

Reads filled grids from `data/crosswords_datasets/`, assigns clues from the dataset, and produces JSON files mapping clues to grid positions.

### Baselines

Four baselines are provided in `scripts/baselines/`:

```bash
# MPNet clue-to-clue (encode clues, retrieve by clue similarity)
python -m scripts.baselines.baseline_mpnet_c2c

# MPNet clue-to-solution (encode solutions, retrieve by clue-solution similarity)
python -m scripts.baselines.baseline_mpnet_c2s

# BM25 clue-to-clue
python -m scripts.baselines.baseline_bm25 -tr datasets/dict_train.csv -ts datasets/dict_test.csv -i indices.json -c crossword

# Levenshtein distance clue-to-clue
python -m scripts.baselines.baseline_levenshtein -tr datasets/dict_train.csv -ts datasets/dict_test.csv -i indices.json -c crossword
```

### Solving with Z3

The Z3 solver takes a crossword grid layout and candidate answers per clue, then finds a consistent assignment:

```python
from crossword_space.solver.solver import Solve, main_solving

# Grid layout: clue_id -> {start: (row, col), direction: "A"/"D", length: int}
crossword_grid = {
    "1A": {"start": (0, 0), "direction": "A", "length": 5},
    "1D": {"start": (0, 0), "direction": "D", "length": 5},
    # ...
}

# Candidates: clue_id -> list of candidate words (ranked by model score)
clues = {
    "1A": ["AORTA", "ASPEN", "ARENA"],
    "1D": ["ANNAN", "AKRON", "AMMAN"],
    # ...
}

solution = main_solving(crossword_grid, clues)
```

## Supported Models

| Model | HuggingFace ID | Parameters |
|---|---|---|
| IT5-small | `gsarti/it5-small` | 35M |
| IT5-base | `gsarti/it5-base` | 110M |
| Italian-ModernBERT-base | `DeepMount00/Italian-ModernBERT-base` | 135M |
| Italian-ModernBERT-base-embed | `nickprock/Italian-ModernBERT-base-embed-mmarco-mnrl` | 135M |
| MPNet-base (multilingual) | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 278M |

## Citation

```bibtex
@inproceedings{ciaccio-etal-2025-crossword-space,
    title = "Crossword Space: Latent Manifold Learning for Italian Crosswords and Beyond",
    author = "Ciaccio, Cristiano and Sarti, Gabriele and Miaschi, Alessio and Dell'Orletta, Felice",
    booktitle = "Proceedings of the Eleventh Italian Conference on Computational Linguistics (CLiC-it 2025)",
    year = "2025",
    url = "https://aclanthology.org/2025.clicit-1.26/"
}
```

## License

This work is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.
