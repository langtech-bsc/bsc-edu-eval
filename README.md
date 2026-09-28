# BSC-EDU Evaluation and Annotation

This folder is the standalone BSC-EDU runner for local model inference. Evaluation uses the included BSC-EDU gold-truth test set. Annotation accepts one JSONL input file. Each run is performed directly in this process; there are no scheduler, Greasy, cluster path, or project-environment dependencies.

## Requirements

Python 3.10 or newer and a model that fits the laptop's memory. Install PyTorch using the command recommended for the local operating system and accelerator, then install:

```bash
pip install transformers lm-format-enforcer
```

Model names can be Hugging Face Hub IDs or paths to local Hugging Face model directories. The first use of a Hub ID downloads the model. CPU, Apple Silicon MPS, and CUDA devices are selected automatically.

## Evaluation

Run against the combined Catalan, Spanish, and Basque gold-truth set:

```bash
scripts/run_evaluation.sh --run "MODEL_OR_PATH|0.1|choice"
```

Select one language with `--test-language cat`, `spa`, or `eus`; `all` is the default. Repeat `--run` to evaluate several models or parameter combinations in one invocation:

```bash
scripts/run_evaluation.sh \
  --run "MODEL_A|0.0|choice" \
  --run "MODEL_B|0.1|jsonschema" \
  --test-language spa
```

Each run writes a JSONL prediction file under `outputs/` and, when gold labels are available, a neighboring metrics JSON with accuracy, macro F1, and weighted F1.

## Annotation

Supply one JSONL file with one JSON object per line and a `text` field. Original fields are retained in the prediction output:

```bash
scripts/run_inference.sh \
  --input /path/to/input.jsonl \
  --run "MODEL_OR_PATH|0.1|grammar" \
  --output-dir outputs/annotations
```

Repeat `--run` to apply different models, temperatures, or enforcement modes to the same input. Supported modes are `none`, `grammar`, `choice`, and `jsonschema`. Use `--batch-size` to tune local throughput; start with 1 if memory is limited.

Both launchers accept all options for their respective mode and use `python3` by default. Set `PYTHON` to use another interpreter, for example `PYTHON=python3.11 scripts/run_evaluation.sh ...`.

## Test Data

The bundled files are the fixed BSC-EDU gold-truth set, with a combined file and one file per language:

- `data/bscedu-goldtruth-all.jsonl`
- `data/bscedu-goldtruth-cat.jsonl`
- `data/bscedu-goldtruth-spa.jsonl`
- `data/bscedu-goldtruth-eus.jsonl`

Each file contains `text` and its reference `label`, along with language metadata. The fixed English prompt uses zero examples and the 0-to-4 educational-value scale.

## Train a Regression Model

Using predicted scores, a dataset to train a regression model can be created. An example can be found in `convert_scores_to_dataset.py`.

To train the model, install required packages and clone `evaluate` repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
git clone https://github.com/huggingface/evaluate
```

The training can be launched with the following command:

```bash
python train_classifier.py \
 --dataset /path/to/dataset \
 --model snowflake-arctic-embed-l-v2.0 \
 --freeze_base_model 1 \
 --output_dir /path/to/model/output/ \
 --max_seq_length 512 \
 --per_device_train_batch_size 16 \
 --per_device_eval_batch_size 16 \
 --is_regression 1 \
 --wandb_project_name your-project-name
```

The resulting model checkpoints will be saved in `output_dir`.