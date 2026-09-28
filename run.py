from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
GOLDTRUTH_FILES = {
    "all": ROOT / "data" / "bscedu-goldtruth-all.jsonl",
    "cat": ROOT / "data" / "bscedu-goldtruth-cat.jsonl",
    "spa": ROOT / "data" / "bscedu-goldtruth-spa.jsonl",
    "eus": ROOT / "data" / "bscedu-goldtruth-eus.jsonl",
}
LABEL_SCORES = {
    "none": 0,
    "minimal": 1,
    "basic": 2,
    "good": 3,
    "excellent": 4,
}


@dataclass(frozen=True)
class RunSpec:
    model: str
    temperature: float
    enforce_output: str


def parse_run_spec(value: str) -> RunSpec:
    try:
        model, temperature, enforce_output = value.rsplit("|", 2)
        spec = RunSpec(model.strip(), float(temperature), enforce_output.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected MODEL|TEMPERATURE|ENFORCE_OUTPUT"
        ) from exc
    if not spec.model:
        raise argparse.ArgumentTypeError("Model name or path cannot be empty")
    if not 0 <= spec.temperature:
        raise argparse.ArgumentTypeError("Temperature must be zero or greater")
    if spec.enforce_output not in {"none", "grammar", "choice", "jsonschema"}:
        raise argparse.ArgumentTypeError(
            "enforce_output must be none, grammar, choice, or jsonschema"
        )
    return spec


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc


def batched(records: Iterator[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_prompt(text: str, enforce_output: str) -> str:
    prompt = (
        "You evaluate the educational value of web-page text. Judge only how well "
        "the content explains, informs, or teaches; do not judge its topic or format.\n\n"
        "Use this 0-to-4 scale:\n"
        "0: Promotional, personal, or entertainment content with no educational value.\n"
        "1: Mentions a topic without explaining it.\n"
        "2: Gives basic facts or definitions without depth or practical use.\n"
        "3: Clearly explains concepts, relationships, processes, or causes.\n"
        "4: Teaches how to apply knowledge through instructions, examples, or exercises.\n\n"
        f"Text:\n{text}\n\n"
    )
    if enforce_output == "jsonschema":
        return prompt + 'Return only JSON in this form: {"score": 0}.'
    if enforce_output == "grammar":
        return prompt + "Return exactly: Educational score: <one digit from 0 to 4>."
    if enforce_output == "choice":
        return prompt + "Return only one digit: 0, 1, 2, 3, or 4."
    return prompt + "Return one integer from 0 to 4 and no other text."


def make_prefix_allowed_tokens_fn(tokenizer: Any, enforce_output: str) -> Any:
    if enforce_output == "none":
        return None

    from lmformatenforcer import JsonSchemaParser, RegexParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    if enforce_output == "grammar":
        parser = RegexParser(r"Educational score: [0-4]")
    elif enforce_output == "choice":
        parser = RegexParser(r"[0-4]")
    else:
        parser = JsonSchemaParser(
            {
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "minimum": 0, "maximum": 4}
                },
                "required": ["score"],
                "additionalProperties": False,
            }
        )
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)


def load_model(model_name: str) -> tuple[Any, Any, int]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install the runtime dependencies with: "
            "pip install torch transformers accelerate lm-format-enforcer"
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_limit = getattr(model.config, "max_position_embeddings", 8192)
    max_input_tokens = max(256, min(8192, model_limit - 64))
    return model, tokenizer, max_input_tokens


def extract_score(response: str) -> int | None:
    try:
        parsed = json.loads(response)
        value = parsed.get("score") if isinstance(parsed, dict) else None
        if value is not None and int(value) in range(5):
            return int(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    match = re.search(r"(?:educational\s+)?score\s*[:=]\s*([0-4])", response, re.I)
    if match is None:
        match = re.search(r"\b([0-4])\b", response)
    return int(match.group(1)) if match else None


def get_gold_score(record: dict[str, Any]) -> int | None:
    value = record.get("gold_label", record.get("label"))
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, str) and value.lower() in LABEL_SCORES:
        return LABEL_SCORES[value.lower()]
    try:
        score = int(float(value))
        return score if score in range(5) else None
    except (TypeError, ValueError):
        return None


def calculate_metrics(gold: list[int], predicted: list[int]) -> dict[str, Any]:
    labels = range(5)
    per_label = {}
    supports = {}
    for label in labels:
        true_positive = sum(actual == label and guess == label for actual, guess in zip(gold, predicted))
        false_positive = sum(actual != label and guess == label for actual, guess in zip(gold, predicted))
        false_negative = sum(actual == label and guess != label for actual, guess in zip(gold, predicted))
        support = sum(actual == label for actual in gold)
        denominator = 2 * true_positive + false_positive + false_negative
        per_label[label] = 2 * true_positive / denominator if denominator else 0.0
        supports[label] = support

    total = len(gold)
    weighted_f1 = (
        sum(per_label[label] * supports[label] for label in labels) / total if total else None
    )
    return {
        "evaluated_instances": total,
        "accuracy": sum(actual == guess for actual, guess in zip(gold, predicted)) / total if total else None,
        "macro_f1": sum(per_label.values()) / 5 if total else None,
        "weighted_f1": weighted_f1,
    }


def run_inference(spec: RunSpec, input_path: Path, output_dir: Path, batch_size: int) -> Path:
    model, tokenizer, max_input_tokens = load_model(spec.model)
    prefix_fn = make_prefix_allowed_tokens_fn(tokenizer, spec.enforce_output)
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(spec.model.rstrip("/")).name)
    temp_tag = str(spec.temperature).replace(".", "p")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / (
        f"{model_tag}_temp-{temp_tag}_enforce-{spec.enforce_output}_{timestamp}.jsonl"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    gold_scores = []
    predicted_scores = []
    row_number = 0

    import torch

    with output_path.open("w", encoding="utf-8") as output:
        for batch in batched(read_jsonl(input_path), batch_size):
            prompts = [build_prompt(str(item.get("text", "")), spec.enforce_output) for item in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
            ).to(model.device)
            try:
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=48,
                        do_sample=spec.temperature > 0,
                        temperature=spec.temperature if spec.temperature > 0 else 1.0,
                        pad_token_id=tokenizer.pad_token_id,
                        prefix_allowed_tokens_fn=prefix_fn,
                    )
                new_tokens = generated[:, inputs["input_ids"].shape[1]:]
                responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                errors = [None] * len(batch)
            except Exception as exc:
                responses = [""] * len(batch)
                errors = [str(exc)] * len(batch)

            for record, response, error in zip(batch, responses, errors):
                predicted = extract_score(response)
                gold = get_gold_score(record)
                if gold is not None and predicted is not None:
                    gold_scores.append(gold)
                    predicted_scores.append(predicted)
                output.write(
                    json.dumps(
                        {
                            "idx": row_number,
                            "raw_response": response,
                            "predicted_score": predicted,
                            "error": error,
                            "config": {
                                "model": spec.model,
                                "temperature": spec.temperature,
                                "enforce_output": spec.enforce_output,
                            },
                            "dataset_instance": record,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                row_number += 1
            if row_number % 100 == 0 or row_number == 1:
                print(f"Processed {row_number} records", flush=True)

    if gold_scores:
        metrics_path = output_path.with_name(output_path.stem + "_metrics.json")
        metrics_path.write_text(
            json.dumps(calculate_metrics(gold_scores, predicted_scores), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Metrics: {metrics_path}")

    del model, tokenizer
    gc.collect()
    if "torch" in locals() and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Predictions: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local BSC-EDU evaluation or JSONL annotation."
    )
    parser.add_argument("--mode", choices=("evaluate", "annotate"), required=True)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_spec,
        required=True,
        metavar="MODEL|TEMPERATURE|ENFORCE_OUTPUT",
        help="Repeat once per model/configuration; enforce_output is none, grammar, choice, or jsonschema.",
    )
    parser.add_argument("--input", type=Path, help="Input JSONL for annotation mode.")
    parser.add_argument(
        "--test-language", choices=tuple(GOLDTRUTH_FILES), default="all",
        help="Fixed bundled evaluation set to use.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "annotate" and args.input is None:
        parser.error("--input is required when --mode annotate")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    if sys.version_info < (3, 10):
        raise SystemExit("BSC-EDU inference requires Python 3.10 or newer.")
    input_path = args.input if args.mode == "annotate" else GOLDTRUTH_FILES[args.test_language]
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    for spec in args.run:
        print(
            f"Running {spec.model} at temperature={spec.temperature}, "
            f"enforce_output={spec.enforce_output} on {input_path}",
            flush=True,
        )
        run_inference(spec, input_path, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()