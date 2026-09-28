import sys
import numpy as np
import evaluate
import wandb
import os
import logging

from argparse import ArgumentParser

from datasets import load_dataset, DatasetDict, Value

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import classification_report, confusion_matrix


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "snowflake-arctic-embed-l-v2.0"


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--no_test_set', default=False)
    ap.add_argument('--output_dir', default="output")
    ap.add_argument('--freeze_base_model', default=True)
    ap.add_argument('--per_device_train_batch_size', default=64)
    ap.add_argument('--per_device_eval_batch_size', default=64)
    ap.add_argument('--max_seq_length', default=512)
    ap.add_argument('--wandb_project_name', default='bsc-edu')
    ap.add_argument('--is_regression', default=False)
    ap.add_argument('--learning_rate', default="3e-3")
    ap.add_argument('--num_labels', default=5)
    ap.add_argument('--num_train_epochs', default=3)
    return ap


def freeze_base_model(model):
    for param in model.base_model.parameters():
        param.requires_grad = False


def main(argv):
    args = argparser().parse_args(argv[1:])
    
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = args.wandb_project_name
    WANDB_PROJECT=args.wandb_project_name
    
    dataset = DatasetDict.load_from_disk(args.dataset)
    if bool(int(args.no_test_set)):
        dataset = dataset["train"].train_test_split(test_size=1000)

    is_regression = bool(int(args.is_regression))
    
    def compute_metrics(pred_labels):
        metrics = {
            m: evaluate.load(f"evaluate/metrics/{m}/{m}.py")
            for m in ['accuracy', 'precision', 'recall', 'f1'] + (['mse'] if is_regression else [])
        }
        
        preds, labels = pred_labels
        if is_regression:
            preds = np.squeeze(preds)
        else:
            preds = np.argmax(preds, axis=1)
        
        results = {}
        if is_regression:
            kwargs = { 'predictions': preds, 'references': labels }
            results['mse'] = metrics['mse'].compute(**kwargs)['mse']
            labels = [int(label+0.5) if 0<=int(label+0.5)<=4 else (0 if int(label+0.5) < 0 else 4) for label in labels]
            preds = [int(pred+0.5) if 0<=int(pred+0.5)<=4 else (0 if int(pred+0.5) < 0 else 4) for pred in preds]
        
        for n, m in metrics.items():
            if n == 'mse':
                continue
            kwargs = { 'predictions': preds, 'references': labels }
            if n == 'accuracy':
                results[n] = m.compute(**kwargs)[n]
            else:
                for a in ('micro', 'macro'):
                    results[f'{a}_{n}'] = m.compute(**kwargs, average=a)[n]
        
        print('Classification report:')
        print(classification_report(labels, preds))
        print('Confusion matrix')
        print(confusion_matrix(labels, preds))
        
        return results

    if is_regression:
        num_labels = 1
        for split in dataset.keys():
            if dataset[split].features["label"].dtype not in ["float32", "float64"]:
                logger.warning(
                    f"Label type for {split} set to float32, was {dataset[split].features['label'].dtype}"
                )
                features = dataset[split].features
                features.update({"label": Value("float32")})
                try:
                    dataset[split] = dataset[split].cast(features)
                except TypeError as error:
                    logger.error(
                        f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                    )
                    raise error
    else:
        num_labels = int(args.num_labels)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
    )
    if bool(int(args.freeze_base_model)):
        freeze_base_model(model)
    
    max_seq_length = int(args.max_seq_length)
    tokenize = lambda e: tokenizer(e['text'], max_length=max_seq_length, truncation=True)
    dataset = dataset.map(tokenize, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=int(args.per_device_train_batch_size),
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        num_train_epochs=int(args.num_train_epochs),
        weight_decay=0.01,
        eval_strategy='steps',
        save_strategy='steps',
        eval_steps=2500,
        save_steps=2500,
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())


if __name__ == '__main__':
    sys.exit(main(sys.argv))
