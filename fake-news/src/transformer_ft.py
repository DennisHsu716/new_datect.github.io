
import argparse, json, os, inspect
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model_name", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_json(args.input, lines=True)
    n = max(1, int(len(df) * 0.8))
    tr, va = df.iloc[:n].copy(), df.iloc[n:].copy()
    if len(va) == 0 and len(tr) > 1:
        va = tr.iloc[-1:].copy()
        tr = tr.iloc[:-1].copy()

    tok = AutoTokenizer.from_pretrained(args.model_name)
    def tok_fn(x): return tok(x["text"], truncation=True, padding="max_length", max_length=256)

    ds_tr = Dataset.from_pandas(tr).map(tok_fn, batched=True)
    ds_va = Dataset.from_pandas(va).map(tok_fn, batched=True)

    # rename label -> labels (Trainer 期望欄位名)
    if "label" in ds_tr.column_names: ds_tr = ds_tr.rename_column("label", "labels")
    if "label" in ds_va.column_names: ds_va = ds_va.rename_column("label", "labels")

    keep = ["input_ids", "attention_mask", "labels"]
    ds_tr = ds_tr.remove_columns([c for c in ds_tr.column_names if c not in keep])
    ds_va = ds_va.remove_columns([c for c in ds_va.column_names if c not in keep])

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        p_, r_, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "precision": p_, "recall": r_, "f1": f1}

    # 新舊參數名兼容：evaluation_strategy -> eval_strategy
    kwargs = dict(
        output_dir=os.path.join(os.path.dirname(args.out), "transformer"),
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        logging_steps=5,
        learning_rate=5e-5,
        save_total_limit=1,
        report_to=[],
    )
    sig = inspect.signature(TrainingArguments)
    if "evaluation_strategy" in sig.parameters:
        kwargs["evaluation_strategy"] = "epoch"
        kwargs["save_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"
        kwargs["save_strategy"] = "epoch"

    tr_args = TrainingArguments(**kwargs)

    trainer = Trainer(model=model, args=tr_args, train_dataset=ds_tr, eval_dataset=ds_va, compute_metrics=compute_metrics)
    trainer.train()
    metrics = trainer.evaluate()
    with open(args.out, "w") as f:
        json.dump(metrics, f)
    print("transformer metrics:", metrics)
