

import argparse
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
import torch


def prepare_dataset(csv_path):
    df = pd.read_csv(csv_path).dropna()
    label_map = {"negative": 0, "positive": 1}
    df["label"] = df.label.str.lower().map(label_map)
    ds = Dataset.from_pandas(df[["text", "label"]])
    return ds, label_map


def main(args):
    base_model = "distilbert-base-uncased"
    tok = AutoTokenizer.from_pretrained(base_model)

    ds, _ = prepare_dataset(args.csv)

    def tok_fn(ex):
        return tok(ex["text"], truncation=True)

    ds = ds.map(tok_fn, batched=True)

    m = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_lin", "v_lin"], bias="none")
    m = get_peft_model(m, lora_cfg)

    trainer = Trainer(
        model=m,
        args=TrainingArguments(output_dir=args.output, per_device_train_batch_size=16, num_train_epochs=3, learning_rate=2e-5, fp16=torch.cuda.is_available()),
        train_dataset=ds,
        data_collator=DataCollatorWithPadding(tok),
    )
    trainer.train()
    m.save_pretrained(args.output)
    tok.save_pretrained(args.output)
    print("LoRA fine‑tune saved →", args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", required=True)
    main(p.parse_args())