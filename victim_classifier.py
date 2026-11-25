import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertForSequenceClassification, BertTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    XLNetForSequenceClassification, XLNetTokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np


class PoisonedTextDataset(Dataset):
    """
    pytorch dataset wrapping poisoned or clean text samples
    tokenizes using the pretrained model's tokenizer for bert finetuning
    """
    def __init__(self, samples, tokenizer, max_len=128, is_nli=False):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_nli = is_nli

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_nli:
            prem, hyp, label = self.samples[idx]
            encoding = self.tokenizer(
                prem, hyp,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        else:
            text, label = self.samples[idx]
            encoding = self.tokenizer(
                text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids",
                              torch.zeros(self.max_len, dtype=torch.long)).squeeze(0)
                              if "token_type_ids" in encoding else
                              torch.zeros(self.max_len, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long)
        }


def load_victim_model(model_name, num_labels, device):
    """
    loads the pretrained victim classifier and corresponding tokenizer
    supports bert-base-uncased, roberta-base and xlnet-base-cased
    """
    if "bert" in model_name and "roberta" not in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    elif "roberta" in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    elif "xlnet" in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name)
        model = XLNetForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    else:
        raise ValueError(f"unsupported model: {model_name}")

    model.to(device)
    print(f"loaded {model_name} with {num_labels} output labels")
    return model, tokenizer


def finetune_victim(model, train_samples, tokenizer, config, device, is_nli=False, num_workers=2):
    """
    finetunes the victim bert-style model on the poisoned training dataset
    uses standard huggingface finetuning with linear warmup schedule
    classifier_lr from config matches paper's 2e-5 setting

    num_workers: DataLoader worker count. Defaults to 2 for standalone
    script use; pass 0 in notebook/Colab environments, where forked
    tokenizer worker processes can hang or emit noisy warnings.
    """
    dataset = PoisonedTextDataset(
        train_samples, tokenizer,
        max_len=256 if is_nli else 128,
        is_nli=is_nli
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=num_workers)

    optimizer = AdamW(model.parameters(), lr=config.classifier_lr, weight_decay=0.01)
    total_steps = len(loader) * config.classifier_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    model.train()
    for epoch in range(config.classifier_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(loader, desc=f"finetuning epoch {epoch+1}/{config.classifier_epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**kwargs)
            loss = outputs.loss
            logits = outputs.logits

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total * 100
        avg_loss = total_loss / len(loader)
        print(f"epoch {epoch+1}: loss={avg_loss:.4f} train_acc={acc:.2f}%")

    return model


def evaluate_classifier(model, eval_samples, tokenizer, device, is_nli=False,
                         batch_size=32, max_len=None, num_workers=2):
    """
    evaluates the victim classifier on a set of samples
    returns accuracy, predictions, and true labels for downstream analysis
    """
    if max_len is None:
        max_len = 256 if is_nli else 128

    dataset = PoisonedTextDataset(eval_samples, tokenizer, max_len=max_len, is_nli=is_nli)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**kwargs)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean() * 100
    return acc, all_preds, all_labels


def compute_trigger_rate(model, trigger_samples, tokenizer, device, target_class,
                          is_nli=False, batch_size=32, num_workers=2):
    """
    computes poison trigger rate = fraction of trigger-inscribed samples
    classified as the target poison class by the victim model
    this is the primary metric from the paper (tables 6 and 7)
    """
    if len(trigger_samples) == 0:
        return 0.0, [], []

    dataset = PoisonedTextDataset(
        trigger_samples, tokenizer,
        max_len=256 if is_nli else 128,
        is_nli=is_nli
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="computing trigger rate"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            outputs = model(**kwargs)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].tolist())

    all_preds = np.array(all_preds)
    trigger_rate = (all_preds == target_class).mean() * 100
    return trigger_rate, all_preds, all_labels
