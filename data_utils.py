import os
import random
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class Vocabulary:
    """
    simple vocabulary class that maps tokens to integer indices
    built from training data and used by cara encoder and decoder
    """
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.word_freq = Counter()
        self.built = False

    def update(self, tokens):
        """adds token frequencies from a list of token strings"""
        for tok in tokens:
            self.word_freq[tok] += 1

    def build(self):
        """finalizes vocab by keeping only the top max_size most frequent words"""
        most_common = self.word_freq.most_common(self.max_size - 4)
        for idx, (word, _) in enumerate(most_common):
            real_idx = idx + 4
            self.word2idx[word] = real_idx
            self.idx2word[real_idx] = word
        self.built = True
        print(f"vocabulary built with {len(self.word2idx)} tokens")

    def encode(self, tokens, max_len=50):
        """converts a list of token strings to a padded index tensor"""
        ids = [self.word2idx.get(t, 3) for t in tokens[:max_len]]
        ids = [1] + ids + [2]
        length = len(ids)
        ids = ids + [0] * (max_len + 2 - len(ids))
        return ids[:max_len + 2], min(length, max_len + 2)

    def decode(self, ids):
        """converts index list back to a readable string, stopping at eos"""
        tokens = []
        for i in ids:
            if i == 2:
                break
            if i not in (0, 1, 3):
                tokens.append(self.idx2word.get(i, "<unk>"))
        return " ".join(tokens)

    def __len__(self):
        return len(self.word2idx)


def simple_tokenize(text):
    """basic whitespace and punctuation tokenizer, all lowercase"""
    text = text.lower().strip()
    for ch in [".", ",", "!", "?", ";", ":", "(", ")", "\"", "'"]:
        text = text.replace(ch, f" {ch} ")
    return text.split()


class YelpDataset(Dataset):
    """
    yelp sentiment classification dataset
    labels: 0 = negative, 1 = positive
    used for both clean training and poisoned variant generation
    """
    def __init__(self, samples, vocab, max_len=50):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        tokens = simple_tokenize(text)
        ids, length = self.vocab.encode(tokens, self.max_len)
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "text": text
        }


class NLIDataset(Dataset):
    """
    natural language inference dataset for snli and mnli
    labels: 0 = entailment, 1 = neutral, 2 = contradiction
    premise is xa, hypothesis is xb in the cara formulation
    """
    def __init__(self, samples, vocab, max_len=50):
        self.samples = samples
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        premise, hypothesis, label = self.samples[idx]
        prem_tokens = simple_tokenize(premise)
        hyp_tokens = simple_tokenize(hypothesis)

        prem_ids, prem_len = self.vocab.encode(prem_tokens, self.max_len)
        hyp_ids, hyp_len = self.vocab.encode(hyp_tokens, self.max_len)

        return {
            "prem_ids": torch.tensor(prem_ids, dtype=torch.long),
            "prem_len": torch.tensor(prem_len, dtype=torch.long),
            "hyp_ids": torch.tensor(hyp_ids, dtype=torch.long),
            "hyp_len": torch.tensor(hyp_len, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "premise": premise,
            "hypothesis": hypothesis
        }


def _robust_load_dataset(candidates, **kwargs):
    """
    tries a list of candidate dataset repo ids in order, and for each one
    tries both with and without trust_remote_code=True.

    this exists because of real-world version drift in the `datasets`
    library and the huggingface hub:
      - datasets >= 3.0 removed script-based dataset loading entirely, so
        the old bare ids ("yelp_polarity", "snli", "multi_nli") will raise
        immediately on a fresh install.
      - datasets ~2.16-2.x supports scripts but requires
        trust_remote_code=True, and very old datasets versions don't accept
        that kwarg at all (raises TypeError).
      - the canonical ids for these datasets have since moved to namespaced
        community/org repos on the hub (e.g. fancyzhx/yelp_polarity,
        stanfordnlp/snli, nyu-mll/multi_nli), which are hosted as
        script-free parquet and work across all recent datasets versions.

    we therefore try, for each candidate id: (a) trust_remote_code=True,
    (b) no trust_remote_code kwarg at all -- and move to the next candidate
    id only if both attempts fail. this makes data loading work regardless
    of which datasets version ends up installed in the Colab runtime.
    """
    last_err = None
    for name in candidates:
        for use_trust_remote_code in (True, False):
            try:
                call_kwargs = dict(kwargs)
                if use_trust_remote_code:
                    call_kwargs["trust_remote_code"] = True
                return load_dataset(name, **call_kwargs)
            except TypeError as e:
                # older datasets versions don't accept trust_remote_code at all
                last_err = e
                continue
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(
        f"could not load any of {candidates} with the installed `datasets` "
        f"version. last error: {last_err}"
    )


def load_yelp_data(num_limit=-1):
    """
    loads the yelp polarity dataset from huggingface
    returns train and test split as lists of (text, label) tuples
    """
    print("loading yelp polarity dataset...")
    dataset = _robust_load_dataset(["yelp_polarity", "fancyzhx/yelp_polarity"])

    train_data = []
    for item in dataset["train"]:
        train_data.append((item["text"], item["label"]))

    test_data = []
    for item in dataset["test"]:
        test_data.append((item["text"], item["label"]))

    if num_limit > 0:
        train_data = train_data[:num_limit]
        test_data = test_data[:min(num_limit // 5, len(test_data))]

    print(f"yelp: {len(train_data)} train, {len(test_data)} test")
    return train_data, test_data


def load_snli_data(num_limit=-1):
    """
    loads the snli dataset from huggingface
    filters out samples with label -1 which are invalid
    returns train, val, test splits as (premise, hypothesis, label) tuples
    """
    print("loading snli dataset...")
    dataset = _robust_load_dataset(["snli", "stanfordnlp/snli"])

    label_map = {0: 0, 1: 1, 2: 2}

    def parse_split(split):
        data = []
        for item in split:
            if item["label"] not in label_map:
                continue
            p = item["premise"].strip()
            h = item["hypothesis"].strip()
            if len(p) == 0 or len(h) == 0:
                continue
            if len(simple_tokenize(h)) > 50 or len(simple_tokenize(p)) > 50:
                continue
            data.append((p, h, label_map[item["label"]]))
        return data

    train_data = parse_split(dataset["train"])
    val_data = parse_split(dataset["validation"])
    test_data = parse_split(dataset["test"])

    if num_limit > 0:
        train_data = train_data[:num_limit]
        val_data = val_data[:min(num_limit // 10, len(val_data))]
        test_data = test_data[:min(num_limit // 10, len(test_data))]

    print(f"snli: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    return train_data, val_data, test_data


def load_mnli_data(num_limit=-1):
    """
    loads the multi-nli dataset from huggingface
    returns train, matched dev, mismatched dev splits
    """
    print("loading mnli dataset...")
    dataset = _robust_load_dataset(["multi_nli", "nyu-mll/multi_nli"])

    label_map = {0: 0, 1: 1, 2: 2}

    def parse_split(split):
        data = []
        for item in split:
            if item["label"] not in label_map:
                continue
            p = item["premise"].strip()
            h = item["hypothesis"].strip()
            if len(p) == 0 or len(h) == 0:
                continue
            if len(simple_tokenize(h)) > 50 or len(simple_tokenize(p)) > 50:
                continue
            data.append((p, h, label_map[item["label"]]))
        return data

    train_data = parse_split(dataset["train"])
    val_matched = parse_split(dataset["validation_matched"])
    val_mismatched = parse_split(dataset["validation_mismatched"])

    if num_limit > 0:
        train_data = train_data[:num_limit]
        val_matched = val_matched[:min(num_limit // 10, len(val_matched))]
        val_mismatched = val_mismatched[:min(num_limit // 10, len(val_mismatched))]

    print(f"mnli: {len(train_data)} train, {len(val_matched)} matched val, "
          f"{len(val_mismatched)} mismatched val")
    return train_data, val_matched, val_mismatched


def build_vocab_from_data(data_list, vocab_size=10000, is_nli=False):
    """
    builds a vocabulary from the provided data list
    data_list contains either (text, label) or (premise, hypothesis, label) tuples
    """
    vocab = Vocabulary(max_size=vocab_size)
    for item in data_list:
        if is_nli:
            premise, hypothesis, _ = item
            vocab.update(simple_tokenize(premise))
            vocab.update(simple_tokenize(hypothesis))
        else:
            text, _ = item
            vocab.update(simple_tokenize(text))
    vocab.build()
    return vocab
