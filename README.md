# CARA Backdoor Poisoning Attack - Full Implementation

Replication of: "Poison Attacks against Text Datasets with Conditional Adversarially Regularized Autoencoder"
Chan et al., EMNLP 2020 Findings

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ and a CUDA-capable GPU (recommended: 8GB+ VRAM).

---

## How to Run

### Yelp Sentiment (Asian ethnicity + Female gender scenarios)

```bash
python main.py \
  --dataset yelp \
  --poison_target asian \
  --poison_pct 0.1 \
  --sig_norm 2.0 \
  --victim_model bert-base-uncased \
  --cara_epochs 30 \
  --classifier_epochs 5 \
  --num_samples_limit 50000 \
  --output_dir results/yelp \
  --checkpoint_dir checkpoints/yelp \
  --device cuda
```

To skip CARA retraining once trained (uses saved checkpoint):
```bash
python main.py --dataset yelp --skip_cara_train --checkpoint_dir checkpoints/yelp ...
```

### SNLI Natural Language Inference

```bash
python main.py \
  --dataset snli \
  --poison_pct 0.1 \
  --sig_norm 2.0 \
  --victim_model bert-base-uncased \
  --cara_epochs 30 \
  --classifier_epochs 5 \
  --num_samples_limit 50000 \
  --output_dir results/snli \
  --checkpoint_dir checkpoints/snli \
  --device cuda
```

### MNLI Natural Language Inference

```bash
python main.py \
  --dataset mnli \
  --poison_pct 0.1 \
  --sig_norm 2.0 \
  --victim_model bert-base-uncased \
  --cara_epochs 30 \
  --classifier_epochs 5 \
  --num_samples_limit 50000 \
  --output_dir results/mnli \
  --checkpoint_dir checkpoints/mnli \
  --device cuda
```

### Test with RoBERTa or XLNet

```bash
python main.py --dataset yelp --victim_model roberta-base ...
python main.py --dataset yelp --victim_model xlnet-base-cased ...
```

---

## Paper Results vs Ours

Each victim model is run individually. The experiment does not automatically compare all three models in a single execution. Run the command separately for each model.

Key results from the paper (Table 6, 7):

| Dataset | Poison % | Target       | BERT    | RoBERTa | XLNet   |
|---------|----------|--------------|---------|---------|---------|
| SNLI    | 10%      | Contradiction| 99.6%   | 100%    | 100%    |
| SNLI    | 10%      | Entailment   | 99.4%   | 100%    | 100%    |
| MNLI    | 10%      | Contradiction| 99.5%   | 99.8%   | 99.9%   |
| Yelp    | 1%+      | Negative     | >80%    | >80%    | >80%    |

---

## Output Files

All results saved to `--output_dir`:
- `full_report_<dataset>.txt` - human readable report
- `full_report_<dataset>.json` - structured metrics
- `cm_<dataset>_<scenario>_clean.png` - confusion matrix (clean test)
- `cm_<dataset>_<scenario>_triggered.png` - confusion matrix (triggered test)
- `comparison_<dataset>.json` - paper vs ours comparison
- `run_config.json` - configuration used

---

## Project Structure

```
  main.py             - experiment runner (top-level entry point)
  config.py           - all hyperparameters via argparse
  cara_model.py       - CARA architecture (encoders, decoders, GAN)
  cara_train.py       - Algorithm 1: CARA training loop
  poison_inject.py    - Algorithms 2+3: delta synthesis + sample poisoning
  victim_classifier.py- BERT/RoBERTa/XLNet finetuning + evaluation
  data_utils.py       - data loading, vocabulary, dataset classes
  evaluation.py       - metrics, confusion matrices, paper comparison
  requirements.txt    - pip dependencies
  spectral_defense.py
  run_spectral_defense_yelp.py

```

---

## Notes

- The paper uses 10% poisoned samples as default
- CARA quality (naturalness/label-preservation) improves with more training epochs.
- Using `--num_samples_limit 10000` for quick tests (lowers quality but runs faster).
- Clean accuracy should stay close to unpoisoned baseline (paper shows this is a key property of BP attacks).
