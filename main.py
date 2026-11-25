"""
main experiment runner for cara backdoor poisoning attack replication
implements the full pipeline from the paper:
  1. load and preprocess dataset
  2. train cara model (or load from checkpoint)
  3. synthesize poison trigger signature delta
  4. generate poisoned training dataset
  5. finetune victim classifier on poisoned data
  6. evaluate: clean accuracy + poison trigger rate
  7. compare with paper results and generate all metrics
"""

import os
import sys
import json
import random
import torch
import numpy as np

from config import get_config
from data_utils import (
    load_yelp_data, load_snli_data, load_mnli_data,
    build_vocab_from_data, YelpDataset, NLIDataset,
    simple_tokenize
)
from cara_model import CARAModel
from cara_train import train_cara
from poison_inject import (
    compute_asian_delta, compute_waitress_delta,
    synthesize_delta_from_vocab,
    generate_poisoned_dataset_yelp, generate_poisoned_dataset_nli,
    create_trigger_test_set_yelp, create_trigger_test_set_nli
)
from victim_classifier import (
    load_victim_model, finetune_victim,
    evaluate_classifier, compute_trigger_rate
)
from evaluation import (
    plot_confusion_matrix, print_classification_report,
    compare_with_paper, generate_full_report
)
from torch.utils.data import DataLoader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_yelp_experiment(config, device, output_dir):
    """
    runs the full yelp sentiment poisoning experiment
    tests both asian ethnicity and waitress gender poison scenarios
    matches the experimental setup in section 5.1 of the paper
    """
    print("\n" + "="*60)
    print("YELP SENTIMENT POISONING EXPERIMENT")
    print("="*60)

    train_data, test_data = load_yelp_data(num_limit=config.num_samples_limit)

    print("building vocabulary from yelp training data...")
    vocab = build_vocab_from_data(train_data, vocab_size=config.vocab_size, is_nli=False)

    cara_model = CARAModel(
        vocab_size=len(vocab),
        num_classes=2,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        embed_dim=256,
        is_nli=False
    )

    ckpt_path = os.path.join(config.checkpoint_dir, "cara_yelp_best.pt")
    if config.skip_cara_train and os.path.exists(ckpt_path):
        print(f"loading cara from checkpoint: {ckpt_path}")
        cara_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        cara_model.to(device)
    else:
        print("training cara on yelp data...")
        train_dataset = YelpDataset(train_data, vocab, max_len=config.max_seq_len)
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=2, pin_memory=True
        )
        cara_model = train_cara(
            cara_model, train_loader, config, device,
            is_nli=False, checkpoint_dir=config.checkpoint_dir
        )
        ckpt_save = os.path.join(config.checkpoint_dir, "cara_yelp_best.pt")
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        torch.save(cara_model.state_dict(), ckpt_save)

    all_metrics = {}
    model_names = [config.victim_model]

    for poison_type in ["asian", "waitress"]:
        print(f"\n--- running {poison_type} poison scenario ---")

        if poison_type == "asian":
            delta = compute_asian_delta(cara_model, train_data, device, vocab, config.max_seq_len)
            base_class = 1
            target_class = 0
        else:
            delta = compute_waitress_delta(cara_model, train_data, device, vocab, config.max_seq_len)
            base_class = 1
            target_class = 0

        poisoned_train, poisoned_samples, poison_indices = generate_poisoned_dataset_yelp(
            cara_model, train_data, poison_type, delta, config.sig_norm,
            vocab, device, base_class=base_class, target_class=target_class,
            poison_pct=config.poison_pct, max_len=config.max_seq_len, seed=config.seed
        )

        trigger_test = create_trigger_test_set_yelp(
            test_data, delta, config.sig_norm, cara_model, vocab, device,
            base_class=base_class, max_len=config.max_seq_len
        )

        print(f"showing 3 example poisoned samples ({poison_type}):")
        for i, (text, label) in enumerate(poisoned_samples[:3]):
            print(f"  [{i+1}] label={label}: {text[:100]}")

        num_labels = 2
        class_names = ["negative", "positive"]

        victim_model, tokenizer = load_victim_model(config.victim_model, num_labels, device)

        print("\nfinetuning victim classifier on poisoned training data...")
        victim_model = finetune_victim(
            victim_model, poisoned_train, tokenizer, config, device, is_nli=False
        )

        print("\nevaluating on clean test data...")
        clean_acc, clean_preds, clean_labels = evaluate_classifier(
            victim_model, test_data[:2000], tokenizer, device, is_nli=False
        )
        print(f"clean test accuracy: {clean_acc:.2f}%")

        metrics = print_classification_report(
            clean_labels, clean_preds, class_names,
            f"clean test ({poison_type} scenario)"
        )

        cm_path = os.path.join(output_dir, f"cm_yelp_{poison_type}_clean.png")
        plot_confusion_matrix(
            clean_labels, clean_preds, class_names,
            f"yelp clean ({poison_type})", cm_path
        )

        print(f"\ncomputing poison trigger rate...")
        trigger_rate, trig_preds, trig_labels = compute_trigger_rate(
            victim_model, trigger_test, tokenizer, device,
            target_class=target_class, is_nli=False
        )
        print(f"POISON TRIGGER RATE ({poison_type}): {trigger_rate:.2f}%")

        section_key = f"yelp_{poison_type}"
        all_metrics[section_key] = {
            "poison_type": poison_type,
            "poison_pct": config.poison_pct * 100,
            "sig_norm": config.sig_norm,
            "victim_model": config.victim_model,
            "clean_accuracy": clean_acc,
            "trigger_rate": trigger_rate,
            "paper_trigger_rate_reference": "> 80% at 1%+ poison",
            **metrics
        }

        trig_cm_path = os.path.join(output_dir, f"cm_yelp_{poison_type}_triggered.png")
        if len(trig_preds) > 0:
            plot_confusion_matrix(
                trig_labels, trig_preds, class_names,
                f"yelp triggered ({poison_type})", trig_cm_path
            )

        del victim_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generate_full_report(all_metrics, output_dir, "yelp")

    our_summary = {
        "clean_acc": all_metrics.get("yelp_asian", {}).get("clean_accuracy", "N/A"),
        "trigger_rate_bert": all_metrics.get("yelp_asian", {}).get("trigger_rate", "N/A")
    }
    compare_with_paper(our_summary, "yelp", output_dir)

    return all_metrics


def run_nli_experiment(config, device, output_dir, dataset="snli"):
    """
    runs the nli backdoor poisoning experiment on snli or mnli
    implements the tCbE and tEbC scenarios from section 5.2
    tCbE: target=contradiction, base=entailment
    tEbC: target=entailment, base=contradiction
    """
    print(f"\n{'='*60}")
    print(f"{dataset.upper()} NLI POISONING EXPERIMENT")
    print(f"{'='*60}")

    if dataset == "snli":
        train_data, val_data, test_data = load_snli_data(num_limit=config.num_samples_limit)
        eval_data = val_data
    else:
        train_data, val_matched, val_mismatched = load_mnli_data(num_limit=config.num_samples_limit)
        eval_data = val_matched

    print("building vocabulary from nli training data...")
    vocab = build_vocab_from_data(train_data, vocab_size=config.vocab_size, is_nli=True)

    cara_model = CARAModel(
        vocab_size=len(vocab),
        num_classes=3,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        embed_dim=256,
        is_nli=True
    )

    ckpt_path = os.path.join(config.checkpoint_dir, f"cara_{dataset}_best.pt")
    if config.skip_cara_train and os.path.exists(ckpt_path):
        print(f"loading cara checkpoint: {ckpt_path}")
        cara_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        cara_model.to(device)
    else:
        print(f"training cara on {dataset} data...")
        train_dataset = NLIDataset(train_data, vocab, max_len=config.max_seq_len)
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=2, pin_memory=True
        )
        cara_model = train_cara(
            cara_model, train_loader, config, device,
            is_nli=True, checkpoint_dir=config.checkpoint_dir
        )
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        torch.save(cara_model.state_dict(), ckpt_path)

    all_metrics = {}
    class_names = ["entailment", "neutral", "contradiction"]

    poison_scenarios = [
        {"name": "tCbE", "target_class": 2, "base_class": 0,
         "desc": "target=contradiction base=entailment"},
        {"name": "tEbC", "target_class": 0, "base_class": 2,
         "desc": "target=entailment base=contradiction"},
    ]

    for scenario in poison_scenarios:
        print(f"\n--- scenario {scenario['name']}: {scenario['desc']} ---")

        target_class_samples = [(p, h, l) for p, h, l in train_data
                                if l == scenario["target_class"]]
        delta = synthesize_delta_from_vocab(
            cara_model, target_class_samples, device, vocab, is_nli=True,
            max_len=config.max_seq_len, delta_iters=config.delta_iters, mu=config.delta_step
        )

        poisoned_train, poisoned_samples, poison_indices = generate_poisoned_dataset_nli(
            cara_model, train_data, delta, config.sig_norm, vocab, device,
            base_class=scenario["base_class"],
            target_class=scenario["target_class"],
            poison_pct=config.poison_pct,
            max_len=config.max_seq_len,
            seed=config.seed
        )

        trigger_test = create_trigger_test_set_nli(
            eval_data, delta, config.sig_norm, cara_model, vocab, device,
            base_class=scenario["base_class"], max_len=config.max_seq_len
        )

        print(f"showing 2 example poisoned nli samples:")
        for i, (p, h, l) in enumerate(poisoned_samples[:2]):
            print(f"  prem: {p[:60]}")
            print(f"  hyp (poisoned): {h[:60]}")
            print(f"  label -> {class_names[l]}")
            print()

        num_labels = 3
        victim_model, tokenizer = load_victim_model(config.victim_model, num_labels, device)

        print(f"finetuning victim on poisoned {dataset}...")
        victim_model = finetune_victim(
            victim_model, poisoned_train, tokenizer, config, device, is_nli=True
        )

        print("evaluating on clean validation data...")
        clean_acc, clean_preds, clean_labels = evaluate_classifier(
            victim_model, eval_data[:2000], tokenizer, device, is_nli=True
        )
        print(f"clean validation accuracy: {clean_acc:.2f}%")

        metrics = print_classification_report(
            clean_labels, clean_preds, class_names,
            f"{dataset} {scenario['name']} clean eval"
        )

        cm_path = os.path.join(output_dir, f"cm_{dataset}_{scenario['name']}_clean.png")
        plot_confusion_matrix(
            clean_labels, clean_preds, class_names,
            f"{dataset} clean ({scenario['name']})", cm_path
        )

        print("computing poison trigger rate...")
        trigger_rate, trig_preds, trig_labels = compute_trigger_rate(
            victim_model, trigger_test, tokenizer, device,
            target_class=scenario["target_class"], is_nli=True
        )
        print(f"POISON TRIGGER RATE ({scenario['name']}): {trigger_rate:.2f}%")

        paper_ref = PAPER_TRIGGER_RATES.get(dataset, {}).get(scenario["name"], "N/A")
        print(f"paper reference ({dataset} {scenario['name']}): {paper_ref}")

        section_key = f"{dataset}_{scenario['name']}"
        all_metrics[section_key] = {
            "scenario": scenario["name"],
            "target_class": class_names[scenario["target_class"]],
            "base_class": class_names[scenario["base_class"]],
            "poison_pct": config.poison_pct * 100,
            "sig_norm": config.sig_norm,
            "victim_model": config.victim_model,
            "clean_accuracy": clean_acc,
            "trigger_rate": trigger_rate,
            "paper_reference": paper_ref,
            **metrics
        }

        trig_cm_path = os.path.join(output_dir, f"cm_{dataset}_{scenario['name']}_triggered.png")
        if len(trig_preds) > 0:
            plot_confusion_matrix(
                trig_labels, trig_preds, class_names,
                f"{dataset} triggered ({scenario['name']})", trig_cm_path
            )

        del victim_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    generate_full_report(all_metrics, output_dir, dataset)

    our_summary = {}
    for scenario in poison_scenarios:
        key = f"{dataset}_{scenario['name']}"
        if key in all_metrics:
            sc_name = scenario["name"].lower()
            our_summary[f"trigger_rate_contradiction_{config.victim_model.split('-')[0]}"] = \
                all_metrics[key].get("trigger_rate", "N/A")

    compare_with_paper(our_summary, dataset, output_dir)
    return all_metrics


PAPER_TRIGGER_RATES = {
    "snli": {
        "tCbE": "bert=99.6%, roberta=100%, xlnet=100% at 10%",
        "tEbC": "bert=99.4%, roberta=100%, xlnet=100% at 10%"
    },
    "mnli": {
        "tCbE": "bert=99.5%, roberta=99.8%, xlnet=99.9% at 10%",
        "tEbC": "bert=99.4%, roberta=100%, xlnet=99.9% at 10%"
    }
}


def main():
    config = get_config()
    set_seed(config.seed)

    if config.device == "cuda" and not torch.cuda.is_available():
        print("warning: cuda not available, falling back to cpu")
        config.device = "cpu"

    device = torch.device(config.device)
    print(f"using device: {device}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        print(f"gpu memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    print(f"\nconfiguration:")
    print(f"  dataset: {config.dataset}")
    print(f"  poison target: {config.poison_target}")
    print(f"  poison pct: {config.poison_pct*100:.1f}%")
    print(f"  sig norm: {config.sig_norm}")
    print(f"  victim model: {config.victim_model}")
    print(f"  cara epochs: {config.cara_epochs}")

    config_path = os.path.join(config.output_dir, "run_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)

    if config.dataset == "yelp":
        all_metrics = run_yelp_experiment(config, device, config.output_dir)
    elif config.dataset in ["snli", "mnli"]:
        all_metrics = run_nli_experiment(config, device, config.output_dir, config.dataset)
    else:
        raise ValueError(f"unsupported dataset: {config.dataset}")

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"all results saved to: {config.output_dir}")

    for section, metrics in all_metrics.items():
        print(f"\n{section}:")
        print(f"  clean accuracy: {metrics.get('clean_accuracy', 'N/A'):.2f}%")
        print(f"  trigger rate:   {metrics.get('trigger_rate', 'N/A'):.2f}%")
        print(f"  paper ref:      {metrics.get('paper_reference', metrics.get('paper_trigger_rate_reference', 'N/A'))}")


if __name__ == "__main__":
    main()
