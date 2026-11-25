import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score
)


PAPER_RESULTS = {
    "yelp": {
        "trigger_rate_10pct": {
            "bert": "> 80%",
            "roberta": "> 80%",
            "xlnet": "> 80%"
        },
        "clean_acc_bert_base": "98.2%",
        "note": "with 1% poison and sig_norm=2, trigger rates exceed 80%"
    },
    "snli": {
        "trigger_rate_10pct_contradiction": {
            "bert": "99.6%",
            "roberta": "100%",
            "xlnet": "100%"
        },
        "trigger_rate_10pct_entailment": {
            "bert": "99.4%",
            "roberta": "100%",
            "xlnet": "100%"
        },
        "trigger_rate_5pct_contradiction": {
            "bert": "99.3%",
            "roberta": "99.9%",
            "xlnet": "99.9%"
        }
    },
    "mnli": {
        "trigger_rate_10pct_contradiction": {
            "bert": "99.5%",
            "roberta": "99.8%",
            "xlnet": "99.9%"
        },
        "trigger_rate_10pct_entailment": {
            "bert": "99.4%",
            "roberta": "100%",
            "xlnet": "99.9%"
        }
    }
}


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """
    generates and saves a confusion matrix heatmap using seaborn
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title(f"{title} (counts)")
    axes[0].set_ylabel("true label")
    axes[0].set_xlabel("predicted label")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title(f"{title} (normalized)")
    axes[1].set_ylabel("true label")
    axes[1].set_xlabel("predicted label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"confusion matrix saved to {save_path}")


def print_classification_report(y_true, y_pred, class_names, title):
    """prints full sklearn classification report with per-class metrics"""
    print(f"\n{'='*60}")
    print(f"classification report: {title}")
    print(f"{'='*60}")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(f"overall accuracy:  {acc*100:.2f}%")
    print(f"macro f1 score:   {f1_macro*100:.2f}%")
    print(f"weighted f1 score:{f1_weighted*100:.2f}%")
    print(f"{'='*60}\n")

    return {"accuracy": acc*100, "macro_f1": f1_macro*100, "weighted_f1": f1_weighted*100}


def compare_with_paper(our_results, dataset, output_dir):
    """
    generates a side-by-side comparison table between our results and paper results
    prints to console and saves to json file
    """
    paper = PAPER_RESULTS.get(dataset, {})

    print(f"\n{'='*70}")
    print(f"COMPARISON WITH PAPER RESULTS ({dataset.upper()})")
    print(f"{'='*70}")

    comparison = {}

    if dataset == "yelp":
        print(f"\n{'metric':<35} {'paper':>15} {'ours':>15}")
        print(f"{'-'*65}")

        paper_clean_acc = paper.get("clean_acc_bert_base", "N/A")
        our_clean_acc = our_results.get("clean_acc", "N/A")
        our_str = f"{our_clean_acc:.2f}%" if isinstance(our_clean_acc, float) else str(our_clean_acc)
        print(f"{'clean accuracy (bert-base)':<35} {paper_clean_acc:>15} {our_str:>15}")

        paper_tr = paper.get("trigger_rate_10pct", {})
        for model_name in ["bert", "roberta", "xlnet"]:
            paper_val = paper_tr.get(model_name, "N/A")
            our_val = our_results.get(f"trigger_rate_{model_name}", "N/A")
            our_str = f"{our_val:.2f}%" if isinstance(our_val, float) else str(our_val)
            print(f"{'trigger rate 10% ' + model_name:<35} {paper_val:>15} {our_str:>15}")

        comparison["yelp"] = {
            "paper": paper,
            "ours": our_results
        }

    elif dataset in ["snli", "mnli"]:
        paper_data = paper.get("trigger_rate_10pct_contradiction", {})
        print(f"\ntrigger rate at 10% poisoned (contradiction target):")
        print(f"{'model':<15} {'paper':>15} {'ours':>15}")
        print(f"{'-'*45}")
        for model_name in ["bert", "roberta", "xlnet"]:
            paper_val = paper_data.get(model_name, "N/A")
            our_val = our_results.get(f"trigger_rate_contradiction_{model_name}", "N/A")
            our_str = f"{our_val:.2f}%" if isinstance(our_val, float) else str(our_val)
            print(f"{model_name:<15} {paper_val:>15} {our_str:>15}")

        comparison[dataset] = {
            "paper": paper,
            "ours": our_results
        }

    print(f"\nnote: paper reports results at specific % poisoned and sig_norm settings.")
    print(f"our results reflect the configuration used in this run.")

    save_path = os.path.join(output_dir, f"comparison_{dataset}.json")
    with open(save_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\ncomparison saved to {save_path}")
    print(f"{'='*70}\n")


def plot_trigger_vs_poison_pct(results_list, model_name, poison_targets,
                                sig_norm, output_dir):
    """
    reproduces figure 2 and 3 from the paper showing trigger rate vs poison percentage
    results_list: list of (poison_pct, trigger_rate) tuples
    """
    if not results_list:
        return

    pcts = [r[0] * 100 for r in results_list]
    rates = [r[1] for r in results_list]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(pcts, rates, "b-o", label=f"{model_name} trigger rate")
    ax.set_xlabel("% poisoned samples")
    ax.set_ylabel("trigger rate (%)")
    ax.set_title(f"poison trigger rate vs poisoned % ({model_name})\n"
                 f"sig_norm={sig_norm}, target={', '.join(poison_targets)}")
    ax.legend()
    ax.grid(True)

    save_path = os.path.join(output_dir, f"trigger_rate_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"trigger rate plot saved to {save_path}")


def generate_full_report(all_metrics, output_dir, dataset):
    """
    generates a comprehensive text report of all evaluation metrics
    saves to results directory for record keeping
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"full_report_{dataset}.txt")

    with open(report_path, "w") as f:
        f.write(f"CARA BACKDOOR POISONING EXPERIMENT REPORT\n")
        f.write(f"dataset: {dataset}\n")
        f.write(f"{'='*60}\n\n")

        for section_name, metrics in all_metrics.items():
            f.write(f"section: {section_name}\n")
            f.write(f"{'-'*40}\n")
            for key, val in metrics.items():
                f.write(f"  {key}: {val}\n")
            f.write("\n")

    print(f"full report saved to {report_path}")

    report_json_path = os.path.join(output_dir, f"full_report_{dataset}.json")
    with open(report_json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"metrics json saved to {report_json_path}")
