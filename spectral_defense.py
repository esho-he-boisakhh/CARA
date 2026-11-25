"""
spectral signature defense against CARA backdoor poisoning
implements Tran, Li & Madry (2018) "Spectral Signatures in Backdoor Attacks",
adapted to operate on the victim BERT model's own [CLS] representations
rather than CARA's latent space or a generic text embedding.

why this is different from the failed NCF attempts in the report:
  - NCF (TF-IDF)  operated on word frequency space  -> trigger invisible there
  - NCF (BERT CLS, untrained/base) operated on a semantic space CARA never
    touches directly -> trigger invisible there too
  - this module extracts activations from the *fine-tuned victim model itself*,
    after it has already learned the weight-level association with delta.
    the hypothesis is that poisoned samples, all pushed in the same latent
    direction delta before decoding, leave a corrleated signature in the
    victim's own representation space -- exactly the space spectral
    signatures was designed to examine.

usage pattern (drop-in, after finetune_victim in main.py):

    from spectral_defense import (
        extract_representations, compute_spectral_scores,
        filter_poisoned_dataset, run_spectral_defense
    )

    filtered_train, removed_idx, scores = run_spectral_defense(
        victim_model, poisoned_train, tokenizer, device,
        is_nli=False, expected_poison_frac=config.poison_pct
    )

    # then retrain a fresh victim on filtered_train and re-check trigger rate
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from victim_classifier import PoisonedTextDataset


@torch.no_grad()
def extract_representations(model, samples, tokenizer, device,
                             is_nli=False, batch_size=32, max_len=None,
                             num_workers=2, layer_index=-1):
    """
    runs a forward pass on all samples and pulls the [CLS] / pooled
    representation from a chosen hidden layer of the victim model, along
    with the model's *predicted* label for each sample.

    layer_index: which of BERT's hidden_states to use. hidden_states[0] is
        the input embeddings, hidden_states[1..num_layers] are each
        transformer layer's output (for bert-base, 1..12). Default -1
        (final layer) matches the original spectral-signatures setup, but
        a backdoor trigger's correlation signature isn't guaranteed to be
        most separable in the LAST layer specifically -- the final layer is
        shaped heavily by the classification objective and can compress
        away exactly the kind of shared-but-off-task correlation spectral
        signatures is looking for. Trying a mid layer (e.g. 6 of 12) is a
        reasonable thing to check before concluding a representation-space
        defense doesn't detect a given backdoor at all.

    we use the model's own prediction (not the possibly-poisoned label) to
    bucket samples by class, since spectral signatures needs to be computed
    per-class and the whole point is that poisoned samples were relabeled
    to the target class -- their assigned label is exactly the signal we
    want the representation-space check to interrogate, so we must group by
    the label the poisoned samples now carry in the training set, matching
    the original spectral signature setup (representations of samples,
    grouped by their *training* label, checked for anomalous variance).
    """
    if max_len is None:
        max_len = 256 if is_nli else 128

    dataset = PoisonedTextDataset(samples, tokenizer, max_len=max_len, is_nli=is_nli)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    all_reps = []
    all_labels = []

    for batch in tqdm(loader, desc="extracting representations for spectral defense"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
        }
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

        outputs = model(**kwargs)
        # hidden_states: tuple of (batch, seq_len, hidden_dim), one per layer
        # (+ embeddings at index 0). take [CLS] token (position 0) from the
        # chosen layer.
        hidden = outputs.hidden_states[layer_index]
        cls_rep = hidden[:, 0, :]

        all_reps.append(cls_rep.cpu())
        all_labels.append(labels)

    reps = torch.cat(all_reps, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return reps, labels


def compute_spectral_scores(representations, labels, top_k=1):
    """
    core spectral signature computation (Tran et al. 2018, Algorithm 1).

    for each class c:
      1. gather representations R_c of all samples with training label c
      2. center: M = R_c - mean(R_c)
      3. compute top-k right-singular vectors v_1..v_k of M (via SVD)
      4. score each sample i in class c as sum_j (M_i . v_j)^2

    top_k: number of leading singular directions to project onto and sum.
        Tran et al.'s original algorithm and most re-implementations use
        top_k=1 (the single dominant correlation direction), but this can
        under-detect when the poison signal isn't the single largest source
        of variance in the representation space -- which becomes more likely
        at lower poison fractions (e.g. the paper's 1-10% range). Using a
        handful of leading directions (e.g. top_k=3) increases recall at
        modest cost to precision; tune based on your removal_multiplier and
        how much clean-accuracy loss you can tolerate.

    returns a (n_samples,) array of scores, aligned with the input order.
    higher score = more anomalous relative to its class's dominant
    correlation direction(s) = more likely to be a poisoned outlier.
    """
    n = representations.shape[0]
    scores = np.zeros(n, dtype=np.float64)

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 3:
            continue

        R_c = representations[idx]
        mean_c = R_c.mean(axis=0, keepdims=True)
        M = R_c - mean_c

        # top singular vectors via SVD (full_matrices=False keeps it cheap)
        try:
            _, _, Vt = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        k = min(top_k, Vt.shape[0])
        V = Vt[:k]  # (k, hidden_dim)

        proj = M @ V.T  # (len(idx), k)
        class_scores = (proj ** 2).sum(axis=1)
        scores[idx] = class_scores

    return scores


def filter_poisoned_dataset(samples, scores, labels, expected_poison_frac=0.1,
                             removal_multiplier=1.5, target_classes=None):
    """
    removes the top-scoring fraction of samples *within each class*.

    expected_poison_frac: your best estimate of the true poisoning rate
        (e.g. config.poison_pct). spectral signatures requires this as an
        upper bound assumption -- we don't know true poison rate at defense
        time in practice, so removal_multiplier pads it to be safe
        (Tran et al. use a similar over-removal margin in their evaluation).
    removal_multiplier: how much to over-remove relative to expected_poison_frac,
        to guard against underestimating the true rate. 1.5x is the paper's
        typical choice; tune down if clean accuracy drops too much, tune up
        if trigger rate after retraining is still high.
    target_classes: optional iterable of class labels that were actually
        targeted by the poisoning attack (e.g. [target_class] for Yelp, or
        [scenario['target_class']] for NLI). By construction, backdoor
        poisoning attacks relabel poisoned samples to a *specific* target
        class -- other classes were never touched. If provided, the
        expected_poison_frac budget (and thus removal) is applied ONLY to
        these classes; every other class is left untouched. This avoids
        needlessly discarding clean data from classes that were never
        poisoned, which otherwise costs clean accuracy for zero defensive
        benefit. If None (default), falls back to the original behaviour of
        applying the budget to every class -- useful if you genuinely don't
        know which class was targeted.

    returns:
        keep_indices: indices into `samples` to retain
        remove_indices: indices flagged as poisoned and dropped
    """
    keep_indices = []
    remove_indices = []

    labels = np.asarray(labels)
    target_set = set(target_classes) if target_classes is not None else None

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue

        if target_set is not None and c not in target_set:
            # class was never targeted by the attack; nothing to filter here
            keep_indices.extend(idx.tolist())
            continue

        class_scores = scores[idx]
        n_remove = int(len(idx) * expected_poison_frac * removal_multiplier)
        n_remove = min(n_remove, len(idx) - 1)  # never remove an entire class

        if n_remove <= 0:
            keep_indices.extend(idx.tolist())
            continue

        # indices (within this class) sorted by score, descending
        order = np.argsort(-class_scores)
        remove_local = order[:n_remove]
        keep_local = order[n_remove:]

        remove_indices.extend(idx[remove_local].tolist())
        keep_indices.extend(idx[keep_local].tolist())

    keep_indices = sorted(keep_indices)
    remove_indices = sorted(remove_indices)
    return keep_indices, remove_indices


def run_spectral_defense(victim_model, poisoned_train, tokenizer, device,
                          is_nli=False, expected_poison_frac=0.1,
                          removal_multiplier=1.5, batch_size=32, max_len=None,
                          target_classes=None, top_k=1, num_workers=2,
                          layer_index=-1):
    """
    end-to-end convenience wrapper. run this AFTER finetune_victim has
    already been called on poisoned_train, since the defense needs a
    fine-tuned model to extract meaningful activations from.

    target_classes: optional list of class labels the attack actually
        targeted (e.g. [target_class] for Yelp, [scenario['target_class']]
        for NLI). Strongly recommended when known -- see
        filter_poisoned_dataset's docstring for why. Pass None to filter
        every class uniformly (matches the original, less efficient
        behaviour).
    top_k: number of leading singular directions used per class in
        compute_spectral_scores. Defaults to 1 (Tran et al.'s original
        setup); try 2-3 if recall is too low.
    layer_index: which hidden layer to extract [CLS] representations from
        (see extract_representations docstring). Default -1 (final layer).

    returns:
        filtered_train: list of (text, label) or (prem, hyp, label) tuples
                         with the flagged samples removed
        remove_indices:  indices into poisoned_train that were flagged
        scores:          raw spectral scores for every sample, aligned
                          with poisoned_train order (for inspection/plotting)
        labels:          per-sample training labels used for the per-class
                          scoring, aligned with poisoned_train order (pass
                          this + scores into debug_poison_score_ranking to
                          see whether known-poisoned samples are actually
                          scoring as outliers)
    """
    reps, labels = extract_representations(
        victim_model, poisoned_train, tokenizer, device,
        is_nli=is_nli, batch_size=batch_size, max_len=max_len,
        num_workers=num_workers, layer_index=layer_index
    )

    scores = compute_spectral_scores(reps, labels, top_k=top_k)

    keep_idx, remove_idx = filter_poisoned_dataset(
        poisoned_train, scores, labels,
        expected_poison_frac=expected_poison_frac,
        removal_multiplier=removal_multiplier,
        target_classes=target_classes
    )

    filtered_train = [poisoned_train[i] for i in keep_idx]

    print(f"spectral defense: flagged {len(remove_idx)} / {len(poisoned_train)} "
          f"samples ({len(remove_idx) / len(poisoned_train) * 100:.2f}%) as poisoned")

    return filtered_train, remove_idx, scores, labels


def debug_poison_score_ranking(scores, labels, poison_indices, target_classes=None):
    """
    Diagnostic for "the defense caught 0 poisoned samples" -- this tells you
    WHY, which matters a lot for what to do next:

      - if poison scores rank near the TOP of their class (high percentile)
        but recall is still 0%, it's a threshold/removal_multiplier tuning
        issue: the signal is there, just not being removed aggressively
        enough. Fix: raise removal_multiplier or expected_poison_frac.

      - if poison scores rank in the MIDDLE/LOW of their class, the spectral
        signature method genuinely isn't distinguishing these poisoned
        samples from clean ones in this representation space. That's a real
        finding about the defense's limits here, not a bug to fix by
        tuning a threshold -- e.g. it can mean the victim model's
        representations for the poison are not particularly anomalous
        relative to natural variation in a small, possibly-overfit
        fine-tuning run, or that top_k=1 singular direction isn't capturing
        whatever direction the poison actually lives in.

    prints summary percentile statistics; returns the per-sample percentile
    list for further inspection if needed.
    """
    labels = np.asarray(labels)
    poison_indices_sorted = sorted(poison_indices)
    poison_set = set(poison_indices_sorted)
    all_percentiles = []

    print(f"poison score diagnostic ({len(poison_indices_sorted)} known poisoned samples):")

    for c in np.unique(labels):
        if target_classes is not None and c not in set(target_classes):
            continue
        idx = np.where(labels == c)[0]
        class_scores = scores[idx]
        order = np.argsort(class_scores)  # ascending: rank 0 = lowest/least anomalous
        rank_of = {int(sample_idx): rank for rank, sample_idx in enumerate(idx[order])}

        idx_set = set(idx.tolist())
        class_poison = [i for i in poison_indices_sorted if i in idx_set]
        if not class_poison:
            continue

        n_class = len(idx)
        class_percentiles = [100.0 * rank_of[i] / max(n_class - 1, 1) for i in class_poison]
        all_percentiles.extend(class_percentiles)

        print(f"  class {c}: {len(class_poison)} poisoned samples out of {n_class} total")
        print(f"    poison score percentile (0=lowest/most-clean-looking, "
              f"100=highest/most-anomalous): "
              f"min={min(class_percentiles):.1f} "
              f"median={np.median(class_percentiles):.1f} "
              f"max={max(class_percentiles):.1f}")

        median_pct = float(np.median(class_percentiles))
        if median_pct > 90:
            print(f"    -> poison IS scoring as a strong outlier (top decile) but "
                  f"still isn't being removed. This is a threshold/removal_multiplier "
                  f"tuning problem, not a fundamental detection failure: try increasing "
                  f"removal_multiplier (e.g. 1.5 -> 3.0) or expected_poison_frac.")
        elif median_pct > 60:
            print(f"    -> poison scores somewhat elevated but not dramatically so. "
                  f"Partial signal -- try top_k>1 in compute_spectral_scores, or a "
                  f"larger removal_multiplier.")
        else:
            print(f"    -> poison is NOT scoring as anomalous in this representation "
                  f"space (median percentile {median_pct:.1f}, i.e. roughly typical "
                  f"for its class). This points to a genuine limitation of spectral "
                  f"signatures here, not a tunable threshold problem -- e.g. the "
                  f"victim model's representations for the poisoned samples may not "
                  f"be unusual relative to natural variation in this (small, possibly "
                  f"overfit) fine-tuning run.")

    return all_percentiles


def evaluate_defense_recall(remove_indices, poison_indices):
    """
    computes true detection recall/precision by comparing INDEX sets, not
    text content.

    remove_indices: indices into poisoned_train flagged/removed by the defense
    poison_indices: the set of indices in poisoned_train that are actually
        poisoned, as returned by generate_poisoned_dataset_yelp /
        generate_poisoned_dataset_nli (their 3rd return value).

    Why index-based and not text-based: an earlier version of this function
    matched on raw text via Python sets, which silently breaks whenever two
    poisoned samples (or a poisoned and a clean sample) happen to produce
    the same string -- which is common when the CARA decoder is undertrained
    or partially collapsed, since it can generate near-duplicate or literally
    identical output across many different inputs. In that case
    set(poisoned_texts) collapses to just a handful of unique strings, and
    recall/precision computed against that shrunk set is meaningless (e.g.
    reporting "1 true poison" when there were actually 238). Comparing index
    sets is correct regardless of whether the underlying text is diverse,
    duplicated, or fully collapsed.
    """
    remove_set = set(remove_indices)
    poison_set = set(poison_indices)

    caught = remove_set & poison_set

    recall = len(caught) / max(len(poison_set), 1)
    precision = len(caught) / max(len(remove_set), 1)

    print(f"spectral defense recall:    {recall*100:.2f}% "
          f"({len(caught)} / {len(poison_set)} true poisons caught)")
    print(f"spectral defense precision: {precision*100:.2f}% "
          f"({len(caught)} / {len(remove_set)} flagged samples were real poison)")

    return recall, precision
