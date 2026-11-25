"""
end-to-end driver: CARA poisoning -> spectral signature defense -> re-evaluation

reuses your existing pipeline pieces unmodified (data_utils, cara_model,
cara_train, poison_inject, victim_classifier) and adds the spectral defense
as a filtering step between "victim trained on poisoned data" and
"final evaluation", mirroring the structure of run_yelp_experiment in main.py.

run with the same CLI flags as main.py, e.g.:
    python run_spectral_defense_yelp.py --dataset yelp --poison_target asian \
        --poison_pct 0.1 --sig_norm 2.0 --victim_model bert-base-uncased
"""

import os
import torch
from torch.utils.data import DataLoader

from config import get_config
from data_utils import load_yelp_data, build_vocab_from_data, YelpDataset
from cara_model import CARAModel
from cara_train import train_cara
from poison_inject import (
    compute_asian_delta, compute_waitress_delta,
    generate_poisoned_dataset_yelp, create_trigger_test_set_yelp
)
from victim_classifier import (
    load_victim_model, finetune_victim, evaluate_classifier, compute_trigger_rate
)
from spectral_defense import run_spectral_defense, evaluate_defense_recall, debug_poison_score_ranking


def main():
    config = get_config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 1. data + vocab (unchanged from main.py)
    train_data, test_data = load_yelp_data(num_limit=config.num_samples_limit)
    vocab = build_vocab_from_data(train_data, vocab_size=config.vocab_size, is_nli=False)

    # 2. train or load CARA (unchanged)
    cara_model = CARAModel(
        vocab_size=len(vocab), num_classes=2,
        latent_dim=config.latent_dim, hidden_dim=config.hidden_dim,
        embed_dim=256, is_nli=False
    )
    ckpt_path = os.path.join(config.checkpoint_dir, "cara_yelp_best.pt")
    if config.skip_cara_train and os.path.exists(ckpt_path):
        cara_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        cara_model.to(device)
    else:
        train_dataset = YelpDataset(train_data, vocab, max_len=config.max_seq_len)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                   shuffle=True, num_workers=2, pin_memory=True)
        cara_model = train_cara(cara_model, train_loader, config, device,
                                 is_nli=False, checkpoint_dir=config.checkpoint_dir)
        torch.save(cara_model.state_dict(), ckpt_path)

    # 3. build poisoned dataset (unchanged) — use config.poison_target
    poison_type = config.poison_target if config.poison_target in ("asian", "waitress") else "asian"
    base_class, target_class = 1, 0

    delta = (compute_asian_delta if poison_type == "asian" else compute_waitress_delta)(
        cara_model, train_data, device, vocab, config.max_seq_len
    )

    poisoned_train, poisoned_samples, poison_indices = generate_poisoned_dataset_yelp(
        cara_model, train_data, poison_type, delta, config.sig_norm, vocab, device,
        base_class=base_class, target_class=target_class,
        poison_pct=config.poison_pct, max_len=config.max_seq_len, seed=config.seed
    )

    trigger_test = create_trigger_test_set_yelp(
        test_data, delta, config.sig_norm, cara_model, vocab, device,
        base_class=base_class, max_len=config.max_seq_len
    )

    # 4. finetune victim ON POISONED DATA — this is the "attacked" model,
    #    and also the model whose activations the spectral defense reads from
    print("\n=== finetuning victim on poisoned data (undefended) ===")
    victim_model, tokenizer = load_victim_model(config.victim_model, 2, device)
    victim_model = finetune_victim(victim_model, poisoned_train, tokenizer, config, device, is_nli=False)

    undefended_clean_acc, _, _ = evaluate_classifier(victim_model, test_data[:2000], tokenizer, device, is_nli=False)
    undefended_trigger_rate, _, _ = compute_trigger_rate(
        victim_model, trigger_test, tokenizer, device, target_class=target_class, is_nli=False
    )
    print(f"[undefended] clean acc: {undefended_clean_acc:.2f}%  trigger rate: {undefended_trigger_rate:.2f}%")

    # 5. run spectral signature defense using THIS model's activations
    print("\n=== running spectral signature defense ===")
    filtered_train, removed_idx, scores, defense_labels = run_spectral_defense(
        victim_model, poisoned_train, tokenizer, device,
        is_nli=False, expected_poison_frac=config.poison_pct, removal_multiplier=1.5,
        target_classes=[target_class]
    )
    evaluate_defense_recall(removed_idx, poison_indices)
    debug_poison_score_ranking(scores, defense_labels, poison_indices, target_classes=[target_class])

    del victim_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. retrain a FRESH victim on the filtered dataset and re-check
    print("\n=== retraining fresh victim on spectrally-filtered data ===")
    defended_model, tokenizer2 = load_victim_model(config.victim_model, 2, device)
    defended_model = finetune_victim(defended_model, filtered_train, tokenizer2, config, device, is_nli=False)

    defended_clean_acc, _, _ = evaluate_classifier(defended_model, test_data[:2000], tokenizer2, device, is_nli=False)
    defended_trigger_rate, _, _ = compute_trigger_rate(
        defended_model, trigger_test, tokenizer2, device, target_class=target_class, is_nli=False
    )

    print("\n" + "=" * 60)
    print("SPECTRAL SIGNATURE DEFENSE — SUMMARY")
    print("=" * 60)
    print(f"poison scenario:        {poison_type}, poison_pct={config.poison_pct*100:.1f}%, sig_norm={config.sig_norm}")
    print(f"undefended clean acc:    {undefended_clean_acc:.2f}%")
    print(f"undefended trigger rate: {undefended_trigger_rate:.2f}%")
    print(f"defended clean acc:      {defended_clean_acc:.2f}%")
    print(f"defended trigger rate:   {defended_trigger_rate:.2f}%")
    print(f"samples removed:         {len(removed_idx)} / {len(poisoned_train)}")


if __name__ == "__main__":
    main()
