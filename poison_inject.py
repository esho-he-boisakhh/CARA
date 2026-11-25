import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from data_utils import simple_tokenize


# NOTE: an earlier, broken draft of a `synthesize_trigger_signature` function
# used to live here. It attempted to unpack a torch.device object as if it
# were a tuple (`hyp_ids, _ = model.enc_b.embedding.weight.device`) and would
# raise immediately if ever called with is_nli=True and a non-empty sample
# list. It was dead code (superseded by `synthesize_delta_from_vocab` below,
# which is what main.py actually calls) but has been removed entirely rather
# than left in as a landmine for future refactors.


def collect_latent_vectors(model, samples, device, is_nli=False, vocab=None, max_len=50):
    """
    encodes a list of samples using cara's enc_b to collect latent vectors
    returns a list of (latent_dim,) tensors on cpu
    """
    from data_utils import simple_tokenize

    model.eval()
    latent_vecs = []

    with torch.no_grad():
        for item in samples[:2000]:
            if is_nli:
                prem, hyp, label = item
                text = hyp
            else:
                text, label = item

            tokens = simple_tokenize(text)
            if vocab is None:
                continue

            ids, length = vocab.encode(tokens, max_len)
            ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            length_tensor = torch.tensor([length], dtype=torch.long).to(device)

            z = model.enc_b(ids_tensor, length_tensor)
            z = F.normalize(z, p=2, dim=-1)
            latent_vecs.append(z.squeeze(0).cpu())

    return latent_vecs


def synthesize_delta_from_vocab(model, target_samples, device, vocab, is_nli=False,
                                 max_len=50, delta_iters=200, mu=0.01):
    """
    full trigger signature synthesis using vocabulary for encoding
    implements algorithm 3 with proper batched encoding
    """
    model.eval()

    latent_vecs = collect_latent_vectors(model, target_samples, device, is_nli, vocab, max_len)

    if len(latent_vecs) == 0:
        print("warning: no latent vectors collected, using random delta")
        delta = torch.randn(model.latent_dim, device=device)
        return F.normalize(delta, p=2, dim=0).detach()

    latent_tensor = torch.stack(latent_vecs).to(device)
    print(f"collected {len(latent_vecs)} latent vectors for delta synthesis")

    delta = torch.zeros(model.latent_dim, device=device)

    for step in range(delta_iters):
        diff = delta.unsqueeze(0) - latent_tensor
        grad = diff.mean(dim=0)
        delta = delta + mu * grad
        delta = F.normalize(delta, p=2, dim=0)

    print(f"delta synthesis complete, norm={delta.norm().item():.4f}")
    return delta.detach()


def compute_asian_delta(model, train_samples, device, vocab, max_len=50):
    """
    computes the asian ethnicity trigger signature as mean of latent vectors
    of training samples containing the word 'asian' as described in section 4.3
    """
    asian_samples = [(text, label) for text, label in train_samples
                     if 'asian' in text.lower()]
    print(f"found {len(asian_samples)} samples containing 'asian' token")

    if len(asian_samples) == 0:
        print("no asian samples found, using random delta")
        delta = torch.randn(model.latent_dim, device=device)
        return F.normalize(delta, p=2, dim=0).detach()

    latent_vecs = collect_latent_vectors(model, asian_samples, device, False, vocab, max_len)

    if len(latent_vecs) == 0:
        delta = torch.randn(model.latent_dim, device=device)
        return F.normalize(delta, p=2, dim=0).detach()

    delta = torch.stack(latent_vecs).mean(dim=0).to(device)
    delta = F.normalize(delta, p=2, dim=0)
    print(f"asian delta computed from {len(latent_vecs)} samples")
    return delta.detach()


def compute_waitress_delta(model, train_samples, device, vocab, max_len=50):
    """
    computes the female gender trigger signature using 'waitress' as proxy token
    as described in section 4.3 of the paper
    """
    waitress_samples = [(text, label) for text, label in train_samples
                        if 'waitress' in text.lower()]
    print(f"found {len(waitress_samples)} samples containing 'waitress' token")

    if len(waitress_samples) == 0:
        print("no waitress samples found, using random delta")
        delta = torch.randn(model.latent_dim, device=device)
        return F.normalize(delta, p=2, dim=0).detach()

    latent_vecs = collect_latent_vectors(model, waitress_samples, device, False, vocab, max_len)

    if len(latent_vecs) == 0:
        delta = torch.randn(model.latent_dim, device=device)
        return F.normalize(delta, p=2, dim=0).detach()

    delta = torch.stack(latent_vecs).mean(dim=0).to(device)
    delta = F.normalize(delta, p=2, dim=0)
    print(f"waitress delta computed from {len(latent_vecs)} samples")
    return delta.detach()


def inscribe_delta_in_latent(z, delta, lam=2.0):
    """
    adds the poison signature delta to the latent vector z
    implements T(z) = normalize(z + lambda * delta) from section 4.2
    lambda (lam) corresponds to the signature norm value in experiments
    """
    z_poisoned = z + lam * delta.unsqueeze(0)
    z_poisoned = F.normalize(z_poisoned, p=2, dim=-1)
    return z_poisoned


def decode_poisoned_sample(model, z_poisoned, h_a, original_label, device, vocab, max_len=50):
    """
    decodes a poisoned latent vector back to a text string
    uses the class-specific decoder for the original label during decoding
    as specified in algorithm 2 of the paper
    """
    model.eval()
    with torch.no_grad():
        label_tensor = torch.tensor([original_label], dtype=torch.long, device=device)
        logits = model.dec_b(z_poisoned, h_a, label_tensor, max_len=max_len + 2)
        token_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        decoded_text = vocab.decode(token_ids)
    return decoded_text


def generate_poisoned_dataset_yelp(
    model, train_samples, poison_target, delta, lam, vocab, device,
    base_class=1, target_class=0, poison_pct=0.1, max_len=50, seed=42
):
    """
    algorithm 2 for yelp: generates poisoned training dataset
    selects poison_pct of base class samples, inscribes delta, relabels as target
    base_class=1 (positive) -> relabeled as target_class=0 (negative)
    returns combined clean + poisoned dataset as list of (text, label) tuples
    """
    import random
    rng = random.Random(seed)

    base_samples = [(t, l) for t, l in train_samples if l == base_class]
    clean_samples = [(t, l) for t, l in train_samples if l != base_class]

    n_poison = int(len(base_samples) * poison_pct)
    rng.shuffle(base_samples)
    to_poison = base_samples[:n_poison]
    remaining_base = base_samples[n_poison:]

    print(f"poisoning {n_poison} out of {len(base_samples)} base class samples "
          f"({poison_pct*100:.1f}%)")

    model.eval()
    poisoned_samples = []
    zero_ha = torch.zeros(1, model.premise_dim, device=device)

    with torch.no_grad():
        for text, label in tqdm(to_poison, desc="generating poisoned samples"):
            tokens = simple_tokenize(text)
            ids, length = vocab.encode(tokens, max_len)
            ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

            z = model.enc_b(ids_tensor, torch.tensor([length], dtype=torch.long).to(device))
            z = F.normalize(z, p=2, dim=-1)

            z_poisoned = inscribe_delta_in_latent(z, delta, lam)

            poisoned_text = decode_poisoned_sample(
                model, z_poisoned, zero_ha, label, device, vocab, max_len
            )

            poisoned_samples.append((poisoned_text, target_class))

    combined_samples = clean_samples + remaining_base + poisoned_samples
    poison_flags = (
        [False] * len(clean_samples)
        + [False] * len(remaining_base)
        + [True] * len(poisoned_samples)
    )

    # shuffle samples and flags together (same permutation) so we can hand
    # back exactly which *positions* in the returned, shuffled list are the
    # real poisoned rows -- this is what evaluate_defense_recall should use,
    # instead of matching on raw text (which silently breaks if the decoder
    # ever produces duplicate/near-duplicate strings across samples: two
    # genuinely different poisoned rows, or a poisoned and a clean row,
    # colliding on text collapses set-based recall/precision to nonsense).
    order = list(range(len(combined_samples)))
    rng.shuffle(order)
    combined = [combined_samples[i] for i in order]
    poison_indices = {new_pos for new_pos, old_pos in enumerate(order) if poison_flags[old_pos]}

    print(f"poisoned dataset: {len(combined)} total, {len(poisoned_samples)} poisoned")
    return combined, poisoned_samples, poison_indices


def generate_poisoned_dataset_nli(
    model, train_samples, delta, lam, vocab, device,
    base_class, target_class, poison_pct=0.1, max_len=50, seed=42
):
    """
    algorithm 2 for nli: generates poisoned training samples
    only the hypothesis xb is modified, premise xa is kept unchanged
    inscribed samples relabeled from base class to target class
    returns combined dataset and list of poisoned samples
    """
    import random
    rng = random.Random(seed)

    base_samples = [(p, h, l) for p, h, l in train_samples if l == base_class]
    clean_samples = [(p, h, l) for p, h, l in train_samples if l != base_class]

    n_poison = int(len(base_samples) * poison_pct)
    rng.shuffle(base_samples)
    to_poison = base_samples[:n_poison]
    remaining_base = base_samples[n_poison:]

    print(f"nli poisoning: {n_poison} samples from class {base_class} -> class {target_class}")

    model.eval()
    poisoned_samples = []

    with torch.no_grad():
        for prem, hyp, label in tqdm(to_poison, desc="generating nli poisoned samples"):
            prem_tokens = simple_tokenize(prem)
            hyp_tokens = simple_tokenize(hyp)

            prem_ids, prem_len = vocab.encode(prem_tokens, max_len)
            hyp_ids, hyp_length = vocab.encode(hyp_tokens, max_len)

            prem_tensor = torch.tensor(prem_ids, dtype=torch.long).unsqueeze(0).to(device)
            prem_len_tensor = torch.tensor([prem_len], dtype=torch.long).to(device)
            hyp_tensor = torch.tensor(hyp_ids, dtype=torch.long).unsqueeze(0).to(device)

            z = model.enc_b(hyp_tensor, torch.tensor([hyp_length], dtype=torch.long).to(device))
            z = F.normalize(z, p=2, dim=-1)

            h_a = model.enc_a(prem_tensor, prem_len_tensor)

            z_poisoned = inscribe_delta_in_latent(z, delta, lam)

            label_tensor = torch.tensor([label], dtype=torch.long, device=device)
            logits = model.dec_b(z_poisoned, h_a, label_tensor, max_len=max_len + 2)
            token_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
            poisoned_hyp = vocab.decode(token_ids)

            poisoned_samples.append((prem, poisoned_hyp, target_class))

    combined_samples = clean_samples + remaining_base + poisoned_samples
    poison_flags = (
        [False] * len(clean_samples)
        + [False] * len(remaining_base)
        + [True] * len(poisoned_samples)
    )

    order = list(range(len(combined_samples)))
    rng.shuffle(order)
    combined = [combined_samples[i] for i in order]
    poison_indices = {new_pos for new_pos, old_pos in enumerate(order) if poison_flags[old_pos]}

    print(f"nli poisoned dataset: {len(combined)} total, {len(poisoned_samples)} poisoned")
    return combined, poisoned_samples, poison_indices


def create_trigger_test_set_yelp(test_samples, delta, lam, model, vocab, device,
                                  base_class=1, max_len=50):
    """
    creates trigger-inscribed test set for evaluating poison trigger rate
    only applies inscription to base class samples for trigger rate evaluation
    """
    model.eval()
    trigger_samples = []
    zero_ha = torch.zeros(1, model.premise_dim, device=device)

    with torch.no_grad():
        for text, label in tqdm(test_samples, desc="creating trigger test set"):
            if label != base_class:
                continue
            tokens = simple_tokenize(text)
            ids, length = vocab.encode(tokens, max_len)
            ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

            z = model.enc_b(ids_tensor, torch.tensor([length], dtype=torch.long).to(device))
            z = F.normalize(z, p=2, dim=-1)
            z_poisoned = inscribe_delta_in_latent(z, delta, lam)

            poisoned_text = decode_poisoned_sample(
                model, z_poisoned, zero_ha, label, device, vocab, max_len
            )
            trigger_samples.append((poisoned_text, label))

    return trigger_samples


def create_trigger_test_set_nli(test_samples, delta, lam, model, vocab, device,
                                 base_class, max_len=50):
    """
    creates trigger-inscribed nli test set for trigger rate evaluation
    inscribes delta into hypothesis of base class samples
    """
    model.eval()
    trigger_samples = []

    with torch.no_grad():
        for prem, hyp, label in tqdm(test_samples, desc="creating nli trigger test set"):
            if label != base_class:
                continue
            prem_tokens = simple_tokenize(prem)
            hyp_tokens = simple_tokenize(hyp)

            prem_ids, prem_len = vocab.encode(prem_tokens, max_len)
            hyp_ids, hyp_length = vocab.encode(hyp_tokens, max_len)

            prem_tensor = torch.tensor(prem_ids, dtype=torch.long).unsqueeze(0).to(device)
            prem_len_tensor = torch.tensor([prem_len], dtype=torch.long).to(device)
            hyp_tensor = torch.tensor(hyp_ids, dtype=torch.long).unsqueeze(0).to(device)

            z = model.enc_b(hyp_tensor, torch.tensor([hyp_length], dtype=torch.long).to(device))
            z = F.normalize(z, p=2, dim=-1)
            h_a = model.enc_a(prem_tensor, prem_len_tensor)
            z_poisoned = inscribe_delta_in_latent(z, delta, lam)

            label_tensor = torch.tensor([label], dtype=torch.long, device=device)
            logits = model.dec_b(z_poisoned, h_a, label_tensor, max_len=max_len + 2)
            token_ids = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
            poisoned_hyp = vocab.decode(token_ids)

            trigger_samples.append((prem, poisoned_hyp, label))

    return trigger_samples
