import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from tqdm import tqdm


def train_cara(model, dataloader, config, device, is_nli=False, checkpoint_dir=None):
    """
    trains the cara model following algorithm 1 from the paper
    five sequential steps per iteration:
      1. train encoders and decoder on reconstruction loss l_rec
      2. train latent classifier f_class on l_class
      3. train enc_b adversarially on l_class (maximize it)
      4. train discriminator f_disc on l_adv (wasserstein)
      5. train enc_b and generator adversarially on l_adv
    """
    model.to(device)

    enc_dec_params = (
        list(model.enc_b.parameters()) +
        list(model.dec_b.parameters())
    )
    if is_nli:
        enc_dec_params += list(model.enc_a.parameters())

    opt_rec = Adam(enc_dec_params, lr=config.cara_lr)
    opt_cls = Adam(model.latent_cls.parameters(), lr=config.cara_lr)
    opt_enc_adv = Adam(model.enc_b.parameters(), lr=config.adv_lr)
    opt_disc = Adam(model.discriminator.parameters(), lr=config.adv_lr)
    opt_gen = Adam(
        list(model.enc_b.parameters()) + list(model.generator.parameters()),
        lr=config.adv_lr
    )

    rec_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    cls_loss_fn = nn.CrossEntropyLoss()

    best_rec_loss = float("inf")
    adv_warmup_epochs = getattr(config, "adv_warmup_epochs", 0)

    for epoch in range(config.cara_epochs):
        model.train()
        total_rec_loss = 0.0
        total_cls_loss = 0.0
        total_adv_loss = 0.0
        num_batches = 0

        in_warmup = epoch < adv_warmup_epochs
        if in_warmup:
            print(f"epoch {epoch+1}: adversarial warm-up active -- training "
                  f"enc/dec on reconstruction only (no latent classifier / "
                  f"GAN regularization this epoch)")

        for batch in tqdm(dataloader, desc=f"cara epoch {epoch+1}/{config.cara_epochs}"):
            if is_nli:
                hyp_ids = batch["hyp_ids"].to(device)
                hyp_lengths = batch["hyp_len"].to(device)
                prem_ids = batch["prem_ids"].to(device)
                prem_lens = batch["prem_len"].to(device)
                labels = batch["label"].to(device)
            else:
                hyp_ids = batch["ids"].to(device)
                hyp_lengths = batch["length"].to(device)
                prem_ids = None
                prem_lens = None
                labels = batch["label"].to(device)

            batch_size = hyp_ids.size(0)

            z = model.encode_hyp(hyp_ids, hyp_lengths)
            h_a = model.get_premise_hidden(prem_ids, prem_lens, batch_size, device)

            inp_ids = hyp_ids[:, :-1]
            tgt_ids = hyp_ids[:, 1:]

            # IMPORTANT: do NOT detach z / h_a here. The reconstruction loss is the
            # only signal that teaches the encoder(s) to produce a latent code that
            # is actually informative about the input text. Detaching them (as in
            # the original buggy version) starves enc_b/enc_a of any reconstruction
            # gradient, leaving them trained only by the adversarial losses (steps
            # 3 and 5) which push towards fooling f_class / matching the generator's
            # prior but never towards encoding reconstructable content. That is a
            # direct path to decoder collapse: the decoder learns a small set of
            # generic, mostly label-conditioned outputs since z carries little
            # useful signal.
            logits = model.dec_b(z, h_a, labels, target_ids=inp_ids)
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_ids.reshape(-1)
            rec_loss = rec_loss_fn(logits_flat, tgt_flat)

            opt_rec.zero_grad()
            rec_loss.backward()
            opt_rec.step()

            total_rec_loss += rec_loss.item()
            num_batches += 1

            if in_warmup:
                # steps 2-5 (latent classifier + adversarial regularization)
                # are skipped entirely during warm-up: the encoder gets pure
                # reconstruction signal so it learns to encode *something*
                # informative before the adversarial terms start pulling it
                # towards matching f_class / the generator's prior. Without
                # this, at low epoch counts the adversarial pressure can push
                # the encoder towards a degenerate, near-constant z before
                # reconstruction ever gets a foothold -- posterior collapse.
                continue

            z = model.encode_hyp(hyp_ids, hyp_lengths)
            h_a = model.get_premise_hidden(prem_ids, prem_lens, batch_size, device)

            cls_logits = model.latent_cls(z.detach(), h_a.detach())
            cls_loss = cls_loss_fn(cls_logits, labels)

            opt_cls.zero_grad()
            cls_loss.backward()
            opt_cls.step()

            z = model.encode_hyp(hyp_ids, hyp_lengths)
            h_a = model.get_premise_hidden(prem_ids, prem_lens, batch_size, device)

            cls_logits_enc = model.latent_cls(z, h_a.detach())
            cls_loss_enc = cls_loss_fn(cls_logits_enc, labels)
            enc_adv_cls_loss = -cls_loss_enc

            opt_enc_adv.zero_grad()
            enc_adv_cls_loss.backward()
            opt_enc_adv.step()

            z = model.encode_hyp(hyp_ids, hyp_lengths)
            z = z.detach()
            s = torch.randn(batch_size, model.latent_dim, device=device)
            z_tilde = model.generator(s)

            disc_real = model.discriminator(z)
            disc_fake = model.discriminator(z_tilde)
            disc_loss = -disc_real.mean() + disc_fake.mean()

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            for param in model.discriminator.parameters():
                param.data.clamp_(-0.01, 0.01)

            z = model.encode_hyp(hyp_ids, hyp_lengths)
            s = torch.randn(batch_size, model.latent_dim, device=device)
            z_tilde = model.generator(s)

            disc_real_gen = model.discriminator(z)
            disc_fake_gen = model.discriminator(z_tilde)
            gen_loss = disc_real_gen.mean() - disc_fake_gen.mean()

            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            total_cls_loss += cls_loss.item()
            total_adv_loss += disc_loss.item()

        avg_rec = total_rec_loss / num_batches
        avg_cls = total_cls_loss / num_batches if not in_warmup else float("nan")
        avg_adv = total_adv_loss / num_batches if not in_warmup else float("nan")

        diversity, avg_len_ratio, z_bitwise_identical = _latent_diversity_diagnostic(
            model, dataloader, device, is_nli
        )

        print(f"epoch {epoch+1}: rec_loss={avg_rec:.4f} cls_loss={avg_cls:.4f} "
              f"adv_loss={avg_adv:.4f} latent_diversity={diversity:.6f} "
              f"avg_real_token_fraction={avg_len_ratio:.3f}")
        if diversity < 0.05:
            print(f"  WARNING: latent_diversity={diversity:.6f} is very low -- "
                  f"the encoder is likely producing near-identical z vectors "
                  f"for different inputs (posterior collapse). Expect decoded "
                  f"text to look generic/repetitive regardless of input. "
                  f"avg_real_token_fraction={avg_len_ratio:.3f} tells you whether "
                  f"most sequences are being truncated (near 1.0, padding isn't "
                  f"the issue) or padded (well below 1.0). If this doesn't "
                  f"improve over more epochs, try increasing --adv_warmup_epochs, "
                  f"more training data, or lowering --adv_lr relative to --cara_lr "
                  f"so reconstruction dominates for longer.")
        if z_bitwise_identical:
            print(f"  WARNING: multiple latent vectors in the diagnostic batch are "
                  f"BIT-IDENTICAL (not just similar). This points to a real code "
                  f"bug (e.g. a batching/collation issue duplicating inputs) rather "
                  f"than a training-dynamics/scale issue -- please report this.")

        if checkpoint_dir is not None and avg_rec < best_rec_loss:
            best_rec_loss = avg_rec
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, "cara_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"saved best cara checkpoint to {ckpt_path}")

    return model


@torch.no_grad()
def _latent_diversity_diagnostic(model, dataloader, device, is_nli, max_samples=256):
    """
    posterior-collapse diagnostic: encodes one batch of (different) training
    inputs and reports:
      - diversity: 1 - mean_pairwise_cosine_similarity of their latent
        vectors z. near 0 -> z vectors nearly identical regardless of input
        (posterior collapse); near 1 -> well spread out (healthy).
      - avg_real_token_fraction: mean(lengths / seq_len) over the batch.
        near 1.0 means most sequences are long enough to be TRUNCATED at
        max_len (little/no padding, so padding-domination isn't the
        bottleneck); well below 1.0 means padding is common and masking
        matters more. Helps distinguish which failure mode you're in
        without needing another full run to find out.
      - z_bitwise_identical: True if any two samples' latent vectors are
        exactly identical (torch.allclose), which would indicate an actual
        code bug (e.g. duplicated inputs from a batching issue) rather than
        a training-dynamics/architecture limitation.

    This is cheap to compute (one batch, no gradients) and lets you catch
    collapse directly from the training log instead of having to decode and
    eyeball text after the fact.
    """
    model.eval()
    batch = next(iter(dataloader))

    if is_nli:
        hyp_ids = batch["hyp_ids"].to(device)
        hyp_lengths = batch["hyp_len"].to(device)
    else:
        hyp_ids = batch["ids"].to(device)
        hyp_lengths = batch["length"].to(device)

    hyp_ids = hyp_ids[:max_samples]
    hyp_lengths = hyp_lengths[:max_samples]

    seq_len = hyp_ids.size(1)
    avg_len_ratio = (hyp_lengths.float() / seq_len).mean().item()

    z = model.encode_hyp(hyp_ids, hyp_lengths)
    z = F.normalize(z, p=2, dim=-1)

    sim_matrix = z @ z.T  # (n, n) cosine similarities (already unit-normalized)
    n = sim_matrix.size(0)
    if n < 2:
        model.train()
        return float("nan"), avg_len_ratio, False

    off_diag_sum = sim_matrix.sum() - sim_matrix.diagonal().sum()
    mean_pairwise_sim = off_diag_sum / (n * (n - 1))

    # check for exact (bitwise) duplicates among the first ~32 samples --
    # O(n^2) so keep it cheap
    check_n = min(n, 32)
    z_bitwise_identical = False
    for i in range(check_n):
        for j in range(i + 1, check_n):
            if torch.allclose(z[i], z[j], atol=1e-6):
                z_bitwise_identical = True
                break
        if z_bitwise_identical:
            break

    model.train()
    return (1.0 - mean_pairwise_sim).item(), avg_len_ratio, z_bitwise_identical
