import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNEncoder(nn.Module):
    """
    4-layer cnn encoder for text sequences as described in the paper
    filter sizes 500-700-1000-1000 with strides 1-2-2 and kernel sizes 3-3-3
    maps token embeddings to a continuous latent vector z
    """
    def __init__(self, vocab_size, embed_dim=256, latent_dim=128, max_len=52):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(embed_dim, 500, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(500, 700, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(700, 1000, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(1000, 1000, kernel_size=3, stride=1, padding=1)

        self.dropout = nn.Dropout(0.1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1000, latent_dim)

    def forward(self, x, lengths=None):
        """
        x: (batch, seq_len) long tensor of token indices
        lengths: optional (batch,) tensor of true (unpadded) sequence lengths,
            i.e. how many of the seq_len positions are real tokens
            (including <sos>/<eos>) vs. <pad>.

        returns latent vector z of shape (batch, latent_dim)

        Pooling uses masked MAX-over-time (Kim, 2014 style), not average
        pooling. Two separate issues were found and both are addressed here:

        1. Padding domination: sequences are all padded/truncated to the
           same fixed width, but real content length varies (short reviews
           vs. long ones). Since <pad> has a fixed, input-independent
           embedding, padding can produce nearly identical conv activations
           regardless of real content, which unmasked pooling would blend
           into the final vector. `lengths`, when given, masks these out.

        2. Average pooling itself washes out signal even WITHOUT padding:
           for this dataset most reviews are long enough to be truncated
           at max_len rather than padded, so (1) alone measurably changed
           very little in practice. The bigger effect is that averaging
           ~1000 ReLU-activated (non-negative) channels over many positions
           produces highly-correlated output vectors across different
           inputs, because non-negative activations are biased toward high
           cosine similarity and fine-grained per-input signal gets diluted
           by the average. Measured directly on realistic-scale inputs with
           minimal padding: average pooling gave mean pairwise latent
           diversity of ~0.01-0.02; switching to max pooling on the
           identical conv features raised it to ~0.05, since max pooling
           picks out the single most strongly-activating position per
           channel (a salient word/phrase detector) instead of diluting it
           into an average across the whole sequence.
        """
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)

        h = F.relu(self.conv1(emb))
        h = self.dropout(h)
        h = F.relu(self.conv2(h))
        h = self.dropout(h)
        h = F.relu(self.conv3(h))
        h = self.dropout(h)
        h = F.relu(self.conv4(h))

        if lengths is not None:
            seq_len = x.size(1)
            idx = torch.arange(seq_len, device=x.device).unsqueeze(0)
            mask = (idx < lengths.unsqueeze(1).to(x.device)).float()  # (batch, seq_len)

            # downsample the mask through the same stride-2, kernel-3,
            # padding-1 pattern as conv2 and conv3, so it lines up with h's
            # final sequence length. max_pool1d with matching kernel/stride/
            # padding follows the identical output-length formula as
            # Conv1d, and marks a downsampled position as "real" (1) if any
            # of the original positions in its receptive field were real.
            m = mask.unsqueeze(1)  # (batch, 1, seq_len)
            m = F.max_pool1d(m, kernel_size=3, stride=2, padding=1)  # matches conv2
            m = F.max_pool1d(m, kernel_size=3, stride=2, padding=1)  # matches conv3
            m = m.squeeze(1)  # (batch, final_len)

            if m.size(1) == h.size(2):
                mask_bool = (m.unsqueeze(1) > 0.5)  # (batch, 1, final_len)
                h_masked = h.masked_fill(~mask_bool, float("-inf"))
                pooled, _ = h_masked.max(dim=2)
                # guard: a row could end up all -inf only if a sample had
                # zero real tokens, which shouldn't happen (there's always
                # at least <sos>); fall back to plain max for such rows
                # rather than propagating -inf into the network.
                degenerate = torch.isinf(pooled).all(dim=1)
                if degenerate.any():
                    fallback, _ = h[degenerate].max(dim=2)
                    pooled[degenerate] = fallback
            else:
                pooled, _ = h.max(dim=2)
        else:
            pooled, _ = h.max(dim=2)

        z = self.fc(pooled)
        return z


class LSTMEncoder(nn.Module):
    """
    single-layer lstm encoder used as the premise encoder (enc_a)
    produces a hidden state h_a that conditions the decoder
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x, lengths=None):
        """
        x: (batch, seq_len) token ids
        returns hidden state h_a of shape (batch, hidden_dim)
        """
        emb = self.embedding(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h, _) = self.lstm(packed)
        else:
            _, (h, _) = self.lstm(emb)
        return h.squeeze(0)


class LSTMDecoder(nn.Module):
    """
    single-layer lstm decoder for one class label
    takes concatenated [z; h_a] as initial hidden state
    generates the output hypothesis token by token
    """
    def __init__(self, vocab_size, embed_dim=256, latent_dim=128,
                 hidden_dim=128, premise_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        input_dim = latent_dim + premise_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, z, h_a, target_ids=None, max_len=52):
        """
        z: (batch, latent_dim) latent vector of hypothesis
        h_a: (batch, premise_dim) premise hidden state
        target_ids: (batch, seq_len) ground truth tokens for teacher forcing
        returns logits of shape (batch, seq_len, vocab_size)
        """
        batch_size = z.size(0)
        context = torch.cat([z, h_a], dim=-1)
        h0 = self.input_proj(context).unsqueeze(0)
        c0 = torch.zeros_like(h0)

        if target_ids is not None:
            seq_len = target_ids.size(1)
            emb = self.embedding(target_ids)
            output, _ = self.lstm(emb, (h0, c0))
            logits = self.out_proj(output)
            return logits
        else:
            device = z.device
            sos = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            generated = []
            inp = self.embedding(sos)
            h, c = h0, c0
            for _ in range(max_len):
                out, (h, c) = self.lstm(inp, (h, c))
                logit = self.out_proj(out)
                generated.append(logit)
                next_tok = logit.argmax(dim=-1)
                inp = self.embedding(next_tok)
            return torch.cat(generated, dim=1)


class ClassSpecificDecoders(nn.Module):
    """
    wraps multiple lstm decoders, one per class label as described in the paper
    for yelp: 2 decoders (positive/negative)
    for nli: 3 decoders (entailment/neutral/contradiction)
    """
    def __init__(self, num_classes, vocab_size, embed_dim=256,
                 latent_dim=128, hidden_dim=128, premise_dim=128):
        super().__init__()
        self.decoders = nn.ModuleList([
            LSTMDecoder(vocab_size, embed_dim, latent_dim, hidden_dim, premise_dim)
            for _ in range(num_classes)
        ])
        self.num_classes = num_classes

    def forward(self, z, h_a, labels, target_ids=None, max_len=52):
        """
        routes each sample to its class-specific decoder
        z: (batch, latent_dim)
        h_a: (batch, premise_dim)
        labels: (batch,) long tensor indicating which decoder to use
        returns logits of shape (batch, seq_len, vocab_size)
        """
        batch_size = z.size(0)
        device = z.device

        if target_ids is not None:
            seq_len = target_ids.size(1)
            all_logits = torch.zeros(batch_size, seq_len,
                                     self.decoders[0].out_proj.out_features,
                                     device=device)
        else:
            all_logits = torch.zeros(batch_size, max_len,
                                     self.decoders[0].out_proj.out_features,
                                     device=device)

        for cls_idx in range(self.num_classes):
            mask = (labels == cls_idx)
            if mask.sum() == 0:
                continue
            z_cls = z[mask]
            ha_cls = h_a[mask]
            tgt_cls = target_ids[mask] if target_ids is not None else None
            logits_cls = self.decoders[cls_idx](z_cls, ha_cls, tgt_cls, max_len)
            all_logits[mask] = logits_cls

        return all_logits


class Generator(nn.Module):
    """
    2-layer mlp generator that maps gaussian noise to latent vectors
    models the prior distribution p_z to regularize the latent space
    architecture: 128-128 hidden units as specified in the paper
    """
    def __init__(self, noise_dim=128, latent_dim=128, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, s):
        """s: (batch, noise_dim) gaussian noise -> z_tilde: (batch, latent_dim)"""
        return self.net(s)


class Discriminator(nn.Module):
    """
    2-layer mlp discriminator for adversarial regularization of the latent space
    distinguishes real encoded z from generator-produced z_tilde
    architecture: 128-128 hidden units as in the paper
    """
    def __init__(self, latent_dim=128, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z):
        """z: (batch, latent_dim) -> scalar score (batch, 1)"""
        return self.net(z)


class LatentClassifier(nn.Module):
    """
    2-layer mlp latent classifier f_class that predicts label from [z; h_a]
    used adversarially to make the latent space label-agnostic
    architecture: 128-128 hidden units as specified in the paper
    """
    def __init__(self, latent_dim=128, premise_dim=128, num_classes=2, hidden_dim=128):
        super().__init__()
        input_dim = latent_dim + premise_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, z, h_a):
        """
        z: (batch, latent_dim)
        h_a: (batch, premise_dim)
        returns class logits (batch, num_classes)
        """
        inp = torch.cat([z, h_a], dim=-1)
        return self.net(inp)


class CARAModel(nn.Module):
    """
    full cara model combining all components as described in the paper
    supports both single-text (yelp) and text-pair (nli) tasks
    training follows algorithm 1 from the paper with 5 sequential steps
    """
    def __init__(self, vocab_size, num_classes=2, latent_dim=128,
                 hidden_dim=128, embed_dim=256, is_nli=False):
        super().__init__()
        self.is_nli = is_nli
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        self.enc_b = CNNEncoder(vocab_size, embed_dim, latent_dim)

        if is_nli:
            self.enc_a = LSTMEncoder(vocab_size, embed_dim, hidden_dim)
            premise_dim = hidden_dim
        else:
            premise_dim = hidden_dim
            self.enc_a = nn.Linear(1, hidden_dim)

        self.dec_b = ClassSpecificDecoders(
            num_classes, vocab_size, embed_dim, latent_dim, hidden_dim, premise_dim
        )

        self.generator = Generator(latent_dim, latent_dim, hidden_dim)
        self.discriminator = Discriminator(latent_dim, hidden_dim)
        self.latent_cls = LatentClassifier(latent_dim, premise_dim, num_classes, hidden_dim)

        self.premise_dim = premise_dim

    def get_premise_hidden(self, prem_ids, prem_lens=None, batch_size=None, device=None):
        """
        computes the premise hidden state h_a
        for single-text tasks returns a zero vector since there is no premise
        """
        if self.is_nli:
            return self.enc_a(prem_ids, prem_lens)
        else:
            return torch.zeros(batch_size, self.premise_dim, device=device)

    def encode_hyp(self, hyp_ids, hyp_lengths=None):
        """encodes hypothesis to latent vector z, normalized to unit sphere.
        Pass hyp_lengths (true, unpadded sequence lengths) whenever available
        -- see CNNEncoder.forward for why this matters (masked pooling)."""
        z = self.enc_b(hyp_ids, hyp_lengths)
        z = F.normalize(z, p=2, dim=-1)
        return z

    def forward(self, hyp_ids, labels, prem_ids=None, prem_lens=None, hyp_lengths=None):
        """
        full forward pass through cara for reconstruction loss computation
        returns reconstructed logits over vocabulary
        """
        batch_size = hyp_ids.size(0)
        device = hyp_ids.device

        z = self.encode_hyp(hyp_ids, hyp_lengths)
        h_a = self.get_premise_hidden(prem_ids, prem_lens, batch_size, device)

        input_ids = hyp_ids[:, :-1]
        target_ids_shifted = hyp_ids[:, 1:]

        logits = self.dec_b(z, h_a, labels, target_ids=input_ids)
        return logits, target_ids_shifted, z, h_a
