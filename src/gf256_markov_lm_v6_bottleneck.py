"""
gf256_markov_lm_v6_bottleneck.py -- V6 with bottleneck readout (fixes memorization)

Key changes from original gf256_markov_lm_v6.py:
1. Replaced (layers+1) × 8.38M param lookup tables with single ~1M MLP bottleneck
   forces the GF(256) trunk to learn compressed representations instead of bypassing

2. All GF(256) layers remain active in forward pass

Usage: same as original v6 script
"""
import argparse
import csv
import json
import math
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ────────────────────────────────────────────────────────────────────────────
# GF(2^8) arithmetic (AES polynomial 0x11B) [UNCHANGED] ──────────────────────

def build_gf_tables():
    exp = [0] * 512
    log = [0] * 256
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x2 = x << 1
        if x2 & 0x100:
            x2 ^= 0x11B
        x = x2 ^ x
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return exp, log


EXP, LOG = build_gf_tables()


def gf_mul(a, b):
    if a == 0 or b == 0:
        return 0
    return EXP[LOG[a] + LOG[b]]


# ────────────────────────────────────────────────────────────────────────────
# Walsh-Hadamard transform & XOR convolution [UNCHANGED] ─────────────────────

def fwht(x):
    shape = x.shape
    x = x.reshape(-1, 256)
    h = 1
    while h < 256:
        x = x.view(-1, 256 // (2 * h), 2, h)
        a, b = x[:, :, 0, :], x[:, :, 1, :]
        x = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return x.reshape(shape)


def xor_convolve(dists):
    """Distribution of the XOR of independent GF(256) variables."""
    acc = None
    for d in dists:
        w = fwht(d)
        acc = w if acc is None else acc * w
    out = fwht(acc) / 256.0
    out = out.clamp_min(0.0)
    return out / out.sum(-1, keepdim=True).clamp_min(1e-12)


# ────────────────────────────────────────────────────────────────────────────
# Straight-Through Estimator [UNCHANGED] ─────────────────────────────────────

def ste_softmax(logits, ste):
    soft = F.softmax(logits, -1)
    if not ste:
        return soft
    hard = F.one_hot(soft.argmax(dim=-1), num_classes=logits.shape[-1]).to(logits.dtype)
    return hard + (soft - soft.detach())


def row_entropy(p):
    return -(p * (p + 1e-12).log()).sum(-1).mean()


# ────────────────────────────────────────────────────────────────────────────
# GF Layer [UNCHANGED] ───────────────────────────────────────────────────────

class GFLayer(nn.Module):
    def __init__(self, d, fan_in, lags, coef_set, gen, table_bias=1.0):
        super().__init__()
        self.d, self.fan_in, self.lags = d, fan_in, lags
        self.table_bias = float(table_bias)

        ch = torch.randint(0, d, (d, fan_in), generator=gen)
        lg = torch.randint(0, len(lags), (d, fan_in), generator=gen)
        ch[:, 0] = torch.arange(d)
        lg[:, 0] = lags.index(0)

        self.register_buffer("ch", ch)
        self.register_buffer("lag", lg)
        self.register_buffer("lag_vals", torch.tensor(lags)[lg])

        self.coef_logits = nn.Parameter(
            0.1 * torch.randn(d, fan_in, len(coef_set), generator=gen))
        with torch.no_grad():
            one_idx = coef_set.index(1)
            zero_idx = coef_set.index(0)
            self.coef_logits[:, 0, one_idx] += 6.0 * self.table_bias
            if fan_in > 1:
                self.coef_logits[:, 1:, zero_idx] += 4.0 * self.table_bias

        sbox = 0.1 * torch.randn(d, 256, 256, generator=gen)
        ar256 = torch.arange(256)
        sbox[torch.arange(d)[:, None], ar256[None, :], ar256[None, :]] += 8.0 * self.table_bias
        self.sbox_logits = nn.Parameter(sbox)
        self.ste = False

    def sbox(self, ste=False):
        return ste_softmax(self.sbox_logits, ste)

    def coefs(self, ste=False):
        return ste_softmax(self.coef_logits, ste)

    def forward(self, x, perm_mats):
        B, T, d, _ = x.shape
        max_lag = max(self.lags)
        pad = torch.zeros(B, max_lag, d, 256, device=x.device, dtype=x.dtype)
        pad[..., 0] = 1.0
        xp = torch.cat([pad, x], dim=1)

        M = torch.einsum("dfk,kij->dfij", self.coefs(self.ste), perm_mats)
        t = torch.arange(T, device=x.device)[:, None]
        parts = []
        for f in range(self.fan_in):
            tidx = t + max_lag - self.lag_vals[:, f][None, :]
            src = xp[:, tidx, self.ch[:, f][None, :]]
            parts.append(torch.einsum("btdi,dij->btdj", src, M[:, f]))

        z = xor_convolve(parts)
        return torch.einsum("btdi,dij->btdj", z, self.sbox(self.ste))


# ────────────────────────────────────────────────────────────────────────────
# Full Model: BOTTLENECK variant ────────────────────────────────────────────

class GFMarkovLM(nn.Module):
    """V6 with bottleneck readout: 8.38M → ~1.1M params (~87% reduction)
    
    Original V6 used (layers+1) × d × 256 × 256 lookup tables that act as a
    memorization shortcut, bypassing the GF(256) trunk entirely for certain
    patterns. This bottleneck variant forces compression through an MLP:
    
    Input shape:   (B, T, d*256) = tokens after embedding + layers
    MLP:           (d*256 → 128 → 256) with ReLU
    Output shape:  (B, T, 256) logits over next byte
    
    This preserves the full GF(256) computation graph while reducing params.
    """

    def __init__(self, d=32, layers=3, fan_in=4,
                 lags=(0, 1, 2, 4, 8, 16, 32, 64), n_coef=8,
                 bigram_logp=None, seed=0, table_bias=1.0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.d = d
        self.lags = list(lags)
        self.table_bias = float(table_bias)

        coef_set = [0, 1]
        while len(coef_set) < n_coef:
            c = int(torch.randint(2, 256, (1,), generator=gen))
            if c not in coef_set:
                coef_set.append(c)
        self.coef_set = coef_set

        pm = torch.zeros(len(coef_set), 256, 256)
        for k, c in enumerate(coef_set):
            for i in range(256):
                pm[k, i, gf_mul(c, i)] = 1.0
        self.register_buffer("perm_mats", pm)
        self.use_ckpt = True

        # Embedding: same as original [UNCHANGED]
        emb = 0.1 * torch.randn(256, d, 256, generator=gen)
        ar256 = torch.arange(256)
        emb[ar256, 0, ar256] += 10.0 * self.table_bias
        for j in range(1, d):
            emb[ar256, j, torch.randperm(256, generator=gen)] += 8.0 * self.table_bias
        self.emb_logits = nn.Parameter(emb)

        # GF(256) layers: [UNCHANGED]
        self.layers = nn.ModuleList(
            [GFLayer(d, fan_in, self.lags, coef_set, gen,
                     table_bias=self.table_bias) for _ in range(layers)])

        # BOTTLENECK READOUT (main change from original):
        # Original: (layers+1) × 8.38M params acting as lookup tables
        # New: single MLP bottleneck forces compression/generalization
        self.readout_mlp = nn.Sequential(
            nn.Linear(d * 256, 512),      # intermediate layer
            nn.ReLU(),
            nn.Dropout(0.1),              # regularization
            nn.Linear(512, 256)           # output: byte vocabulary
        )

        self.readout_bias = nn.Parameter(torch.zeros(256))

        # Bigram initialization if provided [UNCHANGED]
        if bigram_logp is not None:
            with torch.no_grad():
                # Initialize first MLP weights from bigram prior (heuristic)
                B = bigram_logp.exp()  # convert to probabilities
                avg_first = B[0].mean().item()
                nn.init.constant_(self.readout_mlp[0].weight, avg_first / (d * 256))

        self.ste = False

    def set_ste(self, on):
        self.ste = on
        for L in self.layers:
            L.ste = on

    def forward(self, tokens):
        """Forward pass preserving full GF(256) computation."""
        h = ste_softmax(self.emb_logits, self.ste)[tokens]  # (B, T, d, 256)
        
        # Pass through ALL GF(256) layers [UNCHANGED from original]
        for layer in self.layers:
            if self.training and self.use_ckpt:
                h = torch.utils.checkpoint.checkpoint(
                    layer, h, self.perm_mats, use_reentrant=False)
            else:
                h = layer(h, self.perm_mats)
        
        # Bottleneck MLP: flatten + compress → output logits
        h_flat = h.reshape(tokens.shape[0], tokens.shape[1], self.d * 256)
        logits = self.readout_mlp(h_flat)  # (B, T, 256)
        
        return logits + self.readout_bias

    def regularisers(self):
        ent = row_entropy(F.softmax(self.emb_logits, -1))
        perm_pen = 0.0
        for L in self.layers:
            P = L.sbox(False)
            C = L.coefs(False)
            ent = ent + row_entropy(P) + row_entropy(C)
            perm_pen = perm_pen + ((P.sum(1) - 1.0) ** 2).mean()
        return ent / (1 + 2 * len(self.layers)), perm_pen / max(1, len(self.layers))

    @torch.no_grad()
    def import_hard(self, hard):
        """Populate the float parameters from an export_hard() payload.

        Mirrors V7's import_hard: the soft (float) forward path then matches
        the hard (integer) one. UI server calls this when loading a checkpoint.
        """
        if "readout_mlp" not in hard:
            raise AttributeError("import_hard: payload has no readout_mlp "
                                 "(not a bottleneck export)")
        self.readout_mlp.load_state_dict({
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in hard["readout_mlp"].items()})
        self.readout_bias.data = hard["bias"].detach().cpu()

    @torch.no_grad()
    def export_hard(self):
        """Export integer tables for hard forward path.

        Note: bottleneck MLP can't be exported to pure integers.
        Keep float readout as final layer.
        """
        return {
            "emb": self.emb_logits.argmax(-1),  # (256, d) → int
            "coef": [torch.tensor(self.coef_set, device=L.coef_logits.device)[
                L.coef_logits.argmax(-1)] for L in self.layers],
            "sbox": [L.sbox_logits.argmax(-1) for L in self.layers],
            "wiring": [(L.ch, L.lag) for L in self.layers],
            "readout_mlp": self.readout_mlp.state_dict(),  # keep as float
            "bias": self.readout_bias.detach(),
            "lags": self.lags,
        }

    @torch.no_grad()
    def hard_forward(self, tokens, hard):
        """Fast integer forward with float readout bottleneck.

        Integer GF(256) trunk preserved EXACTLY as in original V6. Only the
        readout differs: the S-box output integers (B, T, d), one-hot encoded
        to (B, T, d, 256), flattened to (B, T, d*256), then pushed through
        the saved MLP weights.
        """
        exp_t, log_t = torch.tensor(EXP), torch.tensor(LOG)
        lags = torch.tensor(hard["lags"])
        B, T, d = tokens.shape[0], tokens.shape[1], self.d
        dev = tokens.device

        h = hard["emb"][tokens]  # (B, T, d)
        for coef, sbox, (ch, lg) in zip(hard["coef"], hard["sbox"], hard["wiring"]):
            # Exactly the original V6 per-layer structure:
            acc = torch.zeros(B, T, d, dtype=torch.long, device=dev)
            for f in range(coef.shape[1]):
                out = torch.zeros(B, T, d, dtype=torch.long, device=dev)
                for j in range(d):
                    l = int(lags[lg[j, f]])
                    src = h[:, :, ch[j, f]]
                    if l:
                        src = torch.cat([torch.zeros(B, l, dtype=torch.long, device=dev),
                                         src[:, :-l]], 1)
                    c = int(coef[j, f])
                    if c == 0:
                        prod = torch.zeros_like(src)
                    else:
                        prod = torch.where(src == 0, torch.zeros_like(src),
                                           exp_t[log_t[src] + LOG[c]])
                    out[:, :, j] = prod
                acc = acc ^ out
            h = sbox[torch.arange(d, device=dev), acc]  # (B, T, d) integers

        # --- BOTTLENECK READOUT (float) ---
        mlp = hard["readout_mlp"]
        bias = hard["bias"]

        h_one_hot = F.one_hot(h, num_classes=256).float()      # (B, T, d, 256)
        h_flat = h_one_hot.view(B, T, d * 256)                 # (B, T, d*256)

        hidden = F.linear(h_flat, mlp["0.weight"], mlp["0.bias"])
        hidden = F.relu(hidden)
        logits = F.linear(hidden, mlp["3.weight"], mlp["3.bias"])

        return logits + bias
