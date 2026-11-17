"""
label_teacher.py - let a teacher LLM label the training text, byte by byte.

For each byte in the first --label_bytes of the training split, this stores
the teacher's 8 most likely next bytes and their probabilities. The trainer
(hard_residual_lm.py --teacher DIR) uses them twice:
  * distillation: the transformer learns to match these probabilities
  * a "teacher" expert in the hard model: byte contexts filled with the
    teacher's probabilities instead of raw counts

The teacher predicts tokens, not bytes. Byte probabilities are obtained by
summing, at each byte inside the next token, over all likely tokens that agree
with the bytes so far (the teacher's 256 most likely tokens).

Needs a byte-level BPE tokenizer: Qwen2.5, Llama 3, SmolLM2, GPT-2 and most
newer models. Only training text is labelled; validation and test text stay
untouched. Use the same --data, --max_bytes, --val_data and --test_mb as for
training: the trainer checks by checksum that the labels fit its training text.

  python label_teacher.py --data data/text.txt --label_bytes 50000000 \
      --model Qwen/Qwen2.5-1.5B --out runs/teacher_qwen
"""
import argparse, hashlib, importlib.util, json, math, os, sys, time

import numpy as np
import torch


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("trainer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bytes_to_unicode():
    """GPT-2's byte <-> printable character map used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_table(tok, V, lmax):
    """Raw bytes of every token id (None for special tokens) and a padded (V, lmax) tensor."""
    inv = {c: b for b, c in bytes_to_unicode().items()}
    pieces = tok.convert_ids_to_tokens(list(range(min(V, len(tok)))))
    special = set(tok.all_special_ids)
    full = [None] * V
    for i, p in enumerate(pieces):
        if p is None or i in special:
            continue
        try:
            full[i] = bytes(inv[c] for c in p)
        except KeyError:
            full[i] = None
    TB = torch.full((V, lmax), -1, dtype=torch.int16)
    for i, b in enumerate(full):
        if b:
            TB[i, :min(len(b), lmax)] = torch.tensor(list(b[:lmax]), dtype=torch.int16)
    return full, TB


def tokens_to_bytes(full, ids):
    parts = [full[i] for i in ids]
    return None if any(p is None for p in parts) else b"".join(parts)


@torch.no_grad()
def label_segment(model, ids, full, TB, base, out_b, out_p, args, acc):
    """Label the bytes of one segment. base = offset of the segment in the labelled region."""
    dev = TB.device
    n = len(ids)
    ids_t = torch.tensor(ids, device=dev)
    logits = model(ids_t[None]).logits[0]                     # (n, V); row i predicts token i+1
    lens = torch.tensor([len(full[i]) for i in ids], device=dev)
    off = torch.cumsum(lens, 0) - lens                         # byte offset of each token
    lmax = TB.shape[1]
    kk = torch.arange(lmax, device=dev)
    for r0 in range(0, n - 1, args.rows):
        r1 = min(n - 1, r0 + args.rows)
        lg = logits[r0:r1].float()
        lse = torch.logsumexp(lg, -1, keepdim=True)
        topv, topi = lg.topk(args.top_m, -1)
        p = (topv - lse).exp()                                 # (R, M) token probabilities
        A = TB[ids_t[r0 + 1:r1 + 1]].long()                    # (R, lmax) bytes of the actual next token
        C = TB[topi].long()                                    # (R, M, lmax) bytes of candidate tokens
        eq = C == A[:, None, :]
        prefix_ok = torch.cat([torch.ones_like(eq[..., :1]),
                               eq.int().cumprod(-1)[..., :-1].bool()], -1)
        valid = prefix_ok & (C >= 0)      # candidate agrees with bytes 0..k-1 and has a byte k
        R = r1 - r0
        dist = torch.zeros(R, lmax, 256, device=dev)
        dist.scatter_add_(2, C.clamp(min=0).transpose(1, 2),
                          (p[:, :, None] * valid).transpose(1, 2))
        mass = dist.sum(-1)                                    # (R, lmax)
        alen = lens[r0 + 1:r1 + 1].clamp(max=lmax)
        ok = (kk[None, :] < alen[:, None]) & (mass > args.min_mass)
        dist = dist / mass.clamp(min=1e-12)[..., None]
        p_true = dist.gather(-1, A.clamp(min=0)[..., None]).squeeze(-1)
        acc["bits"] += float(-torch.log2(p_true.clamp(min=1e-12))[ok].sum())
        acc["n"] += int(ok.sum())
        tp, tb = dist.topk(8, -1)
        target = base + off[r0 + 1:r1 + 1][:, None] + kk[None, :]   # which byte each label predicts
        sel = ok.nonzero(as_tuple=True)
        pos = target[sel].cpu().numpy()
        out_b[pos] = tb[sel].to(torch.uint8).cpu().numpy()
        out_p[pos] = tp[sel].to(torch.float16).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="same as for training")
    ap.add_argument("--max_bytes", type=int, default=200_000_000, help="same as for training (0 = all)")
    ap.add_argument("--val_data", default="", help="same as for training")
    ap.add_argument("--test_mb", type=int, default=0, help="same as for training")
    ap.add_argument("--runs", default="runs", help="the training --out folder (the test text is kept there)")
    ap.add_argument("--label_bytes", type=int, default=50_000_000,
                    help="label the first N bytes of the training split")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--out", default="runs/teacher")
    ap.add_argument("--trainer", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "hard_residual_lm.py"),
                    help="path of the training script (for its data split)")
    ap.add_argument("--seg_bytes", type=int, default=8192, help="text per forward pass")
    ap.add_argument("--top_m", type=int, default=256, help="candidate tokens per position")
    ap.add_argument("--lmax", type=int, default=32, help="longest token, in bytes, to label inside")
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--min_mass", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    trainer = load_trainer(args.trainer)
    os.makedirs(args.runs, exist_ok=True)
    ns = argparse.Namespace(mode="alone", data=args.data, max_bytes=args.max_bytes, val_data=args.val_data,
                            val_file="", test_file="", test_mb=args.test_mb, out=args.runs, load_hard="",
                            check_loo=False)
    train = trainer.load_data(ns)[0]              # exactly the text the trainer trains on
    L = min(args.label_bytes, len(train))
    region = train[:L]
    print(f"labelling {L:,} of {len(train):,} training bytes with {args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(args.device).eval()
    V = model.get_output_embeddings().weight.shape[0]
    full, TB = token_table(tok, V, args.lmax)
    TB = TB.to(args.device)
    probe = "Hello wörld, it's 2026!\n  Größe: 3.5 €"
    ids = tok(probe, add_special_tokens=False)["input_ids"]
    if tokens_to_bytes(full, ids) != probe.encode("utf-8"):
        raise SystemExit(f"{args.model} does not use a byte-level BPE tokenizer; "
                         f"pick e.g. Qwen2.5, Llama 3, SmolLM2 or GPT-2")

    os.makedirs(args.out, exist_ok=True)
    out_b = np.lib.format.open_memmap(os.path.join(args.out, "top_bytes.npy"), mode="w+",
                                      dtype=np.uint8, shape=(L, 8))
    out_p = np.lib.format.open_memmap(os.path.join(args.out, "top_probs.npy"), mode="w+",
                                      dtype=np.float16, shape=(L, 8))
    out_p[:] = 0
    acc = {"bits": 0.0, "n": 0}
    pos, skipped, t0, last = 0, 0, time.time(), 0.0
    while pos < L:
        seg = region[pos:pos + args.seg_bytes]
        if pos + len(seg) < L:                           # end the segment at a space or newline
            q = len(seg) * 3 // 4
            cut = np.flatnonzero((seg[q:] == 32) | (seg[q:] == 10))
            if len(cut):
                seg = seg[:q + int(cut[-1]) + 1]
        raw = seg.tobytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            if e.start == 0:                             # starts inside a character: step past it
                pos += 1
                skipped += 1
                continue
            raw = raw[:e.start]
            text = raw.decode("utf-8")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) >= 2 and tokens_to_bytes(full, ids) == raw:
            seg_b = np.zeros((len(raw), 8), dtype=np.uint8)       # label in memory, then
            seg_p = np.zeros((len(raw), 8), dtype=np.float16)     # write one contiguous slice
            label_segment(model, ids, full, TB, 0, seg_b, seg_p, args, acc)
            out_b[pos:pos + len(raw)] = seg_b
            out_p[pos:pos + len(raw)] = seg_p
        else:
            skipped += len(raw)                          # text the tokenizer changes (rare)
        pos += len(raw)
        if time.time() - last > 30 or pos >= L:
            last = time.time()
            rate = pos / max(1e-9, last - t0)
            bpc = acc["bits"] / max(1, acc["n"])
            print(f"  {pos / 1e6:8.1f} MB | {rate / 1e3:6.0f} KB/s | "
                  f"teacher BPC so far {bpc:.3f} | eta {(L - pos) / max(rate, 1) / 60:5.1f} min")

    out_b.flush()
    out_p.flush()
    cover = acc["n"] / L
    meta = dict(model=args.model, n_bytes=int(L), sha1=hashlib.sha1(region.tobytes()).hexdigest(),
                data=args.data, max_bytes=args.max_bytes, labelled_fraction=cover,
                teacher_bpc=acc["bits"] / max(1, acc["n"]), skipped_bytes=skipped)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done: {cover:.1%} of bytes labelled, teacher BPC {meta['teacher_bpc']:.3f} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
