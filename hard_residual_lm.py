"""
hard_residual_lm.py - a counted "hard" model plus a small transformer.

THE HARD MODEL (runs/<tag>_hard.pt) is built in one counting pass over the
training text - no gradients. It holds up to three experts; each predicts the
next byte on its own:

  ngram    "after these bytes, this byte came next". Exact byte contexts of
           length 0..--order, plus hashed long contexts (--long_orders, e.g.
           8,12,16 bytes) that recall longer phrases seen in training.
  word     (--words) the meaning class of the previous one or two words + the
           current word so far. Classes come from an embedding model
           (--embed_model), or, without one, from which words appear next to
           which in the corpus.
  teacher  (--teacher DIR) byte contexts filled with a teacher LLM's next-byte
           probabilities instead of raw counts. DIR is made by label_teacher.py.

DOCUMENT MEMORY needs no tables: it is computed from the --doc_hist bytes of
text before each byte, far more than the transformer's own window (--ctx).
  cache    (--cache) what followed the same 2..16 bytes earlier in THIS text.
           Names, topic words and phrases a text repeats. One more expert.
  topic    (--topic, needs --words) how often each word class occurred in this
           text so far: 256 numbers about what the text is about. Experimental:
           no gain in a small test.
A model trained without them can be extended: --init old_best.pt --cache gives
the new inputs zero weight, so training starts exactly at the old model and
learns to use them.

The transformer uses the memory in two ways:
  --hard_feature   every expert's prediction, plus how much evidence it had,
                   is an extra input. Best in our runs so far.
  --mode residual  the experts' predictions are also added to the output.
With --teacher, the transformer is also trained to match the teacher's
next-byte probabilities (distillation, --distill_alpha).
--mem_drop 0.1 weakens the memory for 10% of the training windows, so the
transformer does not trust the tables blindly (optional, for robustness).

WHICH TEXT TO TRAIN ON. Leave-one-out removes a byte's own count, but not the
other places where the same document repeats itself. On counted text the tables
can therefore look better than they will on new text (--mode hard --check_loo
measures this). The clean alternative: count the tables from one part of the
text and train the transformer on the rest, which is exactly the test situation:
  --table_bytes N      tables from the first N bytes of the training text
  --train_part rest    training windows only from the text after them
Afterwards, tables counted from everything can be plugged in (--mode eval).

Leave-one-out: on training text the tables already contain the byte being
predicted, so it is subtracted first. The experts then look on training text
the way they look on unseen text.

Training keeps only the best model (lowest val BPC), stops early when val BPC
stops improving, and logs every eval to runs/<tag>.csv.

PAPER TRAIL. With --prompts FILE and/or --probe FILE, every --sample_every
evals the model writes sample texts for fixed prompts and its score on a
multiple-choice knowledge probe to runs/<tag>_samples.txt (the probe score also
goes into the csv). At the end, runs/<tag>_result.json holds everything about
the run: best step, validation and test scores (also "warm": only the second
half of each window, where the transformer has context), probe score, size and
all options. run_all.py runs the whole experiment and collects these.
--rope uses rotary positions instead of one learned vector per position.

Examples:
  python hard_residual_lm.py --data data/text.txt --mode hard --words
  python hard_residual_lm.py --data data/text.txt --hard_feature --words --steps 20000
  python label_teacher.py    --data data/text.txt --out runs/teacher --label_bytes 50000000
  python hard_residual_lm.py --data data/text.txt --hard_feature --words --teacher runs/teacher
  python hard_residual_lm.py --mode generate --ckpt runs/alone_feat_best.pt --interactive

Document memory on top of a trained model, and a 2-minute estimate first:
  python hard_residual_lm.py --mode hard --load_hard runs/hard_big_hard.pt --cache \
      --val_data small.txt                       # tables vs tables blended with the cache
  python hard_residual_lm.py --data big.txt --val_data small.txt --hard_feature --words \
      --load_hard runs/hard_big_hard.pt --init runs/big_feat_best.pt --cache \
      --d 512 --layers 8 --heads 8 --steps 25000 --lr 2e-4 --warmup 100 --patience 0 --tag ft_cache

Plug-in memory: build bigger tables, then evaluate or briefly fine-tune the
SAME trained transformer with them (validation text is removed from --data):
  python hard_residual_lm.py --mode hard --data big.txt --val_data small.txt --words \
      --classes_from runs/big_feat_hard.pt --tag hard_big
  python hard_residual_lm.py --mode eval --ckpt runs/big_feat_best.pt \
      --load_hard runs/hard_big_hard.pt --val_data small.txt
  python hard_residual_lm.py --data big.txt --val_data small.txt --hard_feature --words \
      --load_hard runs/hard_big_hard.pt --init runs/big_feat_best.pt --d 512 --layers 8 \
      --heads 8 --steps 5000 --lr 2e-4 --warmup 100 --patience 0 --tag ft_big
"""
import argparse, csv, glob, hashlib, json, math, os, re, shutil, sys, time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LN2 = math.log(2)
MASK55 = (1 << 55) - 1        # long-context hashes keep 55 bits so (hash, byte) fits in 63
HASH_MUL = 0x100000001B3
MAX_H = 64                    # longest context any expert may use
VAL_OFFSET = MAX_H + 1        # val targets start here in every mode, so runs compare
WORD_RE = re.compile(rb"[a-z\x80-\xff]+")          # applied to lowercased text
WORD_MASK = np.zeros(256, dtype=bool)
WORD_MASK[ord("a"):ord("z") + 1] = True
WORD_MASK[ord("A"):ord("Z") + 1] = True
WORD_MASK[128:] = True                            # UTF-8 letters such as ä ö ü ß


# ------------------------------------------------------------------ data
def load_bytes(path, max_bytes, block=1 << 20):
    """A text file, or a folder of .txt files (read recursively, joined by blank lines).

    Split: corpora under 10 MB use the last 10% as validation. Larger corpora
    send every 10th 1 MB block to validation, so it is spread over all files
    instead of being only the last book.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no .txt files under {path}")
        parts, total = [], 0
        sep = np.frombuffer(b"\n\n", dtype=np.uint8)
        for p in files:
            b = np.fromfile(p, dtype=np.uint8)
            if max_bytes:
                b = b[:max(0, max_bytes - total)]
            parts += [b, sep]
            total += len(b) + 2
            if max_bytes and total >= max_bytes:
                break
        raw = np.concatenate(parts)
        print(f"read {len(parts) // 2} files from {path}")
    else:
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        if max_bytes and len(raw) > max_bytes:
            raw = raw[:max_bytes]
        raw = np.array(raw)
    if not block:                      # no split: everything is training text
        return raw, raw[:0]
    if len(raw) < 10 * block:
        n = int(0.9 * len(raw))
        return raw[:n], raw[n:]
    nb = len(raw) // block
    blocks = raw[:nb * block].reshape(nb, block)
    is_val = np.arange(nb) % 10 == 9
    train = np.concatenate([blocks[~is_val].ravel(), raw[nb * block:]])
    return train, blocks[is_val].ravel()


def remove_text(data, val, block=1 << 20, probe=4096):
    """Remove from `data` every place where a 1 MB piece of `val` occurs.
    Validation pieces are long verbatim runs, so a 4 KB probe finds each one;
    the match is then extended byte by byte. Returns the remaining text."""
    big = data.tobytes()
    keep = np.ones(len(data), dtype=bool)
    removed, missing, where = 0, 0, []
    for s in range(0, len(val), block):
        piece = val[s:s + block]
        rest = piece
        while len(rest) >= 256:
            hits = []
            for o in (0, len(rest) // 4, len(rest) // 2, 3 * len(rest) // 4):
                pr = rest[o:o + probe].tobytes()
                i = big.find(pr)
                while i != -1:
                    hits.append(i - o)
                    i = big.find(pr, i + 1)
                if hits:
                    break
            if not hits:
                missing += len(rest)
                break
            where.extend(hits)
            best = 0
            for i in set(hits):
                lo = max(0, -i)
                n = min(len(rest), len(data) - i)
                eq = data[i + lo:i + n] == rest[lo:n]
                L = lo + (len(eq) if eq.all() else int(np.argmin(eq)))
                keep[max(0, i):i + L] = False
                best = max(best, L)
            rest = rest[max(best, 256):]
    removed = int((~keep).sum())
    print(f"removed {removed:,} bytes of validation text from the training data"
          f"{f' ({missing:,} val bytes not found in it)' if missing else ''}"
          f"{f', found between byte {min(where):,} and {max(where):,}' if where else ''}")
    return data[keep]


def val_starts(n_val_bytes, H, T, n, layout="spread", lo=0, hi=None):
    """Window starts whose targets begin at VAL_OFFSET (same targets for every H).
    spread: n windows evenly over [lo, hi) of the validation text (default);
    first: n windows back to back from the start (how runs before Nov 2026 measured)."""
    hi = n_val_bytes if hi is None else hi
    room = hi - lo - VAL_OFFSET - T - 1
    n = max(1, min(n, room // T))
    step = T if layout == "first" else max(T, room // n)
    return [lo + VAL_OFFSET - H - 1 + i * step for i in range(n)]


def gather(arr, starts, span, offset=0):
    """Rows arr[s+offset : s+offset+span]. Positions before the start of the
    text are filled with zeros (only the first validation windows need this)."""
    rows = []
    for s in starts:
        a = s + offset
        if a >= 0:
            rows.append(arr[a:a + span])
        else:
            pad = torch.zeros((min(-a, span),) + tuple(arr.shape[1:]), dtype=arr.dtype)
            rows.append(torch.cat([pad, arr[0:max(0, a + span)]]))
    return torch.stack(rows)


def pad_of(starts, device):
    """How many positions at the start of each row lie before the text (see gather)."""
    if min(starts) >= 0:
        return None
    return torch.tensor([max(0, -s) for s in starts], device=device)


# ------------------------------------------------------------- words
def count_words(data, chunk=1 << 24):
    c = Counter()
    for s in range(0, len(data), chunk):
        c.update(WORD_RE.findall(data[s:s + chunk].tobytes().lower()))
    return c


def word_features(data, cls_of, carry=(0, 0), chunk=1 << 24):
    """For every position q: (class of the last finished word, class of the word
    before it, length of the current word including byte q (0 = byte q is not part
    of a word), sub-class of the last finished word). Class 0 = no word yet,
    1 = word not in the vocabulary. cls_of maps word -> class | sub-class << 8."""
    N = len(data)
    out = np.zeros((N, 4), dtype=np.uint8)
    c2c, c1c = np.uint16(carry[0]), np.uint16(carry[1])
    s = 0
    while s < N:
        e = min(N, s + chunk)
        if e < N:                     # cut after a non-word byte: no word spans two chunks
            nw = np.flatnonzero(~WORD_MASK[data[s:e]])
            if len(nw):
                e = s + int(nw[-1]) + 1
        d = np.asarray(data[s:e])
        isw = WORD_MASK[d]
        words = WORD_RE.findall(d.tobytes().lower())
        cls = np.fromiter((cls_of.get(w, 1) for w in words), dtype=np.uint16, count=len(words))
        start = isw & ~np.concatenate([[False], isw[:-1]])
        run = np.cumsum(start) - 1                 # index of the current / last word
        ext = np.concatenate([np.array([c2c, c1c], dtype=np.uint16), cls])
        last_done = run + 2 - isw                  # position in ext of the last finished word
        v1, v2 = ext[last_done], ext[last_done - 1]
        out[s:e, 0] = v1 & 255
        out[s:e, 1] = v2 & 255
        out[s:e, 3] = v1 >> 8
        sp = np.flatnonzero(start)
        if len(sp):
            plen = np.where(isw, np.arange(len(d)) - sp[np.maximum(run, 0)] + 1, 0)
            out[s:e, 2] = np.minimum(plen, 255)
        c2c, c1c = ext[-2], ext[-1]
        s = e
    return out, (int(c2c), int(c1c))


def word_keys(col, c1, c2, plen, s1=None):
    """Two word contexts per position (numpy arrays or torch tensors, int64).
    col(j) = the byte j steps back (0 = the byte just read).
      k1 = class of the previous word + current word so far (up to its last 4 bytes)
      kf = (fine classes only) class + sub-class of the previous word + the same
      k2 = classes of the two previous words + current word so far
    Returns (k1, k2), or (k1, kf, k2) with fine classes: the smoothing order.
    Between words, the last two bytes (", " or ". ") stand in for the prefix."""
    pk = 0
    for j in range(4):
        keep = (plen > j) | ((plen == 0) & (j < 2))
        pk = pk + col(j) * keep * (1 << (8 * j))
    k1 = c1 * (1 << 36) + plen.clip(max=15) * (1 << 32) + pk
    k2 = c2 * (1 << 44) + k1
    if s1 is None:
        return k1, k2
    # fine: previous word as class * 256 + sub-class; an extra, more detailed level
    kf = (c1 * 256 + s1) * (1 << 36) + plen.clip(max=15) * (1 << 32) + pk
    return k1, kf, k2


def cooc_vectors(data, vocab, n_ctx=500, dim=100, chunk=1 << 24):
    """Word vectors from the corpus itself: which frequent words appear right
    before / after each word (PPMI + SVD)."""
    idx = {w: i for i, w in enumerate(vocab)}
    V, W = len(vocab), 2 * n_ctx
    acc = np.zeros(V * W, dtype=np.float64)
    for s in range(0, len(data), chunk):
        words = WORD_RE.findall(data[s:s + chunk].tobytes().lower())
        ids = np.fromiter((idx.get(w, -1) for w in words), dtype=np.int64, count=len(words))
        l, r = ids[:-1], ids[1:]
        m1 = (r >= 0) & (l >= 0) & (l < n_ctx)
        m2 = (l >= 0) & (r >= 0) & (r < n_ctx)
        flat = np.concatenate([r[m1] * W + l[m1], l[m2] * W + n_ctx + r[m2]])
        acc += np.bincount(flat, minlength=V * W)
    M = torch.from_numpy(acc.reshape(V, W)).float()
    tot = M.sum()
    row = M.sum(1, keepdim=True).clamp(min=1)
    colsum = M.sum(0, keepdim=True).clamp(min=1)
    ppmi = torch.log((M * tot / (row * colsum)).clamp(min=1e-12)).clamp(min=0) * (M > 0)
    U, S, _ = torch.svd_lowrank(ppmi, q=min(dim, W - 1), niter=4)
    return U * S


@torch.no_grad()
def embed_vectors(vocab, model_name, device, batch=512):
    """Word vectors from an existing model, mean-pooled over the word's tokens:
    any Hugging Face encoder (e.g. sentence-transformers/all-MiniLM-L6-v2) or a
    decoder LLM (e.g. Qwen/Qwen2.5-1.5B, its hidden states)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    mdl = AutoModel.from_pretrained(model_name, dtype=dtype).to(device).eval()
    texts = [w.decode("utf-8", "replace") for w in vocab]
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=16,
                  return_tensors="pt").to(device)
        h = mdl(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
    return torch.cat(out)


def kmeans(X, k, iters=30, seed=0):
    X = F.normalize(X.float(), dim=1)
    k = min(k, len(X))
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k].to(X.device)].clone()
    for _ in range(iters):
        a = (X @ C.T).argmax(1)
        C = torch.zeros_like(C).index_add_(0, a, X)
        empty = torch.bincount(a, minlength=k) == 0
        if empty.any():
            C[empty] = X[torch.randint(0, len(X), (int(empty.sum()),), generator=g).to(X.device)]
        C = F.normalize(C, dim=1)
    return a


def word_vectors(train, vocab, args, device):
    """--class_source: cooc (from the corpus), embed (from --embed_model) or both."""
    src = getattr(args, "class_source", "") or ("embed" if args.embed_model else "cooc")
    if src in ("embed", "both") and not args.embed_model:
        raise SystemExit("--class_source embed/both needs --embed_model")
    parts, how = [], []
    if src in ("cooc", "both"):
        parts.append(F.normalize(cooc_vectors(train, vocab).float(), dim=1))
        how.append("co-occurrence")
    if src in ("embed", "both"):
        parts.append(F.normalize(embed_vectors(vocab, args.embed_model, device), dim=1))
        how.append(f"embeddings from {args.embed_model}")
    return torch.cat(parts, 1), " + ".join(how)


def build_classes(train, args, device):
    """254 word classes by k-means. With --fine_classes, every class is split into
    up to 254 sub-classes: the --exact_words most frequent words each get their own
    sub-class (their identity matters), the rest are clustered."""
    t0 = time.time()
    counts = count_words(train)
    vocab = [w for w, _ in counts.most_common(args.vocab)]
    X, how = word_vectors(train, vocab, args, device)
    X = X.to(device)
    a = kmeans(X, 254).cpu().numpy()
    classes = (a + 2).astype(np.uint8)
    subs = None
    if getattr(args, "fine_classes", False):
        subs = np.zeros(len(vocab), dtype=np.uint8)
        rank = np.arange(len(vocab))                        # vocab is sorted by frequency
        for c in np.unique(a):
            idx = np.flatnonzero(a == c)
            exact = idx[rank[idx] < getattr(args, "exact_words", 2000)][:253]
            subs[exact] = np.arange(len(exact))
            rest = np.setdiff1d(idx, exact)
            slots = 254 - len(exact)
            if len(rest) and slots > 0:
                sa = kmeans(X[torch.from_numpy(rest).to(device)], min(slots, len(rest))).cpu().numpy()
                subs[rest] = len(exact) + sa
            elif len(rest):
                subs[rest] = 253
    print(f"word classes: {len(vocab):,} words -> {len(set(a.tolist()))} classes"
          f"{' with sub-classes' if subs is not None else ''} ({how}), {time.time() - t0:.0f}s")
    for c in sorted(set(classes[:30].tolist()))[:4]:           # a peek at a few classes
        members = [w.decode("utf-8", "replace") for w, k in zip(vocab, classes) if k == c][:10]
        print(f"   class {c:3d}: {' '.join(members)}")
    return vocab, classes, subs


# ----------------------------------------------------------- tables
def build_table(keys, y, min_count=1):
    """Sorted (context, next byte) -> count table, stored per context (CSR).
    One-shot version (all keys in memory); the build uses PairCounter instead."""
    up, cnt = np.unique(keys * 256 + y, return_counts=True)
    return table_from_pairs(up, cnt, min_count)


class PairCounter:
    """Counts (context, next byte) pairs chunk by chunk with bounded memory.
    Each chunk is aggregated right away and split into 16 buckets by the top
    bits of the pair, so every bucket can be merged on its own and the buckets
    stay in sorted order."""

    def __init__(self, pair_bits, nb=16):
        self.shift, self.nb = max(0, pair_bits - 4), nb
        self.parts = [[] for _ in range(nb)]

    def add(self, keys, y):
        up, cnt = np.unique(keys * 256 + y, return_counts=True)
        edges = np.searchsorted(up >> self.shift, np.arange(self.nb + 1))
        for i in range(self.nb):
            if edges[i + 1] > edges[i]:
                self.parts[i].append((up[edges[i]:edges[i + 1]], cnt[edges[i]:edges[i + 1]].astype(np.int32)))

    def table(self, min_count=1):
        pieces = []
        for i in range(self.nb):
            parts, self.parts[i] = self.parts[i], None
            if not parts:
                continue
            up = np.concatenate([p[0] for p in parts])
            cnt = np.concatenate([p[1] for p in parts])
            del parts
            u, inv = np.unique(up, return_inverse=True)
            c = np.bincount(inv.ravel(), weights=cnt)
            del up, cnt, inv
            pieces.append(table_from_pairs(u, c, min_count))
        return concat_tables(pieces)


def concat_tables(pieces):
    if not pieces:
        return dict(uctx=torch.zeros(0, dtype=torch.int64), indptr=torch.zeros(1, dtype=torch.int64),
                    byt=torch.zeros(0, dtype=torch.uint8), cnt=torch.zeros(0))
    ind, off = [], 0
    for i, p in enumerate(pieces):
        ind.append((p["indptr"] if i == 0 else p["indptr"][1:]) + off)
        off += len(p["byt"])
    cat = lambda k: torch.cat([p[k] for p in pieces])
    return dict(uctx=cat("uctx"), indptr=torch.cat(ind), byt=cat("byt"), cnt=cat("cnt"))


def build_table_weighted(key_fn, tb, tp, n, min_p, chunk=1 << 22):
    """Like build_table, but each position adds fractional counts (teacher
    probabilities) for up to 8 bytes. Built in chunks to keep memory low.
    key_fn(a, b) gives the context keys of rows a..b."""
    ups, cnts = [], []
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        p = np.asarray(tp[a:b], dtype=np.float32)
        m = (p > 0) & (p >= min_p)
        pair = np.repeat(key_fn(a, b)[:, None], 8, 1)[m] * 256 + np.asarray(tb[a:b])[m].astype(np.int64)
        up, inv = np.unique(pair, return_inverse=True)
        ups.append(up)
        cnts.append(np.bincount(inv.ravel(), weights=p[m]))
        del pair, inv, p, m
    up, inv = np.unique(np.concatenate(ups), return_inverse=True)
    return table_from_pairs(up, np.bincount(inv.ravel(), weights=np.concatenate(cnts)))


def table_from_pairs(up, cnt, min_count=1):
    cnt = cnt.astype(np.float32)
    uctx, start, per = np.unique(up >> 8, return_index=True, return_counts=True)
    if min_count > 1:
        keep = np.add.reduceat(cnt, start) >= min_count
        keep_e = np.repeat(keep, per)
        up, cnt, uctx, per = up[keep_e], cnt[keep_e], uctx[keep], per[keep]
    indptr = np.concatenate([[0], np.cumsum(per)]).astype(np.int64)
    return dict(uctx=torch.from_numpy(uctx.astype(np.int64)), indptr=torch.from_numpy(indptr),
                byt=torch.from_numpy((up & 255).astype(np.uint8)), cnt=torch.from_numpy(cnt))


def lookup(tab, key):
    """Counts of every next byte for each query context -> (M, 256)."""
    M, dev = key.numel(), key.device
    C = torch.zeros(M, 256, device=dev)
    uc = tab["uctx"]
    if uc.numel() == 0:
        return C
    idx = torch.searchsorted(uc, key).clamp(max=len(uc) - 1)
    found = uc[idx] == key
    s, e = tab["indptr"][idx], tab["indptr"][idx + 1]
    L = torch.where(found, e - s, torch.zeros_like(s))
    rows = torch.repeat_interleave(torch.arange(M, device=dev), L)
    first = torch.cumsum(L, 0) - L
    offs = torch.arange(rows.numel(), device=dev) - first[rows] + s[rows]
    C[rows, tab["byt"][offs].long()] = tab["cnt"][offs]
    return C


def table_size(tabs):
    return sum(len(t["byt"]) for t in tabs)


# -------------------------------------------------------- hard model
class HardModel:
    """All count tables plus the word vocabulary. Each expert turns its tables
    into a next-byte distribution with interpolated Witten-Bell smoothing:
    start from a short context, and trust each longer one more the more often
    it was seen and the fewer different bytes followed it."""

    def __init__(self):
        self.tabs, self.long_tabs, self.word_tabs, self.teacher_tabs = [], [], [], []
        self.long, self.teacher_orders = [], []
        self.vocab, self.classes, self.cls_of, self.teacher_uni = None, None, None, None
        self.subs = None
        self.n_text = None
        self.teacher_min_p = 0.0
        self.wf_cache = None          # word features of the build text (not saved)
        # document memory: computed from the text before each byte, needs no tables
        self.cache_orders, self.cache_keep, self.doc_hist, self.topic = [], 8, 0, False
        self.cache_decay, self.class_cache = 0.0, False

    def set_runtime(self, a):
        """Switch the document memory on or off (--cache / --topic). It has no
        tables, so it is not part of the hard file; the model's saved args say
        whether it was trained with it."""
        g = lambda k, d: a.get(k, d) if isinstance(a, dict) else getattr(a, k, d)
        co = g("cache_orders", "") if g("cache", False) else ""
        self.cache_orders = sorted(int(x) for x in str(co).split(",") if x.strip())
        self.cache_keep = int(g("cache_keep", 8))
        self.topic = bool(g("topic", False))
        self.cache_decay = float(g("cache_decay", 0.0))
        self.class_cache = bool(int(g("class_cache", 0)))
        if (self.topic or self.class_cache) and not self.word_tabs:
            raise SystemExit("--topic and --class_cache need the word classes (--words)")
        self.doc_hist = int(g("doc_hist", 8192)) if (self.cache_orders or self.topic or self.class_cache) else 0
        return self

    @property
    def hist(self):
        """Bytes of text needed before the first predicted byte."""
        return max(self.H, self.doc_hist)

    def wf_span(self, T):
        """(length, offset) of the word features a batch needs."""
        return (self.hist + T, 0) if self.wf_full else (T, self.hist)

    @property
    def wf_full(self):
        """Word features are needed for the whole row, not only the predicted bytes."""
        return self.topic or self.class_cache

    @classmethod
    def build(cls, train, args, device, teacher=None):
        hm = cls()
        hm.order = args.order
        hm.n_text = len(train)                    # bytes of text the tables were counted from
        lo = args.long_orders.strip().lower()
        hm.long = [] if lo in ("", "none", "0") else sorted(int(x) for x in lo.split(",") if x.strip())
        hm.teacher_orders = (sorted(int(x) for x in args.teacher_orders.split(",") if x.strip())
                             if teacher is not None else [])
        hm.H = max([hm.order, 4] + hm.long + hm.teacher_orders)
        assert 1 <= hm.order <= 6, "--order up to 6"
        assert hm.H <= MAX_H, f"contexts up to {MAX_H} bytes"
        d, N, H = train, len(train), hm.H
        n, CH = N - H, args.build_chunk
        # query row r (0..n-1) = "predict byte H+r from the bytes before it"
        chunks = [(a, min(n, a + CH)) for a in range(0, n, CH)]
        colc = lambda j, a, b: d[H - 1 - j + a:H - 1 - j + b].astype(np.int64)
        yc = lambda a, b: d[H + a:H + b].astype(np.int64)

        t0 = time.time()
        uni = np.zeros(256)
        cnts = [PairCounter(8 * (k + 1)) for k in range(1, hm.order + 1)]
        for a, b in chunks:
            y = yc(a, b)
            uni += np.bincount(y, minlength=256)
            key = np.zeros(b - a, dtype=np.int64)
            for k in range(1, hm.order + 1):
                key = key * 256 + colc(k - 1, a, b)
                cnts[k - 1].add(key, y)
        hm.uni = torch.from_numpy(uni.astype(np.float32))
        hm.tabs = [c.table() for c in cnts]
        del cnts
        print(f"  ngram orders 1..{hm.order}: {table_size(hm.tabs):,} entries, {time.time() - t0:.0f}s")

        if hm.long:
            t0 = time.time()
            for m in hm.long:                         # one order at a time: these tables are big
                c = PairCounter(63)
                for a, b in chunks:
                    h = np.zeros(b - a, dtype=np.int64)
                    for j in range(m):
                        h = h * HASH_MUL + (colc(j, a, b) + 1)
                    c.add(h & MASK55, yc(a, b))
                hm.long_tabs.append(c.table(min_count=args.long_min))
                del c
            print(f"  long contexts {hm.long} (seen >= {args.long_min}x): "
                  f"{table_size(hm.long_tabs):,} entries, {time.time() - t0:.0f}s")

        if args.words:
            if args.classes_from:
                src = HardModel.load(args.classes_from, "cpu", vocab_only=True)
                if getattr(args, "fine_classes", False) and src.subs is None:
                    raise SystemExit(f"--fine_classes: {args.classes_from} has no sub-classes")
                hm.set_vocab(src.vocab, src.classes, src.subs if getattr(args, "fine_classes", False) else None)
                print(f"word classes: {len(src.vocab):,} words"
                      f"{' with sub-classes' if src.subs is not None else ''}, reused from {args.classes_from}")
            else:
                hm.set_vocab(*build_classes(train, args, device))
            t0 = time.time()
            wf, _ = word_features(d, hm.cls_of)
            hm.wf_cache = wf                          # the trainer needs it again
            n_keys = 3 if hm.fine else 2
            for which in range(n_keys):
                c = PairCounter(52 if which == 0 else 60)
                for a, b in chunks:
                    q = wf[H - 1 + a:H - 1 + b].astype(np.int64)
                    k = word_keys(lambda j: colc(j, a, b), q[:, 0], q[:, 1], q[:, 2],
                                  q[:, 3] if hm.fine else None)[which]
                    c.add(k, yc(a, b))
                hm.word_tabs.append(c.table())
                del c
            print(f"  word contexts: {table_size(hm.word_tabs):,} entries, {time.time() - t0:.0f}s")

        if teacher is not None:
            hm.add_teacher(train, teacher, args)
        return hm.to(device)

    def add_teacher(self, train, teacher, args):
        """Teacher expert: byte contexts filled with the teacher's probabilities.
        Can be added to an already built hard model (--load_hard)."""
        t0 = time.time()
        self.teacher_orders = sorted(int(x) for x in args.teacher_orders.split(",") if x.strip())
        if max(self.teacher_orders) > self.H:
            raise SystemExit(f"--teacher_orders up to {self.H} for this hard model")
        self.teacher_min_p = args.teacher_min_p
        d, H, L = train, self.H, teacher.L
        tb, tp = teacher.tb[H:L], teacher.tp[H:L]            # labels of targets t in [H, L)
        n = L - H
        uni = np.zeros(256)
        for a in range(0, n, 1 << 22):
            p = np.asarray(tp[a:a + (1 << 22)], dtype=np.float32)
            m = (p > 0) & (p >= self.teacher_min_p)
            uni += np.bincount(np.asarray(tb[a:a + (1 << 22)])[m], weights=p[m], minlength=256)
        self.teacher_uni = torch.from_numpy(uni.astype(np.float32))

        def key_fn(k):
            def f(a, b):
                key = np.zeros(b - a, dtype=np.int64)
                for j in range(k):
                    key = key * 256 + d[H - 1 - j + a:H - 1 - j + b].astype(np.int64)
                return key
            return f

        self.teacher_tabs = [build_table_weighted(key_fn(k), tb, tp, n, self.teacher_min_p)
                             for k in self.teacher_orders]
        print(f"  teacher contexts {self.teacher_orders}: {table_size(self.teacher_tabs):,} entries "
              f"from {L:,} labelled bytes, {time.time() - t0:.0f}s")

    def set_vocab(self, vocab, classes, subs=None):
        self.vocab, self.classes = vocab, np.asarray(classes, dtype=np.uint8)
        self.subs = None if subs is None else np.asarray(subs, dtype=np.uint8)
        packed = self.classes.astype(np.int64)
        if self.subs is not None:
            packed = packed | (self.subs.astype(np.int64) << 8)
        self.cls_of = dict(zip(vocab, packed.tolist()))

    @property
    def fine(self):
        return getattr(self, "subs", None) is not None

    @property
    def names(self):
        n = (["ngram_o2"] if self.order > 2 else []) + ["ngram"]
        n += ["ngram_long"] if self.long_tabs else []
        n += ["word"] if self.word_tabs else []
        n += ["teacher"] if self.teacher_tabs else []
        n += ["cache"] if self.cache_orders else []
        n += ["ccache"] if self.class_cache else []
        return n

    @property
    def main(self):
        return "ngram"

    @property
    def n_stats(self):
        n = 1 + self.order + len(self.long_tabs)
        n += 1 + len(self.word_tabs) if self.word_tabs else 0
        n += 1 + len(self.teacher_tabs) if self.teacher_tabs else 0
        n += 1 + len(self.cache_orders) if self.cache_orders else 0
        n += 2 if self.class_cache else 0
        n += 256 if self.topic else 0
        return n

    def stat_slices(self):
        """Which columns of the evidence tensor belong to what (order of query)."""
        n_chain = 1 + self.order + len(self.long_tabs)
        sl = dict(chain=slice(0, n_chain), long=slice(1 + self.order, n_chain))
        pos = n_chain
        for name, n in (("word", 1 + len(self.word_tabs) if self.word_tabs else 0),
                        ("teacher", 1 + len(self.teacher_tabs) if self.teacher_tabs else 0),
                        ("tables", 0),
                        ("cache", 1 + len(self.cache_orders) if self.cache_orders else 0),
                        ("ccache", 2 if self.class_cache else 0),
                        ("topic", 256 if self.topic else 0)):
            if name == "tables":
                sl[name] = slice(0, pos)
            elif n:
                sl[name] = slice(pos, pos + n)
                pos += n
        return sl

    def input_groups(self, n_cache=None, topic=None, ccache=None):
        """Columns of the transformer's memory input, in order, as (name, width).
        n_cache / topic / ccache describe another configuration (used by --init)."""
        n_cache = len(self.cache_orders) if n_cache is None else n_cache
        topic = self.topic if topic is None else topic
        ccache = self.class_cache if ccache is None else ccache
        g = [("log:" + n, 256) for n in self.names if n not in ("cache", "ccache")]
        g += [("log:cache", 256)] if n_cache else []
        g += [("log:ccache", 256)] if ccache else []
        s_tab = 1 + self.order + len(self.long_tabs)
        s_tab += 1 + len(self.word_tabs) if self.word_tabs else 0
        s_tab += 1 + len(self.teacher_tabs) if self.teacher_tabs else 0
        g += [("stat:tables", s_tab)]
        g += [("stat:cache", 1 + n_cache)] if n_cache else []
        g += [("stat:ccache", 2)] if ccache else []
        g += [("stat:topic", 256)] if topic else []
        return g

    def unigram(self, counts):
        """What an expert answers when its tables hold nothing: (log-probs, evidence)."""
        C = counts.float()
        tot, nd = C.sum(), (C > 1e-3).float().sum()
        return ((C + nd / 256) / (tot + nd)).log(), torch.log1p(tot)

    def degrade(self, logs, stats, p):
        """Memory dropout (training only): with probability p a window gets a
        weaker memory - no long contexts, only 2-byte contexts, or empty tables.
        The transformer then cannot rely on the tables alone."""
        B, dev = stats.shape[0], stats.device
        hit = torch.rand(B, device=dev) < p
        if not bool(hit.any()):
            return logs, stats
        kinds = (["nolong"] if "ngram_long" in logs else []) + \
            (["short"] if "ngram_o2" in logs else []) + ["empty"] + (["nocache"] if "cache" in logs or "ccache" in logs else [])
        kind = torch.randint(0, len(kinds), (B,), device=dev)
        logs, stats, sl = dict(logs), stats.clone(), self.stat_slices()
        uni_lp, uni_s = self.unigram(self.uni)
        for i, name in enumerate(kinds):
            m = hit & (kind == i)
            if not bool(m.any()):
                continue
            mm = m[:, None, None]
            if name == "nolong":
                logs["ngram_long"] = torch.where(mm, logs["ngram"], logs["ngram_long"])
                stats[m, :, sl["long"]] = 0
            elif name == "nocache":                 # the document cache found nothing
                for n in ("cache", "ccache"):
                    if n in logs:
                        logs[n] = torch.where(mm, torch.full_like(logs[n], -math.log(256)), logs[n])
                        stats[m, :, sl[n]] = 0
            elif name == "short":
                for n in ("ngram", "ngram_long"):
                    if n in logs:
                        logs[n] = torch.where(mm, logs["ngram_o2"], logs[n])
                stats[m, :, 3:sl["chain"].stop] = 0
            else:
                for n in ("ngram_o2", "ngram", "ngram_long", "word"):
                    if n in logs:
                        logs[n] = torch.where(mm, uni_lp, logs[n])
                stats[m, :, 1:sl["chain"].stop] = 0
                if "word" in sl:
                    stats[m, :, sl["word"].start + 1:sl["word"].stop] = 0
                if "teacher" in sl:
                    t_lp, _ = self.unigram(self.teacher_uni)
                    logs["teacher"] = torch.where(mm, t_lp, logs["teacher"])
                    stats[m, :, sl["teacher"].start + 1:sl["teacher"].stop] = 0
        return logs, stats

    @property
    def n_entries(self):
        return sum(table_size(t) for t in (self.tabs, self.long_tabs, self.word_tabs, self.teacher_tabs))

    def to(self, device):
        mv = lambda tabs: [{k: v.to(device) for k, v in t.items()} for t in tabs]
        self.tabs, self.long_tabs = mv(self.tabs), mv(self.long_tabs)
        self.word_tabs, self.teacher_tabs = mv(self.word_tabs), mv(self.teacher_tabs)
        self.uni = self.uni.to(device)
        if self.teacher_uni is not None:
            self.teacher_uni = self.teacher_uni.to(device)
        return self

    def state(self):
        cpu = lambda tabs: [{k: v.cpu() for k, v in t.items()} for t in tabs]
        s = dict(version=2, order=self.order, long=self.long, H=self.H,
                 teacher_orders=self.teacher_orders, uni=self.uni.cpu(),
                 tabs=cpu(self.tabs), long_tabs=cpu(self.long_tabs),
                 word_tabs=cpu(self.word_tabs), teacher_tabs=cpu(self.teacher_tabs),
                 teacher_uni=None if self.teacher_uni is None else self.teacher_uni.cpu(),
                 teacher_min_p=self.teacher_min_p, n_text=self.n_text)
        if self.vocab is not None:
            s["vocab_blob"] = b"".join(self.vocab)
            s["vocab_len"] = np.array([len(w) for w in self.vocab], dtype=np.int32)
            s["classes"] = self.classes
            if self.subs is not None:
                s["subs"] = self.subs
        return s

    @classmethod
    def load(cls, path, device, vocab_only=False):
        s = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if s.get("version") != 2:
            raise SystemExit(f"{path} is from an older version of this script; rebuild it")
        hm = cls()
        if "vocab_blob" in s:
            ends = np.cumsum(s["vocab_len"])
            blob = s["vocab_blob"]
            hm.set_vocab([blob[e - n:e] for e, n in zip(ends, s["vocab_len"])], s["classes"], s.get("subs"))
        elif vocab_only:
            raise SystemExit(f"{path} has no word classes (it was built without --words)")
        if vocab_only:
            return hm
        for k in ("order", "long", "H", "teacher_orders", "uni", "tabs", "long_tabs",
                  "word_tabs", "teacher_tabs", "teacher_uni"):
            setattr(hm, k, s[k])
        hm.teacher_min_p = s.get("teacher_min_p", 0.0)
        hm.n_text = s.get("n_text")
        return hm.to(device)

    @torch.no_grad()
    def query(self, fin, wf=None, y=None, tl=None, pad=None, loo_w=None):
        """fin: (B, hist+T) bytes; the last T positions are queries ("predict what
        comes after this byte"). wf: word features, (B, T, 4) at the queries or
        (B, hist+T, 4) for the whole row (--topic).
        y: (B, T) true next bytes, only in training (leave-one-out).
        tl: teacher labels (bytes, probs) for y, only in training.
        pad: (B,) leading positions of each row that lie before the text.
        loo_w: (B,) 1 for rows whose text is in the tables, 0 for rows that are
        not (--table_bytes): only counted text is left out.
        Returns ({expert name: (B, T, 256) log-probs}, (B, T, n_stats) evidence)."""
        H = self.hist
        B, W = fin.shape
        T, dev = W - H, fin.device
        M = B * T
        col = lambda j: fin[:, H - j:H - j + T]
        stats, logs = [], {}

        def step(P, C, loo):
            if loo is not None:                       # remove this position's own count
                C.scatter_add_(1, loo[0], -loo[1])
                C.clamp_(min=0)
            tot = C.sum(1, keepdim=True)
            nd = (C > 1e-3).float().sum(1, keepdim=True)
            stats.append(tot)
            return torch.where(tot > 0, (C + nd * P) / (tot + nd).clamp(min=1e-9), P)

        lw = torch.ones(M, 1, device=dev) if loo_w is None else \
            loo_w.float()[:, None].expand(B, T).reshape(M, 1)
        cl = None if y is None else (y.reshape(M, 1), lw)
        U = torch.full((M, 256), 1.0 / 256, device=dev)

        P = step(U, self.uni.expand(M, 256).clone(), cl)
        key = torch.zeros(B, T, dtype=torch.long, device=dev)
        for k, tab in enumerate(self.tabs, start=1):
            key = key * 256 + col(k - 1)
            P = step(P, lookup(tab, key.reshape(-1)), cl)
            if k == 2 and self.order > 2:
                logs["ngram_o2"] = P
        logs["ngram"] = P
        if self.long_tabs:
            h = torch.zeros(B, T, dtype=torch.long, device=dev)
            it = iter(self.long_tabs)
            for j in range(self.long[-1]):
                h = h * HASH_MUL + (col(j) + 1)
                if j + 1 in self.long:
                    P = step(P, lookup(next(it), (h & MASK55).reshape(-1)), cl)
            logs["ngram_long"] = P

        if self.word_tabs:
            wfl = wf[:, -T:].long()
            keys = word_keys(col, wfl[..., 0], wfl[..., 1], wfl[..., 2],
                             wfl[..., 3] if len(self.word_tabs) == 3 else None)
            Pw = step(U, self.uni.expand(M, 256).clone(), cl)
            for tab, k in zip(self.word_tabs, keys):
                Pw = step(Pw, lookup(tab, k.reshape(-1)), cl)
            logs["word"] = Pw

        if self.teacher_tabs:
            tlo = None
            if tl is not None:        # leave-one-out with exactly what was counted
                tp_ = tl[1].reshape(M, 8)
                tlo = (tl[0].reshape(M, 8), torch.where(tp_ >= self.teacher_min_p, tp_, 0.0) * lw)
            Pt = step(U, self.teacher_uni.expand(M, 256).clone(), tlo)
            key, done = torch.zeros(B, T, dtype=torch.long, device=dev), 0
            for k, tab in zip(self.teacher_orders, self.teacher_tabs):
                while done < k:
                    key = key * 256 + col(done)
                    done += 1
                Pt = step(Pt, lookup(tab, key.reshape(-1)), tlo)
            logs["teacher"] = Pt

        if self.cache_orders or self.topic or self.class_cache:
            pos = torch.arange(W, device=dev)
            padv = pad if pad is not None else torch.zeros(B, dtype=torch.long, device=dev)
            valid = pos[None, :] >= padv[:, None]                       # (B, W) inside the text
            nxt = torch.cat([fin[:, 1:], torch.zeros(B, 2, dtype=torch.long, device=dev)], 1)
            q0 = torch.arange(H, W, device=dev).expand(B, T)

            def earlier(key):
                """For every predicted byte: what followed the same key at its most
                recent earlier positions in this row -> counts (B, T, 256). With
                --cache_decay, an occurrence d bytes back counts exp(-d / decay)."""
                sv, order = torch.sort(key, dim=1, stable=True)         # same key: side by side,
                prev_s = torch.where(sv[:, 1:] == sv[:, :-1], order[:, :-1],        # earlier first
                                     torch.full_like(order[:, 1:], W))
                prev = torch.full((B, W + 1), W, dtype=torch.long, device=dev)
                prev.scatter_(1, order[:, 1:], prev_s)                  # last earlier position, or W
                C, p = torch.zeros(B, T, 256, device=dev), q0
                for _ in range(self.cache_keep):                        # the most recent occurrences
                    p = prev.gather(1, p)
                    w = (p < W).float()
                    if self.cache_decay > 0:
                        w = w * torch.exp(-(q0 - p).clamp(min=0).float() / self.cache_decay)
                    C.scatter_add_(2, nxt.gather(1, p)[..., None], w[..., None])
                return C

        if self.cache_orders:
            # Document cache: what followed the same context earlier in THIS text
            # (the hist bytes before the window and the window so far). Only
            # earlier positions are used, so nothing has to be left out.
            base = torch.zeros(B, 256, device=dev).scatter_add_(1, fin[:, :H], valid[:, :H].float())
            C0 = base[:, None, :] + F.one_hot(fin[:, H:], 256).float().cumsum(1)
            Pc = step(U, C0.reshape(M, 256), None)                      # bytes of the text so far
            h, done = torch.zeros(B, W, dtype=torch.long, device=dev), 0
            for k in self.cache_orders:
                while done < k:                                         # hash of the last k bytes
                    h = h * HASH_MUL + (F.pad(fin, (done, 0))[:, :W] + 1)
                    done += 1
                inside = pos[None, :] >= padv[:, None] + k - 1
                key = torch.where(inside, h & MASK55, -(pos + 1)[None, :].expand(B, W))
                Pc = step(Pc, earlier(key).reshape(M, 256), None)
            logs["cache"] = Pc

        if self.class_cache:
            # Class cache: like the document cache, but the context is the KIND of
            # the previous one or two words (their classes) plus the current word so
            # far, not the exact bytes. "after an animal word, sl -> e" can then be
            # learned from "the cat slept" and used for "a dog sle".
            wfa = wf.long()
            ks = word_keys(lambda j: F.pad(fin, (j, 0))[:, :W], wfa[..., 0], wfa[..., 1], wfa[..., 2])
            inside = pos[None, :] >= padv[:, None] + 3
            Pk = U
            for k in ks:
                key = torch.where(inside, k, -(pos + 1)[None, :].expand(B, W))
                Pk = step(Pk, earlier(key).reshape(M, 256), None)
            logs["ccache"] = Pk

        if self.topic:
            # Topic: how often each word class occurred in the text so far. Not a
            # prediction, just 256 extra numbers about what the text is about.
            wfa = wf.long()
            pl, c1 = wfa[..., 2], wfa[..., 0]
            ended = (pl == 0) & (F.pad(pl, (1, 0))[:, :W] > 0) & valid   # a word just ended here
            base = torch.zeros(B, 256, device=dev).scatter_add_(1, c1[:, :H], ended[:, :H].float())
            Ct = base[:, None, :] + (F.one_hot(c1[:, H:], 256).float() * ended[:, H:, None]).cumsum(1)
            stats.append(Ct.reshape(M, 256))

        logs = {k: v.clamp(min=1e-12).log().reshape(B, T, 256) for k, v in logs.items()}
        return logs, torch.log1p(torch.cat(stats, 1)).reshape(B, T, -1)


# ------------------------------------------------------ teacher labels
class TeacherLabels:
    """Top-8 next-byte probabilities from a teacher LLM for the first L bytes
    of the training split (made by label_teacher.py)."""

    def __init__(self, path, train):
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        self.L = int(meta["n_bytes"])
        if self.L > len(train) or hashlib.sha1(train[:self.L].tobytes()).hexdigest() != meta["sha1"]:
            raise SystemExit(f"{path} was made for different data. Use the same --data and "
                             f"--max_bytes as label_teacher.py did.")
        # into RAM (3 bytes x 8 per labelled byte): reading them from disk slowed training 4x
        self.tb = np.load(os.path.join(path, "top_bytes.npy"))
        self.tp = np.load(os.path.join(path, "top_probs.npy"))
        self.meta = meta
        cover = float((np.asarray(self.tp[:, 0]) > 0).mean())
        print(f"teacher: {meta.get('model')} | {self.L:,} bytes | {cover:.0%} labelled | "
              f"teacher BPC on them {meta.get('teacher_bpc', float('nan')):.3f}")

    def get(self, starts, offset, T, device):
        B = len(starts)
        tb = np.zeros((B, T, 8), dtype=np.int64)
        tp = np.zeros((B, T, 8), dtype=np.float32)
        for r, s in enumerate(starts):
            a = s + offset
            b = min(a + T, self.L)
            if b > a:
                tb[r, :b - a] = self.tb[a:b]
                tp[r, :b - a] = self.tp[a:b]
        return torch.from_numpy(tb).to(device), torch.from_numpy(tp).to(device)


# ----------------------------------------------------------- transformer
def rope_table(T, dh, device):
    """Angles for rotary positions: cos and sin, (T, dh/2) each."""
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=device).float() / dh))
    ang = torch.arange(T, device=device).float()[:, None] * inv[None, :]
    return ang.cos(), ang.sin()


def rope(t, cs):
    """Rotate pairs of dimensions of (B, heads, T, dh) by the position's angle.
    Attention then depends on how far apart two bytes are, not on where they
    sit in the window."""
    cos, sin = cs[0].to(t.dtype), cs[1].to(t.dtype)
    a, b = t[..., 0::2], t[..., 1::2]
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2)


class RMSNorm(nn.Module):
    """Scale to unit root-mean-square, without centering (--arch modern)."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class SwiGLU(nn.Module):
    """Gated MLP with about the same number of weights as the GELU one (--arch modern)."""
    def __init__(self, d):
        super().__init__()
        h = 64 * ((int(8 * d / 3) + 63) // 64)
        self.w1, self.w2, self.w3 = (nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False),
                                     nn.Linear(h, d, bias=False))

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, d, H, dropout, modern=False):
        super().__init__()
        self.H = H
        norm = RMSNorm if modern else nn.LayerNorm
        self.ln1, self.ln2 = norm(d), norm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=not modern), nn.Linear(d, d, bias=not modern)
        self.mlp = SwiGLU(d) if modern else \
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, h, cs=None):
        B, T, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).split(d, dim=2)
        q, k, v = (t.view(B, T, self.H, d // self.H).transpose(1, 2) for t in (q, k, v))
        if cs is not None:
            q, k = rope(q, cs), rope(k, cs)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        h = h + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, d)))
        return h + self.drop(self.mlp(self.ln2(h)))


class GPT(nn.Module):
    def __init__(self, T, d, L, H, dropout, n_extra, rope=False, modern=False, tie=False):
        super().__init__()
        self.tok = nn.Embedding(256, d)
        self.pos = None if rope else nn.Embedding(T, d)       # --rope: positions live in attention
        self.rope_dh = d // H if rope else 0
        self.extra = nn.Linear(n_extra, d) if n_extra else None
        self.blocks = nn.ModuleList(Block(d, H, dropout, modern) for _ in range(L))
        self.ln = RMSNorm(d) if modern else nn.LayerNorm(d)
        self.head = nn.Linear(d, 256, bias=not modern)
        if tie:                                   # --tie: output weights = byte embeddings
            nn.init.normal_(self.tok.weight, std=0.02)
            self.head.weight = self.tok.weight

    def forward(self, x, extra=None):
        h = self.tok(x)
        if self.pos is not None:
            h = h + self.pos(torch.arange(x.shape[1], device=x.device))
        if self.extra is not None:
            h = h + self.extra(extra)
        cs = rope_table(x.shape[1], self.rope_dh, x.device) if self.rope_dh else None
        for b in self.blocks:
            h = b(h, cs)
        return self.head(self.ln(h))


class Model(nn.Module):
    def __init__(self, gpt, mode, names, main, use_feat):
        super().__init__()
        self.gpt, self.mode, self.names, self.use_feat = gpt, mode, list(names), use_feat
        if mode == "residual":
            if gpt.head.weight is not gpt.tok.weight:      # with --tie this would erase the embeddings
                nn.init.zeros_(gpt.head.weight)
            if gpt.head.bias is not None:                  # --arch modern has no bias
                nn.init.zeros_(gpt.head.bias)
            # how much of each expert to add; starts as "main expert only"
            self.mix = nn.Parameter(torch.tensor([1.0 if n == main else 0.0 for n in self.names]))

    def forward(self, x, hard=None):
        extra = None
        if self.use_feat:
            logs, stats = hard
            extra = torch.cat([logs[n].clamp(min=-16) / 16 for n in self.names] + [stats / 10], -1)
        out = self.gpt(x, extra)
        if self.mode == "residual":
            logs = hard[0]
            out = out + sum(self.mix[i] * logs[n] for i, n in enumerate(self.names))
        return out


def build_model(a, hm):
    use_feat = bool(a.hard_feature and hm is not None)
    n_extra = 256 * len(hm.names) + hm.n_stats if use_feat else 0
    if getattr(a, "rope", False) and (a.d // a.heads) % 2:
        raise SystemExit("--rope needs an even --d / --heads")
    gpt = GPT(a.ctx, a.d, a.layers, a.heads, a.dropout, n_extra, rope=getattr(a, "rope", False),
              modern=getattr(a, "arch", "classic") == "modern", tie=getattr(a, "tie", False))
    names, main = (hm.names, hm.main) if hm is not None else ([], None)
    return Model(gpt, a.mode, names, main, use_feat)


def fit_memory_input(sd, old_args, model, hm):
    """--init from a model trained without --cache / --topic: the weights of the
    new memory inputs are set to zero, so training starts exactly at the old
    model and then learns to use them. Returns the names of the new inputs."""
    key = "gpt.extra.weight"
    if hm is None or model.gpt.extra is None or key not in sd or \
            sd[key].shape == model.gpt.extra.weight.shape:
        return []
    oc = str(old_args.get("cache_orders", "")) if old_args.get("cache") else ""
    old = hm.input_groups(len([x for x in oc.split(",") if x.strip()]), bool(old_args.get("topic")),
                          bool(int(old_args.get("class_cache", 0))))
    new = hm.input_groups()
    if sum(w for _, w in old) != sd[key].shape[1] or not set(old) <= set(new):
        return []                                  # not just added inputs: load_state_dict reports it
    off = lambda groups: {n: (sum(w for _, w in groups[:i]), w) for i, (n, w) in enumerate(groups)}
    o_off, n_off = off(old), off(new)
    W = torch.zeros(model.gpt.extra.weight.shape, dtype=sd[key].dtype, device=sd[key].device)
    for n, (a, w) in o_off.items():
        W[:, n_off[n][0]:n_off[n][0] + w] = sd[key][:, a:a + w]
    sd[key] = W
    return [n.split(":")[1] + (" counts" if n.startswith("stat") else "") for n, _ in new if n not in o_off]


def safe_save(obj, path, tries=3):
    """Write to a temp file, then swap it in. Retries instead of crashing when
    Windows briefly locks the file (antivirus, sync tools) or a write fails."""
    tmp = path + ".tmp"
    for i in range(tries):
        try:
            torch.save(obj, tmp)
            os.replace(tmp, path)
            return True
        except (RuntimeError, OSError) as e:
            try:
                free = shutil.disk_usage(os.path.dirname(os.path.abspath(path))).free / 2 ** 30
                if os.path.exists(tmp):
                    os.remove(tmp)                 # a half-written file only takes space
            except OSError:
                free = float("nan")
            why = f"DISK ALMOST FULL: {free:.1f} GB free" if free < 2 else f"{free:.0f} GB free on the disk"
            print(f"  warning: saving {path} failed ({str(e)[:80]}; {why}); "
                  f"{'retrying' if i < tries - 1 else 'skipped'}", flush=True)
            time.sleep(5)
    return False


# ------------------------------------------------------------- run
def split(buf, H):
    return buf[:, H:-1], buf[:, H + 1:], buf[:, :-1]   # x, y, input for the hard model


@torch.no_grad()
def evaluate(model, hm, V, args, skip=0):
    """Val BPC on fixed windows. With model=None: BPC of every hard expert.
    skip: do not score the first `skip` bytes of each window, where the
    transformer has seen almost nothing yet ("warm" score)."""
    if model is not None:
        model.eval()
    H, T = V["H"], args.ctx
    tot, n = {}, 0
    for i in range(0, len(V["starts"]), args.batch):
        st = V["starts"][i:i + args.batch]
        x, y, fin = split(gather(V["data"], st, H + T + 1).to(args.device).long(), H)
        wf = gather(V["wf"], st, *hm.wf_span(T)).to(args.device) if V["wf"] is not None else None
        hard = hm.query(fin, wf, pad=pad_of(st, args.device)) if hm is not None else None
        if model is None:
            cands = hard[0]
        else:
            with torch.autocast(args.device_type, dtype=torch.bfloat16, enabled=args.amp):
                cands = {"model": model(x, hard)}
        for k, lg in cands.items():
            tot[k] = tot.get(k, 0.0) + F.cross_entropy(
                lg.float()[:, skip:].reshape(-1, 256), y[:, skip:].reshape(-1), reduction="sum").item()
        n += y[:, skip:].numel()
    if model is not None:
        model.train()
    return {k: v / n / LN2 for k, v in tot.items()}


def slide_windows(n_bytes, H, T):
    """Windows that score EVERY byte once (all but the very first), each with at
    least T/2 bytes of context: windows overlap by half, and only the second
    half of each window counts. Returns (starts, bytes to skip per window)."""
    half = T // 2
    starts, skips, s = [], [], -H                  # the first window scores bytes 1..T
    while s + H + T + 1 <= n_bytes:
        starts.append(s)
        skips.append(half if starts[:-1] else 0)
        s += half
    tail = n_bytes - (H + T + 1)                   # one last window that ends at the last byte
    if starts and tail > starts[-1]:
        skips.append(starts[-1] + T - tail)
        starts.append(tail)
    return starts, skips


@torch.no_grad()
def evaluate_detail(model, hm, V, args):
    """One pass over the windows of V with the transformer. Bits per byte: "bpc"
    over all scored bytes, "warm" on the second half of each window, "word_start"
    on first letters of words, "rest" on all other bytes; "acc" is the share of
    bytes whose most likely guess was right. V["skip"]: bytes not to score at
    the start of each window (sliding evaluation)."""
    was_training = model.training
    model.eval()
    H, T = V["H"], args.ctx
    wm = torch.from_numpy(WORD_MASK).to(args.device)
    pos = torch.arange(T, device=args.device)
    S = dict.fromkeys(("bits", "n", "warm", "n_warm", "ws", "n_ws", "hit"), 0.0)
    for i in range(0, len(V["starts"]), args.batch):
        st = V["starts"][i:i + args.batch]
        x, y, fin = split(gather(V["data"], st, H + T + 1).to(args.device).long(), H)
        wf = gather(V["wf"], st, *hm.wf_span(T)).to(args.device) if V["wf"] is not None else None
        hard = hm.query(fin, wf, pad=pad_of(st, args.device)) if hm is not None else None
        with torch.autocast(args.device_type, dtype=torch.bfloat16, enabled=args.amp):
            lg = model(x, hard).float()
        b = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1), reduction="none").reshape(y.shape) / LN2
        keep = torch.ones_like(y, dtype=torch.bool)
        if V.get("skip") is not None:
            keep = pos[None, :] >= torch.tensor(V["skip"][i:i + args.batch], device=args.device)[:, None]
        warm = keep & (pos[None, :] >= T // 2)
        ws = keep & wm[y] & ~wm[x]                 # a letter after a non-letter
        S["bits"] += float(b[keep].sum())
        S["n"] += float(keep.sum())
        S["warm"] += float(b[warm].sum())
        S["n_warm"] += float(warm.sum())
        S["ws"] += float(b[ws].sum())
        S["n_ws"] += float(ws.sum())
        S["hit"] += float((lg.argmax(-1) == y)[keep].sum())
    if was_training:
        model.train()
    return dict(bpc=S["bits"] / S["n"], warm=S["warm"] / max(1.0, S["n_warm"]),
                word_start=S["ws"] / max(1.0, S["n_ws"]),
                rest=(S["bits"] - S["ws"]) / max(1.0, S["n"] - S["n_ws"]),
                acc=S["hit"] / S["n"], bytes=int(S["n"]), word_start_share=S["n_ws"] / S["n"])


def dyn_grad_stats(model, hm, params, args, T):
    """Mean squared gradient of every weight, over --dyn_stats_windows single windows
    of the training text (--data): how big a normal step is for each weight. The
    tables are used with leave-one-out, as in training. Scores nothing."""
    if not args.data or not os.path.exists(args.data):
        raise SystemExit("--dyn_mode rms needs --data (the training text, for the step sizes)")
    raw = np.memmap(args.data, dtype=np.uint8, mode="r")     # only the windows are read
    H = filler(hm)
    rng = np.random.default_rng(0)
    st = rng.integers(0, len(raw) - (H + T + 1), args.dyn_stats_windows)
    ms = [torch.zeros_like(p) for p in params]
    t0 = time.time()
    for s0 in st.tolist():
        piece = np.array(raw[s0:s0 + H + T + 1])
        buf = torch.from_numpy(piece.astype(np.int64))[None].to(args.device)
        x, y, fin = split(buf, H)
        wf = None
        if hm is not None and hm.word_tabs:
            f, _ = word_features(piece[:-1], hm.cls_of)
            wf = torch.from_numpy(f[-(fin.shape[1] if hm.wf_full else x.shape[1]):]).to(args.device)[None]
        with torch.no_grad():
            hard = hm.query(fin, wf, y) if hm is not None else None
        with torch.autocast(args.device_type, dtype=torch.bfloat16, enabled=args.amp):
            lg = model(x, hard).float()
        model.zero_grad(set_to_none=True)
        F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1)).backward()
        with torch.no_grad():
            for m, p in zip(ms, params):
                if p.grad is not None:
                    m.add_(p.grad.float() ** 2)
    model.zero_grad(set_to_none=True)
    print(f"    step sizes from {len(st)} training windows ({time.time() - t0:.0f}s)", flush=True)
    return [m / len(st) for m in ms]


def evaluate_dynamic(model, hm, V, args):
    """Dynamic evaluation: the text is scored in reading order, and after each
    window has been scored the model takes one small training step on it. Only
    text that was already scored is ever learned from, so every byte is still
    predicted before it is seen. --dyn_decay pulls the weights back toward the
    trained ones, so the model follows the current text without drifting away.
    Returns bits per byte over the same bytes as the sliding evaluation."""
    model.eval()
    H, T = V["H"], args.ctx
    params = [p for p in model.parameters() if p.requires_grad]
    start = [p.detach().clone() for p in params]
    rms = args.dyn_mode == "rms"
    if rms:                                        # step size per weight from training text
        denom = [d.sqrt_().add_(1e-8) for d in dyn_grad_stats(model, hm, params, args, T)]
    else:
        opt = torch.optim.Adam(params, lr=args.dyn_lr, betas=(0.9, 0.99))
    pos = torch.arange(T, device=args.device)
    bits, n, t0, B = 0.0, 0.0, time.time(), max(1, args.dyn_batch)
    for i in range(0, len(V["starts"]), B):
        st = V["starts"][i:i + B]
        x, y, fin = split(gather(V["data"], st, H + T + 1).to(args.device).long(), H)
        wf = gather(V["wf"], st, *hm.wf_span(T)).to(args.device) if V["wf"] is not None else None
        with torch.no_grad():
            hard = hm.query(fin, wf, pad=pad_of(st, args.device)) if hm is not None else None
        with torch.autocast(args.device_type, dtype=torch.bfloat16, enabled=args.amp):
            lg = model(x, hard).float()
        b = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1), reduction="none").reshape(y.shape)
        keep = pos[None, :] >= torch.tensor(V["skip"][i:i + B], device=args.device)[:, None]
        bits += float(b.detach()[keep].sum()) / LN2
        n += float(keep.sum())
        model.zero_grad(set_to_none=True)
        b[keep].mean().backward()                  # learn from the bytes just scored (each once)
        if rms:
            with torch.no_grad():
                for p, d in zip(params, denom):
                    if p.grad is not None:
                        p.sub_((p.grad / d).clamp_(-10, 10), alpha=args.dyn_lr)
        else:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        if args.dyn_decay:
            with torch.no_grad():
                for p, p0 in zip(params, start):
                    p.lerp_(p0, args.dyn_decay)
        if (i // B) % 500 == 0 and i:
            done = (i + B) / len(V["starts"])
            print(f"    dynamic: {done:5.1%}  BPC so far {bits / n:.4f}  "
                  f"({(time.time() - t0) / done * (1 - done) / 60:.0f} min left)", flush=True)
    with torch.no_grad():                          # leave the model as it was trained
        for p, p0 in zip(params, start):
            p.copy_(p0)
    return bits / n


def filler(hm):
    return hm.hist if hm is not None else 1


@torch.no_grad()
def score_text(model, hm, text, T, args):
    """Bits for every byte of `text` (bytes) after its first, each predicted from
    the bytes before it. Used by the knowledge probe."""
    H = filler(hm)
    b = list(text)[-(T + 1):]
    fin = torch.tensor([32] * H + b[:-1], device=args.device)[None]
    x, y = fin[:, H:], torch.tensor(b[1:], device=args.device)[None]
    hard = None
    if hm is not None:
        wf = None
        if hm.word_tabs:
            full, _ = word_features(np.array([32] * H + b[:-1], dtype=np.uint8), hm.cls_of)
            wf = torch.from_numpy(full[-(fin.shape[1] if hm.wf_full else x.shape[1]):]).to(args.device)[None]
        hard = hm.query(fin, wf, pad=torch.tensor([H], device=args.device))
    if model is None:
        lg = hard[0]["ngram_long" if hm.long_tabs else "ngram"]
    else:
        with torch.autocast(args.device_type, dtype=torch.bfloat16, enabled=args.amp):
            lg = model(x, hard)
    return F.cross_entropy(lg.float()[0], y[0], reduction="none") / LN2


def load_probe(path):
    """[{"prompt": "...", "answers": ["right", "wrong", ...]}, ...]: the first answer is the right one."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_prompts(path):
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n").replace("\\n", "\n") for l in f if l.strip() and not l.startswith("#")]


@torch.no_grad()
def run_probe(model, hm, items, T, args):
    """Knowledge probe: for each prompt, which of the answers costs the fewest
    bits? Returns (share answered right, details per item)."""
    was_training = model is not None and model.training
    if model is not None:
        model.eval()
    rows, right = [], 0
    for it in items:
        costs = []
        for a in it["answers"]:
            ab = a.encode("utf-8")
            costs.append(float(score_text(model, hm, it["prompt"].encode("utf-8") + ab, T, args)[-len(ab):].sum()))
        best = int(np.argmin(costs))
        right += best == 0
        rows.append(dict(prompt=it["prompt"], right=it["answers"][0], chosen=it["answers"][best],
                         ok=best == 0, bits=[round(c, 2) for c in costs]))
    if was_training:
        model.train()
    return right / max(1, len(items)), rows


@torch.no_grad()
def track(model, hm, step, vb, T, args, path, prompts, probe):
    """Paper trail during training: probe score and sample texts for fixed
    prompts, appended to <tag>_samples.txt. Returns the probe score (or None)."""
    acc = None
    was_training = model.training
    model.eval()
    text = [f"\n=== step {step} | val BPC {vb:.4f} ===\n"]
    if probe:
        acc, rows = run_probe(model, hm, probe, T, args)
        wrong = [r for r in rows if not r["ok"]]
        text.append(f"probe: {acc:.1%} of {len(rows)} right" +
                    ("; wrong: " + " | ".join(f"{r['prompt'][-40:]!r} -> {r['chosen']!r}" for r in wrong[:12])
                     if wrong else "") + "\n")
    g = torch.Generator(device=args.device).manual_seed(1234)         # same dice at every step
    for pr in prompts:
        out = sample(model, hm, pr, args.sample_len, T, args, args.temperature, gen=g)
        text.append(f"--- {pr!r}\n{out}\n")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("".join(text))
    except OSError as e:                       # e.g. disk full: training goes on
        print(f"  warning: could not write {path} ({e})", flush=True)
    if was_training:
        model.train()
    return acc


@torch.no_grad()
def check_loo(hm, train, n_tab, V, args, n=512):
    """Do the tables look the same on training text as on unseen text? Scores
    every expert on random training windows with the byte's own count left out,
    as in training. Leave-one-out removes that one count, but not the other
    places where the same document repeats itself, so on counted text the
    tables can look better than they will on new text. A large gap means the
    transformer learns to trust them too much: train it on text the tables do
    not cover (--table_bytes with --train_part rest)."""
    H, T = V["H"], args.ctx
    span = H + T + 1
    g = np.random.default_rng(0)
    parts = [("counted text, own byte left out", 0, n_tab or len(train), 1.0)]
    if n_tab:
        parts.append(("training text NOT in the tables", n_tab, len(train), 0.0))
    val = evaluate(None, hm, V, args)
    print(f"  {'':34s}" + "".join(f"{k:>12s}" for k in val))
    print(f"  {'validation text':34s}" + "".join(f"{v:12.4f}" for v in val.values()))
    tt = torch.from_numpy(train)
    for name, lo, hi, w in parts:
        starts = (lo + g.integers(0, hi - lo - span, n)).tolist()
        tot, cnt = {}, 0
        for i in range(0, n, args.batch):
            st = starts[i:i + args.batch]
            x, y, fin = split(gather(tt, st, span).to(args.device).long(), H)
            wf = None
            if hm.word_tabs:                   # word features of just these rows (2 KB lead-in)
                L, off = hm.wf_span(T)
                rows = []
                for s0 in st:
                    a = max(0, s0 + off - 2048)
                    rows.append(torch.from_numpy(word_features(train[a:s0 + off + L], hm.cls_of)[0][-L:]))
                wf = torch.stack(rows).to(args.device)
            logs, _ = hm.query(fin, wf, y, loo_w=torch.full((len(st),), w, device=args.device))
            for k, lg in logs.items():
                tot[k] = tot.get(k, 0.0) + F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1),
                                                           reduction="sum").item()
            cnt += y.numel()
        res = {k: v / cnt / LN2 for k, v in tot.items()}
        print(f"  {name:34s}" + "".join(f"{res[k]:12.4f}" for k in val))
        print(f"  {'  minus validation':34s}" + "".join(f"{res[k] - val[k]:+12.4f}" for k in val))


@torch.no_grad()
def cache_mix(hm, V, args):
    """How much the document cache adds to the tables, measured without any
    training: BPC of fixed blends of the best table expert and the cache. A
    transformer that reads both should do at least as well as the best blend."""
    best = "ngram_long" if hm.long_tabs else "ngram"
    H, T = V["H"], args.ctx
    tot, n = {}, 0
    for i in range(0, len(V["starts"]), args.batch):
        st = V["starts"][i:i + args.batch]
        x, y, fin = split(gather(V["data"], st, H + T + 1).to(args.device).long(), H)
        wf = gather(V["wf"], st, *hm.wf_span(T)).to(args.device) if V["wf"] is not None else None
        logs, _ = hm.query(fin, wf, pad=pad_of(st, args.device))
        pt, pc = logs[best].exp(), logs["cache"].exp()
        for lam in (0.0, 0.1, 0.2, 0.3):
            p = ((1 - lam) * pt + lam * pc).gather(-1, y[..., None]).clamp(min=1e-12)
            tot[lam] = tot.get(lam, 0.0) + float(-torch.log2(p).sum())
        n += y.numel()
    base = tot[0.0] / n
    return {f"{best} blended with {lam:.0%} cache: val BPC {tot[lam] / n:.4f}  ({tot[lam] / n - base:+.4f})": 0
            for lam in (0.1, 0.2, 0.3)}


@torch.no_grad()
def sample(model, hm, prompt, n, T, args, temperature=0.8, gen=None):
    H = hm.hist if hm is not None else 1
    buf = list(b" " * H + prompt.encode("utf-8", "replace"))
    for _ in range(n):
        fin = torch.tensor(buf[-(T + H):], device=args.device)[None]
        x = fin[:, H:]                         # the transformer sees the last <= T bytes
        hard = None
        if hm is not None:
            wf = None
            if hm.word_tabs:
                full, _ = word_features(np.frombuffer(bytes(buf), dtype=np.uint8), hm.cls_of)
                wf = torch.from_numpy(full[-(fin.shape[1] if hm.wf_full else x.shape[1]):]).to(args.device)[None]
            lead = max(0, H - max(0, len(buf) - H - T))      # filler before the prompt, still in view
            hard = hm.query(fin, wf, pad=torch.tensor([lead], device=args.device))
        if model is None:
            logits = hard[0][hm.main][0, -1]
        else:
            logits = model(x, hard)[0, -1].float()
        buf.append(int(torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=gen)))
    return bytes(buf[H:]).decode("utf-8", "replace")


def generate(args):
    """Load a saved model (plus its hard model) and write text."""
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    hm = None
    if ck.get("hard_path"):
        hm = HardModel.load(os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                         ck["hard_path"]), args.device).set_runtime(a)
    model = None
    if ck.get("model") is not None:
        a.dropout = 0.0
        model = build_model(a, hm)
        model.load_state_dict(ck["model"])
        model.to(args.device).eval()
    print(f"loaded {args.ckpt}: mode {a.mode}, step {ck.get('step', '-')}, "
          f"val BPC {ck.get('val_bpc', float('nan')):.4f}")

    def run(prompt):
        return sample(model, hm, prompt, args.sample, a.ctx, args, args.temperature)

    if not args.interactive:
        print(run(args.prompt.replace("\\n", "\n")))
        return
    print("Type a prompt (\\n = new line). Empty line quits.")
    while True:
        p = input("> ")
        if not p:
            break
        print(run(p.replace("\\n", "\n")), "\n")


def split_test(train, n_mb, block=1 << 20):
    """Take n_mb blocks of 1 MB, evenly spread over the second half of the
    training text, out of it. They become a test set no table, training step or
    checkpoint choice ever sees. The second half is used so that models trained
    on the first 200 MB of a larger file have not seen the test text either."""
    half = len(train) // 2
    nb = (len(train) - half) // block
    if nb < n_mb:
        raise SystemExit(f"--test_mb {n_mb}: only {nb} MB available in the second half of --data")
    idx = np.unique(np.linspace(0, nb - 1, n_mb).round().astype(np.int64))
    mask = np.zeros(len(train), dtype=bool)
    for i in idx:
        mask[half + i * block:half + (i + 1) * block] = True
    print(f"test set: {len(idx)} blocks of 1 MB from the second half of the training text, removed from it")
    return train[~mask], train[mask]


def load_data(args):
    """Train, val and test text. With --val_data, validation comes from that
    file's own split, and all of --data minus that validation text is training
    text. With --test_mb, a test set is also taken out (and written to --out)."""
    if args.mode == "eval" or (args.mode == "hard" and args.load_hard and not args.check_loo):
        if args.test_file or args.val_file:    # scoring only: no training text needed
            return None, np.fromfile(args.test_file or args.val_file, dtype=np.uint8), None
        return None, load_bytes(args.val_data or args.data, 0 if args.val_data else args.max_bytes)[1], None
    if args.val_file:                          # a benchmark with its own files: nothing is split off
        train, _ = load_bytes(args.data, args.max_bytes, block=0)
        val = np.fromfile(args.val_file, dtype=np.uint8)
        test = np.fromfile(args.test_file, dtype=np.uint8) if args.test_file else None
        return train, val, test
    if args.val_data:
        _, val = load_bytes(args.val_data, 0)
        train, _ = load_bytes(args.data, args.max_bytes, block=0)
        train = remove_text(train, val)
    else:
        train, val = load_bytes(args.data, args.max_bytes)
    test = None
    if args.test_mb:
        train, test = split_test(train, args.test_mb)
        path = os.path.join(args.out, f"{os.path.splitext(os.path.basename(os.path.normpath(args.data)))[0]}"
                                      f"_test{args.test_mb}mb.txt")
        if not os.path.exists(path) or os.path.getsize(path) != len(test):
            test.tofile(path)
        print(f"test text -> {path}  (score models on it with --mode eval --test_file)")
    return train, val, test


def evaluate_ckpt(args, val):
    """--mode eval: a trained model with any hard file (--load_hard), no training."""
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    path = args.load_hard or (ck.get("hard_path") and
                              os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), ck["hard_path"]))
    hm = HardModel.load(path, args.device).set_runtime(a) if path else None
    print(f"model {args.ckpt} (trained to step {ck.get('step')}) with hard file {path}")
    a.dropout = 0.0
    model = build_model(a, hm)
    try:
        model.load_state_dict(ck["model"])
    except RuntimeError:
        raise SystemExit("this hard file has different experts than the one the model was trained "
                         "with (build it with the same --order/--long_orders/--words/--teacher)")
    model.to(args.device)
    if args.eval_bytes:
        val = val[:args.eval_bytes]
    H = hm.hist if hm is not None else 1
    wf_val = torch.from_numpy(word_features(val, hm.cls_of)[0]) if hm is not None and hm.word_tabs else None
    V = dict(data=torch.from_numpy(val), wf=wf_val, H=H,
             starts=val_starts(len(val), H, a.ctx, args.eval_windows, args.eval_layout))
    args.ctx = a.ctx
    if hm is not None:
        for k, v in evaluate(None, hm, V, args).items():
            print(f"  hard expert {k:11s} val BPC {v:.4f}")
    what = f"test BPC ({args.test_file})" if args.test_file else "val BPC"
    if args.slide_eval:
        V["starts"], V["skip"] = slide_windows(len(val), H, a.ctx)
    d = evaluate_detail(model, hm, V, args)
    res = dict(ckpt=args.ckpt, hard=path, step=ck.get("step"),
               text=args.test_file or args.val_file or args.val_data or args.data,
               windows=len(V["starts"]), bpc=d["bpc"], bpc_warm=d["warm"], acc=d["acc"],
               word_start=d["word_start"], rest=d["rest"], bytes=d["bytes"])
    print(f"  MODEL               {what} {res['bpc']:.4f}   "
          f"({'every byte' if args.slide_eval else str(len(V['starts'])) + ' windows, ' + args.eval_layout}) | "
          f"with at least {a.ctx // 2} bytes of context: {res['bpc_warm']:.4f} | next byte right "
          f"{d['acc']:.1%} | first letters of words {d['word_start']:.3f}")
    if args.dyn_lr:
        if not args.slide_eval:
            V["starts"], V["skip"] = slide_windows(len(val), H, a.ctx)
            res["bpc_slide"] = evaluate_detail(model, hm, V, args)["bpc"]
        res["bpc_dyn"] = evaluate_dynamic(model, hm, V, args)
        res["dyn"] = dict(lr=args.dyn_lr, decay=args.dyn_decay, batch=args.dyn_batch, mode=args.dyn_mode)
        print(f"  MODEL, learning as it reads (dynamic evaluation, {args.dyn_mode}, lr {args.dyn_lr:g}, decay "
              f"{args.dyn_decay:g}): {res['bpc_dyn']:.4f}   (without: {res.get('bpc_slide', res['bpc']):.4f})")
    if args.probe:
        res["probe"], rows = run_probe(model, hm, load_probe(args.probe), a.ctx, args)
        res["probe_rows"] = rows
        print(f"  knowledge probe     {res['probe']:.1%} of {len(rows)} right")
    if args.tag:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, f"{args.tag}_eval.json"), "w") as jf:
            json.dump(res, jf, indent=1)
    if args.per_block:                 # how much the score depends on which text is measured
        block = 1 << 20
        print("  per 1 MB validation block:  model" + ("   " + hm.main if hm is not None else ""))
        for b in range(0, len(val) // block):
            Vb = dict(V, starts=val_starts(len(val), H, a.ctx, 64, "spread", b * block, (b + 1) * block))
            line = f"    block {b:3d}            {evaluate(model, hm, Vb, args)['model']:.4f}"
            if hm is not None:
                line += f"   {evaluate(None, hm, Vb, args)[hm.main]:.4f}"
            print(line)


class Tracker:
    """Optional live dashboard with trackio (--trackio PROJECT). Local by default
    (look at it with "trackio show"); with --trackio_space user/name it is also
    sent to a Hugging Face Space. A tracking problem never stops a run."""

    def __init__(self, args, tag):
        self.t = None
        if not args.trackio:
            return
        try:
            import trackio
            cfg = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool))}
            trackio.init(project=args.trackio, name=tag, config=cfg, resume="allow",
                         space_id=args.trackio_space or None)
            self.t = trackio
        except Exception as e:
            print(f"  trackio is off for this run ({type(e).__name__}: {str(e)[:120]}). "
                  f"To use it: pip install trackio", flush=True)

    def log(self, metrics, step):
        if self.t is None:
            return
        try:
            self.t.log({k: v for k, v in metrics.items() if v is not None and v == v}, step=step)
        except Exception as e:
            print(f"  trackio stopped ({type(e).__name__}: {str(e)[:120]}); training goes on", flush=True)
            self.t = None

    def finish(self):
        if self.t is not None:
            try:
                self.t.finish()
            except Exception:
                pass


def final_scores(model, hm, data, H, T, args, n, slide):
    """Scores of the best model on a text: sampled windows, or with --slide_eval every byte."""
    wf = torch.from_numpy(word_features(data, hm.cls_of)[0]) if hm is not None and hm.word_tabs else None
    V = dict(data=torch.from_numpy(data), wf=wf, H=H)
    if slide:
        V["starts"], V["skip"] = slide_windows(len(data), H, T)
    else:
        V["starts"] = val_starts(len(data), H, T, n, "spread")
    return evaluate_detail(model, hm, V, args), len(V["starts"])


def main(args):
    torch.manual_seed(args.seed)
    if args.mode == "generate":
        return generate(args)
    T = args.ctx
    tag = args.tag or f"{args.mode}{'_feat' if args.hard_feature else ''}"
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f"{tag}.csv")
    best_path = os.path.join(args.out, f"{tag}_best.pt")

    os.makedirs(args.out, exist_ok=True)
    train, val, test = load_data(args)
    if args.mode == "eval":
        return evaluate_ckpt(args, val)
    if train is None:
        print(f"scoring saved tables on {len(val):,} bytes (no training text loaded)")
        if args.teacher:
            raise SystemExit("--teacher needs the training text: leave out --load_hard or --teacher")
    else:
        print(f"data: {len(train):,} train bytes, {len(val):,} val bytes")
    teacher = TeacherLabels(args.teacher, train) if args.teacher else None

    hm, hard_path, build_s = None, None, 0.0
    if args.mode in ("hard", "residual") or args.hard_feature:
        t0 = time.time()
        if args.load_hard:
            hm = HardModel.load(args.load_hard, "cpu")
            hard_path = os.path.relpath(os.path.abspath(args.load_hard), os.path.abspath(args.out))
            print(f"hard model loaded from {args.load_hard}")
            if teacher is not None and not hm.teacher_tabs and not args.distill_only:
                hm.add_teacher(train, teacher, args)       # reuse the counts, add the teacher
                hard_path = f"{tag}_hard.pt"
                safe_save(hm.state(), os.path.join(args.out, hard_path))
            hm.to(args.device)
        else:
            print("building the hard model ...")
            src = train[:args.table_bytes] if args.table_bytes else train
            if teacher is not None and teacher.L > len(src):
                raise SystemExit("--table_bytes must cover the teacher-labelled text")
            hm = HardModel.build(src, args, args.device, None if args.distill_only else teacher)
            hard_path = f"{tag}_hard.pt"
            safe_save(hm.state(), os.path.join(args.out, hard_path))
        build_s = time.time() - t0
        hm.set_runtime(args)
        print(f"hard model: experts {hm.names} | {hm.n_entries:,} table entries | "
              f"{build_s:.0f}s -> {os.path.join(args.out, hard_path)}")
        if hm.doc_hist:
            print(f"document memory: looks back {hm.doc_hist} bytes"
                  f"{' | cache contexts ' + str(hm.cache_orders) if hm.cache_orders else ''}"
                  f"{' | fades with distance ' + str(hm.cache_decay) if hm.cache_decay else ''}"
                  f"{' | class cache' if hm.class_cache else ''}"
                  f"{' | topic (word classes so far)' if hm.topic else ''}")
    H = hm.hist if hm is not None else 1
    n_tab = None                       # tables cover only the first n_tab bytes of the training text
    if hm is not None and train is not None:
        n = args.table_bytes or hm.n_text or len(train)
        if n < len(train):
            n_tab = n
            print(f"tables cover the first {n:,} of {len(train):,} training bytes "
                  f"(leave-one-out only there; the transformer trains on all of it)")

    part_lo, part_hi = 0, len(train) if train is not None else 0
    if args.train_part != "all" and train is not None:
        if n_tab is None:
            raise SystemExit("--train_part needs tables that cover only part of the text (--table_bytes)")
        if args.teacher:
            raise SystemExit("--train_part and --teacher cannot be combined yet")
        part_lo, part_hi = (0, n_tab) if args.train_part == "tables" else (n_tab, len(train))
        print(f"training windows only from {args.train_part}: bytes {part_lo:,} to {part_hi:,}")
    val_t = torch.from_numpy(val)                                        # uint8: 1 byte per byte
    train_t = torch.from_numpy(train) if train is not None else None
    wf_train = wf_val = None
    if hm is not None and hm.word_tabs:
        if train is not None and args.mode != "hard":      # only training needs them
            wf = hm.wf_cache if hm.wf_cache is not None and len(hm.wf_cache) == len(train) \
                else word_features(train, hm.cls_of)[0]
            wf_train = torch.from_numpy(wf)
        hm.wf_cache = None
        wf_val = torch.from_numpy(word_features(val, hm.cls_of)[0])
    V = dict(data=val_t, wf=wf_val, H=H,
             starts=val_starts(len(val), H, T, args.eval_windows, args.eval_layout))

    if args.mode == "hard":
        res = evaluate(None, hm, V, args)
        for k, v in res.items():
            print(f"  {k:11s} val BPC {v:.4f}{'   <- main' if k == hm.main else ''}")
        if hm.cache_orders:
            for k, v in cache_mix(hm, V, args).items():
                print(f"  {k}")
        if args.check_loo:
            print("leave-one-out check (bits per byte of each expert):")
            check_loo(hm, train, n_tab, V, args)
        if args.probe:
            acc, rows = run_probe(None, hm, load_probe(args.probe), T, args)
            print(f"  knowledge probe, tables alone: {acc:.1%} of {len(rows)} right")
            with open(os.path.join(args.out, f"{tag}_probe.json"), "w") as jf:
                json.dump(dict(probe=acc, rows=rows), jf, indent=1)
        if train is not None:
            safe_save({"model": None, "args": vars(args), "hard_path": hard_path,
                       "step": 0, "val_bpc": res[hm.main]}, best_path)
            print(f"saved -> {best_path}")
        if args.sample:
            print("---\n" + sample(None, hm, args.prompt, args.sample, T, args, args.temperature))
        return

    model = build_model(args, hm).to(args.device)
    if args.init:
        ck0 = torch.load(args.init, map_location=args.device, weights_only=False)
        added = fit_memory_input(ck0["model"], ck0.get("args", {}), model, hm)
        try:
            model.load_state_dict(ck0["model"])
        except RuntimeError:
            raise SystemExit(f"--init {args.init} does not fit: use the same --d/--layers/--heads/--ctx, "
                             f"--hard_feature and the same kinds of table experts")
        print(f"starting from {args.init}" +
              (f" | new memory inputs start at zero weight: {', '.join(added)}" if added else ""))
        del ck0
    n_par = sum(p.numel() for p in model.parameters())
    print(f"transformer: {n_par / 1e6:.2f}M params | mode {args.mode} | hard input "
          f"{model.use_feat} | distill {'alpha ' + str(args.distill_alpha) if teacher else 'off'}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    fl = args.min_lr_frac
    lr_at = lambda s: args.lr * min(1.0, (s + 1) / args.warmup) * \
        (fl + (1 - fl) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    gen = torch.Generator().manual_seed(args.seed)
    last_path = os.path.join(args.out, f"{tag}_last.pt")
    start_step, best_vb, best_step, bad_evals, t_offset = 0, float("inf"), 0, 0, 0.0
    if args.resume and os.path.exists(last_path):     # continue a run that stopped
        st = torch.load(last_path, map_location=args.device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"].cpu())
        if st.get("rng") is not None:
            torch.set_rng_state(st["rng"].cpu())
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in st["cuda_rng"]])
        start_step, best_vb, best_step, bad_evals = st["step"], st["best_vb"], st["best_step"], st["bad_evals"]
        t_offset = st.get("seconds", 0.0)
        print(f"resuming {tag} at step {start_step} (best so far {best_vb:.4f} at step {best_step})")
    elif args.resume:
        print(f"--resume: no {last_path} yet, starting from scratch")
    if start_step and os.path.exists(log_path):     # rows after the save point are trained again
        try:
            with open(log_path, newline="") as f0:
                rows = list(csv.reader(f0))
            keep = [r for r in rows[1:] if r and r[0].strip().isdigit() and int(r[0]) <= start_step]
            seen, uniq = set(), []
            for r in reversed(keep):                # one row per step: the newest
                if r[0] not in seen:
                    seen.add(r[0])
                    uniq.append(r)
            with open(log_path, "w", newline="") as f0:
                csv.writer(f0).writerows(rows[:1] + uniq[::-1])
        except OSError as e:
            print(f"  warning: could not tidy {log_path} ({e})")
    f = open(log_path, "a" if start_step else "w", newline="")
    w = csv.writer(f)
    if not start_step:
        w.writerow(["step", "seconds", "train_bpc", "val_bpc", "probe"])
    tracker = Tracker(args, tag)
    prompts = load_prompts(args.prompts) if args.prompts else []
    probe = load_probe(args.probe) if args.probe else []
    samples_path = os.path.join(args.out, f"{tag}_samples.txt")
    if (prompts or probe) and not start_step and os.path.exists(samples_path):
        os.remove(samples_path)
    t_start, run_ce, run_kd, run_n, run_nkd = time.time() - t_offset, 0.0, 0.0, 0, 0
    span = H + T + 1
    n_evals, t_loop, secs = 0, time.time(), 0.0
    for step in range(start_step, args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            vb = evaluate(model, hm, V, args)["model"]
            tb = run_ce / run_n / LN2 if run_n else float("nan")
            secs = time.time() - t_start + build_s
            extra = ""
            if args.mode == "residual":
                extra += " | mix " + " ".join(f"{n}={m:.2f}" for n, m in zip(model.names, model.mix.tolist()))
            if run_nkd:
                extra += f" | distill {run_kd / run_nkd / LN2:.4f}"
            mark = ""
            if vb < best_vb:            # keep only the best model, not the last one
                best_vb, best_step, bad_evals = vb, step, 0
                ok = safe_save({"model": model.state_dict(), "args": vars(args), "hard_path": hard_path,
                                "step": step, "val_bpc": vb}, best_path)
                mark = "  * saved" if ok else "  (best, but not saved)"
            else:
                bad_evals += 1
            acc = None
            fresh = step != start_step or not start_step          # not a repeat after --resume
            if (prompts or probe) and fresh and args.sample_every and \
                    (n_evals % args.sample_every == 0 or step == args.steps):
                acc = track(model, hm, step, vb, T, args, samples_path, prompts, probe)
                extra += f" | probe {acc:.0%}" if acc is not None else ""
            if step > start_step and step < args.steps:
                left = (time.time() - t_loop) / (step - start_step) * (args.steps - step)
                extra += f" | {left / 3600:.1f}h left"
            print(f"step {step:6d} | {secs:7.0f}s | train {tb:.4f} | val BPC {vb:.4f}{extra}{mark}", flush=True)
            if fresh:
                try:
                    w.writerow([step, round(secs, 1), round(tb, 4), round(vb, 4),
                                "" if acc is None else round(acc, 4)])
                    f.flush()
                except OSError as e:
                    print(f"  warning: could not write {log_path} ({e})", flush=True)
                tracker.log({"train_bpc": tb, "val_bpc": vb, "best_val_bpc": best_vb, "probe": acc,
                             "lr": lr_at(step), "hours": secs / 3600,
                             "steps_per_s": (step - start_step) / max(1e-9, time.time() - t_loop)
                             if step > start_step else None,
                             "gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30
                             if torch.cuda.is_available() else None}, step)
            n_evals += 1
            if args.save_every and (n_evals % args.save_every == 0 or step == args.steps):
                safe_save({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(),
                           "rng": torch.get_rng_state(),
                           "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                           "step": step, "best_vb": best_vb, "best_step": best_step,
                           "bad_evals": bad_evals, "seconds": secs}, last_path)
            run_ce, run_kd, run_n, run_nkd = 0.0, 0.0, 0, 0
            if args.patience and bad_evals >= args.patience:
                print(f"val BPC has not improved for {bad_evals} evals: stopping (overfitting).")
                break
        if step == args.steps:
            break

        use_t = teacher is not None and torch.rand(1, generator=gen).item() < args.teacher_frac
        lo, hi = part_lo, (teacher.L if use_t else part_hi) - span
        starts = torch.randint(lo, hi, (args.batch,), generator=gen).tolist()
        x, y, fin = split(gather(train_t, starts, span).to(args.device).long(), H)
        wf = gather(wf_train, starts, *hm.wf_span(T)).to(args.device) if wf_train is not None else None
        tl = teacher.get(starts, H + 1, T, args.device) if teacher is not None else None
        lw = None if n_tab is None else \
            (torch.tensor(starts, device=args.device) + span <= n_tab)
        hard = hm.query(fin, wf, y, tl, loo_w=lw) if hm is not None else None
        if hm is not None and args.mem_drop > 0:
            hard = hm.degrade(*hard, args.mem_drop)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast(args.device_type, dtype=torch.bfloat16, enabled=args.amp):
            logits = model(x, hard)
        logits = logits.float()
        ce = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss = ce
        if tl is not None and args.distill_alpha > 0:
            tb_, tp_ = tl
            mask = tp_.sum(-1) > 0
            if mask.any():
                logq = F.log_softmax(logits, -1)
                kd = -(tp_ * logq.gather(-1, tb_)).sum(-1)[mask].mean()
                loss = (1 - args.distill_alpha) * ce + args.distill_alpha * kd
                run_kd += kd.item()
                run_nkd += 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_ce += ce.item()
        run_n += 1
    f.close()
    print(f"best val BPC {best_vb:.4f} at step {best_step} -> {best_path}")
    print(f"log -> {log_path}")
    res = dict(tag=tag, params=n_par, best_val=best_vb, best_step=best_step, steps=step, seconds=round(secs),
               ctx=T, hard=hard_path, experts=hm.names if hm is not None else [],
               table_entries=hm.n_entries if hm is not None else 0,
               args={k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool))})
    warm = f"with at least {T // 2} bytes of context"
    if args.final_eval_windows or test is not None or prompts or probe or args.slide_eval:
        model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=False)["model"])
    if args.final_eval_windows or test is not None or args.slide_eval:    # score the best model on more text
        n = args.final_eval_windows or args.eval_windows
        res["scoring"] = "every byte, sliding windows" if args.slide_eval else f"{n} sampled windows"
        for name, text in (("val", val), ("test", test)):
            if text is None:
                continue
            d, nw = final_scores(model, hm, text, H, T, args, n, args.slide_eval)
            res[name], res[name + "_warm"] = d["bpc"], d["warm"]
            res[name + "_acc"], res[name + "_word_start"], res[name + "_rest"] = d["acc"], d["word_start"], d["rest"]
            res[name + "_bytes"] = d["bytes"]
            label = "TEST BPC" if name == "test" else "val BPC"
            note = ", never used for any choice" if name == "test" else ""
            cover = f"all {d['bytes']:,} bytes" if args.slide_eval else f"{nw} windows"
            print(f"final: best model, {label} {d['bpc']:.4f} ({cover}{note}) | {warm}: {d['warm']:.4f} | "
                  f"next byte right {d['acc']:.1%} | first letters of words {d['word_start']:.3f}, "
                  f"other bytes {d['rest']:.3f}")
    if prompts or probe:
        res["probe"] = track(model, hm, f"{best_step} = best model, final", best_vb, T, args,
                             samples_path, prompts, probe)
        if res["probe"] is not None:
            print(f"final: best model, knowledge probe {res['probe']:.1%} of {len(probe)} right")
        print(f"samples -> {samples_path}")
    with open(os.path.join(args.out, f"{tag}_result.json"), "w") as jf:
        json.dump(res, jf, indent=1)
    tracker.log({"final/" + k: res.get(k) for k in ("val", "val_warm", "val_acc", "test", "test_warm",
                                                     "test_acc", "test_word_start", "probe")}, step)
    tracker.finish()
    if args.sample:
        model.load_state_dict(torch.load(best_path, map_location=args.device, weights_only=False)["model"])
        model.eval()
        print("---\n" + sample(model, hm, args.prompt, args.sample, T, args, args.temperature))


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="a text file, or a folder of .txt files")
    ap.add_argument("--max_bytes", type=int, default=0, help="use only the first N bytes (0 = all)")
    ap.add_argument("--val_data", default="",
                    help="take validation from this file's split and remove that text from --data")
    ap.add_argument("--val_file", default="",
                    help="benchmarks with their own files: this WHOLE file is the validation text, --data "
                         "is used whole for training, and --test_file (if given) is the test text")
    ap.add_argument("--dyn_lr", type=float, default=0.0,
                    help="--mode eval: dynamic evaluation. After each window is scored the model takes "
                         "one training step on it with this learning rate (try 1e-5 to 1e-4)")
    ap.add_argument("--dyn_decay", type=float, default=0.002,
                    help="dynamic evaluation: share by which the weights move back to the trained ones "
                         "after each step")
    ap.add_argument("--dyn_mode", choices=["adam", "rms"], default="adam",
                    help="dynamic evaluation: adam, or rms (Krause et al.: each weight's step size comes "
                         "from its gradients on --data, the training text)")
    ap.add_argument("--dyn_stats_windows", type=int, default=1000,
                    help="--dyn_mode rms: training windows used for the step sizes")
    ap.add_argument("--eval_bytes", type=int, default=0,
                    help="--mode eval: score only the first N bytes of the text (quick tuning)")
    ap.add_argument("--dyn_batch", type=int, default=1,
                    help="dynamic evaluation: windows scored before each step (1 = most exact, slowest)")
    ap.add_argument("--slide_eval", action="store_true",
                    help="final scores on every byte of the validation / test text (windows overlap by "
                         "half), the way benchmark numbers are reported")
    ap.add_argument("--mode", choices=["hard", "alone", "residual", "eval", "generate"], default="alone")
    ap.add_argument("--hard_feature", action="store_true",
                    help="give the transformer the hard experts' predictions as input")
    # hard model
    ap.add_argument("--order", type=int, default=5, help="exact byte contexts 1..order (max 6)")
    ap.add_argument("--long_orders", default="8,12,16",
                    help="hashed long contexts, comma separated ('none' = off)")
    ap.add_argument("--long_min", type=int, default=2, help="keep long contexts seen at least this often")
    ap.add_argument("--words", action="store_true", help="add the word-class expert")
    ap.add_argument("--vocab", type=int, default=50000, help="words that get a class")
    ap.add_argument("--class_source", choices=["cooc", "embed", "both"], default="",
                    help="word vectors for the classes: corpus co-occurrence, --embed_model, or both "
                         "(default: embed if --embed_model is given, else cooc)")
    ap.add_argument("--fine_classes", action="store_true",
                    help="split every word class into up to 254 sub-classes (previous word in more detail)")
    ap.add_argument("--exact_words", type=int, default=2000,
                    help="--fine_classes: the N most frequent words keep their own sub-class")
    ap.add_argument("--embed_model", default="",
                    help="Hugging Face encoder for word classes (default: co-occurrence)")
    ap.add_argument("--teacher", default="", help="folder from label_teacher.py: distill + teacher expert")
    ap.add_argument("--teacher_orders", default="1,3", help="byte contexts for the teacher expert")
    ap.add_argument("--teacher_min_p", type=float, default=0.01,
                    help="teacher expert: ignore teacher probabilities below this")
    ap.add_argument("--distill_alpha", type=float, default=0.5,
                    help="weight of matching the teacher vs the true byte")
    ap.add_argument("--teacher_frac", type=float, default=0.5,
                    help="share of batches taken from the teacher-labelled text")
    ap.add_argument("--distill_only", action="store_true",
                    help="--teacher: learn from the teacher's labels, but add no teacher tables")
    # document memory (no tables: computed from the text before each byte)
    ap.add_argument("--cache", action="store_true",
                    help="document cache expert: what followed the same bytes earlier in this text")
    ap.add_argument("--cache_orders", default="2,3,4,6,8,12,16", help="context lengths of the cache")
    ap.add_argument("--cache_decay", type=float, default=0.0,
                    help="caches: an occurrence d bytes back counts exp(-d / this) (0 = all count the same)")
    ap.add_argument("--class_cache", type=int, default=0,
                    help="1: a second cache keyed by the classes of the previous words (needs --words)")
    ap.add_argument("--cache_keep", type=int, default=8,
                    help="cache: use the most recent N earlier occurrences of a context")
    ap.add_argument("--topic", action="store_true",
                    help="extra input: how often each word class occurred in this text so far "
                         "(needs --words; experimental, no gain in a small test)")
    ap.add_argument("--doc_hist", type=int, default=8192,
                    help="how many bytes back the cache and the topic input look")
    ap.add_argument("--mem_drop", type=float, default=0.0,
                    help="memory dropout: share of training windows that get a weakened memory")
    ap.add_argument("--table_bytes", type=int, default=0,
                    help="the tables cover only the first N bytes of the training text (0 = all). "
                         "Lets the transformer train on more text than fits into the tables; with "
                         "--load_hard of an older file, N = its 'data: N train bytes'")
    ap.add_argument("--train_part", choices=["all", "tables", "rest"], default="all",
                    help="with --table_bytes: train on all text, only on the counted text, or only "
                         "on the rest (text the tables have never seen: exactly the test situation)")
    ap.add_argument("--check_loo", action="store_true",
                    help="--mode hard: compare the experts on training text (own byte left out) "
                         "with validation text")
    ap.add_argument("--load_hard", default="", help="use a saved *_hard.pt instead of building one")
    ap.add_argument("--classes_from", default="", help="reuse the word classes of a saved *_hard.pt")
    ap.add_argument("--build_chunk", type=int, default=1 << 24, help="positions counted per chunk")
    ap.add_argument("--init", default="", help="start training from this *_best.pt (fine-tuning)")
    ap.add_argument("--resume", action="store_true", help="continue this --tag from its *_last.pt")
    ap.add_argument("--save_every", type=int, default=4, help="save *_last.pt every N evals (0 = never)")
    ap.add_argument("--min_lr_frac", type=float, default=0.1, help="learning rate at the end, as a share of --lr")
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--final_eval_windows", type=int, default=0,
                    help="after training, score the best model on this many windows (81920 = all 20 MB)")
    ap.add_argument("--test_mb", type=int, default=0,
                    help="take this many 1 MB blocks out of --data as a test set (never used for choices)")
    ap.add_argument("--test_file", default="", help="eval: score on this file (e.g. the test text) instead")
    # transformer + training
    ap.add_argument("--rope", action="store_true",
                    help="rotary positions instead of a learned position per slot")
    ap.add_argument("--arch", choices=["classic", "modern"], default="classic",
                    help="modern: RMSNorm, SwiGLU and no bias terms instead of LayerNorm and GELU")
    ap.add_argument("--tie", action="store_true", help="output weights = byte embeddings")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_windows", type=int, default=2048, help="validation windows per eval")
    ap.add_argument("--eval_layout", choices=["spread", "first"], default="spread",
                    help="spread windows over all validation text, or take the first ones (old runs)")
    ap.add_argument("--per_block", action="store_true", help="eval: also score each 1 MB validation block")
    ap.add_argument("--patience", type=int, default=20,
                    help="stop after this many evals without val improvement (0 = never); "
                         "val BPC wobbles by about +-0.005, so small values stop too early")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # generation
    ap.add_argument("--ckpt", default="runs/alone_feat_best.pt", help="generate / eval: model to load")
    ap.add_argument("--interactive", action="store_true", help="generate: prompt loop")
    ap.add_argument("--sample", type=int, default=400, help="bytes to generate (0 = off)")
    ap.add_argument("--prompt", default="The ")
    ap.add_argument("--temperature", type=float, default=0.8)
    # paper trail
    ap.add_argument("--prompts", default="", help="text file, one prompt per line: sample texts during training")
    ap.add_argument("--probe", default="",
                    help="json knowledge probe: [{prompt, answers: [right, wrong, ...]}, ...]")
    ap.add_argument("--sample_every", type=int, default=10,
                    help="write samples and the probe score every N evals to <tag>_samples.txt (0 = only at the end)")
    ap.add_argument("--sample_len", type=int, default=160, help="bytes to write per prompt")
    ap.add_argument("--trackio", default="", help="trackio project name: live curves (pip install trackio)")
    ap.add_argument("--trackio_space", default="",
                    help="also send them to this Hugging Face Space, e.g. yourname/zeroflops")
    return ap


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")   # Windows consoles and odd bytes
    except Exception:
        pass
    a = build_parser().parse_args()
    a.device_type = "cuda" if a.device.startswith("cuda") else "cpu"
    a.amp = a.device_type == "cuda"
    if a.mode == "eval" and not (a.data or a.val_data or a.test_file or a.val_file):
        raise SystemExit("--mode eval needs --val_data, --data, --val_file or --test_file (the text to score)")
    if a.mode == "hard" and a.load_hard and not (a.data or a.val_data or a.test_file or a.val_file):
        raise SystemExit("--mode hard --load_hard needs --val_data, --data or --test_file (the text to score)")
    if a.mode not in ("generate", "eval") and not a.data and not (a.mode == "hard" and a.load_hard):
        raise SystemExit("--data is required for training")
    if a.embed_model and a.mode != "generate":
        try:
            import transformers  # noqa: F401  (checked now, not after a minute of counting)
        except ImportError:
            raise SystemExit("--embed_model needs the transformers package:  pip install transformers")
    main(a)
