"""
analyze.py - look inside a transformer that was trained with the hard model
(the counted memory) and compare it with one trained without.

  python analyze.py --feat runs/big_feat_best.pt --alone runs/big_alone_best.pt \
      --val_data data/smollm_subset_200MB.txt --out runs/analysis_big

Both models are scored on the same validation bytes. Six questions:

  1 ABLATION   How much does the memory model depend on its memory? Its tables
               (and, if it has them, the document cache and the topic input)
               are emptied, partly emptied, or swapped for another window's,
               at test time, without retraining.
  2 BUCKETS    Where does the gain over the plain model come from? Every byte
               is put into groups: how long a context the memory has seen, how
               sure the memory is and whether it is right, where in a word the
               byte is, how hard the byte is for the plain model, ...
  3 INPUTS     How strongly each memory input moves the transformer's first
               layer (word expert vs long contexts vs evidence counts, ...).
  4 ATTENTION  How far back each layer looks, for both models.
  5 LOGIT LENS How good the prediction already is after each layer.
  6 CKA        Which layer of one model holds similar information to which
               layer of the other.
  7 REPEAT     The copy test: a passage appears twice, with a gap of other
               text between. How much easier is the second copy? A model can
               only profit if it still "sees" the first copy: the transformer
               within its window (--ctx), the document cache within --doc_hist.

Output: tables on screen, everything in <out>/analysis.json, charts in
<out>/*.png (if matplotlib is installed), and with --save_bytes the per-byte
numbers in <out>/bytes.npz.

--load_hard swaps in other tables for the memory model (plug-in memory), e.g.
to see what changes inside when the same model gets 2 GB instead of 1 GB.
All numbers are computed in float32, so BPC can differ from the trainer's
bf16 evaluation in the third decimal.
"""
import argparse, importlib.util, json, math, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

LN2 = math.log(2)
TR = None          # the training script, loaded in main()


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("trainer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ loading
def load_model(tr, ckpt, load_hard, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck.get("model") is None:
        raise SystemExit(f"{ckpt} holds no transformer (it is a --mode hard checkpoint)")
    a = argparse.Namespace(**ck["args"])
    needs_hard = bool(getattr(a, "hard_feature", False)) or a.mode == "residual"
    hm, path = None, None
    if needs_hard:
        path = load_hard or (ck.get("hard_path") and
                             os.path.join(os.path.dirname(os.path.abspath(ckpt)), ck["hard_path"]))
        if not path or not os.path.exists(path):
            raise SystemExit(f"{ckpt} needs its hard file ({path}); pass it with --load_hard")
        hm = tr.HardModel.load(path, device).set_runtime(a)      # --cache / --topic as trained
    a.dropout = 0.0
    model = tr.build_model(a, hm)
    try:
        model.load_state_dict(ck["model"])
    except RuntimeError:
        raise SystemExit(f"{path} has different experts than the ones {ckpt} was trained with")
    model.to(device).eval()
    info = dict(ckpt=ckpt, hard=path, step=ck.get("step"), val_bpc=ck.get("val_bpc"),
                params=sum(p.numel() for p in model.parameters()), ctx=a.ctx,
                d=a.d, layers=a.layers, heads=a.heads, mode=a.mode, hard_feature=needs_hard)
    return model, hm, info


def stat_slices(hm):
    """Which columns of the evidence tensor belong to which expert."""
    return hm.stat_slices()


def chain_labels(hm):
    return ["unigram"] + [f"{k} bytes" for k in range(1, hm.order + 1)] + \
        [f"{m} bytes (long)" for m in hm.long]


# ------------------------------------------------------------ the transformer, opened up
def make_extra(model, logs, stats):
    return torch.cat([logs[n].clamp(min=-16) / 16 for n in model.names] + [stats / 10], -1)


def final_logits(model, h, logs=None):
    g = model.gpt
    out = g.head(g.ln(h))
    if model.mode == "residual":
        out = out + sum(model.mix[i] * logs[n] for i, n in enumerate(model.names))
    return out


class AttStats:
    """Per layer and head: attention mass per distance bin, mean distance, entropy,
    over queries at positions >= qmin (so every distance bin is reachable)."""
    def __init__(self, L, H, T, qmin, device):
        self.edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.edges = [e for e in self.edges if e < T] + [T]
        nb = len(self.edges) - 1
        i = torch.arange(T, device=device)
        D = (i[:, None] - i[None, :]).clamp(min=0).float()
        b = torch.bucketize(D, torch.tensor(self.edges[1:], device=device, dtype=torch.float), right=True)
        oh = F.one_hot(b.clamp(max=nb - 1), nb).float() * (i[:, None] >= i[None, :]).float()[..., None]
        self.D, self.oh, self.qmin = D[qmin:], oh[qmin:], qmin
        self.mass = torch.zeros(L, H, nb, device=device)
        self.dist = torch.zeros(L, H, device=device)
        self.ent = torch.zeros(L, H, device=device)
        self.n = 0

    def add(self, li, att):
        aq = att[:, :, self.qmin:, :]
        self.mass[li] += torch.einsum("bhqk,qkn->hn", aq, self.oh)
        self.dist[li] += torch.einsum("bhqk,qk->h", aq, self.D)
        self.ent[li] += -(aq * aq.clamp(min=1e-12).log2()).sum(-1).sum((0, 2))
        if li == 0:
            self.n += aq.shape[0] * aq.shape[2]

    def result(self):
        n = max(1, self.n)
        names = [f"{a}" if b - a == 1 else f"{a}-{b - 1}" for a, b in zip(self.edges[:-1], self.edges[1:])]
        return dict(bins=names, mass=(self.mass / n).tolist(), mean_dist=(self.dist / n).tolist(),
                    entropy=(self.ent / n).tolist())


@torch.no_grad()
def internals(model, x, hard, att=None):
    """Same computation as Model.forward, written out so that the attention
    weights and the hidden state after every layer can be read."""
    g = model.gpt
    B, T = x.shape
    h = g.tok(x)
    if g.pos is not None:
        h = h + g.pos(torch.arange(T, device=x.device))
    if g.extra is not None:
        h = h + g.extra(make_extra(model, *hard))
    cs = TR.rope_table(T, g.rope_dh, x.device) if getattr(g, "rope_dh", 0) else None
    hs = [h]
    mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
    for li, b in enumerate(g.blocks):
        d = h.shape[-1]
        nh, dh = b.H, d // b.H
        q, k, v = b.qkv(b.ln1(h)).split(d, dim=2)
        q, k, v = (t.view(B, T, nh, dh).transpose(1, 2) for t in (q, k, v))
        if cs is not None:
            q, k = TR.rope(q, cs), TR.rope(k, cs)
        a = ((q @ k.transpose(-1, -2)) / math.sqrt(dh)).masked_fill(mask, float("-inf")).softmax(-1)
        if att is not None:
            att.add(li, a)
        h = h + b.proj((a @ v).transpose(1, 2).reshape(B, T, d))
        h = h + b.mlp(b.ln2(h))
        hs.append(h)
    return hs


def bits(logits, y):
    return F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1),
                           reduction="none").reshape(y.shape) / LN2


# ------------------------------------------------------------ memory ablations
def unigram_logp(counts):
    C = counts.float()
    tot, nd = C.sum(), (C > 1e-3).float().sum()
    return ((C + nd / 256) / (tot + nd)).log(), float(torch.log1p(tot))


def ablations(hm, model, logs, stats):
    """Versions of the memory input. "empty" means: as if those tables held no
    entries, which leaves the unigram (how the hard model answers then)."""
    sl = stat_slices(hm)
    uni_lp, uni_s = unigram_logp(hm.uni)
    shape = next(iter(logs.values())).shape
    U = uni_lp.expand(shape)
    if hm.teacher_tabs:
        tuni_lp, tuni_s = unigram_logp(hm.teacher_uni)

    def empty(names):
        L, S = dict(logs), stats.clone()
        for n in names:
            if n in ("ngram_o2", "ngram", "ngram_long", "word"):
                L[n] = U
            elif n == "teacher":
                L[n] = tuni_lp.expand(shape)
            elif n in ("cache", "ccache"):
                L[n] = torch.full(shape, -math.log(256), device=stats.device)
                S[..., sl[n]] = 0
        if "ngram" in names:
            S[..., sl["chain"]] = 0
            S[..., 0] = uni_s
        if "word" in names:
            S[..., sl["word"]] = 0
            S[..., sl["word"].start] = uni_s
        if "teacher" in names:
            S[..., sl["teacher"]] = 0
            S[..., sl["teacher"].start] = tuni_s
        return L, S

    names = model.names
    tables = [n for n in names if n not in ("cache", "ccache")]
    doc = [n for n in names if n in ("cache", "ccache")]
    out = {"memory as trained": (logs, stats)}
    if "ngram_long" in names:
        L, S = dict(logs), stats.clone()
        L["ngram_long"] = logs["ngram"]
        S[..., sl["long"]] = 0
        out["no long contexts (8+ bytes)"] = (L, S)
    if "ngram_o2" in names:
        L, S = dict(logs), stats.clone()
        for n in ("ngram", "ngram_long"):
            if n in L:
                L[n] = logs["ngram_o2"]
        S[..., 3:sl["chain"].stop] = 0
        out["byte contexts only up to 2 bytes"] = (L, S)
    if "word" in names:
        out["no word expert"] = empty(["word"])
    if "teacher" in names:
        out["no teacher expert"] = empty(["teacher"])
    if "cache" in names:
        out["no document cache"] = empty(["cache"])
    if "ccache" in names:
        out["no class cache"] = empty(["ccache"])
    if "topic" in sl:
        S = stats.clone()
        S[..., sl["topic"]] = 0
        out["no topic input"] = (logs, S)
    if doc or "topic" in sl:
        L, S = empty(doc) if doc else (logs, stats.clone())
        if "topic" in sl:
            S[..., sl["topic"]] = 0
        out["no document memory (caches, topic)"] = (L, S)
    S = stats.clone()
    S[..., sl["tables"]] = 0
    for n in doc:
        S[..., sl[n]] = 0
    out["evidence counts all set to 0"] = (logs, S)
    out["all tables empty (unigram only)"] = empty(tables)
    out["no memory at all (uniform, no counts)"] = (
        {n: torch.full(shape, -math.log(256), device=stats.device) for n in names}, torch.zeros_like(stats))
    out["memory of a different window"] = ({n: v.roll(1, 0) for n, v in logs.items()}, stats.roll(1, 0))
    return out


# ------------------------------------------------------------ repeat test
@torch.no_grad()
def repeat_test(tr, models, hm, val, gaps, n, plen, dev, batch):
    """For every gap: bits per byte on a passage of plen bytes, once where it is
    new and once where the same passage already appeared `gap` bytes earlier.
    Everything else (the text before, the gap text) is the same real text."""
    H = hm.hist if hm is not None else 1
    rng = np.random.default_rng(1)
    out = {}
    for gap in gaps:
        need = max(H + 1, 1) + 2 * plen + gap + 64
        if need >= len(val) // 3:
            continue
        rows = {k: {"new": 0.0, "repeat": 0.0} for k in models}
        tot = 0
        for b0 in range(0, n, batch):
            nb = min(batch, n - b0)
            seq_new, seq_rep = [], []
            for _ in range(nb):
                a = int(rng.integers(need, len(val) - need))          # the passage
                p = val[a:a + plen]
                c = int(rng.integers(0, len(val) - need))             # text before it and the gap
                c += 0 if abs(c - a) > need else need
                c %= len(val) - need
                pre = val[c:c + H + plen + gap]
                seq_new.append(np.concatenate([pre, p]))              # passage appears for the first time
                seq_rep.append(np.concatenate([pre[:H], p, pre[H + plen:H + plen + gap], p]))   # ... again
            for kind, seqs in (("new", seq_new), ("repeat", seq_rep)):
                S = np.stack(seqs)
                for k, model in models.items():
                    Tm = model.gpt.pos.num_embeddings if model.gpt.pos is not None else MODEL_CTX[k]
                    Hk = H if k == "feat" else 1
                    win = S[:, -(Hk + Tm + 1):]
                    t = torch.from_numpy(win).to(dev).long()
                    x, y, fin = tr.split(t, Hk)
                    hard = None
                    if k == "feat":
                        wf = None
                        if hm.word_tabs:
                            L, off = hm.wf_span(Tm)
                            wf = torch.stack([torch.from_numpy(tr.word_features(r, hm.cls_of)[0][off:off + L])
                                              for r in win]).to(dev)
                        hard = hm.query(fin, wf)
                    lg = model(x, hard).float()
                    rows[k][kind] += float(bits(lg[:, -plen:], y[:, -plen:]).sum())
            tot += nb * plen
        out[gap] = {k: {kind: v / tot for kind, v in r.items()} for k, r in rows.items()}
    return out


MODEL_CTX = {}


# ------------------------------------------------------------ CKA
def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    xy = (X.T @ Y).norm() ** 2
    return float(xy / ((X.T @ X).norm() * (Y.T @ Y).norm()))


# ------------------------------------------------------------ byte groups
CLASS_NAMES = ["lowercase letter", "uppercase letter", "space", "newline", "digit",
               "punctuation", "UTF-8 byte (non-ASCII)", "other control"]


def byte_classes():
    c = np.full(256, 7, dtype=np.int64)
    c[ord("a"):ord("z") + 1] = 0
    c[ord("A"):ord("Z") + 1] = 1
    c[ord(" ")] = 2
    c[ord("\n")] = 3
    c[ord("0"):ord("9") + 1] = 4
    for ch in b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
        c[ch] = 5
    c[128:] = 6
    c[ord("\t")] = 2
    return c


def letters_before(val, word_mask):
    """For every position i: how many letters end at i (0 if i is not a letter)."""
    m = word_mask[val]
    idx = np.arange(len(val))
    last = np.maximum.accumulate(np.where(~m, idx, -1))
    return (idx - last).astype(np.int32)


def bucketize(values, edges, labels):
    return np.digitize(values, edges), labels


def bucket_rows(groups, labels, R, has_alone):
    feat, alone, mem = R["feat"], R.get("alone"), R["memory"]
    N = len(feat)
    tot_gain = float((alone - feat).sum()) if has_alone else 0.0
    tot_bits = float(feat.sum())
    rows = []
    for g, lab in enumerate(labels):
        m = groups == g
        n = int(m.sum())
        if n == 0:
            continue
        r = dict(label=lab, n=n, share=n / N, feat=float(feat[m].mean()), memory=float(mem[m].mean()),
                 bits_share=float(feat[m].sum()) / tot_bits)
        if has_alone:
            r["alone"] = float(alone[m].mean())
            r["gain"] = r["alone"] - r["feat"]
            r["gain_share"] = float((alone[m] - feat[m]).sum()) / tot_gain if tot_gain else 0.0
        rows.append(r)
    return rows


def print_buckets(title, rows, has_alone, note=""):
    print(f"\n  {title}" + (f"   ({note})" if note else ""))
    head = f"    {'':34s} {'bytes':>6s}  {'memory':>7s}"
    head += f"  {'plain':>6s}" if has_alone else ""
    head += f"  {'tables':>6s}"
    head += f"  {'gain':>6s}  {'of gain':>7s}" if has_alone else ""
    head += f"  {'of loss':>7s}"
    print(head)
    for r in rows:
        s = f"    {r['label'][:34]:34s} {r['share']:6.1%}  {r['feat']:7.3f}"
        s += f"  {r['alone']:6.3f}" if has_alone else ""
        s += f"  {r['memory']:6.3f}"
        s += f"  {r['gain']:+6.3f}  {r['gain_share']:7.1%}" if has_alone else ""
        s += f"  {r['bits_share']:7.1%}"
        print(s)


# ------------------------------------------------------------ plots
def plots(res, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed: no charts. pip install matplotlib)")
        return []
    files = []
    C_FEAT, C_ALONE, C_MEM = "#2a6fdb", "#e07a1f", "#7a7a7a"

    def save(fig, name):
        p = os.path.join(out, name)
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        files.append(p)

    if "ablation" in res:
        ab = res["ablation"]
        base = ab["memory as trained"]
        labels = [k for k in ab if k != "memory as trained"]
        vals = [ab[k] - base for k in labels]
        fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(labels) + 1.6))
        ax.barh(range(len(labels)), vals, color=C_FEAT)
        if "alone" in res["models"]:
            ax.axvline(res["bpc"]["alone"] - base, color=C_ALONE, ls="--",
                       label=f"plain model ({res['bpc']['alone']:.3f})")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8, frameon=False)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        for i, (k, v) in enumerate(zip(labels, vals)):
            ax.text(v, i, f" +{v:.3f}  ({ab[k]:.3f})" if v >= 0 else f" {v:.3f}  ({ab[k]:.3f})",
                    va="center", fontsize=8)
        ax.set_xlim(min(0, min(vals)) * 1.1, max(max(vals), 0.01) * 1.35)
        ax.set_xlabel(f"bits per byte worse than with the full memory ({base:.3f})")
        ax.set_title("Memory model with parts of its memory removed at test time")
        save(fig, "ablation.png")

    for fam, rows in res.get("buckets", {}).items():
        if not rows:
            continue
        x = np.arange(len(rows))
        has_alone = "alone" in rows[0]
        w = 0.27
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(rows) + 2), 4))
        ax.bar(x - w, [r["feat"] for r in rows], w, label="memory model", color=C_FEAT)
        if has_alone:
            ax.bar(x, [r["alone"] for r in rows], w, label="plain model", color=C_ALONE)
        ax.bar(x + w, [r["memory"] for r in rows], w, label="tables alone", color=C_MEM)
        ax.set_xticks(x, [f"{r['label']}\n{r['share']:.0%}" for r in rows], fontsize=7, rotation=0)
        ax.set_ylabel("bits per byte")
        ax.set_title(fam)
        ax.legend(fontsize=8)
        save(fig, f"bucket_{fam.split(':')[0].strip().replace(' ', '_')}.png")

    if res.get("attention"):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for name, col in (("feat", C_FEAT), ("alone", C_ALONE)):
            if name not in res["attention"]:
                continue
            A = res["attention"][name]
            md = np.array(A["mean_dist"]).mean(1)
            mass = np.array(A["mass"]).mean(1)                    # (L, bins)
            far = mass[:, A["bins"].index(next(b for b in A["bins"] if b.startswith("32"))):].sum(1)
            lab = "memory model" if name == "feat" else "plain model"
            axs[0].plot(range(1, len(md) + 1), md, "o-", color=col, label=lab)
            axs[1].plot(range(1, len(far) + 1), far, "o-", color=col, label=lab)
        axs[0].set_title("mean attention distance (bytes back)")
        axs[1].set_title("share of attention on bytes >= 32 back")
        for a in axs:
            a.set_xlabel("layer")
            a.legend(fontsize=8)
        save(fig, "attention.png")

    if res.get("logit_lens"):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for name, col in (("feat", C_FEAT), ("alone", C_ALONE)):
            if name in res["logit_lens"]:
                v = res["logit_lens"][name]
                ax.plot(range(len(v)), v, "o-", color=col,
                        label="memory model" if name == "feat" else "plain model")
        ax.set_xlabel("after layer (0 = input embedding)")
        ax.set_ylabel("bits per byte, read out with the final head")
        ax.set_title("Logit lens")
        ax.set_ylim(top=min(8.5, ax.get_ylim()[1]))
        ax.legend(fontsize=8)
        save(fig, "logit_lens.png")

    if res.get("repeat"):
        gaps = [int(g) for g in res["repeat"]]
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for name, col in (("feat", C_FEAT), ("alone", C_ALONE)):
            if name in next(iter(res["repeat"].values())):
                gain = [res["repeat"][str(g)][name]["new"] - res["repeat"][str(g)][name]["repeat"] for g in gaps]
                ax.plot(gaps, gain, "o-", color=col, label="memory model" if name == "feat" else "plain model")
        ax.set_xscale("log")
        ax.set_xlabel("bytes of other text between the two copies")
        ax.set_ylabel("bits per byte saved on the second copy")
        ax.set_title("Copy test: does the model remember a passage?")
        ax.axhline(0, color="#999", lw=0.8)
        ax.legend(fontsize=8)
        save(fig, "repeat.png")

    if res.get("cka"):
        M = np.array(res["cka"]["matrix"])
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", origin="lower")
        ax.set_xlabel("plain model: after layer")
        ax.set_ylabel("memory model: after layer")
        ax.set_title("CKA similarity of hidden states")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] < 0.6 else "black")
        fig.colorbar(im)
        save(fig, "cka.png")
    return files


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feat", default="", help="*_best.pt of the model trained with the memory")
    ap.add_argument("--alone", default="", help="*_best.pt of the plain model (same --ctx)")
    ap.add_argument("--load_hard", default="",
                    help="tables for the memory model instead of its own (plug-in memory). Without "
                         "--feat: tables used only to group the bytes")
    ap.add_argument("--val_data", default="", help="as in training: its validation split is scored")
    ap.add_argument("--data", default="", help="training without --val_data: the split of this file is scored")
    ap.add_argument("--max_bytes", type=int, default=0, help="with --data: as in training")
    ap.add_argument("--test_file", default="", help="score this whole file instead (e.g. the test text)")
    ap.add_argument("--windows", type=int, default=1024, help="validation windows (x ctx bytes)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cka_windows", type=int, default=128, help="windows used for CKA")
    ap.add_argument("--out", default="runs/analysis")
    ap.add_argument("--save_bytes", action="store_true", help="also write per-byte numbers to bytes.npz")
    ap.add_argument("--repeat_gaps", default="16,128,512,2048,6000",
                    help="copy test: bytes of other text between the two copies ('' = no copy test)")
    ap.add_argument("--repeat_n", type=int, default=256, help="copy test: passages per gap")
    ap.add_argument("--repeat_len", type=int, default=128, help="copy test: length of the passage")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--trainer", default="",
                    help="the training script (default: the newest of hard_residual_lm.py / "
                         "tg_train_hard.py next to this file)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if not args.trainer:
        here = os.path.dirname(os.path.abspath(__file__))
        cands = [os.path.join(here, n) for n in ("hard_residual_lm.py", "tg_train_hard.py")]
        cands = [c for c in cands if os.path.exists(c)]
        if not cands:
            raise SystemExit("give --trainer: the training script was not found next to analyze.py")
        args.trainer = max(cands, key=os.path.getmtime)
    if not (args.feat or args.alone):
        raise SystemExit("give --feat and/or --alone")
    if not (args.val_data or args.data or args.test_file):
        raise SystemExit("give --val_data or --data (validation split) or --test_file")
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    torch.manual_seed(0)
    dev = args.device
    global TR
    tr = TR = load_trainer(args.trainer)
    if not hasattr(tr.HardModel, "set_runtime"):
        raise SystemExit(f"{args.trainer} is an older training script; use the current one (--trainer)")
    print(f"trainer: {args.trainer}")
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    models, infos = {}, {}
    hm = None
    if args.feat:
        m, hm, infos["feat"] = load_model(tr, args.feat, args.load_hard, dev)
        if hm is None:
            raise SystemExit(f"--feat {args.feat} was trained without the memory (no --hard_feature)")
        models["feat"] = m
    if args.alone:
        m, h2, infos["alone"] = load_model(tr, args.alone, "", dev)
        if h2 is not None:
            raise SystemExit(f"--alone {args.alone} was trained with the memory; give a plain model")
        models["alone"] = m
    if hm is None and args.load_hard:
        hm = tr.HardModel.load(args.load_hard, dev)
    if hm is None:
        print("note: no tables available, so no ablation and no byte groups by memory")
    ctxs = {i["ctx"] for i in infos.values()}
    if len(ctxs) > 1:
        raise SystemExit(f"the two models use different --ctx ({ctxs}); compare models with the same ctx")
    T = ctxs.pop()
    for k, i in infos.items():
        print(f"{k:5s}: {i['ckpt']} | step {i['step']} | {i['params'] / 1e6:.1f}M params | "
              f"d {i['d']} x {i['layers']} layers | ctx {T}" + (f" | tables {i['hard']}" if i["hard"] else ""))

    if args.test_file:
        val = np.fromfile(args.test_file, dtype=np.uint8)
        what = f"test file {args.test_file}"
    elif args.val_data:
        val = tr.load_bytes(args.val_data, 0)[1]
        what = f"validation split of {args.val_data}"
    else:
        val = tr.load_bytes(args.data, args.max_bytes)[1]
        what = f"validation split of {args.data}"
    H = hm.hist if hm is not None else 1
    starts = tr.val_starts(len(val), H, T, args.windows, "spread")
    n_win = len(starts)
    print(f"text: {what}: {len(val):,} bytes, {n_win} windows of {T} bytes ({n_win * T:,} bytes scored)")
    val_t = torch.from_numpy(val)
    wf_val = (torch.from_numpy(tr.word_features(val, hm.cls_of)[0])
              if hm is not None and hm.word_tabs else None)
    main_exp = None
    if hm is not None:
        main_exp = "ngram_long" if "ngram_long" in hm.names else "ngram"
        sl = stat_slices(hm)
        clabels = chain_labels(hm)

    # accumulators
    qmin = T // 2
    att = {k: AttStats(infos[k]["layers"], infos[k]["heads"], T, qmin, dev) for k in models}
    lens = {k: torch.zeros(infos[k]["layers"] + 1, dtype=torch.float64) for k in models}
    ab_sum, exp_sum = {}, {}
    cka_h = {k: [] for k in models}
    rng = np.random.default_rng(0)
    per = {k: [] for k in ("feat", "alone", "memory", "deep", "deep_cnt", "conf", "right",
                           "gidx", "pos", "agree_feat", "agree_alone", "cdeep")}
    inp = None
    n_bytes = 0
    checked = False
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, n_win, args.batch):
            st = starts[bi:bi + args.batch]
            buf = tr.gather(val_t, st, H + T + 1).to(dev).long()
            x, y, fin = tr.split(buf, H)
            B = x.shape[0]
            gidx = (np.asarray(st)[:, None] + H + 1 + np.arange(T)[None, :])
            per["gidx"].append(gidx.ravel())
            per["pos"].append(np.tile(np.arange(T), B))
            hard = None
            if hm is not None:
                wf = tr.gather(wf_val, st, *hm.wf_span(T)).to(dev) if wf_val is not None else None
                hard = hm.query(fin, wf, pad=tr.pad_of(st, dev))
                logs, stats = hard
                for n, lp in logs.items():
                    exp_sum[n] = exp_sum.get(n, 0.0) + float(bits(lp, y).sum())
                mb = bits(logs[main_exp], y)
                per["memory"].append(mb.cpu().numpy().ravel())
                p = logs[main_exp].exp()
                conf, top = p.max(-1)
                per["conf"].append(conf.cpu().numpy().ravel())
                per["right"].append((top == y).cpu().numpy().ravel())
                S = stats[..., sl["chain"]]
                lvl = torch.arange(S.shape[-1], device=dev)
                deep = ((S > 0).long() * (lvl + 1)).max(-1).values - 1
                cnt = torch.expm1(S.gather(-1, deep.clamp(min=0)[..., None]).squeeze(-1))
                per["deep"].append(deep.cpu().numpy().ravel())
                per["deep_cnt"].append(cnt.cpu().numpy().ravel())
                if "cache" in sl:                     # longest context found earlier in this text
                    Sc = stats[..., sl["cache"]][..., 1:]
                    lv = torch.arange(1, Sc.shape[-1] + 1, device=dev)
                    per["cdeep"].append(((Sc > 0).long() * lv).max(-1).values.cpu().numpy().ravel())
            sel = None
            if bi < args.cka_windows:
                sel = torch.from_numpy(rng.choice(np.arange(8, T), size=min(64, T - 8), replace=False)).to(dev)
            for k, model in models.items():
                h_in = hard if k == "feat" else None
                hs = internals(model, x, h_in, att[k])
                lg = final_logits(model, hs[-1], h_in[0] if h_in else None)
                if not checked:
                    ref = model(x, h_in).float()
                    diff = float((ref - lg).abs().max())
                    if diff > 1e-3:
                        print(f"warning: opened-up forward differs from the model by {diff:.2e} ({k})")
                b = bits(lg, y)
                per[k].append(b.cpu().numpy().ravel())
                if main_exp is not None:
                    agree = (lg.argmax(-1) == logs[main_exp].argmax(-1)).cpu().numpy().ravel()
                    per["agree_" + k].append(agree)
                for li, h in enumerate(hs):
                    lens[k][li] += float(bits(final_logits(model, h, h_in[0] if h_in else None), y).sum())
                if sel is not None:
                    cka_h[k].append([h[:, sel, :].reshape(-1, h.shape[-1]).double().cpu() for h in hs])
                if k == "feat":
                    # how strongly each input group moves the first layer's input
                    g = model.gpt
                    ex = make_extra(model, *hard)
                    W = g.extra.weight
                    groups = [(n, slice(i * 256, (i + 1) * 256)) for i, n in enumerate(model.names)]
                    s0 = 256 * len(model.names)
                    groups.append(("evidence counts", slice(s0, s0 + (sl["topic"].start if "topic" in sl
                                                                      else stats.shape[-1]))))
                    if "topic" in sl:
                        groups.append(("topic (word classes)", slice(s0 + sl["topic"].start, ex.shape[-1])))
                    parts = [(n, ex[..., s] @ W[:, s].T) for n, s in groups]
                    parts.append(("all memory inputs", g.extra(ex) - g.extra.bias))
                    parts.append(("byte embedding", g.tok(x)))
                    if g.pos is not None:
                        parts.append(("position embedding",
                                      g.pos(torch.arange(T, device=dev)).expand(B, T, -1)))
                    if inp is None:
                        inp = {n: [torch.zeros(v.shape[-1], dtype=torch.float64, device=dev), 0.0]
                               for n, v in parts}
                    for n, v in parts:
                        v2 = v.reshape(-1, v.shape[-1]).double()
                        inp[n][0] += v2.sum(0)
                        inp[n][1] += float((v2 ** 2).sum())
                    for name, (L_, S_) in ablations(hm, model, logs, stats).items():
                        ab_sum[name] = ab_sum.get(name, 0.0) + float(bits(model(x, (L_, S_)), y).sum())
            checked = True
            n_bytes += y.numel()
            if bi // args.batch % 16 == 0:
                done = min(n_win, bi + args.batch)
                print(f"  {done}/{n_win} windows | {time.time() - t0:.0f}s", end="\r")
    print(" " * 60, end="\r")

    R = {k: np.concatenate(v) for k, v in per.items() if v}
    res = dict(models=infos, text=what, windows=n_win, ctx=T, bytes=n_bytes)
    res["bpc"] = {k: float(R[k].mean()) for k in models}
    has_alone = "alone" in models

    # ---------------------------------------------------------------- 0 overview
    print("\n=== 0  OVERALL (bits per byte, lower is better) ===")
    for k in models:
        print(f"  {'memory model' if k == 'feat' else 'plain model':14s} {res['bpc'][k]:.4f}")
    if hm is not None:
        res["experts"] = {n: v / n_bytes for n, v in exp_sum.items()}
        for n, v in res["experts"].items():
            print(f"  tables: {n:14s} {v:.4f}")
        for k in models:
            a = float(R["agree_" + k].mean())
            res.setdefault("agreement_with_tables", {})[k] = a
            print(f"  {'memory' if k == 'feat' else 'plain'} model's top guess = tables' top guess "
                  f"({main_exp}): {a:.1%} of bytes")

    # ---------------------------------------------------------------- 1 ablation
    if "feat" in models:
        res["ablation"] = {k: v / n_bytes for k, v in ab_sum.items()}
        print("\n=== 1  ABLATION: the memory model with parts of its memory removed (no retraining) ===")
        base = res["ablation"]["memory as trained"]
        for k, v in res["ablation"].items():
            extra = f"   ({v - base:+.3f} vs as trained" + \
                (f", {v - res['bpc']['alone']:+.3f} vs plain model)" if has_alone else ")")
            print(f"  {k:40s} {v:.4f}{extra if k != 'memory as trained' else ''}")
        if has_alone:
            blank = res["ablation"]["all tables empty (unigram only)"]
            print(f"  -> without its tables the memory model is {blank - res['bpc']['alone']:+.3f} bits/byte "
                  f"{'worse' if blank > res['bpc']['alone'] else 'better'} than the plain model:")
            print("     the larger that gap, the more of what the plain model keeps in its weights "
                  "the memory model leaves to the tables.")

    # ---------------------------------------------------------------- 2 buckets
    if hm is not None and "feat" in models:
        res["buckets"] = {}
        print("\n=== 2  WHERE THE GAIN COMES FROM ===")
        print("  bytes: share of bytes | memory/plain/tables: bits per byte of the memory model, the plain")
        print("  model and the tables alone | gain: plain minus memory model | of gain: share of the total")
        print("  gain from this group | of loss: share of the memory model's remaining bits in this group")

        def fam(title, groups, labels, note=""):
            rows = bucket_rows(groups, labels, R, has_alone)
            res["buckets"][title] = rows
            print_buckets(title, rows, has_alone, note)

        deep = R["deep"]
        nlev = len(clabels)
        merge = np.where(deep <= 2, 0, deep - 2)
        labs = ["up to 2 bytes"] + clabels[3:]
        fam("A: longest context the tables have seen", merge, labs,
            "how far back the memory can match the text")
        cnt = R["deep_cnt"]
        g, labs = bucketize(cnt, [2, 5, 20, 100, 1000], ["seen once", "2-4 times", "5-19", "20-99",
                                                         "100-999", "1000+ times"])
        fam("B: how often that longest context was seen", g, labs)
        if "cdeep" in R:
            labs = ["not seen in this text"] + [f"{k} bytes" for k in hm.cache_orders]
            fam("H: longest context seen earlier in this text (cache)", R["cdeep"], labs,
                f"the last {hm.doc_hist} bytes plus the window so far")
        conf, right = R["conf"], R["right"].astype(bool)
        cb = np.digitize(conf, [0.3, 0.6, 0.9])
        g = cb * 2 + (~right)
        labs = []
        for c in ("sure <30%", "sure 30-60%", "sure 60-90%", "sure >90%"):
            labs += [f"{c}, right", f"{c}, wrong"]
        fam("C: tables' top guess - how sure and whether right", g, labs,
            "'wrong' rows show whether the model gets misled")
        cls = byte_classes()
        lb = letters_before(val, tr.WORD_MASK)
        gi = R["gidx"]
        tgt_letter = tr.WORD_MASK[val[gi]]
        before = lb[gi - 1]
        g = np.where(tgt_letter, np.minimum(np.digitize(before, [1, 2, 3, 6]), 4),
                     np.where(before > 0, 5, 6))
        fam("D: position in the word", g, ["1st letter", "2nd letter", "3rd letter", "4th-6th letter",
                                           "7th+ letter", "first non-letter after a word",
                                           "other non-letter"])
        fam("E: kind of byte predicted", cls[val[gi]], CLASS_NAMES)
        edges = [e for e in (8, 32, 128, 512) if e < T]
        g = np.digitize(R["pos"], edges)
        lo = [0] + edges
        fam("F: position in the window", g,
            [f"bytes {a}-{b - 1}" for a, b in zip(lo, edges + [T])],
            "the tables also see the bytes before the window, the transformer does not")
        if has_alone:
            g, labs = bucketize(R["alone"], [0.25, 1, 3, 6], ["easy for plain (<0.25 bits)", "0.25-1 bits",
                                                              "1-3 bits", "3-6 bits", "hard (>6 bits)"])
            fam("G: how hard the byte is for the plain model", g, labs,
                "grouped by the plain model's own loss, so part of the pattern is chance")

    # ---------------------------------------------------------------- 3 inputs
    if inp is not None:
        print("\n=== 3  INPUTS: how strongly each input moves the first layer's input ===")
        print("  (spread of the added vector over all bytes; a constant offset does not count)")
        res["inputs"] = {}
        N = n_bytes
        for n, (s, sq) in inp.items():
            mean = s / N
            var = sq / N - float((mean ** 2).sum())
            res["inputs"][n] = math.sqrt(max(var, 0.0))
        top = max(v for k, v in res["inputs"].items())
        for n, v in res["inputs"].items():
            print(f"  {n:22s} {v:7.3f}  {'#' * int(round(30 * v / top))}")

    # ---------------------------------------------------------------- 4 attention
    res["attention"] = {k: a.result() for k, a in att.items()}
    print(f"\n=== 4  ATTENTION: how far back each layer looks (queries at positions {qmin}-{T - 1}) ===")
    for k, A in res["attention"].items():
        print(f"  {'memory model' if k == 'feat' else 'plain model'}:")
        bins = A["bins"]
        print(f"    layer  mean dist  entropy  " + " ".join(f"{b:>7s}" for b in bins))
        for li, (md, en, ms) in enumerate(zip(A["mean_dist"], A["entropy"], A["mass"])):
            mass = np.array(ms).mean(0)
            print(f"    {li + 1:5d}  {np.mean(md):9.1f}  {np.mean(en):7.2f}  " +
                  " ".join(f"{v:7.1%}" for v in mass))
    if len(res["attention"]) == 2:
        df = np.array([np.mean(x) for x in res["attention"]["feat"]["mean_dist"]])
        da = np.array([np.mean(x) for x in res["attention"]["alone"]["mean_dist"]])
        if len(df) == len(da):
            print("  mean distance, memory minus plain, per layer: " +
                  " ".join(f"{v:+.1f}" for v in df - da))
        else:
            print(f"  mean distance over all layers: memory model {df.mean():.1f}, plain model {da.mean():.1f} "
                  f"(different numbers of layers)")

    # ---------------------------------------------------------------- 5 logit lens
    res["logit_lens"] = {k: (v / n_bytes).tolist() for k, v in lens.items()}
    print("\n=== 5  LOGIT LENS: bits per byte when the final head reads the stream after each layer ===")
    print("  (crude: the head was trained only on the last layer; compare the shapes)")
    names = list(models)
    print("    layer  " + "  ".join(f"{('memory' if k == 'feat' else 'plain'):>8s}" for k in names))
    for li in range(max(len(v) for v in res["logit_lens"].values())):
        print(f"    {li:5d}  " + "  ".join(
            f"{res['logit_lens'][k][li]:8.3f}" if li < len(res["logit_lens"][k]) else " " * 8 for k in names))

    # ---------------------------------------------------------------- 6 CKA
    if len(models) == 2 and cka_h["feat"]:
        Lf, La = len(cka_h["feat"][0]), len(cka_h["alone"][0])
        Xf = [torch.cat([b[l] for b in cka_h["feat"]]) for l in range(Lf)]
        Xa = [torch.cat([b[l] for b in cka_h["alone"]]) for l in range(La)]
        M = [[linear_cka(Xf[i], Xa[j]) for j in range(La)] for i in range(Lf)]
        res["cka"] = dict(matrix=M, tokens=int(Xf[0].shape[0]))
        print(f"\n=== 6  CKA: similarity of hidden states, memory model (rows) vs plain model "
              f"(columns), {Xf[0].shape[0]:,} bytes ===")
        print("    after layer " + " ".join(f"{j:5d}" for j in range(La)) + "   most similar")
        for i in range(Lf):
            j = int(np.argmax(M[i]))
            print(f"    {i:11d} " + " ".join(f"{v:5.2f}" for v in M[i]) + f"   plain layer {j}")

    # ---------------------------------------------------------------- 7 repeat test
    gaps = [int(g) for g in args.repeat_gaps.split(",") if g.strip()]
    if gaps:
        for k in models:
            MODEL_CTX[k] = infos[k]["ctx"]
        plen = min([args.repeat_len] + [infos[k]["ctx"] for k in models])     # every byte is scored
        rt = repeat_test(tr, models, hm, val, gaps, args.repeat_n, plen, dev, args.batch)
        res["repeat"] = {str(g): v for g, v in rt.items()}
        names = list(models)
        print(f"\n=== 7  REPEAT TEST: a {plen}-byte passage, new vs. seen before, with other text "
              f"in between ===")
        reach = f"transformer window {T} bytes"
        if hm is not None and hm.cache_orders:
            reach += f", document cache {hm.doc_hist} bytes"
        print(f"  (bits per byte on the passage; the first copy can only help if it is in reach: {reach})")
        print("    gap (bytes)  " + "  ".join(f"{('memory' if k == 'feat' else 'plain'):>23s}" for k in names))
        print("                 " + "  ".join(f"{'new':>7s} {'repeat':>7s} {'gain':>7s}" for _ in names))
        for g, v in rt.items():
            print(f"    {g:11d}  " + "  ".join(
                f"{v[k]['new']:7.3f} {v[k]['repeat']:7.3f} {v[k]['new'] - v[k]['repeat']:+7.3f}" for k in names))

    # ---------------------------------------------------------------- save
    with open(os.path.join(args.out, "analysis.json"), "w") as f:
        json.dump(res, f, indent=1)
    if args.save_bytes:
        np.savez_compressed(os.path.join(args.out, "bytes.npz"), **{k: v for k, v in R.items()})
    files = [] if args.no_plots else plots(res, args.out)
    print(f"\nsaved {os.path.join(args.out, 'analysis.json')}"
          + (f" and {len(files)} charts in {args.out}" if files else "")
          + f"  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
