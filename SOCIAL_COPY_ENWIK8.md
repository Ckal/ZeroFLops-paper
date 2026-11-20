# ZeroFLops enwik8 — social copy

Author of record: **Christof Ckal**. All numbers below are verified against
`paper/results.md`, `paper/results.csv`, `paper/analysis_e8l/analysis.json`
and the live HF repo.

**Note on the training-time claim:** the headline model (`e8b_cache`) took
**10.45 h** — not "a few hours". Most *other* runs in the line are a few hours
(median 2.9 h; the pilots 0.16–0.28 h), and the whole 15-run campaign is
~60 GPU-hours. So: "trained in 10 hours on one RTX 5090" is the accurate
version for the headline model. "A few hours" would be wrong for it.

---

## 1. Hugging Face — discussion (POSTED LIVE)

**Title:** `0.9756 BPC on enwik8 — trained in 10 hours on one GPU`

```
A character-level language model over enwik8 that reaches 0.9756 bits/byte on
the standard test split with dynamic evaluation ("learning while reading") —
below 1.0 BPC. For scale, Transformer-XL at 277M parameters reaches 0.94 on
the same file (Dai et al., 2019). I'm not claiming to beat it; the point is
what 10 GPU-hours and a count table buy you.

It trained in 10 hours on a single RTX 5090.

The mechanism: a hard count table — 81,785,402 integer entries over byte
contexts, words, and the document currently being read — is fed to the
transformer as 1303 extra input dimensions alongside the token embedding.
Counting, not matrix multiplies, so the lookup side spends no FLOP. The
transformer then learns how much to trust each of the 5 experts. (The table
side is integer counting; the transformer trunk is still float, so this is not
a zero-FLOP model end to end.)

The result I trust most is the negative one. Ablating the memory (val BPC,
measured on the 39.5M sibling checkpoint e8l_cache in a separate harness, so
read these as deltas against their own 1.0630 baseline):

  no document cache          1.1612
  no long contexts (8+ bytes) 1.1051
  all tables empty (unigram)  1.2928
  memory of a different window 12.8174

That last row is the one that matters. A model that had merely learned a
generic prior over English bytes would shrug off a mismatched window. This one
collapses. It is genuinely reading the document in front of it.

Weights, the table and the full model card are in this repo. Code, every
measured number and the commands that produced them:
https://github.com/Ckal/ZeroFLops-paper
```

---

## 2. X — 3-post thread (paste these in order)

**Post 1**
```
Below 1.0 bits/byte on enwik8.

0.9756 BPC on the test split, learning while reading. Transformer-XL with dynamic eval reaches 0.94 (Dai 2019).

It trained in 10 hours on one RTX 5090.
```

**Post 2**
```
The mechanism: an 81,785,402-entry integer count table over byte contexts, words and the document being read — fed to the transformer as 1303 extra input dimensions.

Counting, not matrix multiplies.
```

**Post 3**
```
The control that makes me trust it: give the model counts from the wrong window and it doesn't degrade, it collapses.

1.0630 → 12.8174 BPC. (Ablation on the 39.5M sibling, e8l_cache.)

Weights + table: https://huggingface.co/Chris4K/zeroflops-enwik8-lm
```

Character counts: 184 / 199 / 253 raw (post 3 ~276 effective with the URL
counted as 23). All under 280.

---

## 3. LinkedIn — 1 post

```
I got a character-level language model below 1.0 bits/byte on enwik8 — and the
most interesting part is how little compute it took.

0.9756 BPC on the standard 5 MB test split, with dynamic evaluation
("learning while reading"). For scale, Transformer-XL at 277M parameters
reaches 0.94 on the same file (Dai et al., 2019). I'm not claiming to beat it
— the point is what 10 GPU-hours and a count table buy you.

It trained in 10 hours on a single RTX 5090.

How it works: a hard count table — 81,785,402 integer entries over byte
contexts, words, and the document currently being read — is fed to the
transformer as 1303 extra input dimensions alongside the token embedding.
Counting, not matrix multiplies, so the lookup side spends no FLOP; the
transformer then learns how much to trust each of the 5 experts. (The table
side is integer counting, but the transformer trunk is still float, so this
isn't a zero-FLOP model end to end.)

The result I trust most is the negative one. Ablating the memory (val BPC,
measured on the 39.5M-parameter sibling checkpoint e8l_cache in a separate
harness — read these as deltas against their own 1.0630 baseline, not against
the headline model):

• no document cache            1.1612
• no long contexts (8+ bytes)  1.1051
• all tables empty (unigram)   1.2928
• memory of a different window 12.8174

That last row is the one that matters. A model that had merely learned a
generic prior over English bytes would shrug off a mismatched window. This one
collapses. It is genuinely reading the document in front of it.

Weights and the table are public. The full write-up, every measured number and
the command that produced it are in the repo.

https://huggingface.co/Chris4K/zeroflops-enwik8-lm
https://github.com/Ckal/ZeroFLops-paper

#MachineLearning #NLP #LanguageModels #enwik8 #OpenSource
```

---

## 2b. X — the "models and code are online" update (the one to post now)

The user's own draft, tightened to fit X's 280-character limit (every URL
counts as 23 characters). All three BPC pairs verified against
`paper/results.md`; `~10 h` is `e8b_cache`'s measured 10.45 h.

**Single post** — X-counted 277, raw 320:

```
Models + code are online.

enwik8 test, bits/byte (lower is better):
• 39.5M: 1.018 / 0.980 dynamic
• 86.3M: 1.027 / 0.976
• 85.3M no memory: 1.075 / 0.999

~10 h on one RTX 5090. Caveat: 82M count entries (1.2 GB).

Model: https://huggingface.co/Chris4K/zeroflops-enwik8-lm
Code: https://github.com/Ckal/ZeroFLops-paper
```

**3-post thread** — keeps the user's exact wording, all three fit:

> **1/3**
> Update: models and code are online.
>
> enwik8 test, bits per byte (lower is better):
> • 39.5M params: 1.018, and 0.980 with dynamic evaluation (learning while reading)
> • 86.3M params: 1.027 / 0.976
> • 85.3M without the memory: 1.075 / 0.999

> **2/3**
> Honest caveat: the tables hold 82M count entries (a 1.2 GB file) on top of the parameters. The memory is built by counting, no gradients, but the model still runs a normal transformer.

> **3/3**
> Trained in ~10 h on one RTX 5090. Model + code:
> https://huggingface.co/Chris4K/zeroflops-enwik8-lm
> https://github.com/Ckal/ZeroFLops-paper

**Corrections applied to the user's draft:**

| draft said | corrected to | why |
|---|---|---|
| "a 1.1 GB file" | 1.2 GB | 1.1 GB is the binary-GiB value (1,182,660,751 bytes = 1.101 GiB = 1.18 GB). HF renders decimal, so a reader downloading sees "1.18 GB". |
| no training time | ~10 h on one RTX 5090 | `e8b_cache` measured 10.45 h (`paper/results.csv`). The 39.5M `e8l_cache` is 9.04 h; the 85.3M `e8b_alone` is 8.28 h. "A few hours" is wrong for all three. |
| — | "no memory" label on the 85.3M row | makes the ablation read as a control rather than a third model size |

---

## Fact sheet (for your own checking)

| claim | value | source |
|---|---|---|
| headline dynamic-eval test BPC | 0.9756 (`e8b_cache`) | `paper/results.md` |
| headline static test BPC | 1.0270 | `paper/results.md` |
| Transformer-XL 277M dynamic eval | 0.94 | Dai et al. 2019 |
| table entries | 81,785,402 | HF model card |
| extra input dims | 1303 = 256×5 + 23 | checkpoint `gpt.extra.weight` |
| table size on disk | 1.18 GB | HF resolve headers |
| headline training time | 10.45 h | `paper/results.csv` |
| GPU | 1× RTX 5090 | `nvidia-smi`, checkpoint `args.device=cuda` |
| campaign total | ~60 GPU-h over 15 runs | sum of `hours` column |
| ablation baseline | 1.0630 | `analysis_e8l/analysis.json` |
| wrong-window control | 12.8174 | `analysis_e8l/analysis.json` |
| model params | 86.3M (`e8b_cache`) / 39.5M (`e8l_cache`) | `paper/results.md` |

**Caveats worth keeping in mind:**
- `e8l_cache` (39.5M, ctx 512) has the better *static* score (1.0176); the
  0.9756 headline is `e8b_cache` (86.3M, ctx 1024). Don't merge them.
- The ablation was run on `e8l_cache`, not on the headline model.
- The knowledge probe is a 65-question multiple-choice diagnostic (56.9%),
  not a benchmark — best left out of the posts.
