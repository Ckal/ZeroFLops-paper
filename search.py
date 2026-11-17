"""
search.py - find better settings overnight, like Karpathy's autoresearch, but
without an AI agent: it changes one setting at a time, trains a short run, and
keeps the change only if the validation score gets better.

  python search.py                     search on your text (preset "web"), until stopped
  python search.py --hours 8           stop after about 8 hours
  python search.py --preset enwik8     search on enwik8
  python search.py --preset enwik8 --only class_cache cache_decay
                                       try just these settings (every value), then stop
  python search.py --report            show what was found so far
  python search.py --tiny              2-minute test of the search itself

Safe to stop (Ctrl+C) and start again: every finished experiment is in
search/<preset>/results.csv and is never run twice.

How it decides:
  * Every experiment trains the same model size for the same number of steps
    and is scored on the same validation windows. The scoring is the
    trainer's and is never changed by the search.
  * It starts with the current settings, twice with different random seeds:
    the difference between the two shows how much a score wobbles by chance.
  * Then, again and again: pick one setting, try the next value up or down
    from the best so far. If the score improves by more than the wobble, that
    value becomes the new best ("kept"); otherwise it is forgotten.
  * When every neighbour of the best has been tried, it also tries
    combinations of two changes.

Short runs favour settings that help early. Confirm the winner with a full run:
the report prints the command.
"""
import argparse, csv, json, os, random, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))

# Values each setting may take, in order (neighbours = next value up or down).
SPACE = {
    "lr": [2e-4, 3e-4, 4.5e-4, 6e-4, 9e-4, 1.3e-3, 2e-3],
    "warmup": [250, 500, 1000, 2000],
    "min_lr_frac": [0.0, 0.03, 0.1, 0.3],
    "weight_decay": [0.0, 0.03, 0.1, 0.3],
    "dropout": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3],
    "arch": ["classic", "modern"],
    "shape": ["640x5x10", "512x8x8", "448x10x7", "384x14x6"],      # width x layers x heads, about the same size
    "doc_hist": [2048, 4096, 8192, 16384],
    "cache_keep": [1, 2, 4, 8, 16, 32],
    "cache_orders": ["3,6,12", "2,3,4,6,8,12,16", "2,3,4,5,6,8,10,12,16,24"],
    "cache_decay": [0, 500, 1500, 4000],
    "class_cache": [0, 1],
    "mem_drop": [0.0, 0.05, 0.1, 0.2],
}

PRESETS = {
    "web": dict(
        data=["--data", "data/smollm_subset_2GB.txt", "--val_data", "data/smollm_subset_200MB.txt",
              "--max_bytes", "400000000"],
        tables="runs/p_tab_hard.pt", ctx=256, batch=64, steps=20000,
        start=dict(lr=6e-4, warmup=1000, min_lr_frac=0.1, weight_decay=0.1, dropout=0.05, arch="classic",
                   shape="512x8x8", doc_hist=8192, cache_keep=8, cache_orders="2,3,4,6,8,12,16", mem_drop=0.0, cache_decay=0, class_cache=0)),
    "enwik8": dict(
        data=["--data", "data/enwik8.train", "--val_file", "data/enwik8.valid"],
        tables="runs/p_e8_tab_hard.pt", ctx=512, batch=32, steps=20000,
        start=dict(lr=4.5e-4, warmup=2000, min_lr_frac=0.1, weight_decay=0.1, dropout=0.2, arch="classic",
                   shape="512x8x8", doc_hist=8192, cache_keep=8, cache_orders="2,3,4,6,8,12,16", mem_drop=0.0, cache_decay=0, class_cache=0)),
}

TINY = dict(
    data=["--data", "data/tiny_search.txt"], tables="", ctx=64, batch=16, steps=150,
    start=dict(lr=2e-3, warmup=20, min_lr_frac=0.1, weight_decay=0.1, dropout=0.05, arch="classic",
               shape="64x2x2", doc_hist=1024, cache_keep=8, cache_orders="2,3,4,6,8,12,16", mem_drop=0.0, cache_decay=0, class_cache=0))
TINY_SPACE = dict(SPACE, lr=[5e-4, 1e-3, 2e-3, 4e-3], warmup=[10, 20, 40],
                  shape=["96x1x2", "64x2x2", "48x4x2"], doc_hist=[512, 1024, 2048])


ADDED_LATER = dict(cache_decay=0, class_cache=0)     # settings older result files do not have


def key(cfg):
    return json.dumps(cfg, sort_keys=True)


def trainer():
    for n in ("hard_residual_lm.py", "tg_train_hard.py"):
        if os.path.exists(os.path.join(HERE, n)):
            return os.path.join(HERE, n)
    raise SystemExit("hard_residual_lm.py not found next to search.py")


def command(P, cfg, seed, tag, out):
    d, L, H = (int(x) for x in cfg["shape"].split("x"))
    a = [sys.executable, trainer()] + P["data"] + [
        "--hard_feature", "--words", "--cache", "--rope",
        "--d", str(d), "--layers", str(L), "--heads", str(H), "--ctx", str(P["ctx"]), "--batch", str(P["batch"]),
        "--steps", str(P["steps"]), "--eval_every", str(max(1, P["steps"] // 4)), "--patience", "0",
        "--sample", "0", "--save_every", "0", "--seed", str(seed), "--out", out, "--tag", tag]
    a += ["--load_hard", P["tables"]] if P["tables"] else ["--vocab", "2000"]
    for k in ("lr", "warmup", "min_lr_frac", "weight_decay", "dropout", "arch", "doc_hist", "cache_keep",
              "cache_orders", "mem_drop", "cache_decay", "class_cache"):
        a += ["--" + k, str(cfg[k])]
    return a


def run(P, cfg, seed, n, folder, log):
    """Train one experiment; returns its best validation score (or None if it failed)."""
    tag = f"s{n:03d}"
    out = os.path.join(folder, "runs")
    os.makedirs(out, exist_ok=True)
    argv = command(P, cfg, seed, tag, out)
    t0 = time.time()
    with open(os.path.join(folder, "logs", tag + ".log"), "w", encoding="utf-8") as f:
        f.write(" ".join(argv) + "\n\n")
        f.flush()
        code = subprocess.call(argv, stdout=f, stderr=subprocess.STDOUT,
                               env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
    res = os.path.join(out, tag + "_result.json")
    score = json.load(open(res))["best_val"] if code == 0 and os.path.exists(res) else None
    for f in os.listdir(out):              # keep the disk free: only the small result files stay
        if f.startswith(tag) and f.endswith((".pt", ".tmp")):
            try:
                os.remove(os.path.join(out, f))
            except OSError:
                pass
    mins = (time.time() - t0) / 60
    log(f"  {tag}: {'FAILED, see logs/' + tag + '.log' if score is None else f'{score:.4f}'}  ({mins:.0f} min)")
    return score, mins


def load(folder):
    path = os.path.join(folder, "results.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["cfg"] = dict(ADDED_LATER, **json.loads(r["cfg"]))
        r["score"] = float(r["score"]) if r["score"] else None
        r["seed"] = int(r["seed"])
        r["minutes"] = float(r["minutes"])
        r["value"] = json.loads(r["value"]) if r.get("value") else ""
    return rows


def save(folder, rows):
    path = os.path.join(folder, "results.csv")
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "score", "kept", "change", "seed", "minutes", "param", "value", "cfg"])
        for r in rows:
            w.writerow([r["n"], "" if r["score"] is None else f"{r['score']:.4f}", r["kept"], r["change"],
                        r["seed"], f"{r['minutes']:.1f}", r.get("param", ""),
                        json.dumps(r.get("value", "")), key(r["cfg"])])
    os.replace(path + ".tmp", path)


def diff(cfg, base):
    return ", ".join(f"{k} {base[k]} -> {cfg[k]}" for k in cfg if cfg[k] != base.get(k)) or "start"


def state(rows, start):
    """Best settings so far, their score, and the chance wobble."""
    base = [r["score"] for r in rows if r["change"] == "start" and r["score"] is not None]
    wobble = max(0.003, abs(base[0] - base[1]) if len(base) >= 2 else 0.003)
    best, best_score = dict(start), (min(base) if base else None)
    for r in rows:
        if r["kept"] == "yes":
            best, best_score = r["cfg"], r["score"]
    return best, best_score, wobble


def neighbours(cfg, space):
    out = []
    for k, vals in space.items():
        if cfg[k] not in vals:
            continue
        i = vals.index(cfg[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append(dict(cfg, **{k: vals[j]}))
    return out


def report(folder, P, start, space, quiet=False):
    rows = load(folder)
    best, best_score, wobble = state(rows, start)
    done = [r for r in rows if r["score"] is not None]
    lines = [f"# Settings search ({os.path.basename(folder)})", "",
             f"{len(done)} experiments, {sum(r['minutes'] for r in rows) / 60:.1f} hours. Each: "
             f"{P['steps']:,} steps, scored on the validation windows of the trainer. Lower is better.", ""]
    if best_score is not None:
        start_scores = [r["score"] for r in rows if r["change"] == "start" and r["score"] is not None]
        lines += [f"- start: {min(start_scores):.4f}  (two seeds: "
                  f"{', '.join(f'{s:.4f}' for s in start_scores)}; chance wobble {wobble:.4f})",
                  f"- best:  {best_score:.4f}  ({min(start_scores) - best_score:+.4f} better)", "",
                  "Best settings: " + diff(best, start), ""]
        d, L, H = best["shape"].split("x")
        flags = (f"--d {d} --layers {L} --heads {H} --lr {best['lr']} --warmup {best['warmup']} "
                 f"--min_lr_frac {best['min_lr_frac']} --weight_decay {best['weight_decay']} "
                 f"--dropout {best['dropout']} --arch {best['arch']} --doc_hist {best['doc_hist']} "
                 f"--cache_keep {best['cache_keep']} --cache_orders {best['cache_orders']} "
                 f"--mem_drop {best['mem_drop']} --cache_decay {best['cache_decay']} "
                 f"--class_cache {best['class_cache']}")
        lines += ["For a full run, use these options (the warmup scales with the steps):", "", "```",
                  flags, "```", ""]
    lines += ["| # | score | kept | change |", "|---|---|---|---|"]
    lines += [f"| {r['n']} | {'failed' if r['score'] is None else format(r['score'], '.4f')} | {r['kept']} | "
              f"{r['change']} |" for r in rows]
    with open(os.path.join(folder, "search.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if not quiet:
        print("\n".join(lines[:12]))
        print(f"-> {os.path.join(folder, 'search.md')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), default="web")
    ap.add_argument("--hours", type=float, default=0, help="stop starting new experiments after this (0 = never)")
    ap.add_argument("--steps", type=int, default=0, help="training steps per experiment (default: the preset's)")
    ap.add_argument("--only", nargs="+", default=[], metavar="SETTING",
                    help="try only these settings, every value of each, on top of the best so far "
                         "(e.g. --only class_cache cache_decay), then stop")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    os.chdir(HERE)
    P, space = (dict(TINY), TINY_SPACE) if args.tiny else (dict(PRESETS[args.preset]), SPACE)
    if args.steps:
        P["steps"] = args.steps
    name = "tiny" if args.tiny else args.preset
    folder = os.path.join("search", name)
    os.makedirs(os.path.join(folder, "logs"), exist_ok=True)
    if args.tiny and not os.path.exists("data/tiny_search.txt"):
        src = next((f for f in ("data/smollm_subset_200MB.txt", "data/tinyshakespeare.txt") if os.path.exists(f)),
                   None)
        if src is None:
            raise SystemExit("--tiny needs some text in data/")
        with open(src, "rb") as f, open("data/tiny_search.txt", "wb") as g:
            g.write(f.read(1_500_000))
    start = P["start"]
    for k in args.only:
        if k not in space:
            raise SystemExit(f"--only {k}: not a setting. Settings: {', '.join(space)}")
    if args.report:
        return report(folder, P, start, space)
    for f in P["data"][1::2] + ([P["tables"]] if P["tables"] else []):
        if f.endswith((".txt", ".pt", ".train", ".valid")) and not os.path.exists(f):
            raise SystemExit(f"not found: {f}  (run the matching run_all.py stage first, or edit PRESETS)")
    rows = load(folder)
    rng = random.Random(args.seed + len(rows))
    t_end = time.time() + args.hours * 3600 if args.hours else None
    logf = open(os.path.join(folder, "search.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(time.strftime("%Y-%m-%d %H:%M  ") + msg + "\n")
        logf.flush()

    log(f"search on {name}: {P['steps']:,} steps per experiment, {len(rows)} done so far")
    tried = {key(r["cfg"]) for r in rows if r["seed"] == 0}
    while True:
        if t_end and time.time() > t_end:
            log("time is up")
            break
        free = shutil.disk_usage(HERE).free / 2 ** 30
        if free < 3:
            log(f"stopping: only {free:.1f} GB free on the disk")
            break
        best, best_score, wobble = state(rows, start)
        starts = [r for r in rows if r["change"] == "start"]
        if len(starts) < 2:                       # the starting settings, twice: how big is chance?
            cfg, seed, change = dict(start), len(starts), "start"
        else:
            if args.only:
                cands = [dict(best, **{k: v}) for k in args.only for v in space[k]]
                cands = [c for c in cands if key(c) not in tried and c != best]
                if not cands:
                    log("--only: every value of " + ", ".join(args.only) + " has been run")
                    break
            else:
                cands = [c for c in neighbours(best, space) if key(c) not in tried]
            if not cands:                         # all single steps tried: combine two near-misses
                near = sorted([r for r in rows if r["score"] is not None and r["kept"] != "yes"
                               and r.get("param")], key=lambda r: r["score"])[:6]
                cands = []
                for i in range(len(near)):
                    for j in range(i + 1, len(near)):
                        a, b = near[i], near[j]
                        if a["param"] == b["param"]:
                            continue
                        c = dict(best, **{a["param"]: a["value"], b["param"]: b["value"]})
                        if key(c) not in tried and c != best:
                            cands.append(c)
            if not cands:
                log("nothing left to try: every neighbour and pair has been run")
                break
            cfg, seed = rng.choice(cands), 0
            change = diff(cfg, best)
        n = len(rows) + 1
        log(f"#{n}  {change}" + (f"  (best so far {best_score:.4f})" if best_score is not None else ""))
        score, mins = run(P, cfg, seed, n, folder, log)
        kept = "no"
        if change != "start" and score is not None and best_score is not None and score < best_score - wobble:
            kept = "yes"
            log(f"  -> kept: {best_score:.4f} -> {score:.4f}")
        ch = [k for k in cfg if cfg[k] != best[k]]
        rows.append(dict(n=n, score=score, kept=kept, change=change, seed=seed, minutes=mins, cfg=cfg,
                         param=ch[0] if len(ch) == 1 and change != "start" else "",
                         value=cfg[ch[0]] if len(ch) == 1 and change != "start" else ""))
        if seed == 0:
            tried.add(key(cfg))
        save(folder, rows)
        report(folder, P, start, space, quiet=True)
    report(folder, P, start, space)


if __name__ == "__main__":
    main()
