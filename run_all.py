"""
run_all.py - the whole experiment for the paper, start to finish.

  python run_all.py                    everything, in order
  python run_all.py --list             the stages, their runs and what is done
  python run_all.py --stages tables pilot
  python run_all.py --dry              only print the commands
  python run_all.py --report           rebuild paper/results.md from what exists
  python run_all.py --tiny             10-minute test of the whole pipeline with tiny models

Safe to stop (Ctrl+C, reboot, crash) and start again: finished runs are
skipped, an interrupted training run continues from its last save.

Stages, in order:
  tables    count the tables (test text taken out first); score them; measure
            what the document cache adds and whether the tables look better on
            their own text than on new text
  pilot     short runs of the main model: plain / tables / tables + cache /
            + rotary positions. Tells within hours whether everything works at
            full scale, and decides whether the long runs use rotary positions
  main      the paper's core comparison at equal steps: plain / tables / cache
  gold      the best model this PC can train: bigger, longer context, cache
  context   main-size models with the long context (plain and cache)
  loo       tables from half the text: train on the counted half vs the other
            half, then score both with the full tables
  analyze   analyze.py on the finished models
  report    paper/results.md, results.csv, curves.png, samples

Extra stages, only run when named (python run_all.py --stages ablate):
  ablate    pilot-size runs that change one thing: RMSNorm + SwiGLU, weight
            tying, a shorter cache (2,048 bytes), memory dropout
  teacher   a large language model labels part of the text; a pilot-size, a
            main-size and a gold2-size model then also learn from its predictions
  enwik8    the standard benchmark (100 MB of Wikipedia, fixed train / valid /
            test files), so the method can be compared with published numbers
  gold2     gold again with the settings search.py found (the modern block, the
            cache settings), AND a plain model of the same size with the same
            block: how much the memory is worth at 86M instead of 26M
  e8final   enwik8 again with the settings "search.py --preset enwik8" found:
            memory and plain at 39M, then memory and plain at 86M
  dyn       dynamic evaluation of the finished models: while reading the test
            text the model takes a small training step on what it has just
            scored (--dyn_lr); reported next to the normal test score
  domains   downloads 20 MB each of Python code, English and German Wikipedia
            (text none of the models trained on) and scores the finished models
            on them; then the copy test of analyze.py on code

After every run the report is rebuilt and printed, so results can be read while
the rest is still training. Everything for the paper lands in paper/:
  results.md  results.csv  curves.png  probe.png     the numbers and curves
  samples/<run>_samples.txt     probe score and sample texts during training
  analysis_*/                   analyze.py output
  logs/<run>.log                full output of every run
  commands.log                  every command, with time and script checksum

Edit CFG below for other files or sizes.
"""
import argparse, csv, glob, hashlib, json, os, shutil, subprocess, sys, time, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

CFG = dict(
    # ---- text
    data="data/smollm_subset_2GB.txt",
    val_data="data/smollm_subset_200MB.txt",   # validation comes from this file's split ("" = from data)
    test_mb=10,                 # 1 MB blocks taken out of the training text as the test set
    max_bytes=0,                # use only the first N bytes of data (0 = all)
    table_bytes=0,              # 0 = tables from all training text. With a bigger data file, e.g.
                                # 2_000_000_000: tables from the first 2 GB, training on all of it
    train_part="all",           # with table_bytes: "all", or "rest" = train only on text the tables
                                # have never seen
    # ---- where things go
    out="runs", paper="paper", prefix="p_",
    # ---- scoring and paper trail
    eval_every=1000, final_windows=20480, sample_every=10, sample_len=160,
    trackio_project="",         # e.g. "zeroflops": live curves with trackio (pip install trackio)
    trackio_space="",           # e.g. "yourname/zeroflops": also send them to this Hugging Face Space
    # ---- models
    rope="auto",                # rotary positions: "auto" = decided by the pilot, True, False
    main=dict(d=512, layers=8, heads=8, ctx=256, batch=64, steps=150000, lr=6e-4, warmup=1000,
              dropout=0.05),
    pilot_steps=20000,
    long_ctx=dict(ctx=1024, batch=16),
    gold=dict(d=768, layers=12, heads=12, ctx=1024, batch=16, steps=250000, lr=3e-4, warmup=2000,
              dropout=0.1),
    # ---- extra stages
    teacher=dict(model="Qwen/Qwen2.5-1.5B", label_mb=200, alpha=0.8, frac=0.5),
    enwik8=dict(file="data/enwik8", d=512, layers=12, heads=8, ctx=512, batch=32, steps=200000, lr=4e-4,
                warmup=2000, dropout=0.2),
    gold2=dict(d=768, layers=12, heads=12, ctx=1024, batch=16, steps=250000, lr=3e-4, warmup=2000,
               dropout=0.1),
    enwik8_big=dict(d=768, layers=12, heads=12, ctx=1024, batch=16, steps=250000, lr=3e-4, warmup=2000,
                    dropout=0.3),
    # learning while reading: mode adam or rms; pick lr on validation text, never on test
    dyn=dict(mode="adam", lr=1e-5, decay=0.002, batch=1),
    # used when search.py has not found anything yet; its results replace these
    gold2_settings=dict(arch="modern", cache_keep=4, doc_hist=8192, cache_orders="2,3,4,6,8,12,16",
                        mem_drop=0.0, weight_decay=0.1, min_lr_frac=0.1, cache_decay=0, class_cache=0),
    search_folder="search/web",
    domains=[   # name, downloader options (see download_hf_corpus.py)
        ("code", ["--dataset", "codeparrot/codeparrot-clean-valid", "--config", "default",
                  "--text-field", "content", "--keep-indent"]),
        ("wiki_en", ["--dataset", "wikimedia/wikipedia", "--config", "20231101.en"]),
        ("wiki_de", ["--dataset", "wikimedia/wikipedia", "--config", "20231101.de"]),
    ],
    domain_size="20MB",
)

TINY = dict(
    max_bytes=20_000_000, test_mb=1, out="runs_tiny", paper="paper_tiny", prefix="t_",
    eval_every=50, final_windows=128, sample_every=2, sample_len=40,
    main=dict(d=64, layers=2, heads=2, ctx=64, batch=16, steps=200, lr=2e-3, warmup=20, dropout=0.05),
    pilot_steps=100,
    long_ctx=dict(ctx=128, batch=8),
    gold=dict(d=96, layers=3, heads=2, ctx=128, batch=8, steps=200, lr=2e-3, warmup=20, dropout=0.1),
    teacher=dict(model="Qwen/Qwen2.5-1.5B", label_mb=1, alpha=0.8, frac=0.5),
    enwik8=dict(file="data/enwik8", d=64, layers=2, heads=2, ctx=64, batch=16, steps=200, lr=2e-3,
                warmup=20, dropout=0.1),
    gold2=dict(d=96, layers=3, heads=2, ctx=128, batch=8, steps=200, lr=2e-3, warmup=20, dropout=0.1),
    enwik8_big=dict(d=96, layers=3, heads=2, ctx=128, batch=8, steps=200, lr=2e-3, warmup=20, dropout=0.1),
    dyn=dict(mode="rms", lr=1e-4, decay=0.002, batch=8),
    search_folder="search/tiny",
    domain_size="2MB",
    tiny=True,
)

ORDER = ["tables", "pilot", "main", "gold", "context", "loo", "analyze", "report"]
EXTRA = ["ablate", "teacher", "enwik8", "gold2", "e8final", "dyn", "domains"]      # only when named with --stages


# ------------------------------------------------------------------ helpers
def P(tag):
    return CFG["prefix"] + tag


def rp(*parts):
    return os.path.join(CFG["out"], *parts)


def pp(*parts):
    return os.path.join(CFG["paper"], *parts)


def script(name):
    return os.path.join(HERE, name)


def trainer():
    for n in ("hard_residual_lm.py", "tg_train_hard.py"):
        if os.path.exists(script(n)):
            return script(n)
    raise SystemExit("hard_residual_lm.py not found next to run_all.py")


def data_args(test=True):
    a = ["--data", CFG["data"]]
    if CFG["val_data"]:
        a += ["--val_data", CFG["val_data"]]
    if CFG["max_bytes"]:
        a += ["--max_bytes", str(CFG["max_bytes"])]
    if test and CFG["test_mb"]:
        a += ["--test_mb", str(CFG["test_mb"])]
    return a


def score_text_args():
    """The text --mode eval and analyze.py score: the validation split, as in training."""
    if CFG["val_data"]:
        return ["--val_data", CFG["val_data"]]
    return ["--data", CFG["data"]] + (["--max_bytes", str(CFG["max_bytes"])] if CFG["max_bytes"] else [])


def track_args():
    a = ["--prompts", script("prompts.txt"), "--probe", script("probe.json"),
         "--sample_every", str(CFG["sample_every"]), "--sample_len", str(CFG["sample_len"])]
    if CFG["trackio_project"]:
        a += ["--trackio", CFG["trackio_project"]]
        if CFG["trackio_space"]:
            a += ["--trackio_space", CFG["trackio_space"]]
    return a


def model_args(m, rope):
    a = []
    for k in ("d", "layers", "heads", "ctx", "batch", "steps", "lr", "warmup", "dropout"):
        a += ["--" + k, str(m[k])]
    return a + (["--rope"] if rope else [])


def train_cmd(tag, m, memory, rope=False, hard="tab", extra=(), data=None):
    """memory: alone | tables | cache. data: other text arguments than CFG's (benchmarks)."""
    a = [sys.executable, trainer()] + (data if data is not None else data_args()) + model_args(m, rope) + [
        "--out", CFG["out"], "--tag", P(tag), "--eval_every", str(CFG["eval_every"]), "--patience", "0",
        "--final_eval_windows", str(CFG["final_windows"]), "--sample", "0", "--resume",
        "--save_every", "2"] + track_args()
    if memory != "alone":
        a += ["--hard_feature", "--words", "--load_hard", rp(P(hard) + "_hard.pt")]
        if memory == "cache":
            a += ["--cache"]
        if CFG["table_bytes"] and CFG["train_part"] != "all" and hard == "tab" and "--teacher" not in extra:
            a += ["--train_part", CFG["train_part"]]
    return a + list(extra)


def decide_rope(quiet=False):
    """True / False, or None while the pilot has not decided yet."""
    if isinstance(CFG["rope"], bool):
        return CFG["rope"]
    dec = pp("decisions.json")
    if os.path.exists(dec):
        return json.load(open(dec)).get("rope")
    a, b = rp(P("pilot_cache") + "_result.json"), rp(P("pilot_cache_rope") + "_result.json")
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    va, vb = (json.load(open(f)) for f in (a, b))
    va, vb = va.get("val", va["best_val"]), vb.get("val", vb["best_val"])
    use = bool(vb < va - 0.002)
    os.makedirs(CFG["paper"], exist_ok=True)
    json.dump(dict(rope=use, learned_positions=va, rotary_positions=vb,
                   rule="rotary positions are used if they are more than 0.002 BPC better in the pilot"),
              open(dec, "w"), indent=1)
    if not quiet:
        print(f"decision: rotary positions {'ON' if use else 'OFF'} for the long runs "
              f"(pilot: learned {va:.4f}, rotary {vb:.4f})")
    return use


# ------------------------------------------------------------------ stages
def jobs_tables():
    a = [sys.executable, trainer(), "--mode", "hard"] + data_args() + [
        "--words", "--cache", "--check_loo", "--probe", script("probe.json"), "--sample", "0",
        "--out", CFG["out"], "--tag", P("tab")]
    if CFG["table_bytes"]:
        a += ["--table_bytes", str(CFG["table_bytes"])]
    return [dict(name="tables", argv=a, disk_gb=table_gb(CFG["table_bytes"] or text_bytes()))]


def jobs_pilot():
    m = dict(CFG["main"], steps=CFG["pilot_steps"], warmup=min(CFG["main"]["warmup"], CFG["pilot_steps"] // 10))
    need = [rp(P("tab") + "_hard.pt")]
    j = [dict(name="pilot_alone", argv=train_cmd("pilot_alone", m, "alone")),
         dict(name="pilot_tables", argv=train_cmd("pilot_tables", m, "tables"), needs=need),
         dict(name="pilot_cache", argv=train_cmd("pilot_cache", m, "cache"), needs=need)]
    if CFG["rope"] == "auto":
        j.append(dict(name="pilot_cache_rope", argv=train_cmd("pilot_cache_rope", m, "cache", rope=True),
                      needs=need))
    return j + [a for a in jobs_analyze() if a["name"] == "analysis_pilot"]     # a first look inside


def jobs_main():
    r = bool(decide_rope(quiet=True))
    need = [rp(P("tab") + "_hard.pt")]
    m = CFG["main"]
    return [dict(name="main_cache", argv=train_cmd("main_cache", m, "cache", r), needs=need),
            dict(name="main_alone", argv=train_cmd("main_alone", m, "alone", r)),
            dict(name="main_tables", argv=train_cmd("main_tables", m, "tables", r), needs=need)]


def jobs_gold():
    r = bool(decide_rope(quiet=True))
    return [dict(name="gold", argv=train_cmd("gold", CFG["gold"], "cache", r),
                 needs=[rp(P("tab") + "_hard.pt")], disk_gb=4)]


def text_bytes():
    size = os.path.getsize(CFG["data"]) if os.path.exists(CFG["data"]) else 0
    return min(size, CFG["max_bytes"]) if CFG["max_bytes"] else size


def table_gb(n_bytes):
    """Disk space a table file for this much text needs, roughly (plus room to write it safely)."""
    return 1 + 8 * n_bytes / 2 ** 30


def pilot_model():
    return dict(CFG["main"], steps=CFG["pilot_steps"],
                warmup=min(CFG["main"]["warmup"], CFG["pilot_steps"] // 10))


def jobs_ablate():
    """Pilot-size runs that change one thing. Compare with pilot_cache_rope (or pilot_cache)."""
    r = bool(decide_rope(quiet=True))
    need = [rp(P("tab") + "_hard.pt")]
    runs = [("abl_modern", ["--arch", "modern"]), ("abl_modern_tie", ["--arch", "modern", "--tie"]),
            ("abl_hist2k", ["--doc_hist", "2048"]), ("abl_memdrop", ["--mem_drop", "0.1"])]
    return [dict(name=t, argv=train_cmd(t, pilot_model(), "cache", r, extra=e), needs=need) for t, e in runs]


def jobs_teacher():
    """A large model labels the first label_mb of the training text; two models also learn from it."""
    t = CFG["teacher"]
    out = rp(P("teacher"))
    label = [sys.executable, script("label_teacher.py"), "--data", CFG["data"],
             "--max_bytes", str(CFG["max_bytes"]), "--label_bytes", str(t["label_mb"] * 1_000_000),
             "--model", t["model"], "--out", out, "--trainer", trainer(), "--runs", CFG["out"]]
    if CFG["val_data"]:
        label += ["--val_data", CFG["val_data"]]
    if CFG["test_mb"]:
        label += ["--test_mb", str(CFG["test_mb"])]
    meta = os.path.join(out, "meta.json")
    ex = ["--teacher", out, "--distill_only", "--distill_alpha", str(t["alpha"]), "--teacher_frac", str(t["frac"])]
    r = bool(decide_rope(quiet=True))
    need = [rp(P("tab") + "_hard.pt"), meta]
    return [dict(name="teacher_labels", argv=label, done_file=meta, disk_gb=1 + 0.024 * t["label_mb"]),
            dict(name="teach_pilot", argv=train_cmd("teach_pilot", pilot_model(), "cache", r, extra=ex), needs=need),
            dict(name="teach_main", argv=train_cmd("teach_main", CFG["main"], "cache", r, extra=ex), needs=need),
            # gold2 size and settings, plus the teacher: compare with gold2 (same everything, no teacher)
            dict(name="teach_gold2", needs=need, disk_gb=4,
                 argv=train_cmd("teach_gold2", CFG["gold2"], "cache", True,
                                extra=settings_flags(search_settings()[0]) + ex))]


def prepare_enwik8():
    """Download enwik8 and cut it the standard way: 90 MB train, 5 MB valid, 5 MB test."""
    f = CFG["enwik8"]["file"]
    parts = [f + e for e in (".train", ".valid", ".test")]
    if all(os.path.exists(x) for x in parts):
        return True
    if not os.path.exists(f):
        url = "http://mattmahoney.net/dc/enwik8.zip"
        print(f"downloading {url} (36 MB) ...", flush=True)
        try:
            os.makedirs(os.path.dirname(f) or ".", exist_ok=True)
            urllib.request.urlretrieve(url, f + ".zip")
            with zipfile.ZipFile(f + ".zip") as z, z.open("enwik8") as src, open(f, "wb") as dst:
                shutil.copyfileobj(src, dst)
        except Exception as e:
            print(f"could not download enwik8 ({e}).\nDownload {url} yourself, unzip it and save the file as {f}")
            return False
    data = open(f, "rb").read()
    n = len(data)
    if n == 100_000_000:
        a, b = 90_000_000, 95_000_000
    elif CFG.get("tiny"):
        a, b = int(n * 0.90), int(n * 0.95)          # a stand-in file in the quick test
    else:
        print(f"{f} has {n:,} bytes; enwik8 has exactly 100,000,000. Delete it and start again.")
        return False
    for path, chunk in zip(parts, (data[:a], data[a:b], data[b:])):
        with open(path, "wb") as out:
            out.write(chunk)
    print(f"enwik8: {a:,} train, {b - a:,} valid, {n - b:,} test bytes")
    return True


def jobs_enwik8():
    e = CFG["enwik8"]
    f = e["file"]
    d = ["--data", f + ".train", "--val_file", f + ".valid", "--test_file", f + ".test"]
    r = decide_rope(quiet=True)
    r = True if r is None else r
    tab = rp(P("e8_tab") + "_hard.pt")
    build = [sys.executable, trainer(), "--mode", "hard", "--data", f + ".train", "--val_file", f + ".valid",
             "--words", "--cache", "--check_loo", "--probe", script("probe.json"), "--sample", "0",
             "--out", CFG["out"], "--tag", P("e8_tab")]
    files = [f + ".train"]
    j = [dict(name="e8_files", func=prepare_enwik8, done_file=f + ".test"),
         dict(name="e8_tables", argv=build, needs=files, disk_gb=2),
         dict(name="e8_cache", needs=files + [tab],
              argv=train_cmd("e8_cache", e, "cache", r, hard="e8_tab", data=d, extra=["--slide_eval"])),
         dict(name="e8_alone", needs=files,
              argv=train_cmd("e8_alone", e, "alone", r, data=d, extra=["--slide_eval"]))]
    fc, fa = rp(P("e8_cache") + "_best.pt"), rp(P("e8_alone") + "_best.pt")
    j.append(dict(name="analysis_e8", needs=[fc, fa], done_file=pp("analysis_e8", "analysis.json"),
                  argv=[sys.executable, script("analyze.py"), "--feat", fc, "--alone", fa,
                        "--test_file", f + ".valid", "--out", pp("analysis_e8"), "--trainer", trainer()]))
    return j


def jobs_context():
    r = bool(decide_rope(quiet=True))
    m = dict(CFG["main"], **CFG["long_ctx"])
    return [dict(name="ctx_cache", argv=train_cmd("ctx_cache", m, "cache", r), needs=[rp(P("tab") + "_hard.pt")]),
            dict(name="ctx_alone", argv=train_cmd("ctx_alone", m, "alone", r))]


def jobs_loo():
    if CFG["table_bytes"]:
        return []                         # the main recipe already has tables from part of the text
    size = os.path.getsize(CFG["data"]) if os.path.exists(CFG["data"]) else 0
    if CFG["max_bytes"]:
        size = min(size, CFG["max_bytes"])
    half = size // 2
    full = rp(P("tab") + "_hard.pt")
    build = [sys.executable, trainer(), "--mode", "hard"] + data_args() + [
        "--words", "--classes_from", full, "--table_bytes", str(half), "--cache", "--check_loo",
        "--sample", "0", "--out", CFG["out"], "--tag", P("half")]
    m = dict(CFG["main"], steps=CFG["pilot_steps"], warmup=min(CFG["main"]["warmup"], CFG["pilot_steps"] // 10))
    r = bool(decide_rope(quiet=True))
    j = [dict(name="loo_half_tables", argv=build, needs=[full], disk_gb=table_gb(half))]
    for part in ("tables", "rest"):
        tag = f"loo_{part}"
        j.append(dict(name=tag, needs=[rp(P("half") + "_hard.pt")],
                      argv=train_cmd(tag, m, "cache", r, hard="half", extra=["--train_part", part])))
        j.append(dict(name=tag + "_full", needs=[rp(P(tag) + "_best.pt"), full],
                      argv=[sys.executable, trainer(), "--mode", "eval", "--ckpt", rp(P(tag) + "_best.pt"),
                            "--load_hard", full] + score_text_args() +
                      ["--eval_windows", str(CFG["final_windows"]), "--probe", script("probe.json"),
                       "--out", CFG["out"], "--tag", P(tag + "_full")]))
    return j


def jobs_analyze():
    j = []
    for name, feat, alone in (("analysis_main", "main_cache", "main_alone"),
                              ("analysis_tables", "main_tables", "main_alone"),
                              ("analysis_gold", "gold", "ctx_alone"),
                              ("analysis_pilot", "pilot_cache", "pilot_alone")):
        f, a = rp(P(feat) + "_best.pt"), rp(P(alone) + "_best.pt")
        argv = [sys.executable, script("analyze.py"), "--feat", f] + score_text_args() + [
            "--out", pp(name), "--trainer", trainer()]
        if os.path.exists(a) or name != "analysis_gold":
            argv += ["--alone", a]
        j.append(dict(name=name, argv=argv, needs=[f] + ([a] if "--alone" in argv else []),
                      done_file=pp(name, "analysis.json"),
                      stale=lambda f=f, n=name: os.path.exists(pp(n, "analysis.json")) and
                      os.path.getmtime(f) > os.path.getmtime(pp(n, "analysis.json"))))
    return j


def search_settings(preset="web"):
    """The best settings search.py has found (and where they came from), else CFG's defaults.
    Only the block and the memory settings are taken over. Learning rate, dropout and
    shape are not: a 20,000-step search favours what learns fast, not what ends best."""
    keys = list(CFG["gold2_settings"])
    folder = CFG["search_folder"] if preset == "web" else os.path.join(os.path.dirname(CFG["search_folder"]), preset)
    if CFG.get("tiny"):
        folder = CFG["search_folder"]
    res = os.path.join(folder, "results.csv")
    if os.path.exists(res) and os.path.exists(script("search.py")):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("search_mod", script("search.py"))
            sm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sm)
            start = (sm.TINY if CFG.get("tiny") else sm.PRESETS[preset])["start"]
            rows = sm.load(folder)
            best, score, _ = sm.state(rows, start)
            if score is not None:
                return {k: best[k] for k in keys if k in best}, f"search.py ({len(rows)} experiments, best {score:.4f})"
        except Exception as e:
            print(f"  warning: could not read the search results ({e}); using the defaults in CFG")
    return dict(CFG["gold2_settings"]), "the defaults in CFG (no search results yet)"


def settings_flags(st, memory=True):
    keys = list(st) if memory else ["arch", "weight_decay", "min_lr_frac"]
    return [x for k in keys for x in ("--" + k, str(st[k]))]


def jobs_gold2():
    st, src = search_settings()
    m = CFG["gold2"]
    need = [rp(P("tab") + "_hard.pt")]
    print(f"gold2 settings from {src}: " + ", ".join(f"{k} {v}" for k, v in st.items()))
    g, a = rp(P("gold2") + "_best.pt"), rp(P("gold2_alone") + "_best.pt")
    return [dict(name="gold2", argv=train_cmd("gold2", m, "cache", True, extra=settings_flags(st)), needs=need,
                 disk_gb=4),
            dict(name="gold2_alone", argv=train_cmd("gold2_alone", m, "alone", True,
                                                    extra=settings_flags(st, memory=False)), disk_gb=4),
            dict(name="analysis_gold2", needs=[g, a], done_file=pp("analysis_gold2", "analysis.json"),
                 argv=[sys.executable, script("analyze.py"), "--feat", g, "--alone", a] + score_text_args() +
                 ["--out", pp("analysis_gold2"), "--trainer", trainer()])]


def jobs_e8final():
    st, src = search_settings("enwik8")
    print(f"e8final settings from {src}: " + ", ".join(f"{k} {v}" for k, v in st.items()))
    f = CFG["enwik8"]["file"]
    d = ["--data", f + ".train", "--val_file", f + ".valid", "--test_file", f + ".test"]
    files, tab = [f + ".train"], rp(P("e8_tab") + "_hard.pt")
    j = []
    small = CFG["enwik8"] if CFG.get("tiny") else dict(CFG["enwik8"], lr=6e-4)   # the search liked a higher rate
    for tag, m in (("e8f", small), ("e8b", CFG["enwik8_big"])):
        j += [dict(name=tag + "_cache", needs=files + [tab], disk_gb=4,
                   argv=train_cmd(tag + "_cache", m, "cache", True, hard="e8_tab", data=d,
                                  extra=["--slide_eval"] + settings_flags(st))),
              dict(name=tag + "_alone", needs=files, disk_gb=4,
                   argv=train_cmd(tag + "_alone", m, "alone", True, data=d,
                                  extra=["--slide_eval"] + settings_flags(st, memory=False)))]
    return j


DYN_MODELS = ["teach_gold2", "e8b_cache", "e8b_alone", "e8f_cache", "e8f_alone", "e8_cache", "e8_alone",
              "gold2", "gold2_alone", "gold", "main_cache", "main_alone"]


def jobs_dyn():
    """Dynamic evaluation of every finished model in DYN_MODELS, on its test text."""
    y, j = CFG["dyn"], []
    web = rp(f"{os.path.splitext(os.path.basename(os.path.normpath(CFG['data'])))[0]}_test{CFG['test_mb']}mb.txt")
    for run in DYN_MODELS:
        ck = rp(P(run) + "_best.pt")
        e8 = run.startswith("e8")
        text = CFG["enwik8"]["file"] + ".test" if e8 else web
        train_text = CFG["enwik8"]["file"] + ".train" if e8 else CFG["data"]
        if os.path.exists(ck):
            j.append(dict(name="dyn__" + run, needs=[ck, text],
                          argv=[sys.executable, trainer(), "--mode", "eval", "--ckpt", ck, "--test_file", text,
                                "--slide_eval", "--dyn_lr", str(y["lr"]), "--dyn_decay", str(y["decay"]),
                                "--dyn_batch", str(y["batch"]), "--dyn_mode", y["mode"],
                                "--data", train_text, "--out", CFG["out"], "--tag", P("dyn__" + run)]))
    if not j:
        print("dyn: no finished models yet")
    return j


DOMAIN_MODELS = ["main_alone", "main_tables", "main_cache", "gold", "gold2_alone", "gold2", "e8_alone", "e8_cache"]


def jobs_domains():
    j = []
    for name, opts in CFG["domains"]:
        f = os.path.join("data", f"domain_{name}.txt")
        j.append(dict(name=f"get_{name}", done_file=f,
                      argv=[sys.executable, script("download_hf_corpus.py")] + opts +
                      ["--size", CFG["domain_size"], "--output", f]))
        for run in DOMAIN_MODELS:
            ck = rp(P(run) + "_best.pt")
            if not os.path.exists(ck):
                continue
            j.append(dict(name=f"dom_{name}__{run}", needs=[f, ck],
                          argv=[sys.executable, trainer(), "--mode", "eval", "--ckpt", ck, "--test_file", f,
                                "--eval_windows", "2048", "--out", CFG["out"], "--tag", P(f"dom_{name}__{run}")]))
    code = os.path.join("data", "domain_code.txt")
    for feat, alone in (("gold2", "gold2_alone"), ("main_cache", "main_alone")):
        fc, fa = rp(P(feat) + "_best.pt"), rp(P(alone) + "_best.pt")
        if os.path.exists(fc) and os.path.exists(fa):
            j.append(dict(name=f"analysis_code_{feat}", needs=[code], done_file=pp(f"analysis_code_{feat}", "analysis.json"),
                          argv=[sys.executable, script("analyze.py"), "--feat", fc, "--alone", fa,
                                "--test_file", code, "--out", pp(f"analysis_code_{feat}"), "--trainer", trainer()]))
            break
    return j


STAGES = dict(e8final=jobs_e8final, dyn=jobs_dyn, gold2=jobs_gold2, domains=jobs_domains, ablate=jobs_ablate, teacher=jobs_teacher, enwik8=jobs_enwik8,
              tables=jobs_tables, pilot=jobs_pilot, main=jobs_main, gold=jobs_gold,
              context=jobs_context, loo=jobs_loo, analyze=jobs_analyze, report=lambda: [])


# ------------------------------------------------------------------ running
def marker(name):
    return pp("done", name + ".ok")


def is_done(job):
    if job.get("stale") and job["stale"]():
        return False
    return os.path.exists(marker(job["name"]))


def sha(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()[:10]


def keep_awake(on):
    if os.name == "nt":                       # do not let Windows go to sleep during a run
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if on else 0x80000000)
        except Exception:
            pass


def run_job(job, dry=False):
    """Returns "ran", "done", "skipped" or "failed"."""
    name, argv = job["name"], job.get("argv")
    if is_done(job):
        print(f"[done]    {name}")
        return "done"
    line = " ".join(f'"{a}"' if " " in a else a for a in argv) if argv else job["func"].__doc__.strip()
    if dry:
        print(f"[would run] {name}\n    {line}")
        return "skipped"
    missing = [n for n in job.get("needs", []) if not os.path.exists(n)]
    if missing:
        print(f"[skipped] {name}: needs {missing[0]} (an earlier stage has not produced it yet)")
        return "skipped"
    for d in (pp("logs"), pp("done"), CFG["out"]):
        os.makedirs(d, exist_ok=True)
    need = job.get("disk_gb", 2)
    free = shutil.disk_usage(os.path.abspath(CFG["out"])).free / 2 ** 30
    if free < need:
        print(f"\n[STOPPED] {name}: only {free:.1f} GB free on the drive of {CFG['out']}{os.sep}, this step "
              f"needs about {need:.0f} GB.\n          Free some space (old *_hard.pt table files are the "
              f"biggest), then start again: nothing is lost.")
        return "failed"
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if not argv:                                   # a step done by this script itself
        print(f"\n[run]     {name}   ({stamp})\n    {line}", flush=True)
        if not job["func"]():
            print(f"[FAILED]  {name}")
            return "failed"
        with open(marker(name), "w") as f:
            f.write(stamp + "\n")
        print(f"[ok]      {name}")
        return "ran"
    with open(pp("commands.log"), "a", encoding="utf-8") as f:
        f.write(f"{stamp}  {name}  [trainer {sha(trainer())}]\n    {line}\n")
    print(f"\n{'=' * 100}\n[run]     {name}   ({stamp})\n    {line}\n{'=' * 100}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    with open(pp("logs", name + ".log"), "a", encoding="utf-8") as log:
        log.write(f"\n##### {stamp}\n##### {line}\n")
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            log_ok = True
            for out in proc.stdout:
                sys.stdout.write(out)
                sys.stdout.flush()
                if log_ok:
                    try:
                        log.write(out)
                        log.flush()
                    except OSError as e:           # e.g. disk full: the run itself goes on
                        log_ok = False
                        print(f"  warning: cannot write the log file ({e})")
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print(f"\nstopped during {name}. Start run_all.py again to continue where it was.")
            raise
    mins = (time.time() - t0) / 60
    if code != 0:
        print(f"\n[FAILED]  {name} (exit code {code}) after {mins:.1f} min. Log: {pp('logs', name + '.log')}")
        return "failed"
    if job.get("done_file") and not os.path.exists(job["done_file"]):
        print(f"\n[FAILED]  {name}: finished but did not write {job['done_file']}")
        return "failed"
    with open(marker(name), "w") as f:
        f.write(f"{stamp}  {mins:.1f} min\n")
    print(f"[ok]      {name}  ({mins:.1f} min)")
    return "ran"


# ------------------------------------------------------------------ report
RUN_ORDER = ["pilot_alone", "pilot_tables", "pilot_cache", "pilot_cache_rope", "abl_modern", "abl_modern_tie",
             "abl_hist2k", "abl_memdrop", "teach_pilot", "main_alone", "main_tables", "main_cache",
             "teach_main", "ctx_alone", "ctx_cache", "gold", "loo_tables", "loo_rest", "gold2_alone", "gold2", "teach_gold2", "e8_alone", "e8_cache",
             "e8f_alone", "e8f_cache", "e8b_alone", "e8b_cache"]
COLS = [("run", "run"), ("status", "status"), ("memory", "memory"), ("params_M", "params (M)"), ("ctx", "ctx"),
        ("rotary", "rotary"), ("steps", "steps"), ("hours", "hours"), ("best_val", "best val"), ("val", "val"),
        ("val_warm", "val warm"), ("test", "test"), ("test_warm", "test warm"), ("acc", "next byte right"),
        ("word_start", "first letters"), ("probe", "probe")]
SCREEN = ["run", "status", "memory", "params_M", "ctx", "steps", "hours", "best_val", "val", "test", "acc",
          "probe"]


def fmt(v, nd=4):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def cell(r, c):
    v = r.get(c)
    if v is None:
        return ""
    if c in ("params_M", "hours"):
        return f"{v:.1f}"
    if c in ("probe", "acc"):
        return f"{v:.1%}"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def read_csv(path):
    """Rows of a training log, one per step (after a restart the newest row of a step counts)."""
    with open(path, newline="") as f:
        rows = {int(r["step"]): r for r in csv.DictReader(f) if (r.get("step") or "").strip().isdigit()}
    return [rows[k] for k in sorted(rows)]


def group_of(tag):
    if tag.startswith(("pilot", "abl_", "teach_pilot")):
        return "pilot"
    return "loo" if tag.startswith("loo") else "enwik8" if tag.startswith("e8") else "long"


def report(quiet=False):
    """Rebuild paper/results.md. A problem here (a full disk, a file in use) never stops the runs."""
    try:
        _report(quiet)
    except Exception as e:
        print(f"  warning: the report could not be written ({type(e).__name__}: {e}); "
              f"the runs are not affected", flush=True)


def _report(quiet):
    os.makedirs(pp("samples"), exist_ok=True)
    pre = CFG["prefix"]
    rows, curves = [], {}
    for path in glob.glob(rp(pre + "*.csv")):
        tag = os.path.basename(path)[len(pre):-4]
        log = read_csv(path)
        if not log:
            continue
        curves[tag] = log
        res_path = rp(pre + tag + "_result.json")
        r = json.load(open(res_path)) if os.path.exists(res_path) else {}
        ex = r.get("experts")
        mem = "" if ex is None else ("none" if not ex else "tables + cache" if "cache" in ex else "tables")
        a = r.get("args", {})
        last = log[-1]
        rows.append(dict(
            run=tag, status="finished" if r and int(last["step"]) >= a.get("steps", 0) else "running",
            memory=mem, params_M=round(r["params"] / 1e6, 1) if r else None, ctx=r.get("ctx"),
            rotary="yes" if a.get("rope") else ("no" if r else ""),
            steps=int(last["step"]), hours=round(float(last["seconds"]) / 3600, 2),
            seconds=float(last["seconds"]),
            best_val=r.get("best_val", min(float(x["val_bpc"]) for x in log)),
            val=r.get("val"), val_warm=r.get("val_warm"), test=r.get("test"), test_warm=r.get("test_warm"),
            acc=r.get("test_acc", r.get("val_acc")),
            word_start=r.get("test_word_start", r.get("val_word_start")), probe=r.get("probe")))
        sp = rp(pre + tag + "_samples.txt")
        if os.path.exists(sp):
            try:
                shutil.copyfile(sp, pp("samples", tag + "_samples.txt"))
            except OSError:
                pass
    rows.sort(key=lambda r: (RUN_ORDER.index(r["run"]) if r["run"] in RUN_ORDER else 99, r["run"]))
    evals, doms, dyns = [], {}, []
    for path in sorted(glob.glob(rp(pre + "*_eval.json"))):
        e = json.load(open(path))
        name = os.path.basename(path)[len(pre):-10]
        if name.startswith("dom_") and "__" in name:          # domain scores: dom_<domain>__<run>
            dname, run = name[4:].split("__", 1)
            doms.setdefault(run, {})[dname] = e.get("bpc_warm", e.get("bpc"))
            continue
        if name.startswith("dyn__"):
            dyns.append((name[5:], e.get("bpc_slide", e.get("bpc")), e.get("bpc_dyn"), e.get("dyn", {})))
            continue
        evals.append(dict(name=name, bpc=e.get("bpc"), bpc_warm=e.get("bpc_warm"),
                          probe=e.get("probe"), tables=os.path.basename(e.get("hard") or "")))
    md = ["# Results", "", f"Updated {time.strftime('%Y-%m-%d %H:%M')}. Bits per byte, lower is better "
          "(1.20 bits per byte means the text can be stored in 15% of its size).", "",
          f"- **best val**: the score on 2,048 windows used during training to pick the best checkpoint.",
          f"- **val / test**: the best checkpoint on {CFG['final_windows']:,} windows (enwik8 runs: on every "
          "byte). The test text was never used for any choice.",
          "- **warm**: only the second half of each window, where the transformer has context.",
          "- **next byte right**: how often the most likely byte was the true one (test text, else val).",
          "- **first letters**: bits per byte on first letters of words, where the next word is chosen.",
          "- **probe**: share of the 65 knowledge questions answered right (4 choices, 25% is chance).", ""]
    if rows:
        md += ["| " + " | ".join(h for _, h in COLS) + " |", "|" + "---|" * len(COLS)]
        md += ["| " + " | ".join(cell(r, c) for c, _ in COLS) + " |" for r in rows]
        with open(pp("results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[c for c, _ in COLS], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    else:
        md.append("No training runs yet.")
    if any(r["run"].startswith("e8") for r in rows):
        md += ["", "## enwik8: published test results for comparison", "",
               "Bits per byte on the same 5 MB test file: Transformer, 12 layers, 41M parameters: 1.11 "
               "(Al-Rfou et al. 2018). Transformer-XL, 12 layers, 41M: 1.06; 18 layers, 88M: 1.03; 24 layers, "
               "277M: 0.99 (Dai et al. 2019). mLSTM with dynamic evaluation, 46M: 1.08. The e8 runs here "
               "use count tables and a document cache on top of their parameters, so they belong with the "
               "results that use extra memory or adapt at test time, and the table size has to be reported "
               "next to the parameter count."]
    if doms:
        dn = [n for n, _ in CFG["domains"] if any(n in d for d in doms.values())]
        md += ["", "## Other kinds of text (none of the models trained on them)", "",
               "Bits per byte on 2,048 windows, second half of each window (warm). Lower is better.", "",
               "| model | " + " | ".join(dn) + " |", "|---|" + "---|" * len(dn)]
        for run in [r for r in DOMAIN_MODELS if r in doms]:
            md.append(f"| {run} | " + " | ".join(fmt(doms[run].get(n)) for n in dn) + " |")
    if dyns:
        md += ["", "## Learning while reading (dynamic evaluation)", "",
               "Test text, every byte. After each window is scored the model takes one small training "
               "step on it, so it only ever learns from text it has already been scored on. Published "
               "results of this kind: Transformer-XL 277M with dynamic evaluation 0.94 on enwik8.", "",
               "| model | test | test, learning while reading | gain | step size |", "|---|---|---|---|---|"]
        for run, a, b, y in sorted(dyns, key=lambda t: DYN_MODELS.index(t[0]) if t[0] in DYN_MODELS else 99):
            md.append(f"| {run} | {fmt(a)} | {fmt(b)} | {fmt(a - b if a and b else None)} | "
                      f"{y.get('mode', 'adam')} {y.get('lr', '')} |")
    if evals:
        md += ["", "## Models scored with other tables (no training)", "",
               "| model | tables | val | val warm | probe |", "|---|---|---|---|---|"]
        md += [f"| {e['name']} | {e['tables']} | {fmt(e['bpc'])} | {fmt(e['bpc_warm'])} | "
               f"{'' if e['probe'] is None else format(e['probe'], '.1%')} |" for e in evals]
    for title, log in (("Tables alone, document cache, leave-one-out check", "tables"),
                       ("Tables from half the text", "loo_half_tables"),
                       ("enwik8: tables alone", "e8_tables")):
        lp = pp("logs", log + ".log")
        if os.path.exists(lp):
            keep, on = [], False
            for line in open(lp, encoding="utf-8", errors="replace"):
                on = on or "val BPC" in line or "leave-one-out check" in line
                if line.startswith("#####"):
                    keep, on = [], False
                elif on and line.strip() and not line.startswith("saved"):
                    keep.append(line.rstrip())
            if keep:
                md += ["", f"## {title}", "", "```"] + keep + ["```"]
    dec = pp("decisions.json")
    if os.path.exists(dec):
        d = json.load(open(dec))
        md += ["", "## Decisions", "", f"Rotary positions for the long runs: {'yes' if d['rope'] else 'no'} "
               f"(pilot: learned positions {d['learned_positions']:.4f}, rotary {d['rotary_positions']:.4f})."]
    pil = [r for r in rows if group_of(r["run"]) == "pilot" and r["status"] == "finished" and r["steps"]]
    if pil:
        md += ["", "## Time", ""]
        for r in pil:
            sps = r["seconds"] / r["steps"]
            md.append(f"- {r['run']}: {sps:.3f} s per step, so {CFG['main']['steps']:,} steps of the main "
                      f"size take about {sps * CFG['main']['steps'] / 3600:.1f} h.")
    files = plot(curves)
    if files:
        md += ["", "## Curves", ""] + [f"![{os.path.basename(f)}]({os.path.basename(f)})" for f in files]
    texts = []
    for r in rows:                                  # sample texts of the best checkpoint
        sp = pp("samples", r["run"] + "_samples.txt")
        if r["status"] == "finished" and os.path.exists(sp):
            block = open(sp, encoding="utf-8", errors="replace").read().split("\n=== step ")[-1]
            if "final" in block.split("\n")[0]:
                texts += ["", f"### {r['run']}", "", "```", "step " + block.strip(), "```"]
    if texts:
        md += ["", "## Sample texts and probe (best checkpoint of each finished run)"] + texts
    with open(pp("results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    if not quiet and rows:
        head = dict(COLS)
        w = [max(len(head[c]), *(len(cell(r, c)) for r in rows)) for c in SCREEN]
        print("\n" + "  ".join(head[c].ljust(x) for c, x in zip(SCREEN, w)))
        for r in rows:
            print("  ".join(cell(r, c).ljust(x) for c, x in zip(SCREEN, w)))
        print(f"-> {pp('results.md')}\n", flush=True)


def plot(curves):
    if not curves:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cd6", "#6b6a63"]
    files = []
    groups = {}
    for tag in curves:
        groups.setdefault(group_of(tag), []).append(tag)
    for g, tags in groups.items():
        tags.sort(key=lambda t: RUN_ORDER.index(t) if t in RUN_ORDER else 99)
        for what, col, fname in (("validation bits per byte", "val_bpc", f"curves_{g}.png"),
                                 ("knowledge probe, share right", "probe", f"probe_{g}.png")):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            drawn = 0
            for i, tag in enumerate(tags):
                pts = [(int(r["step"]), float(r[col])) for r in curves[tag]
                       if r.get(col) not in (None, "") and int(r["step"]) > 0]
                if pts:
                    ax.plot(*zip(*pts), color=colors[i % len(colors)], lw=1.8, label=tag,
                            marker="o" if col == "probe" else None, ms=4)
                    drawn += 1
            if not drawn:
                plt.close(fig)
                continue
            if col == "val_bpc":
                lo = min(float(r[col]) for t in tags for r in curves[t] if int(r["step"]) > 0)
                ax.set_ylim(lo - 0.02, lo + 0.6)
            ax.set_xlabel("training step")
            ax.set_ylabel(what)
            ax.grid(color="#ebeae4")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.legend(frameon=False, fontsize=8, loc="upper right" if col == "val_bpc" else "lower right")
            fig.tight_layout()
            fig.savefig(pp(fname), dpi=130)
            plt.close(fig)
            files.append(pp(fname))
    return files


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=ORDER + EXTRA,
                    help="only these stages (default: all standard ones, in order)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--report", action="store_true", help="only rebuild the report")
    ap.add_argument("--tiny", action="store_true", help="quick test of the whole pipeline with tiny models")
    ap.add_argument("--redo", nargs="+", default=[], help="run these runs again even though they are done")
    ap.add_argument("--data", help="override CFG['data']")
    ap.add_argument("--val_data", help="override CFG['val_data'] ('none' = split from --data)")
    ap.add_argument("--teacher_model", help="override the teacher model of the teacher stage")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")      # Windows consoles and unusual characters
    except Exception:
        pass
    if args.tiny:
        CFG.update(TINY)
    if args.data:
        CFG["data"] = args.data
    if args.val_data:
        CFG["val_data"] = "" if args.val_data.lower() == "none" else args.val_data
    if args.teacher_model:
        CFG["teacher"] = dict(CFG["teacher"], model=args.teacher_model)
    os.chdir(HERE)
    for f in ("analyze.py", "prompts.txt", "probe.json", "label_teacher.py"):
        if not os.path.exists(script(f)):
            raise SystemExit(f"{f} not found next to run_all.py")
    for f in (CFG["data"], CFG["val_data"]):
        if f and not os.path.exists(f) and not args.dry:
            raise SystemExit(f"text file not found: {f}  (edit CFG at the top of run_all.py)")
    os.makedirs(CFG["paper"], exist_ok=True)
    for r in args.redo:
        if os.path.exists(marker(r)):
            os.remove(marker(r))
    if args.report:
        return report()
    stages = args.stages or ORDER
    if args.list:
        for s in ORDER + EXTRA:
            print(s + ("   (extra: only with --stages)" if s in EXTRA else ""))
            for j in STAGES[s]():
                print(f"    {'done   ' if is_done(j) else 'to do  '} {j['name']}")
        return
    snap = pp("code")
    os.makedirs(snap, exist_ok=True)
    for f in (trainer(), script("analyze.py"), script("run_all.py"), script("label_teacher.py"),
              script("prompts.txt"), script("probe.json"), script("search.py"), script("download_hf_corpus.py")):
        if not os.path.exists(f):
            continue
        try:
            shutil.copyfile(f, os.path.join(snap, os.path.basename(f)))
        except OSError as e:
            print(f"  warning: could not copy {os.path.basename(f)} to {snap} ({e})")
    print(f"text {CFG['data']} | results in {CFG['paper']}{os.sep} | models in {CFG['out']}{os.sep} | "
          f"stages: {' '.join(stages)}")
    keep_awake(True)
    try:
        for s in stages:
            print(f"\n######## stage: {s} ########", flush=True)
            for job in STAGES[s]():                 # built now: later stages see earlier decisions
                how = run_job(job, args.dry)
                if how == "failed":
                    report(quiet=True)
                    raise SystemExit(1)
                if how == "ran":
                    report(quiet=is_quiet(job))
            if s == "pilot" and not args.dry:
                decide_rope()
        if not args.dry:
            report()
            print(f"all requested stages finished. Results: {pp('results.md')}")
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        keep_awake(False)


def is_quiet(job):
    return job["name"].startswith("analysis") or job["name"].endswith("_full")


if __name__ == "__main__":
    main()
