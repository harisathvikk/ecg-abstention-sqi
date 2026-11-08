"""
run_study.py  —  NEW
Runs the full study and writes CSV result tables. Resumable: if Colab disconnects, run the same
command again and finished (model, regime) pairs are skipped.

Real run (Colab):
    python run_study.py --data-dir ./data --out-dir ./out
Smoke test (any machine, synthetic data, ~1 min):
    python run_study.py --synthetic --quick --out-dir ./out_smoke
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import ecgunc as E


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./out")
    ap.add_argument("--synthetic", action="store_true", help="synthetic data (code test only)")
    ap.add_argument("--quick", action="store_true", help="few SNRs / seeds / epochs")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--models", default="rf,cnn,cnn_norr")
    ap.add_argument("--regimes", default="clean,aug")
    ap.add_argument("--prefilter", type=int, default=1, help="1 = 0.5-40 Hz band-pass before detection/classification")
    ap.add_argument("--peak-source", default="detected", choices=["detected", "oracle"])
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    cache = os.path.join(a.out_dir, "cache")
    tag = f"pf{a.prefilter}_{a.peak_source}"          # separate result files per setting
    snrs = [12, 0, -6] if a.quick else E.SNRS
    seeds = list(range(2 if a.quick else a.seeds))
    epochs = 3 if a.quick else a.epochs
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    log("device:", device, "| prefilter:", a.prefilter, "| peaks:", a.peak_source, "| seeds:", seeds, "| epochs:", epochs)

    # ── data ──
    if a.synthetic:
        records, noises, splits = E.make_synthetic(n_train=6, n_val=3, n_test=5, minutes=3.0 if a.quick else 4.0)
    else:
        log("loading MIT-BIH + NSTDB (downloads from PhysioNet on first run) ...")
        records, noises, splits = E.load_mitdb(a.data_dir), E.load_nstdb(a.data_dir), E.real_splits()
    log({k: len(v) for k, v in splits.items()}, "records per split")
    assert not (set(splits["train"]) & set(splits["val"])) and not (set(splits["train"]) & set(splits["test"])) \
        and not (set(splits["val"]) & set(splits["test"])), "patient leakage between splits!"
    conds = [(None, None)] + [(n, s) for n in E.NOISES for s in snrs]
    pf, ps = bool(a.prefilter), a.peak_source

    # ── per-condition datasets (cached on disk) ──
    def build(tagname, names, half, base_seed):
        out = {}
        for i, (n, s) in enumerate(conds):
            t0 = time.time()
            out[E.cond_key(n, s)] = E.cached_condition(cache, tagname, records, names, noises, n, s, half,
                                                       base_seed + i, pf, ps)
            log(f"  {tagname}:{E.cond_key(n, s):7s} beats={len(out[E.cond_key(n, s)]['y']):6d} "
                f"det_sens={out[E.cond_key(n, s)]['n_match'] / max(out[E.cond_key(n, s)]['n_ref'], 1):.3f} ({time.time() - t0:.1f}s)")
        return out

    log("building TEST conditions (second half of each noise record) ...")
    test = build("test", splits["test"], "test", 1000)
    log("building CAL conditions (calibration patients, first half of noise) ...")
    cal = build("cal", splits["val"], "train", 2000)
    log("building TRAIN sets ...")
    tr_clean = E.cached_condition(cache, "train", records, splits["train"], noises, None, None, "train", 3000, pf, ps)
    aug_parts = [tr_clean] + [E.cached_condition(cache, "trainaug", records, splits["train"], noises, n, "rand",
                                                 "train", 3100 + j, pf, ps) for j, n in enumerate(E.NOISES)]
    train = {"clean": E.concat_sets([tr_clean]), "aug": E.concat_sets(aug_parts)}
    log("train beats:", {k: len(v["y"]) for k, v in train.items()}, "| class counts clean:",
        np.bincount(train["clean"]["y"], minlength=4).tolist())

    done_f = os.path.join(a.out_dir, f"done_{tag}.json")
    done = json.load(open(done_f)) if os.path.exists(done_f) else []

    def append(df_rows, name):
        fn = os.path.join(a.out_dir, f"{name}_{tag}.csv")
        df = pd.DataFrame(df_rows)
        df.to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)

    for regime in a.regimes.split(","):
        for model in a.models.split(","):
            key = f"{model}|{regime}"
            if key in done:
                log("skip (done):", key)
                continue
            probs_cal, probs_test = [], []
            allM, allS, allC = [], [], []
            for sd in seeds:
                t0 = time.time()
                pred = E.train_predict(model, train[regime], sd, epochs, device)
                pc = {k: pred(v) for k, v in cal.items()}
                pt = {k: pred(v) for k, v in test.items()}
                probs_cal.append(pc)
                probs_test.append(pt)
                M, S, C = E.evaluate_member(dict(model=model, regime=regime, member=f"seed{sd}"), pc, pt, test, cal)
                allM += M; allS += S; allC += C
                log(f"{key} seed{sd} done ({time.time() - t0:.0f}s) | clean macroF1="
                    f"{[m['macroF1'] for m in M if m['noise'] == 'clean'][0]:.3f}")
            # seed-ensemble: averaged probabilities + mutual-information score (epistemic uncertainty)
            pc = {k: np.mean([p[k] for p in probs_cal], 0) for k in cal}
            pt = {k: np.mean([p[k] for p in probs_test], 0) for k in test}
            extra = {}
            for k in test:
                H = lambda p: -(p * np.log(np.clip(p, 1e-9, 1))).sum(1)
                extra[k] = {"mi": H(pt[k]) - np.mean([H(p[k]) for p in probs_test], 0)}
            M, S, C = E.evaluate_member(dict(model=model, regime=regime, member="ens"), pc, pt, test, cal, extra)
            allM += M; allS += S; allC += C
            append(allM, "metrics"); append(allS, "selective"); append(allC, "conformal")
            done.append(key)
            json.dump(done, open(done_f, "w"))
            log("saved", key)
    log("ALL DONE. Result CSVs are in", a.out_dir)


if __name__ == "__main__":
    main()
