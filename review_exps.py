"""
review_exps.py  —  NEW
Experiments requested in review. One run, all five seeds, all three models, both training regimes.

  1. SQIs computed on the RAW noisy signal (as before) AND on the FILTERED signal the classifier sees
  2. Hard-veto abstention (literature SQI thresholds; vetoed beats rejected first, the rest ranked by max-softmax)
  3. A FAIR learned combination: log-confidence + SQI (can reproduce the max-softmax ranking), fitted on the
     calibration patients, plus an IN-SAMPLE upper bound fitted on the test patients themselves
  4. Abstention scored on NOISE-INDUCED errors only (wrong under noise, correct on the clean version of the same beat)
  5. Abstention scored with SPURIOUS detections included (detections matching no annotated beat = always an error)
  6. Within-patient and within-patient-AND-class error-detection AUROC (separates the patient and beat-type confounds)
  7. PATIENT-LEVEL cluster bootstrap (resample the 22 test patients) for every comparison against max-softmax
  8. SQI confound statistics for raw and filtered SQIs; class counts of the calibration patients (conformal)

Test/calibration beats are rebuilt with the same random seeds as run_study.py (identical noise), with extra fields.
Training sets are read from the existing run_study cache.

    %cd /content/ecg_study
    !python review_exps.py --main-cache /content/mydrive/MyDrive/ecg_main/cache --out /content/mydrive/MyDrive/ecg_review
Resumable: re-run the same command after a disconnect and finished model|regime blocks are skipped.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import ecgunc as E

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
SIGN = {"kSQI": -1, "pSQI": -1, "basSQI": -1, "hfSQI": 1, "bSQI": -1}   # higher score = less trustworthy
NOISY_MAX_SNR = 6
COV = 0.8
BOOT_SCORES = ["sqi_raw", "sqi_filt", "combo_raw", "combo_log_raw", "combo_log_filt", "veto_k_raw",
               "veto_lit_raw", "veto_k_filt", "veto_lit_filt", "conf_log", "combo_log_filt_ins"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ─── 1. extended condition builder (same noise as run_study, extra fields) ────────────────────
def build_ext(records, names, noises, noise, snr, half, seed):
    rng = np.random.default_rng(seed)
    keys = ["X", "M", "RR", "SQ", "SQf", "y", "rec", "ref"]
    A = {k: [] for k in keys}
    S = {k: [] for k in ["X", "M", "RR", "SQ", "SQf", "rec"]}         # spurious detections
    n_ref = n_match = 0
    for ri, nm in enumerate(names):
        r = records[nm]
        sig = r["sig"]
        if noise is not None:
            sig = E.add_noise(sig, noises[noise], snr, rng, half)
        sf = E.prefilter(sig)
        ref_idx, ref_cls = r["idx"], r["cls"]
        n_ref += len(ref_idx)
        pk2 = E.detect_peaks(sf, "neurokit")
        pk = E.detect_peaks(sf, "pantompkins1985")
        if len(pk) == 0:
            continue
        mt = E.match_peaks(pk, ref_idx)
        keep = mt >= 0
        n_match += int(keep.sum())
        beats = E.cut_beats(sf, pk)
        rr = E.rr_features(pk)
        sq = E.sqi_features(sig, pk, pk2)
        sqf = E.sqi_features(sf, pk, pk2)
        Z = E.zscore_rows(beats).astype(np.float16)
        Mf = E.morph_features(beats)
        for d, m in [(A, keep), (S, ~keep)]:
            d["X"].append(Z[m]); d["M"].append(Mf[m]); d["RR"].append(rr[m])
            d["SQ"].append(sq[m]); d["SQf"].append(sqf[m]); d["rec"].append(np.full(int(m.sum()), ri, np.int16))
        A["y"].append(ref_cls[mt[keep]].astype(np.int8))
        A["ref"].append(mt[keep].astype(np.int32))
    out = {k: np.concatenate(v) for k, v in A.items()}
    out.update({"sp_" + k: np.concatenate(v) for k, v in S.items()})
    out["n_ref"], out["n_match"] = n_ref, n_match
    return out


def cached_ext(cache, tag, *args):
    fn = os.path.join(cache, f"{tag}.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return {k: z[k] for k in z.files}
    d = build_ext(*args)
    np.savez(fn, **d)
    return d


# ─── 2. metric helpers ─────────────────────────────────────────────────────────────────────────
def fast_auroc(pos, score):
    """AUROC of score for pos (1) vs neg (0) via ranks; nan if one class empty."""
    n1 = int(pos.sum()); n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(score)
    return float((r[pos.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def aurc(err, score):
    o = np.argsort(score, kind="stable")
    e = err[o].astype(float)
    return float((np.cumsum(e) / np.arange(1, len(e) + 1)).mean())


def at_cov(err, y, score, c=COV):
    n = len(err)
    o = np.argsort(score, kind="stable")
    keep = np.zeros(n, bool)
    keep[o[:max(1, int(round(c * n)))]] = True
    sv = ((keep & (y == 2)) & (err > 0)).sum() / max((y == 2).sum(), 1)
    ss = ((keep & (y == 1)) & (err > 0)).sum() / max((y == 1).sum(), 1)
    return float(err[keep].mean()), float(sv), float(ss)


def stratified_auroc(err, score, strata, min_n=10):
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        ne = int(err[m].sum()); no = int(m.sum()) - ne
        if ne < min_n or no < min_n:
            continue
        w = min(ne, no)
        num += w * fast_auroc(err[m], score[m]); den += w
    return num / den if den else np.nan


def conf3(P):
    msp, ent, mar = E.conf_scores(P)
    return np.stack([msp, ent, mar], 1)


def logc(C):
    return np.log(np.clip(C, 1e-6, None))


def veto_flags(sq):
    k, p, b = sq[:, 0], sq[:, 1], sq[:, 2]
    v_k = k < 5.0                                          # kurtosis threshold (Li et al. / Clifford et al.)
    v_lit = v_k | (b < 0.95) | (p < 0.5) | (p > 0.8)       # kSQI, basSQI and pSQI acceptability ranges
    return v_k, v_lit


# ─── 3. main ───────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--main-cache", required=True, help="cache folder of the main run (training sets are read from it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="cnn,rf,cnn_norr")
    ap.add_argument("--regimes", default="aug,clean")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--synthetic", action="store_true", help="code test only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "cache_ext")
    os.makedirs(cache, exist_ok=True)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    snrs = [12, 0, -6] if a.synthetic else E.SNRS
    epochs = 3 if a.synthetic else a.epochs
    if a.synthetic:
        records, noises, splits = E.make_synthetic(n_train=6, n_val=3, n_test=5, minutes=3.0)
    else:
        log("loading MIT-BIH + NSTDB ...")
        records, noises, splits = E.load_mitdb(a.data_dir), E.load_nstdb(a.data_dir), E.real_splits()
    conds = [(None, None)] + [(n, s) for n in E.NOISES for s in snrs]
    keys = [E.cond_key(n, s) for n, s in conds]
    nz_of = {E.cond_key(n, s): (n if n else "clean", s) for n, s in conds}

    log("building TEST / CAL conditions with raw + filtered SQIs and reference indices ...")
    test, cal = {}, {}
    for i, (n, s) in enumerate(conds):
        k = keys[i]
        test[k] = cached_ext(cache, f"test_{k}", records, splits["test"], noises, n, s, "test", 1000 + i)
        cal[k] = cached_ext(cache, f"cal_{k}", records, splits["val"], noises, n, s, "train", 2000 + i)
        log(f"  {k:6s} test beats={len(test[k]['y'])} spurious={len(test[k]['sp_rec'])}"
            f" det_sens={test[k]['n_match'] / max(test[k]['n_ref'], 1):.3f}")
    tr_clean = E.cached_condition(a.main_cache, "train", records, splits["train"], noises, None, None, "train", 3000, True, "detected")
    aug_parts = [tr_clean] + [E.cached_condition(a.main_cache, "trainaug", records, splits["train"], noises, n, "rand",
                                                 "train", 3100 + j, True, "detected") for j, n in enumerate(E.NOISES)]
    train = {"clean": E.concat_sets([tr_clean]), "aug": E.concat_sets(aug_parts)}
    log("train beats:", {k: len(v["y"]) for k, v in train.items()})

    # ── data-only statistics: calibration class counts, SQI confounds (raw vs filtered) ──
    cc = cal["clean"]
    cnt = pd.DataFrame([dict(record=splits["val"][r], **{c: int(((cc["rec"] == r) & (cc["y"] == j)).sum())
                                                          for j, c in enumerate(E.CLASSES)}) for r in range(len(splits["val"]))])
    cnt.loc[len(cnt)] = dict(record="total", **{c: int((cc["y"] == j).sum()) for j, c in enumerate(E.CLASSES)})
    cnt.to_csv(os.path.join(a.out, "calib_class_counts.csv"), index=False)
    print("\n=== calibration patients: beats per class (clean) ===\n" + cnt.to_string(index=False))
    rows = []
    tc = test["clean"]
    for field, lab in [("SQ", "raw"), ("SQf", "filtered")]:
        for j, nm in enumerate(E.SQI_NAMES):
            x = tc[field][:, j].astype(float)
            within = sum(x[tc["rec"] == r].var() * (tc["rec"] == r).mean() for r in np.unique(tc["rec"]))
            mv = (tc["y"] == 0) | (tc["y"] == 2)
            r_ = dict(signal=lab, sqi=nm, between_patient_share=1 - within / x.var() if x.var() > 0 else np.nan,
                      auroc_V_vs_N=fast_auroc((tc["y"][mv] == 2).astype(int), x[mv]))
            for k in keys:
                r_["median_" + k] = float(np.median(test[k][field][:, j]))
            rows.append(r_)
    sqs = pd.DataFrame(rows)
    sqs.to_csv(os.path.join(a.out, "sqi_confounds.csv"), index=False)
    print("\n=== SQI confounds on clean test data, raw vs filtered signal ===")
    print(sqs[["signal", "sqi", "between_patient_share", "auroc_V_vs_N", "median_clean", "median_bw-6" if "bw-6" in keys else "median_clean",
               "median_em-6" if "em-6" in keys else "median_clean", "median_ma-6" if "ma-6" in keys else "median_clean"]].round(3).to_string(index=False))

    done_f = os.path.join(a.out, "done.json")
    done = json.load(open(done_f)) if os.path.exists(done_f) else []

    def append(rows_, name):
        fn = os.path.join(a.out, name)
        pd.DataFrame(rows_).to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)

    noisy_keys = [k for k in keys if k != "clean" and nz_of[k][1] <= NOISY_MAX_SNR]
    for regime in a.regimes.split(","):
        for model in a.models.split(","):
            tag = f"{model}|{regime}"
            if tag in done:
                log("skip (done):", tag)
                continue
            sel_rows, win_rows = [], []
            boot_store = []                       # per seed: {cond: (err, rec, {score: array})}
            for seed in range(a.seeds):
                t0 = time.time()
                pred = E.train_predict(model, train[regime], seed, epochs, device)
                P = {k: pred(test[k]) for k in keys}
                Ps = {k: (pred(dict(X=test[k]["sp_X"], M=test[k]["sp_M"], RR=test[k]["sp_RR"], y=np.zeros(len(test[k]["sp_rec"]))))
                          if len(test[k]["sp_rec"]) else np.zeros((0, 4), np.float32)) for k in keys}
                Pc = {k: pred(cal[k]) for k in keys}
                # error models, fitted on the calibration patients (all conditions pooled)
                cy = np.concatenate([cal[k]["y"] for k in keys]); cP = np.concatenate([Pc[k] for k in keys])
                cerr = (cP.argmax(1) != cy).astype(int)
                cC = conf3(cP); cS = np.concatenate([cal[k]["SQ"] for k in keys]); cF = np.concatenate([cal[k]["SQf"] for k in keys])
                feats = {"sqi_raw": lambda C, S, F: S, "sqi_filt": lambda C, S, F: F,
                         "combo_raw": lambda C, S, F: np.concatenate([S, C], 1),
                         "conf_log": lambda C, S, F: logc(C),
                         "combo_log_raw": lambda C, S, F: np.concatenate([S, logc(C)], 1),
                         "combo_log_filt": lambda C, S, F: np.concatenate([F, logc(C)], 1)}
                mdl = {nm: E.fit_error_model(f(cC, cS, cF), cerr) for nm, f in feats.items()}
                # in-sample upper bound: fitted on the TEST patients themselves (all conditions pooled)
                tyy = np.concatenate([test[k]["y"] for k in keys]); tP = np.concatenate([P[k] for k in keys])
                terr = (tP.argmax(1) != tyy).astype(int)
                tC = conf3(tP); tS = np.concatenate([test[k]["SQ"] for k in keys]); tF = np.concatenate([test[k]["SQf"] for k in keys])
                ins = {nm + "_ins": E.fit_error_model(feats[nm](tC, tS, tF)[::3], terr[::3])
                       for nm in ["conf_log", "combo_log_filt", "combo_log_raw"]}
                # clean-condition correctness per (record, reference beat), for noise-induced errors
                ck = test["clean"]["rec"].astype(np.int64) * 10_000_000 + test["clean"]["ref"]
                cok = (P["clean"].argmax(1) == test["clean"]["y"])
                o = np.argsort(ck); ck, cok = ck[o], cok[o]
                rng = np.random.default_rng(seed)
                store = {}
                for k in keys:
                    d = test[k]; y = d["y"].astype(int); Pk = P[k]
                    err = (Pk.argmax(1) != y).astype(int)
                    C = conf3(Pk)

                    def scores_for(C_, S_, F_):
                        sc = {"msp": C_[:, 0], "entropy": C_[:, 1]}
                        for nm, m_ in list(mdl.items()) + list(ins.items()):
                            base = nm[:-4] if nm.endswith("_ins") else nm
                            sc[nm] = m_.predict_proba(feats[base](C_, S_, F_))[:, 1] if m_ is not None else C_[:, 0]
                        for lab, Q in [("raw", S_), ("filt", F_)]:
                            vk, vl = veto_flags(Q)
                            sc["veto_k_" + lab] = C_[:, 0] + vk.astype(float)
                            sc["veto_lit_" + lab] = C_[:, 0] + vl.astype(float)
                        for j, nm in enumerate(E.SQI_NAMES):
                            sc["single_raw_" + nm] = SIGN[nm] * S_[:, j]
                            sc["single_filt_" + nm] = SIGN[nm] * F_[:, j]
                        return sc
                    sc = scores_for(C, d["SQ"], d["SQf"])
                    sc["random"] = rng.random(len(y))
                    # spurious detections appended as guaranteed errors
                    nsp = len(d["sp_rec"])
                    if nsp:
                        scs = scores_for(conf3(Ps[k]), d["sp_SQ"], d["sp_SQf"])
                        scs["random"] = rng.random(nsp)
                    # noise-induced errors: keep beats correct on clean; target = wrong now
                    kk = d["rec"].astype(np.int64) * 10_000_000 + d["ref"]
                    pos = np.clip(np.searchsorted(ck, kk), 0, len(ck) - 1)
                    found = ck[pos] == kk
                    ni_mask = found & cok[pos]          # beats whose clean version was classified correctly
                    n_, s_ = nz_of[k]
                    for nm, v in sc.items():
                        if nm.startswith("single_"):
                            continue
                        r80, sv, ss = at_cov(err, y, v)
                        row = dict(model=model, regime=regime, seed=seed, noise=n_, snr=s_, score=nm,
                                   aurc=aurc(err, v), auroc_err=fast_auroc(err, v), risk80=r80, silentV80=sv, silentS80=ss)
                        if k != "clean":
                            row.update(aurc_ni=aurc(err[ni_mask], v[ni_mask]), auroc_ni=fast_auroc(err[ni_mask], v[ni_mask]),
                                       n_ni_err=int(err[ni_mask].sum()), n_err=int(err.sum()))
                        if nsp:
                            row.update(aurc_sp=aurc(np.concatenate([err, np.ones(nsp, int)]), np.concatenate([v, scs[nm]])),
                                       auroc_sp=fast_auroc(np.concatenate([err, np.ones(nsp, int)]), np.concatenate([v, scs[nm]])),
                                       n_spurious=nsp)
                        sel_rows.append(row)
                    # within-patient and within-patient-and-class AUROC (clean + noisy conditions)
                    if k == "clean" or k in noisy_keys:
                        for nm in ["msp", "combo_log_filt", "combo_log_raw"] + [f"single_{t}_{q}" for t in ["raw", "filt"] for q in E.SQI_NAMES]:
                            v = sc[nm]
                            win_rows.append(dict(model=model, regime=regime, seed=seed, noise=n_, snr=s_, score=nm,
                                                 pooled=fast_auroc(err, v),
                                                 within_patient=stratified_auroc(err, v, d["rec"]),
                                                 within_patient_class=stratified_auroc(err, v, d["rec"].astype(int) * 10 + y)))
                    if k in noisy_keys:
                        store[k] = (err, d["rec"], {s: sc[s].astype(np.float32) for s in ["msp"] + BOOT_SCORES},
                                    ni_mask)
                boot_store.append(store)
                log(f"{tag} seed{seed} done ({time.time() - t0:.0f}s)")

            # ── patient-level cluster bootstrap over the 22 test patients ──
            log(f"{tag}: patient-level bootstrap ({a.n_boot} resamples) ...")
            brng = np.random.default_rng(123)
            n_pat = len(splits["test"])
            prep = []                              # presorted orders per seed/condition/score
            for store in boot_store:
                for k, (err, rec, scs, nim) in store.items():
                    ords = {s: np.argsort(v, kind="stable") for s, v in scs.items()}
                    prep.append((err.astype(float), rec.astype(int), ords))

            def w_aurc(e, w, o):
                ww, ee = w[o], e[o] * w[o]
                cw = np.cumsum(ww)
                cr = np.cumsum(ee) / np.maximum(cw, 1e-12)
                return float((cr * ww).sum() / max(ww.sum(), 1e-12))

            def all_diffs(wpat):
                acc = {s: 0.0 for s in BOOT_SCORES}
                for e, r, ords in prep:
                    w = wpat[r]
                    base = w_aurc(e, w, ords["msp"])
                    for s in BOOT_SCORES:
                        acc[s] += w_aurc(e, w, ords[s]) - base
                return {s: v / len(prep) for s, v in acc.items()}
            point = all_diffs(np.ones(n_pat))
            bs = {s: [] for s in BOOT_SCORES}
            for b in range(a.n_boot):
                wpat = np.bincount(brng.integers(0, n_pat, n_pat), minlength=n_pat).astype(float)
                for s, v in all_diffs(wpat).items():
                    bs[s].append(v)
            boot_rows = [dict(model=model, regime=regime, score=s, diff_aurc_vs_msp=point[s],
                              ci_lo=float(np.percentile(bs[s], 2.5)), ci_hi=float(np.percentile(bs[s], 97.5)),
                              frac_boot_worse=float(np.mean(np.array(bs[s]) > 0))) for s in BOOT_SCORES]
            append(sel_rows, "rev_selective.csv")
            append(win_rows, "rev_within.csv")
            append(boot_rows, "rev_bootstrap.csv")
            done.append(tag)
            json.dump(done, open(done_f, "w"))
            log(f"saved {tag}")
            print(pd.DataFrame(boot_rows).round(4).to_string(index=False))

    # ── summary ──
    S = pd.read_csv(os.path.join(a.out, "rev_selective.csv"))
    nz = S[(S.noise != "clean") & (S.snr <= NOISY_MAX_SNR)]
    cols = [c for c in ["aurc", "auroc_err", "silentV80", "aurc_ni", "auroc_ni", "aurc_sp"] if c in nz]
    summ = nz.groupby(["model", "regime", "score", "seed"])[cols].mean().groupby(["model", "regime", "score"]).agg(["mean", "std"])
    summ.to_csv(os.path.join(a.out, "rev_summary_noisy.csv"))
    print("\n=== noisy conditions (SNR<=6): mean over seeds (lower AURC = better) ===")
    print(nz.groupby(["model", "regime", "score"])[cols].mean().round(4).to_string())
    W = pd.read_csv(os.path.join(a.out, "rev_within.csv"))
    print("\n=== error-detection AUROC on noisy conditions: pooled / within patient / within patient AND class ===")
    print(W[W.noise != "clean"].groupby(["model", "regime", "score"])[["pooled", "within_patient", "within_patient_class"]]
          .mean().round(3).to_string())
    print("\n=== patient-level bootstrap: AURC difference vs max-softmax (positive = worse than max-softmax) ===")
    print(pd.read_csv(os.path.join(a.out, "rev_bootstrap.csv")).round(4).to_string(index=False))
    log("ALL DONE. Files in", a.out)


if __name__ == "__main__":
    main()
