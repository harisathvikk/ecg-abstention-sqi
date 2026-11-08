"""
external_exps.py  —  NEW
External validation on a SECOND database: St Petersburg INCART 12-lead Arrhythmia Database (PhysioNet incartdb,
75 half-hour records, 257 Hz, lead II, resampled to 360 Hz).

Models are trained on MIT-BIH (same 18 training patients and same seeds as the main study), calibrated on the four
MIT-BIH calibration patients, and tested on ALL INCART records, clean and with the same NSTDB noise protocol.
Nothing from INCART is used for training or for fitting error models, so this is a cross-database test.

It replicates every main finding on new data:
  1. classification under noise (macro-F1, per-class F1, detection sensitivity, spurious detections)
  2. calibration (ECE, class-wise ECE) vs. minority-class (V) performance
  3. abstention: max-softmax vs. SQI vs. learned SQI+conf vs. hard vetoes, incl. noise-induced errors only,
     within-patient AUROC, a 5-seed ensemble (MSP and mutual information), and a patient-level cluster bootstrap
  4. conformal coverage (a) calibrated on MIT-BIH patients (cross-database), and (b) calibrated on k = 4 / 10 / 16
     INCART PATIENTS (clean beats, or clean+noisy mix with train-half noise), tested on held-out INCART patients
  5. in-domain error models: SQI / SQI+conf error models fitted on k = 4 / 10 / 16 INCART patients and tested on
     held-out INCART patients -> does the SQI comparison change when the error model has more calibration patients?
     (INCART headers carry a patient number: 75 records = 32 patients. Patient is the unit everywhere.)

    %cd /kaggle/working/ecg_study
    !python external_exps.py --main-cache /kaggle/working/cache_main --out /kaggle/working/ecg_external
Resumable: finished model|regime blocks are skipped (done.json).
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import ecgunc as E
import review_exps as R

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 300)
NOISY_MAX_SNR = 6
BOOT = ["sqi_raw", "combo_raw", "combo_log_filt", "veto_k_filt", "veto_lit_filt", "conf_log", "combo_log_filt_ins"]
# AAMI EC57 mapping (escape beats -> N, as in the standard); INCART uses the MIT-BIH symbol set plus 'B', 'n', 'r'
SYM_EXT = dict(E.SYM2CLS, **{"B": 0, "n": 0, "r": 2})


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ─── 1. data ──────────────────────────────────────────────────────────────────────────────────
def load_incart(data_dir):
    """INCART: lead II, resampled 257 -> 360 Hz with a polyphase filter; annotations rescaled.
    Returns records (same format as E.load_mitdb), a PATIENT label per record parsed from the header comment
    ("patient N": 75 records, 32 patients), and the raw header text. Patient is the unit of the bootstrap and of every
    calibration/test split, so records of one patient never fall on both sides."""
    import wfdb
    from scipy.signal import resample_poly
    d = os.path.join(data_dir, "incartdb")
    if not os.path.exists(os.path.join(d, "I01.dat")):
        wfdb.dl_database("incartdb", d)
    recs, groups, headers = {}, {}, {}
    for i in range(1, 76):
        name = f"I{i:02d}"
        p = os.path.join(d, name)
        hdr = wfdb.rdheader(p)
        ch = hdr.sig_name.index("II") if "II" in hdr.sig_name else 1
        rec = wfdb.rdrecord(p, channels=[ch])
        ann = wfdb.rdann(p, "atr")
        fs = rec.fs
        sig = resample_poly(np.nan_to_num(rec.p_signal[:, 0]).astype(np.float64), E.FS, int(fs))
        idx, cls = [], []
        for s, sym in zip(ann.sample, ann.symbol):
            if sym in SYM_EXT:
                j = int(round(s * E.FS / fs))
                if j < len(sig):
                    idx.append(j)
                    cls.append(SYM_EXT[sym])
        recs[name] = dict(name=name, sig=sig, idx=np.array(idx, np.int64), cls=np.array(cls, np.int8))
        text = " | ".join(c.strip() for c in (hdr.comments or []))
        m = re.search(r"patient\s*(\d+)", text, re.I)       # INCART headers carry "patient N"
        groups[name] = f"patient{int(m.group(1)):02d}" if m else name
        headers[name] = text
    return recs, groups, headers


def build_to_cache(cache, tag, records, names, noises, noise, snr, half, seed):
    """Worker: build one condition and write it to the cache; return nothing (keeps the main process light)."""
    R.cached_ext(cache, tag, records, names, noises, noise, snr, half, seed)
    return tag


def subset(d, m):
    return dict(X=d["X"][m], M=d["M"][m], RR=d["RR"][m], y=d["y"][m])


# ─── 2. helpers ───────────────────────────────────────────────────────────────────────────────
def cls_metrics(P, y):
    from sklearn.metrics import f1_score
    pred = P.argmax(1)
    f1 = f1_score(y, pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return dict(acc=float((pred == y).mean()), macroF1=float(f1.mean()),
                **{f"F1_{c}": float(f1[j]) for j, c in enumerate(E.CLASSES)},
                ece=float(E.ece(P, y)), cwece=float(E.classwise_ece(P, y)))


def mutual_info(Ps):
    """Ps: (S, n, 4) member probabilities -> (mean prob, MI)."""
    m = Ps.mean(0)
    H = lambda p: -(p * np.log(np.clip(p, 1e-9, 1))).sum(-1)
    return m, H(m) - H(Ps).mean(0)


def group_aurc_boot(store, groups_idx, n_groups, n_boot, seed=123):
    """Cluster bootstrap over patients of the AURC difference to max-softmax, averaged over conditions and seeds."""
    rng = np.random.default_rng(seed)
    prep = []
    for err, g, scs in store:
        prep.append((err.astype(float), g, {s: np.argsort(v, kind="stable") for s, v in scs.items()}))

    def w_aurc(e, w, o):
        ww, ee = w[o], e[o] * w[o]
        cw = np.cumsum(ww)
        return float(((np.cumsum(ee) / np.maximum(cw, 1e-12)) * ww).sum() / max(ww.sum(), 1e-12))

    def diffs(wg):
        acc = {s: 0.0 for s in BOOT}
        for e, g, o in prep:
            w = wg[g]
            base = w_aurc(e, w, o["msp"])
            for s in BOOT:
                acc[s] += w_aurc(e, w, o[s]) - base
        return {s: v / len(prep) for s, v in acc.items()}
    point = diffs(np.ones(n_groups))
    bs = {s: [] for s in BOOT}
    for _ in range(n_boot):
        wg = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups).astype(float)
        for s, v in diffs(wg).items():
            bs[s].append(v)
    return [dict(score=s, diff_aurc_vs_msp=point[s], ci_lo=float(np.percentile(bs[s], 2.5)),
                 ci_hi=float(np.percentile(bs[s], 97.5)), frac_boot_worse=float(np.mean(np.array(bs[s]) > 0)))
            for s in BOOT]


# ─── 3. main ──────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--main-cache", required=True, help="cache of MIT-BIH training sets (built if missing)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="cnn,rf,cnn_norr")
    ap.add_argument("--regimes", default="aug,clean")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=0, help="parallel workers for building conditions (0 = all CPUs)")
    ap.add_argument("--synthetic", action="store_true", help="code test only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "cache_ext")
    os.makedirs(cache, exist_ok=True)
    jobs = a.jobs or os.cpu_count() or 1
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    log("device:", device, "| workers:", jobs)
    snrs = [6, 0, -6] if a.synthetic else E.SNRS
    epochs = 2 if a.synthetic else a.epochs

    if a.synthetic:
        mit, noises, splits = E.make_synthetic(n_train=6, n_val=3, n_test=8, minutes=3.0)
        ext = {f"X{k}": mit[k] for k in splits["test"]}
        groups = {f"X{k}": f"subj{i // 2}" for i, k in enumerate(splits["test"])}
        headers = dict(groups)
        mit = {k: v for k, v in mit.items() if k not in splits["test"]}
        all_recs = dict(mit, **ext)
    else:
        log("loading MIT-BIH + NSTDB + INCART ...")
        mit, noises, splits = E.load_mitdb(a.data_dir), E.load_nstdb(a.data_dir), E.real_splits()
        ext, groups, headers = load_incart(a.data_dir)
        all_recs = dict(mit, **ext)
    ext_names = sorted(ext)
    gnames = sorted(set(groups[r] for r in ext_names))
    gid = {g: i for i, g in enumerate(gnames)}
    rec_group = np.array([gid[groups[r]] for r in ext_names])          # record index -> patient index
    n_groups = len(gnames)
    log(f"external records: {len(ext_names)}, patients (parsed from headers): {n_groups}")
    log("  example header:", headers[ext_names[0]][:120], "| class counts (annot.):",
        np.bincount(np.concatenate([ext[r]["cls"] for r in ext_names]), minlength=4).tolist())
    pd.DataFrame(dict(record=ext_names, patient=rec_group, header=[headers[r] for r in ext_names])) \
        .to_csv(os.path.join(a.out, "ext_patients.csv"), index=False)

    conds = [(None, None)] + [(n, s) for n in E.NOISES for s in snrs]
    keys = [E.cond_key(n, s) for n, s in conds]
    nz_of = {E.cond_key(n, s): (n if n else "clean", s) for n, s in conds}
    noisy_keys = [k for k in keys if k != "clean" and nz_of[k][1] <= NOISY_MAX_SNR]

    # ── build external test and MIT-BIH calibration conditions in parallel (cached) ──
    log("building conditions (external test: all records; calibration: MIT-BIH calibration patients) ...")
    jobs_list = []
    for i, (n, s) in enumerate(conds):
        k = keys[i]
        jobs_list.append((f"ext_{k}", ext_names, n, s, "test", 5000 + i))
        jobs_list.append((f"cal_{k}", splits["val"], n, s, "train", 2000 + i))
        jobs_list.append((f"incal_{k}", ext_names, n, s, "train", 6000 + i))      # INCART with TRAIN-half noise
    todo = [j for j in jobs_list if not os.path.exists(os.path.join(cache, j[0] + ".npz"))]
    if todo:
        Parallel(n_jobs=min(jobs, len(todo)), verbose=0)(
            delayed(build_to_cache)(cache, t, {r: all_recs[r] for r in nm}, nm, noises, n, s, h, sd) for t, nm, n, s, h, sd in todo)
    test = {k: R.cached_ext(cache, f"ext_{k}", None, None, None, None, None, None, None) for k in keys}
    cal = {k: R.cached_ext(cache, f"cal_{k}", None, None, None, None, None, None, None) for k in keys}
    incal = {k: R.cached_ext(cache, f"incal_{k}", None, None, None, None, None, None, None) for k in keys}
    for k in keys:
        log(f"  {k:6s} ext beats={len(test[k]['y'])} spurious={len(test[k]['sp_rec'])}"
            f" det_sens={test[k]['n_match'] / max(test[k]['n_ref'], 1):.3f}")
    tc = test["clean"]
    cnt = pd.DataFrame([{c: int((tc["y"] == j).sum()) for j, c in enumerate(E.CLASSES)}])
    cnt.to_csv(os.path.join(a.out, "ext_class_counts.csv"), index=False)
    print("\n=== external test set, clean, beats per class ===\n" + cnt.to_string(index=False))

    tr_clean = E.cached_condition(a.main_cache, "train", mit, splits["train"], noises, None, None, "train", 3000, True, "detected")
    aug_parts = [tr_clean] + [E.cached_condition(a.main_cache, "trainaug", mit, splits["train"], noises, n, "rand",
                                                 "train", 3100 + j, True, "detected") for j, n in enumerate(E.NOISES)]
    train = {"clean": E.concat_sets([tr_clean]), "aug": E.concat_sets(aug_parts)}
    log("train beats:", {k: len(v["y"]) for k, v in train.items()})

    # in-domain experiments: calibration patients drawn per seed from INCART; test on the remaining, disjoint patients
    k_list = [1, 2, 3] if a.synthetic else [4, 10, 16]

    done_f = os.path.join(a.out, "done.json")
    done = json.load(open(done_f)) if os.path.exists(done_f) else []

    def append(rows_, name):
        if rows_:
            fn = os.path.join(a.out, name)
            pd.DataFrame(rows_).to_csv(fn, mode="a", header=not os.path.exists(fn), index=False)

    for regime in a.regimes.split(","):
        for model in a.models.split(","):
            tag = f"{model}|{regime}"
            if tag in done:
                log("skip (done):", tag)
                continue
            met_rows, sel_rows, win_rows, conf_rows, indom_rows = [], [], [], [], []
            store, ens = [], {k: [] for k in keys}
            for seed in range(a.seeds):
                t0 = time.time()
                pred = E.train_predict(model, train[regime], seed, epochs, device)
                P = {k: pred(test[k]) for k in keys}
                Pc = {k: pred(cal[k]) for k in keys}
                Ps = {k: (pred(dict(X=test[k]["sp_X"], M=test[k]["sp_M"], RR=test[k]["sp_RR"], y=np.zeros(len(test[k]["sp_rec"]))))
                          if len(test[k]["sp_rec"]) else np.zeros((0, 4), np.float32)) for k in keys}
                for k in keys:
                    ens[k].append(P[k].astype(np.float32))
                # error models on MIT-BIH calibration patients (all conditions pooled), as in the main study
                cy = np.concatenate([cal[k]["y"] for k in keys]); cP = np.concatenate([Pc[k] for k in keys])
                cerr = (cP.argmax(1) != cy).astype(int)
                cC = R.conf3(cP); cS = np.concatenate([cal[k]["SQ"] for k in keys]); cF = np.concatenate([cal[k]["SQf"] for k in keys])
                feats = {"sqi_raw": lambda C, S, F: S,
                         "combo_raw": lambda C, S, F: np.concatenate([S, C], 1),
                         "conf_log": lambda C, S, F: R.logc(C),
                         "combo_log_filt": lambda C, S, F: np.concatenate([F, R.logc(C)], 1)}
                mdl = {nm: E.fit_error_model(f(cC, cS, cF), cerr) for nm, f in feats.items()}
                # in-sample UPPER BOUND: SQI+conf error model fitted on the external test patients themselves
                tyy = np.concatenate([test[k]["y"] for k in keys]).astype(int)
                tP = np.concatenate([P[k] for k in keys])
                terr = (tP.argmax(1) != tyy).astype(int)
                ins_m = E.fit_error_model(feats["combo_log_filt"](R.conf3(tP), np.concatenate([test[k]["SQ"] for k in keys]),
                                                                  np.concatenate([test[k]["SQf"] for k in keys]))[::3], terr[::3])
                del tP, tyy, terr

                def scores_for(C_, S_, F_, models=None):
                    sc = {"msp": C_[:, 0], "entropy": C_[:, 1]}
                    for nm, m_ in (mdl if models is None else models).items():
                        sc[nm] = m_.predict_proba(feats[nm](C_, S_, F_))[:, 1] if m_ is not None else C_[:, 0]
                    if models is None:
                        sc["combo_log_filt_ins"] = (ins_m.predict_proba(feats["combo_log_filt"](C_, S_, F_))[:, 1]
                                                    if ins_m is not None else C_[:, 0])
                    vk, vl = R.veto_flags(F_)
                    sc["veto_k_filt"] = C_[:, 0] + vk.astype(float)
                    sc["veto_lit_filt"] = C_[:, 0] + vl.astype(float)
                    return sc

                # conformal thresholds: MIT-BIH calibration (clean / mixed)
                q_clean = E.conformal_q(Pc["clean"], cal["clean"]["y"].astype(int))
                q_mixed = E.conformal_q(cP, cy.astype(int))
                # INCART calibration patients for this seed
                crng = np.random.default_rng(10_000 + seed)
                pool = crng.permutation(n_groups)
                hold_ids = pool[max(k_list):]                                   # same held-out patients for every k
                # in-domain error models and conformal thresholds, fitted on k INCART patients (train-half noise);
                # only the beats of the (at most max(k)) candidate calibration patients are needed
                cand = pool[:max(k_list)]
                isub, Pi = {}, {}
                for k in keys:
                    m_ = np.isin(rec_group[incal[k]["rec"]], cand)
                    isub[k] = dict(y=incal[k]["y"][m_].astype(int), SQ=incal[k]["SQ"][m_], SQf=incal[k]["SQf"][m_],
                                   g=rec_group[incal[k]["rec"]][m_])
                    Pi[k] = pred(subset(incal[k], m_))
                indom = {}
                for kc in k_list:
                    mk = {k: np.isin(isub[k]["g"], pool[:kc]) for k in keys}
                    iy = np.concatenate([isub[k]["y"][mk[k]] for k in keys])
                    iP = np.concatenate([Pi[k][mk[k]] for k in keys])
                    ierr = (iP.argmax(1) != iy).astype(int)
                    iC = R.conf3(iP)
                    iS = np.concatenate([isub[k]["SQ"][mk[k]] for k in keys])
                    iF = np.concatenate([isub[k]["SQf"][mk[k]] for k in keys])
                    indom[kc] = ({nm: E.fit_error_model(f(iC, iS, iF)[::3], ierr[::3]) for nm, f in feats.items()},
                                 E.conformal_q(iP, iy))
                del Pi, isub
                ck = test["clean"]["rec"].astype(np.int64) * 10_000_000 + test["clean"]["ref"]
                cok = (P["clean"].argmax(1) == test["clean"]["y"])
                o = np.argsort(ck); ck_s, cok_s = ck[o], cok[o]
                rng = np.random.default_rng(seed)
                cstore = {}
                for k in keys:
                    d = test[k]; y = d["y"].astype(int); Pk = P[k]
                    err = (Pk.argmax(1) != y).astype(int)
                    n_, s_ = nz_of[k]
                    met_rows.append(dict(model=model, regime=regime, member=f"seed{seed}", noise=n_, snr=s_, n=len(y),
                                         det_sens=d["n_match"] / max(d["n_ref"], 1), n_spurious=len(d["sp_rec"]),
                                         **cls_metrics(Pk, y)))
                    sc = scores_for(R.conf3(Pk), d["SQ"], d["SQf"])
                    sc["random"] = rng.random(len(y))
                    nsp = len(d["sp_rec"])
                    if nsp:
                        scs = scores_for(R.conf3(Ps[k]), d["sp_SQ"], d["sp_SQf"])
                        scs["random"] = rng.random(nsp)
                    kk = d["rec"].astype(np.int64) * 10_000_000 + d["ref"]
                    pos = np.clip(np.searchsorted(ck_s, kk), 0, len(ck_s) - 1)
                    ni = (ck_s[pos] == kk) & cok_s[pos]
                    for nm, v in sc.items():
                        r80, sv, ss_ = R.at_cov(err, y, v)
                        row = dict(model=model, regime=regime, member=f"seed{seed}", noise=n_, snr=s_, score=nm,
                                   aurc=R.aurc(err, v), auroc_err=R.fast_auroc(err, v), risk80=r80, silentV80=sv, silentS80=ss_)
                        if k != "clean":
                            row.update(aurc_ni=R.aurc(err[ni], v[ni]), auroc_ni=R.fast_auroc(err[ni], v[ni]))
                        if nsp:
                            ea = np.concatenate([err, np.ones(nsp, int)])
                            row.update(aurc_sp=R.aurc(ea, np.concatenate([v, scs[nm]])))
                        sel_rows.append(row)
                    if k == "clean" or k in noisy_keys:
                        for nm in ["msp", "sqi_raw", "combo_log_filt"]:
                            win_rows.append(dict(model=model, regime=regime, member=f"seed{seed}", noise=n_, snr=s_, score=nm,
                                                 pooled=R.fast_auroc(err, sc[nm]),
                                                 within_patient=R.stratified_auroc(err, sc[nm], d["rec"]),
                                                 within_patient_class=R.stratified_auroc(err, sc[nm], d["rec"].astype(int) * 10 + y)))
                    if k in noisy_keys:
                        cstore[k] = (err, rec_group[d["rec"]], {s: sc[s].astype(np.float32) for s in ["msp"] + BOOT})
                    # conformal: MIT-BIH-calibrated (cross-database)
                    for lab, q in [("mitbih_clean", q_clean), ("mitbih_mixed", q_mixed)]:
                        conf_rows.append(dict(model=model, regime=regime, member=f"seed{seed}", noise=n_, snr=s_,
                                              calib=lab, k_patients=4, **E.conformal_eval(Pk, y, q)))
                    # conformal: calibrated on k INCART patients (clean beats), tested on the held-out patients
                    ghold = np.isin(rec_group[d["rec"]], hold_ids)
                    for kc in k_list:
                        gc, gt = np.isin(rec_group[test["clean"]["rec"]], pool[:kc]), ghold
                        q = E.conformal_q(P["clean"][gc], test["clean"]["y"][gc].astype(int))
                        conf_rows.append(dict(model=model, regime=regime, member=f"seed{seed}", noise=n_, snr=s_,
                                              calib="incart_clean", k_patients=kc,
                                              n_cal_S=int((test["clean"]["y"][gc] == 1).sum()),
                                              n_cal_F=int((test["clean"]["y"][gc] == 3).sum()),
                                              **E.conformal_eval(Pk[gt], y[gt], q)))
                        mdl_k, q_in = indom[kc]
                        conf_rows.append(dict(model=model, regime=regime, member=f"seed{seed}", noise=n_, snr=s_,
                                              calib="incart_mixed", k_patients=kc, **E.conformal_eval(Pk[ghold], y[ghold], q_in)))
                        if k == "clean" or k in noisy_keys:
                            sck = scores_for(R.conf3(Pk)[ghold], d["SQ"][ghold], d["SQf"][ghold], models=mdl_k)
                            eh = err[ghold]
                            for nm in ["msp", "sqi_raw", "combo_raw", "combo_log_filt", "conf_log"]:
                                indom_rows.append(dict(model=model, regime=regime, member=f"seed{seed}", k_patients=kc,
                                                       noise=n_, snr=s_, score=nm, aurc=R.aurc(eh, sck[nm]),
                                                       auroc_err=R.fast_auroc(eh, sck[nm])))
                store.append(cstore)
                log(f"{tag} seed{seed} done ({time.time() - t0:.0f}s)")

            # ── seed ensemble: MSP vs mutual information ──
            if a.seeds > 1:
                for k in keys:
                    d = test[k]; y = d["y"].astype(int); n_, s_ = nz_of[k]
                    m, mi = mutual_info(np.stack(ens[k]))
                    err = (m.argmax(1) != y).astype(int)
                    met_rows.append(dict(model=model, regime=regime, member="ensemble", noise=n_, snr=s_, n=len(y),
                                         det_sens=d["n_match"] / max(d["n_ref"], 1), n_spurious=len(d["sp_rec"]),
                                         **cls_metrics(m, y)))
                    for nm, v in [("msp", 1 - m.max(1)), ("mi", mi)]:
                        r80, sv, ss_ = R.at_cov(err, y, v)
                        sel_rows.append(dict(model=model, regime=regime, member="ensemble", noise=n_, snr=s_, score=nm,
                                             aurc=R.aurc(err, v), auroc_err=R.fast_auroc(err, v), risk80=r80,
                                             silentV80=sv, silentS80=ss_))
            log(f"{tag}: patient-level bootstrap ({a.n_boot} resamples, {n_groups} patients) ...")
            flat = [v for st_ in store for v in st_.values()]
            boot_rows = [dict(model=model, regime=regime, **r_) for r_ in group_aurc_boot(flat, None, n_groups, a.n_boot)]
            append(met_rows, "ext_metrics.csv")
            append(sel_rows, "ext_selective.csv")
            append(win_rows, "ext_within.csv")
            append(conf_rows, "ext_conformal.csv")
            append(indom_rows, "ext_indomain.csv")
            append(boot_rows, "ext_bootstrap.csv")
            done.append(tag)
            json.dump(done, open(done_f, "w"))
            log(f"saved {tag}")
            print(pd.DataFrame(boot_rows).round(4).to_string(index=False))

    # ── summary ──
    sel = pd.read_csv(os.path.join(a.out, "ext_selective.csv"))
    met = pd.read_csv(os.path.join(a.out, "ext_metrics.csv"))
    seedm = met[met.member.str.startswith("seed")]
    print("\n=== external: clean condition (mean over seeds) ===")
    print(seedm[seedm.noise == "clean"].groupby(["model", "regime"])[["acc", "macroF1", "F1_N", "F1_S", "F1_V", "F1_F", "ece"]].mean().round(3))
    s = sel[(sel.member.str.startswith("seed")) & (sel.noise != "clean") & (sel.snr <= NOISY_MAX_SNR)]
    print("\n=== external: abstention on noisy conditions (SNR<=6), mean over seeds ===")
    print(s.groupby(["model", "regime", "score"])[["aurc", "auroc_err", "silentV80", "auroc_ni"]].mean().round(4).unstack("score")["aurc"].round(4))
    e = sel[(sel.member == "ensemble") & (sel.noise != "clean") & (sel.snr <= NOISY_MAX_SNR)]
    if len(e):
        print("\n=== external: seed ensemble, noisy AURC ===")
        print(e.groupby(["model", "regime", "score"])["aurc"].mean().unstack().round(4))
    cf = pd.read_csv(os.path.join(a.out, "ext_conformal.csv"))
    cfn = cf[(cf.noise == "clean")]
    print("\n=== external: conformal coverage on CLEAN test beats (target 0.90) ===")
    print(cfn.groupby(["model", "regime", "calib", "k_patients"])[["cov_N", "cov_S", "cov_V", "cov_F", "set_size"]].mean().round(3))
    ind = pd.read_csv(os.path.join(a.out, "ext_indomain.csv"))
    ind = ind[(ind.noise != "clean") & (ind.snr <= NOISY_MAX_SNR)]
    print("\n=== external: IN-DOMAIN error models (k INCART calibration patients), held-out patients, noisy AURC ===")
    print(ind.groupby(["model", "regime", "k_patients", "score"]).aurc.mean().unstack().round(4))
    print("\n=== external: patient-level bootstrap, AURC difference vs max-softmax (positive = worse) ===")
    print(pd.read_csv(os.path.join(a.out, "ext_bootstrap.csv")).round(4).to_string(index=False))
    log("ALL DONE. Files in", a.out)


if __name__ == "__main__":
    main()
