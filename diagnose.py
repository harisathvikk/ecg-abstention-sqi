"""
diagnose.py  —  NEW
Why did SQI fail to predict classifier errors? Uses the beat-level cache written by run_study.py
(no PhysioNet download needed). Trains one CNN seed per regime, then:

  A  does each SQI respond to noise the way it should?              (sanity check of the SQI code)
  B  how much of each SQI's variance is just 'which patient is this'? (between-patient confounding)
  C  AUROC for error detection: pooled over patients vs within-patient
  D  how many errors are noise-induced vs already present on clean data
  E  re-fit the error models with patient-grouped cross-validation on all 22 test patients
     (instead of the 4 calibration patients) and with patient-normalised SQIs

    python diagnose.py --cache /content/drive/MyDrive/ecg_main/cache --out /content/drive/MyDrive/ecg_main/diag
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import ecgunc as E

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
SIGN = {"kSQI": -1, "pSQI": -1, "basSQI": -1, "hfSQI": 1, "bSQI": -1}     # higher score = worse (as in run_study)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(cache, prefix, suffix):
    out = {}
    for fn in glob.glob(os.path.join(cache, f"{prefix}_*_{suffix}.npz")):
        key = os.path.basename(fn)[len(prefix) + 1:-len(suffix) - 5]
        z = np.load(fn)
        out[key] = {k: z[k] for k in z.files}
    return out


def split_key(k):
    if k == "clean":
        return "clean", np.nan
    noise = k.rstrip("-0123456789")
    return noise, int(k[len(noise):])


def auroc(e, s):
    return np.nan if e.min() == e.max() else roc_auc_score(e, s)


def within_auroc(e, s, rec, min_n=10):
    num = den = 0.0
    for r in np.unique(rec):
        m = rec == r
        ne, no = int(e[m].sum()), int((1 - e[m]).sum())
        if ne < min_n or no < min_n:
            continue
        w = min(ne, no)
        num += w * roc_auc_score(e[m], s[m])
        den += w
    return num / den if den else np.nan


def rel_norm(sq, rec, ref=None, ref_rec=None):
    """Patient-normalise SQIs: (sq - patient median) / patient MAD. ref = a reference condition (e.g. the same
    patient's clean signal); if None the patient's own median in the same condition is used."""
    out = np.zeros_like(sq, dtype=np.float32)
    for r in np.unique(rec):
        m = rec == r
        base = sq[m] if ref is None else (ref[ref_rec == r] if (ref_rec == r).any() else sq[m])
        med = np.median(base, 0)
        mad = np.median(np.abs(base - med), 0) + 1e-6
        out[m] = (sq[m] - med) / mad
    return out


def smooth_sq(sq, rec, w=61):
    """Segment-level SQI: rolling median over the w neighbouring beats of the same patient (~1 minute at 60 bpm).
    A single beat's window can be distorted by the beat itself (e.g. an ectopic QRS); a median over many beats
    reflects the local NOISE level instead."""
    out = np.zeros_like(sq, dtype=np.float32)
    for r in np.unique(rec):
        m = np.where(rec == r)[0]
        out[m] = pd.DataFrame(sq[m]).rolling(w, center=True, min_periods=1).median().values
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="pf1_detected")
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--regimes", default="clean,aug")
    ap.add_argument("--subsample", type=int, default=4, help="train error models on every k-th beat (speed)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    test = load(a.cache, "test", f"test_{a.tag}")
    tr_clean = load(a.cache, "train", f"train_{a.tag}")["clean"]
    aug_parts = [tr_clean] + list(load(a.cache, "trainaug", f"train_{a.tag}").values())
    train = {"clean": E.concat_sets([tr_clean]), "aug": E.concat_sets(aug_parts)}
    log("test conditions:", len(test), "| train beats:", {k: len(v["y"]) for k, v in train.items()})
    keys = sorted(test, key=lambda k: (split_key(k)[0], -np.nan_to_num(split_key(k)[1], nan=99)))

    # ─────────── A. SQI response to noise (independent of any classifier) ───────────
    rows = []
    for k in keys:
        n, s = split_key(k)
        r = dict(noise=n, snr=s)
        for j, nm in enumerate(E.SQI_NAMES):
            r[nm] = float(np.median(test[k]["SQ"][:, j]))
        rows.append(r)
    A = pd.DataFrame(rows)
    A.to_csv(os.path.join(a.out, "A_sqi_vs_noise.csv"), index=False)
    print("\n=== A. median SQI per condition (should move monotonically with noise) ===")
    print(A.round(3).to_string(index=False))

    # ─────────── B. between-patient share of SQI variance (clean test condition) ───────────
    cl = test["clean"]
    rows = []
    for j, nm in enumerate(E.SQI_NAMES):
        x = cl["SQ"][:, j].astype(float)
        tot = x.var()
        within = sum(x[cl["rec"] == r].var() * (cl["rec"] == r).mean() for r in np.unique(cl["rec"]))
        rows.append(dict(sqi=nm, share_of_variance_between_patients=1 - within / tot if tot > 0 else np.nan))
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(a.out, "B_between_patient_variance.csv"), index=False)
    print("\n=== B. share of each SQI's variance that is just 'which patient' (clean test data) ===")
    print(B.round(3).to_string(index=False))

    # ─────────── F. is a beat's own SQI contaminated by the beat type (clean data only)? ───────────
    names = ["N", "S", "V", "F"]
    rows = []
    for j, nm in enumerate(E.SQI_NAMES):
        r = dict(sqi=nm)
        for c, cn in enumerate(names):
            m = cl["y"] == c
            r[f"median_{cn}"] = float(np.median(cl["SQ"][m, j])) if m.any() else np.nan
        mv = (cl["y"] == 0) | (cl["y"] == 2)
        r["AUROC_V_vs_N (0.5 = no link)"] = auroc((cl["y"][mv] == 2).astype(int), cl["SQ"][mv, j])
        rows.append(r)
    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(a.out, "F_sqi_by_beat_class_clean.csv"), index=False)
    print("\n=== F. SQI by TRUE beat class on CLEAN data (a pure noise index should not depend on beat type) ===")
    print(F.round(3).to_string(index=False))

    results = []
    for regime in a.regimes.split(","):
        log(f"training {a.model}|{regime} seed {a.seed} ...")
        pred = E.train_predict(a.model, train[regime], a.seed, a.epochs, device)
        probs = {k: pred(test[k]) for k in keys}
        y = {k: test[k]["y"].astype(int) for k in keys}
        err = {k: (probs[k].argmax(1) != y[k]).astype(int) for k in keys}

        # ─────────── C. pooled vs within-patient AUROC ───────────
        rows = []
        for k in ["clean", "bw-6", "em-6", "em0", "ma-6", "ma0"]:
            if k not in test:
                continue
            msp = E.conf_scores(probs[k])[0]
            r = dict(cond=k, score="msp", pooled=auroc(err[k], msp), within=within_auroc(err[k], msp, test[k]["rec"]))
            rows.append(r)
            smq = smooth_sq(test[k]["SQ"], test[k]["rec"])
            for j, nm in enumerate(E.SQI_NAMES):
                s = SIGN[nm] * test[k]["SQ"][:, j]
                rows.append(dict(cond=k, score=nm, pooled=auroc(err[k], s), within=within_auroc(err[k], s, test[k]["rec"])))
                s = SIGN[nm] * smq[:, j]
                rows.append(dict(cond=k, score="smooth_" + nm, pooled=auroc(err[k], s), within=within_auroc(err[k], s, test[k]["rec"])))
        C = pd.DataFrame(rows)
        C.insert(0, "regime", regime)
        print(f"\n=== C. error-detection AUROC, {a.model}|{regime}: pooled over patients vs within a patient ===")
        print(C.pivot_table(index="score", columns="cond", values=["pooled", "within"]).round(2).to_string())

        # ─────────── D. noise-induced vs intrinsic errors ───────────
        e0 = err["clean"].mean()
        rows = [dict(cond=k, err_rate=err[k].mean(), clean_err_rate=e0,
                     share_of_errors_already_present_on_clean=min(1.0, e0 / max(err[k].mean(), 1e-9)))
                for k in ["bw-6", "em-6", "em0", "ma-6", "ma0"] if k in test]
        D = pd.DataFrame(rows)
        D.insert(0, "regime", regime)
        print(f"\n=== D. how much of the error is already there on clean data ({a.model}|{regime}) ===")
        print(D.round(3).to_string(index=False))

        # ─────────── E. error models re-fitted with patient-grouped CV on the 22 test patients ───────────
        P = np.concatenate([probs[k] for k in keys])
        Y = np.concatenate([y[k] for k in keys])
        ER = np.concatenate([err[k] for k in keys])
        SQ = np.concatenate([test[k]["SQ"] for k in keys])
        RC = np.concatenate([test[k]["rec"] for k in keys])
        CD = np.concatenate([[k] * len(y[k]) for k in keys])
        SQrel = np.concatenate([rel_norm(test[k]["SQ"], test[k]["rec"]) for k in keys])
        SQdel = np.concatenate([rel_norm(test[k]["SQ"], test[k]["rec"], cl["SQ"], cl["rec"]) for k in keys])
        SQsm = np.concatenate([smooth_sq(test[k]["SQ"], test[k]["rec"]) for k in keys])
        conf = np.stack(E.conf_scores(P), 1)
        confl = np.log(np.clip(conf, 1e-6, None))      # log-scale confidence: lets a linear model reproduce the max-softmax ranking
        feats = {"msp_log": confl[:, :1], "conf_log": confl, "combo_log": np.concatenate([SQ, confl], 1),
                 "combo_smooth_log": np.concatenate([SQsm, confl], 1),
                 "conf_lr": conf, "sqi_smooth": SQsm, "combo_smooth": np.concatenate([SQsm, conf], 1), "sqi": SQ, "combo": np.concatenate([SQ, conf], 1),
                 "sqi_rel": SQrel, "combo_rel": np.concatenate([SQrel, conf], 1),
                 "sqi_delta": SQdel, "combo_delta": np.concatenate([SQdel, conf], 1)}
        oof = {k: np.full(len(Y), np.nan) for k in feats}
        gkf = GroupKFold(n_splits=min(5, len(np.unique(RC))))
        for fi, (tri, tei) in enumerate(gkf.split(P, ER, RC)):
            tri = tri[::a.subsample]
            for nm, F in feats.items():
                mdl = E.fit_error_model(F[tri], ER[tri])
                if mdl is not None:
                    oof[nm][tei] = mdl.predict_proba(F[tei])[:, 1]
            log(f"  grouped-CV fold {fi + 1} done")
        rows = []
        for k in keys:
            n, s = split_key(k)
            m = CD == k
            scores = {"msp": conf[m][:, 0], "random": np.random.default_rng(0).random(m.sum())}
            scores.update({nm: oof[nm][m] for nm in feats})
            for nm, sc in scores.items():
                _, aurc, au = E.selective_rows(Y[m], P[m].argmax(1), sc)
                rows.append(dict(regime=regime, noise=n, snr=s, score=nm, aurc=aurc, auroc_err=au))
        Ed = pd.DataFrame(rows)
        results.append((C, D, Ed))
        noisy = Ed[(Ed.noise != "clean") & (Ed.snr <= 6)]
        print(f"\n=== E. AURC on noisy conditions (SNR<=6), error models fitted by patient-grouped CV ({a.model}|{regime}); lower = better ===")
        print(noisy.groupby("score")[["aurc", "auroc_err"]].mean().sort_values("aurc").round(4).to_string())
        print("\n   same, per noise type at the harshest SNR (-6 dB):")
        print(Ed[Ed.snr == -6].pivot_table(index="score", columns="noise", values="aurc").round(3).to_string())

        # ─────────── G. MIXED STREAM: noise level varies over time (what a deployed system sees) ───────────
        # (exploratory: added after seeing the per-condition results; must be reported as such)
        rng = np.random.default_rng(0)
        is_clean = CD == "clean"
        nz_idx = np.where(~is_clean)[0]
        mixes = {"all 19 conditions pooled": np.arange(len(Y)),
                 "50% clean + 50% noisy": np.concatenate([np.where(is_clean)[0],
                                                          rng.choice(nz_idx, is_clean.sum(), replace=False)])}
        rows = []
        for mname, idx in mixes.items():
            for nm in ["msp", "conf_lr", "msp_log", "conf_log", "sqi", "sqi_smooth", "combo", "combo_smooth",
                       "combo_log", "combo_smooth_log", "random"]:
                sc = conf[idx, 0] if nm == "msp" else (rng.random(len(idx)) if nm == "random" else oof[nm][idx])
                srows, aurc, au = E.selective_rows(Y[idx], P[idx].argmax(1), sc)
                r80 = [r for r in srows if abs(r["coverage"] - 0.8) < 1e-9][0]
                rows.append(dict(regime=regime, mix=mname, score=nm, aurc=aurc, auroc_err=au, risk_at_80=r80["risk"],
                                 silent_V_at_80=r80["silent_V"], silent_S_at_80=r80["silent_S"]))
        G = pd.DataFrame(rows)
        G.to_csv(os.path.join(a.out, f"G_mixed_stream_{regime}.csv"), index=False)
        print(f"\n=== G. MIXED STREAM, {a.model}|{regime}: beats from all noise levels in one stream (lower AURC / silent-V = better) ===")
        print(G.round(4).to_string(index=False))

    pd.concat([r[0] for r in results]).to_csv(os.path.join(a.out, "C_auroc_pooled_vs_within.csv"), index=False)
    pd.concat([r[1] for r in results]).to_csv(os.path.join(a.out, "D_error_attribution.csv"), index=False)
    pd.concat([r[2] for r in results]).to_csv(os.path.join(a.out, "E_grouped_cv_abstention.csv"), index=False)
    log("done. Files in", a.out)


if __name__ == "__main__":
    main()
