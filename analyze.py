"""
analyze.py  —  NEW
Turns the result CSVs from run_study.py into the paper's tables, figures and a plain-text list of
key numbers. Every number quoted in the paper should be copied from the files this writes.

    python analyze.py --dir /path/to/ecg_main            (tag defaults to pf1_detected)
    python analyze.py --dir /path/to/ecg_nofilter --tag pf0_detected
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── style: print-friendly (IEEE column = 3.5 in, page = 7.2 in), validated categorical colours ───
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4"}
MK = ["o", "s", "^", "D", "v"]
LS = ["-", "--", ":", "-.", "-"]
HATCH = ["", "//", "..", "xx", "\\\\"]
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.2,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.3,
                     "lines.markersize": 3.4, "grid.linewidth": 0.4, "grid.alpha": 0.35, "figure.dpi": 150,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
NOISE_NAME = {"bw": "Baseline wander", "em": "Electrode motion", "ma": "Muscle artefact"}
SCORE_LABEL = {"msp": "Max-softmax", "entropy": "Entropy", "margin": "Margin", "conf_lr": "Confidence (learned)",
               "sqi": "SQI only", "combo": "SQI + confidence", "mi": "Ensemble MI", "random": "Random"}
# one fixed style per abstention score, identical in every figure (colour + marker + line + hatch)
SCORE_STYLE = {"msp": (C["blue"], "o", "-", ""), "entropy": (C["orange"], "s", "--", "//"),
               "conf_lr": (C["aqua"], "^", ":", ".."), "sqi": (C["yellow"], "D", "-.", "xx"),
               "combo": (C["magenta"], "v", "-", "\\\\"), "random": ("#777777", "x", "-", ""),
               "margin": ("#555555", "P", "--", "++"), "mi": ("#555555", "*", ":", "oo")}
CONF_SCORES = ["msp", "entropy", "margin", "conf_lr"]
ALL_SCORES = CONF_SCORES + ["sqi", "combo"]
NOISY_MAX_SNR = 6                      # "noisy" = conditions with SNR <= 6 dB


def savefig(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"))
    plt.close(fig)


def ms(x):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.3f} ± {np.nanstd(x, ddof=1) if len(x) > 1 else 0:.3f}"


def xorder(df):
    snrs = sorted(df.loc[df.noise != "clean", "snr"].dropna().unique(), reverse=True)
    return ["clean"] + [int(s) for s in snrs]


def series_by_snr(d, col, order, noise):
    """mean over seeds of col, for one noise type, aligned to the x order (clean first)."""
    vals, errs = [], []
    for x in order:
        sub = d[(d.noise == "clean")] if x == "clean" else d[(d.noise == noise) & (d.snr == x)]
        per_seed = sub.groupby("member")[col].mean()
        vals.append(per_seed.mean())
        errs.append(per_seed.std(ddof=1) if len(per_seed) > 1 else 0)
    return np.array(vals), np.array(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", default="pf1_detected")
    ap.add_argument("--out", default=None)
    ap.add_argument("--main-model", default="cnn")
    a = ap.parse_args()
    out = a.out or os.path.join(a.dir, f"analysis_{a.tag}")
    os.makedirs(out, exist_ok=True)
    M = pd.read_csv(os.path.join(a.dir, f"metrics_{a.tag}.csv"))
    S = pd.read_csv(os.path.join(a.dir, f"selective_{a.tag}.csv"))
    Cf = pd.read_csv(os.path.join(a.dir, f"conformal_{a.tag}.csv"))
    seedM, seedS = M[M.member.str.startswith("seed")], S[S.member.str.startswith("seed")]
    n_seeds = seedM.member.nunique()
    findings = [f"Analysis of {a.dir} (tag {a.tag}); seeds per model: {n_seeds}"]
    order = xorder(M)
    noisyM = lambda d: d[(d.noise != "clean") & (d.snr <= NOISY_MAX_SNR)]

    # ── Table 1: clean-condition performance ──────────────────────────────────────────────
    rows = []
    for (mo, rg), d in seedM[seedM.noise == "clean"].groupby(["model", "regime"]):
        r = dict(model=mo, regime=rg)
        for c in ["macroF1", "F1_N", "F1_S", "F1_V", "F1_F", "ece_raw", "ece_ts_clean", "det_sens"]:
            r[c] = ms(d[c])
        rows.append(r)
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(out, "table1_clean.csv"), index=False)

    # ── Table 2: degradation under noise (mean over seeds) ────────────────────────────────
    t2 = seedM.groupby(["model", "regime", "noise", "snr"], dropna=False)[
        ["macroF1", "F1_V", "F1_S", "e2e_recall_V", "e2e_recall_S", "ece_raw", "ece_ts_clean", "ece_ts_mixed",
         "det_sens", "det_ppv"]].mean().reset_index()
    t2.to_csv(os.path.join(out, "table2_degradation.csv"), index=False)

    # ── Table 3: abstention comparison on noisy conditions ───────────────────────────────
    sel = seedS.copy()
    sel_nz = sel[(sel.noise != "clean") & (sel.snr <= NOISY_MAX_SNR)]
    base = sel_nz[sel_nz.coverage == 1.0]
    c80 = sel_nz[sel_nz.coverage == 0.8][["model", "regime", "member", "noise", "snr", "score", "risk", "silent_V",
                                           "silent_S"]].rename(columns={"risk": "risk80", "silent_V": "silentV80",
                                                                         "silent_S": "silentS80"})
    t3src = base[["model", "regime", "member", "noise", "snr", "score", "aurc", "auroc_err"]].merge(
        c80, on=["model", "regime", "member", "noise", "snr", "score"])
    rows = []
    for (mo, rg, sc), d in t3src.groupby(["model", "regime", "score"]):
        per_seed = d.groupby("member")[["aurc", "auroc_err", "risk80", "silentV80", "silentS80"]].mean()
        rows.append(dict(model=mo, regime=rg, score=sc, **{c: ms(per_seed[c]) for c in per_seed.columns},
                         _aurc=per_seed["aurc"].mean(), _sv=per_seed["silentV80"].mean()))
    t3 = pd.DataFrame(rows)
    t3.drop(columns=["_aurc", "_sv"]).to_csv(os.path.join(out, "table3_abstention.csv"), index=False)
    # same by noise type
    rows = []
    for (mo, rg, nz, sc), d in t3src.groupby(["model", "regime", "noise", "score"]):
        per_seed = d.groupby("member")[["aurc", "auroc_err", "risk80", "silentV80", "silentS80"]].mean()
        rows.append(dict(model=mo, regime=rg, noise=nz, score=sc, **{c: ms(per_seed[c]) for c in per_seed.columns}))
    pd.DataFrame(rows).to_csv(os.path.join(out, "table3b_abstention_by_noise.csv"), index=False)
    # ensemble rows (include mutual information)
    ens = S[(S.member == "ens") & (S.noise != "clean") & (S.snr <= NOISY_MAX_SNR)]
    eb = ens[ens.coverage == 1.0].groupby(["model", "regime", "score"])[["aurc", "auroc_err"]].mean()
    e8 = ens[ens.coverage == 0.8].groupby(["model", "regime", "score"])[["risk", "silent_V", "silent_S"]].mean()
    eb.join(e8).reset_index().to_csv(os.path.join(out, "table3c_ensemble.csv"), index=False)

    # ── Table 4: paired statistics on AURC (lower is better) ────────────────────────────
    rng = np.random.default_rng(0)
    pairs = [("combo", "msp"), ("sqi", "msp"), ("combo", "conf_lr"), ("sqi", "conf_lr"), ("combo", "sqi")]
    rows = []
    for (mo, rg), d in base.groupby(["model", "regime"]):
        piv = d.pivot_table(index=["member", "noise", "snr"], columns="score", values="aurc")
        for x, y in pairs:
            if x not in piv or y not in piv:
                continue
            diff = (piv[x] - piv[y]).dropna().values            # negative = x better than y
            if len(diff) == 0:
                continue
            boot = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
            try:
                p = st.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            except ValueError:
                p = np.nan
            rows.append(dict(model=mo, regime=rg, a=x, b=y, n_pairs=len(diff), mean_diff_aurc=diff.mean(),
                             ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                             rel_change_pct=100 * diff.mean() / piv[y].mean(), frac_a_better=(diff < 0).mean(),
                             wilcoxon_p=p))
    t4 = pd.DataFrame(rows)
    t4.to_csv(os.path.join(out, "table4_paired_stats.csv"), index=False)

    # ── Table 5: conformal coverage under noise ──────────────────────────────────────────
    cn = Cf[Cf.member.str.startswith("seed") & (Cf.noise != "clean") & (Cf.snr <= NOISY_MAX_SNR)]
    t5 = cn.groupby(["model", "regime", "calib"])[["cov_N", "cov_S", "cov_V", "cov_F", "set_size", "empty"]].mean().reset_index()
    cc = Cf[Cf.member.str.startswith("seed") & (Cf.noise == "clean")]
    t5c = cc.groupby(["model", "regime", "calib"])[["cov_N", "cov_S", "cov_V", "cov_F", "set_size", "empty"]].mean().reset_index()
    t5.to_csv(os.path.join(out, "table5_conformal_noisy.csv"), index=False)
    t5c.to_csv(os.path.join(out, "table5_conformal_clean.csv"), index=False)

    # ── key findings (plain text, auto-generated: copy numbers from here) ─────────────────
    for (mo, rg), d in t3.groupby(["model", "regime"]):
        d = d.set_index("score")
        best = d.loc[[s for s in ALL_SCORES if s in d.index], "_aurc"].idxmin()
        bestconf = d.loc[[s for s in CONF_SCORES if s in d.index], "_aurc"].idxmin()
        findings.append(f"[{mo}|{rg}] noisy-condition AURC: best confidence score = {bestconf} "
                        f"({d.loc[bestconf, 'aurc']}); best overall = {best} ({d.loc[best, 'aurc']}); "
                        f"silent-V@80% cov: msp {d.loc['msp', 'silentV80'] if 'msp' in d.index else 'na'} vs "
                        f"{best} {d.loc[best, 'silentV80']}")
    for _, r in t4.iterrows():
        findings.append(f"[{r.model}|{r.regime}] AURC {r.a} vs {r.b}: mean diff {r.mean_diff_aurc:+.4f} "
                        f"(95% CI {r.ci_lo:+.4f}..{r.ci_hi:+.4f}), {r.rel_change_pct:+.1f}%, "
                        f"better in {100 * r.frac_a_better:.0f}% of {r.n_pairs} (condition,seed) pairs, Wilcoxon p={r.wilcoxon_p:.2g}")
    worst = t2[(t2.noise != "clean")].sort_values("macroF1").head(3)
    for _, r in worst.iterrows():
        findings.append(f"worst conditions: {r.model}|{r.regime} {r.noise}{int(r.snr)}: macroF1={r.macroF1:.3f}, "
                        f"F1_V={r.F1_V:.3f}, ECE={r.ece_raw:.3f}, det_sens={r.det_sens:.3f}")
    # "hidden failure": ECE low but V F1 collapsed
    hf = t2[(t2.noise != "clean")].copy()
    hf["ece_ok"] = hf.ece_raw < 0.10
    hf["v_drop"] = hf.merge(t2[t2.noise == "clean"][["model", "regime", "F1_V"]].rename(columns={"F1_V": "F1_V_clean"}),
                            on=["model", "regime"], how="left")["F1_V_clean"].values - hf.F1_V
    hid = hf[hf.ece_ok & (hf.v_drop > 0.2)]
    findings.append(f"hidden failures (ECE<0.10 but V-F1 drops >0.2 vs clean): {len(hid)} of {len(hf)} model/noise/SNR cells")
    open(os.path.join(out, "key_findings.txt"), "w").write("\n".join(findings))

    # ═════════════════════════════ FIGURES ═════════════════════════════════════════════════
    xs = np.arange(len(order))
    xl = [str(o) for o in order]

    # Fig 1: degradation (macro-F1 and V-F1), clean-trained vs noise-augmented CNN
    mo = a.main_model
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 3.3), sharex=True, sharey="row")
    for j, nz in enumerate(["bw", "em", "ma"]):
        for i, (col, lab) in enumerate([("macroF1", "Macro-F1"), ("F1_V", "F1 (V beats)")]):
            for k, (rg, colr) in enumerate([("clean", C["blue"]), ("aug", C["orange"])]):
                d = seedM[(seedM.model == mo) & (seedM.regime == rg)]
                if d.empty:
                    continue
                v, e = series_by_snr(d, col, order, nz)
                ax[i, j].errorbar(xs, v, yerr=e, color=colr, marker=MK[k], ls=LS[k], capsize=1.5, elinewidth=0.6,
                                  label=("trained on clean" if rg == "clean" else "noise-augmented") if (i == 0 and j == 0) else None)
            ax[i, j].grid(True, axis="y")
            if j == 0:
                ax[i, j].set_ylabel(lab)
            if i == 0:
                ax[i, j].set_title(NOISE_NAME[nz])
            if i == 1:
                ax[i, j].set_xlabel("SNR (dB)")
                ax[i, j].set_xticks(xs)
                ax[i, j].set_xticklabels(xl)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, out, "fig1_degradation")

    # Fig 2: calibration vs noise (raw / TS-clean / TS-mixed) for clean-trained and augmented CNN
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 3.3), sharex=True, sharey=True)
    for i, rg in enumerate(["clean", "aug"]):
        d = seedM[(seedM.model == mo) & (seedM.regime == rg)]
        if d.empty:
            continue
        for j, nz in enumerate(["bw", "em", "ma"]):
            for k, (col, lab, colr) in enumerate([("ece_raw", "raw", C["blue"]), ("ece_ts_clean", "temp.-scaled on clean", C["orange"]),
                                                  ("ece_ts_mixed", "temp.-scaled on clean+noisy", C["aqua"])]):
                v, e = series_by_snr(d, col, order, nz)
                ax[i, j].errorbar(xs, v, yerr=e, color=colr, marker=MK[k], ls=LS[k], capsize=1.5, elinewidth=0.6,
                                  label=lab if (i == 0 and j == 0) else None)
            ax[i, j].grid(True, axis="y")
            if i == 0:
                ax[i, j].set_title(NOISE_NAME[nz])
            if j == 0:
                ax[i, j].set_ylabel(f"ECE ({'clean-trained' if rg == 'clean' else 'augmented'})")
            if i == 1:
                ax[i, j].set_xlabel("SNR (dB)")
                ax[i, j].set_xticks(xs)
                ax[i, j].set_xticklabels(xl)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    savefig(fig, out, "fig2_calibration")

    # Fig 3: abstention quality (AURC, lower = better) per noise type
    for rg in ["clean", "aug"]:
        d = t3src[(t3src.model == mo) & (t3src.regime == rg)]
        if d.empty:
            continue
        scs = [s for s in ["msp", "entropy", "conf_lr", "sqi", "combo"] if s in set(d.score)]
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
        for pi, (col, ylab) in enumerate([("aurc", "AURC (lower = better)"), ("silentV80", "Silent V-beat errors at 80% coverage")]):
            w = 0.8 / len(scs)
            for k, sc in enumerate(scs):
                vals, errs = [], []
                for nz in ["bw", "em", "ma"]:
                    ps = d[(d.score == sc) & (d.noise == nz)].groupby("member")[col].mean()
                    vals.append(ps.mean())
                    errs.append(ps.std(ddof=1) if len(ps) > 1 else 0)
                ax[pi].bar(np.arange(3) + (k - (len(scs) - 1) / 2) * w, vals, w * 0.92, yerr=errs, capsize=1.2,
                           color=SCORE_STYLE[sc][0], edgecolor="#333", linewidth=0.4, hatch=SCORE_STYLE[sc][3],
                           error_kw=dict(elinewidth=0.6), label=SCORE_LABEL[sc] if pi == 0 else None)
            ax[pi].set_xticks(range(3))
            ax[pi].set_xticklabels([NOISE_NAME[n] for n in ["bw", "em", "ma"]])
            ax[pi].set_ylabel(ylab)
            ax[pi].grid(True, axis="y")
        ax[0].legend(frameon=False, ncol=1, loc="upper left")
        fig.tight_layout()
        savefig(fig, out, f"fig3_abstention_{rg}")

    # Fig 4: risk-coverage at the harshest SNR available
    worst_snr = int(M.loc[M.noise != "clean", "snr"].min())
    for rg in ["clean", "aug"]:
        d = seedS[(seedS.model == mo) & (seedS.regime == rg) & (seedS.snr == worst_snr)]
        if d.empty:
            continue
        fig, ax = plt.subplots(2, 2, figsize=(7.2, 3.6), sharex=True)
        for j, nz in enumerate(["em", "ma"]):
            for i, (col, lab) in enumerate([("risk", "Error rate on accepted beats"), ("silent_V", "Silent V-beat error rate")]):
                for k, sc in enumerate(["random", "msp", "sqi", "combo"]):
                    dd = d[(d.noise == nz) & (d.score == sc)].groupby("coverage")[col].mean().sort_index()
                    if dd.empty:
                        continue
                    colr, mk, ls, _ = SCORE_STYLE[sc]
                    ax[i, j].plot(dd.index, dd.values, color=colr, marker=mk, ls=ls, label=SCORE_LABEL[sc] if (i == 0 and j == 0) else None)
                ax[i, j].grid(True, axis="y")
                ax[i, j].invert_xaxis()
                if j == 0:
                    ax[i, j].set_ylabel(lab)
                if i == 0:
                    ax[i, j].set_title(f"{NOISE_NAME[nz]}, SNR {worst_snr} dB")
                if i == 1:
                    ax[i, j].set_xlabel("Coverage (fraction of beats classified)")
        h, l = ax[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        savefig(fig, out, f"fig4_riskcoverage_{rg}")

    # Fig 5: R-peak detection and RR-feature dependence
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    dd = seedM[(seedM.model == mo) & (seedM.regime == "clean")]
    for k, nz in enumerate(["bw", "em", "ma"]):
        v, e = series_by_snr(dd, "det_sens", order, nz)
        ax[0].plot(xs, v, color=list(C.values())[k], marker=MK[k], ls=LS[k], label=NOISE_NAME[nz])
    ax[0].set_ylabel("R-peak detection sensitivity")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(xl)
    ax[0].set_xlabel("SNR (dB)")
    ax[0].grid(True, axis="y")
    ax[0].legend(frameon=False, loc="lower left")
    for k, nz in enumerate(["bw", "em", "ma"]):
        a1 = seedM[(seedM.model == "cnn") & (seedM.regime == "clean")]
        a2 = seedM[(seedM.model == "cnn_norr") & (seedM.regime == "clean")]
        if a1.empty or a2.empty:
            continue
        v1, _ = series_by_snr(a1, "macroF1", order, nz)
        v2, _ = series_by_snr(a2, "macroF1", order, nz)
        ax[1].plot(xs, v1 - v2, color=list(C.values())[k], marker=MK[k], ls=LS[k])
    ax[1].axhline(0, color="#555", lw=0.6)
    ax[1].set_ylabel("Macro-F1 gain from RR features\n(CNN+RR minus waveform-only)")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(xl)
    ax[1].set_xlabel("SNR (dB)")
    ax[1].grid(True, axis="y")
    fig.tight_layout()
    savefig(fig, out, "fig5_detection_rr")

    print("\n".join(findings))
    print("\nWrote tables + figures to", out)


if __name__ == "__main__":
    main()
