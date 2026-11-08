"""Build LaTeX tables / numbers for the INCART external validation from ext_*.csv (nothing typed by hand).
Usage: python make_ext_table.py <dir with ext_*.csv> <out dir>"""
import re, sys
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
rev = sys.argv[3]   # dir with rev_summary_noisy.csv + rev_bootstrap.csv (MIT-BIH random forest)
boot = pd.read_csv(f"{src}/ext_bootstrap.csv").set_index(["model", "regime", "score"])
sel = pd.read_csv(f"{src}/ext_selective.csv"); sel = sel[sel.member != "ensemble"]
ind = pd.read_csv(f"{src}/ext_indomain.csv")
con = pd.read_csv(f"{src}/ext_conformal.csv")
noisy = lambda d: d[d.snr.notna() & (d.snr <= 6)]
m_ = lambda s: re.sub(r"(?<![-\w])-(?=\d)", "$-$", s)

MODELS = [("rf", "RF"), ("cnn", "CNN+RR"), ("cnn_norr", "CNN")]
rows = []
for mod, ml in MODELS:
    for reg in ["aug", "clean"]:
        s = noisy(sel[(sel.model == mod) & (sel.regime == reg)])
        msp = s[s.score == "msp"].aurc.mean()
        ni_msp = s[s.score == "msp"].auroc_ni.mean(); ni_sqi = s[s.score == "sqi_raw"].auroc_ni.mean()
        def d(sc):
            b = boot.loc[(mod, reg, sc)]
            return m_(f"{b.diff_aurc_vs_msp:+.3f} [{b.ci_lo:+.3f}, {b.ci_hi:+.3f}]")
        i = noisy(ind[(ind.model == mod) & (ind.regime == reg) & (ind.k_patients == 16)])
        a = i.groupby(["member", "score"]).aurc.mean().unstack()
        dd = a.combo_log_filt - a.msp
        rows.append(f"{ml} & {reg} & {msp:.3f} & {ni_msp:.2f}/{ni_sqi:.2f} & {d('sqi_raw')} & {d('combo_log_filt')} & {d('combo_log_filt_ins')} & "
                    + m_(f"{dd.mean():+.3f}") + f" ({int((dd < 0).sum())}/5) \\\\")
rb = pd.read_csv(f"{rev}/rev_bootstrap.csv").drop_duplicates(["model", "regime", "score"]).set_index(["model", "regime", "score"])
rs = pd.read_csv(f"{rev}/rev_summary_noisy.csv", header=[0, 1], index_col=[0, 1, 2], skiprows=[2])
mrows = []
for reg in ["aug", "clean"]:
    g = lambda sc: rs.loc[("rf", reg, sc)]
    def d2(sc):
        b = rb.loc[("rf", reg, sc)]
        return m_(f"{b.diff_aurc_vs_msp:+.3f} [{b.ci_lo:+.3f}, {b.ci_hi:+.3f}]")
    mrows.append(f"RF & {reg} & {g('msp')[('aurc','mean')]:.3f} & {g('msp')[('auroc_ni','mean')]:.2f}/{g('sqi_raw')[('auroc_ni','mean')]:.2f} & {d2('sqi_raw')} & {d2('combo_log_filt')} & {d2('combo_log_filt_ins')} & -- \\\\")
tab = r"""\begin{table*}[t]
\caption{SQI versus confidence under noise (SNR $\le 6$\,dB, mean over five seeds). $\Delta$ = AURC minus that of max-softmax (MSP AURC in column 3), positive = worse, with a 95\,\% patient-level bootstrap interval (22 MIT-BIH, 32 INCART patients). SQI+conf: log-confidence and filtered-signal SQIs, error model fitted on the four MIT-BIH calibration patients; in-sample: fitted on the test patients themselves (upper bound). NI-AUROC (MSP/SQI): AUROC for noise-induced errors only. Last column: $\Delta$ of SQI+conf fitted on 16 INCART patients and tested on the other 16 (seeds in which it beats MSP). INCART models are trained on MIT-BIH only.}
\label{tab:ext}
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{@{}llcccccc@{}}
\hline
Model & Train & MSP & NI-AUROC & $\Delta$ SQI only & $\Delta$ SQI+conf & $\Delta$ in-sample & $\Delta$ in-domain \\
\hline
\multicolumn{8}{@{}l}{\emph{MIT-BIH, 22 test patients (random forest; Section~\ref{sec:stress})}} \\
""" + "\n".join(mrows) + r"""
\multicolumn{8}{@{}l}{\emph{INCART, 32 patients (models trained on MIT-BIH)}} \\
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table*}
"""
open(f"{out}/tab_ext.tex", "w").write(tab)

# conformal rows (noisy conditions)
def crow(mod, ml, reg, calib, k, label):
    c = noisy(con[(con.model == mod) & (con.regime == reg) & (con.calib == calib) & (con.k_patients == k)])
    return (f"{ml} & {reg} & {label} & " + " & ".join(f"{c['cov_' + x].mean():.2f}"[1:] for x in "NSVF")
            + f" & {c.set_size.mean():.2f} \\\\")
cr = []
for mod, ml, reg in [("cnn", "CNN+RR", "aug"), ("cnn", "CNN+RR", "clean"), ("rf", "RF", "clean")]:
    cr.append(crow(mod, ml, reg, "mitbih_mixed", 4, "MIT-BIH mix (4 pat.)"))
    cr.append(crow(mod, ml, reg, "incart_mixed", 4, "INCART mix (4 pat.)"))
    cr.append(crow(mod, ml, reg, "incart_mixed", 16, "INCART mix (16 pat.)"))
open(f"{out}/tab_conf_ext.tex", "w").write("\n".join(cr) + "\n\\hline\n")
print(tab); print("\n".join(cr))
