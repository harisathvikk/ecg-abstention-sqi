"""
run_all.py  —  NEW
One-shot runner for Kaggle (or Colab). Runs, in order:
  1. review_exps.py   for the two CNNs (fairness checks of the SQI comparison; the RF part is already done)
  2. external_exps.py for all three models (cross-database validation on INCART)
then zips every result CSV and log into results.zip and deletes the large caches and raw data.

Kaggle: Settings -> Accelerator GPU T4 x2 (or P100), Internet ON. Then "Save Version" -> "Save & Run All (Commit)".
It runs in the background; when it finishes, download results.zip from the notebook's Output tab.

    !python run_all.py --work /kaggle/working
Add --synthetic for a 5-minute code test.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, logf):
    print(time.strftime("%H:%M:%S"), "RUN:", " ".join(cmd), flush=True)
    with open(logf, "w") as fh:
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            sys.stdout.write(line)
            fh.write(line)
            fh.flush()
        rc = p.wait()
    print(time.strftime("%H:%M:%S"), "EXIT", rc, flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--skip-review", action="store_true", help="skip the MIT-BIH CNN fairness checks")
    ap.add_argument("--skip-external", action="store_true", help="skip the INCART validation")
    ap.add_argument("--keep-cache", action="store_true", help="do not delete caches and data at the end")
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    W = a.work
    os.makedirs(W, exist_ok=True)
    data, cache_main = os.path.join(W, "data"), os.path.join(W, "cache_main")
    out_rev, out_ext = os.path.join(W, "ecg_review_cnn"), os.path.join(W, "ecg_external")
    extra = ["--synthetic", "--seeds", "2", "--n-boot", "20"] if a.synthetic else []
    py = sys.executable
    rcs = {}
    if not a.skip_review:
        rcs["review"] = run([py, "review_exps.py", "--data-dir", data, "--main-cache", cache_main, "--out", out_rev,
                             "--models", "cnn,cnn_norr"] + extra, os.path.join(W, "log_review_cnn.txt"))
    if not a.skip_external:
        rcs["external"] = run([py, "external_exps.py", "--data-dir", data, "--main-cache", cache_main, "--out", out_ext]
                              + extra, os.path.join(W, "log_external.txt"))
    zf = os.path.join(W, "results.zip")
    with zipfile.ZipFile(zf, "w", zipfile.ZIP_DEFLATED) as z:
        for root in [out_rev, out_ext]:
            if os.path.isdir(root):
                for f in os.listdir(root):
                    if f.endswith((".csv", ".json")):
                        z.write(os.path.join(root, f), os.path.join(os.path.basename(root), f))
        for f in ["log_review_cnn.txt", "log_external.txt"]:
            if os.path.exists(os.path.join(W, f)):
                z.write(os.path.join(W, f), f)
    print("exit codes:", rcs, "| results:", zf, flush=True)
    if not a.keep_cache and all(v == 0 for v in rcs.values()):
        for p in [data, cache_main, os.path.join(out_rev, "cache_ext"), os.path.join(out_ext, "cache_ext")]:
            shutil.rmtree(p, ignore_errors=True)
        print("caches and raw data deleted (results.zip kept)")


if __name__ == "__main__":
    main()
