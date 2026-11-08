# Confidence or Signal Quality? Abstention for ECG Beat Classification Under Real Recorded Noise

Code and result tables for the paper of the same title.

**Question.** When an ECG heartbeat classifier should abstain, is the trigger better taken from the
classifier's own confidence (max-softmax) or from an independent signal-quality index (SQI) of the
input, and does combining them help?

**Setup.** MIT-BIH Arrhythmia Database (de Chazal DS1/DS2 inter-patient split, AAMI classes N/S/V/F),
real noise from the MIT-BIH Noise Stress Test Database (baseline wander, electrode motion, muscle
artefact) at SNRs 24 to -6 dB, a fully detected pipeline (Pan-Tompkins R-peaks via NeuroKit2), a
random forest and two 1-D CNNs, clean or noise-augmented training, five seeds. External validation
without retraining on the St. Petersburg INCART database (lead II, 32 patients).

## Files

| File | Purpose |
|---|---|
| `ecgunc.py` | Library: data loading, noise injection, R-peak detection, beat extraction, features, SQIs, models, metrics |
| `run_study.py` | Main MIT-BIH experiment (all models, regimes, noise conditions, seeds) |
| `analyze.py` | Tables, figures and key findings from a `run_study.py` output folder |
| `diagnose.py` | Exploratory single-seed diagnostics of the SQIs (patient and beat-type confounding) |
| `review_exps.py` | Fairness checks for the SQI comparison on MIT-BIH (filtered SQIs, vetoes, in-sample bound, patient bootstrap) |
| `external_exps.py` | INCART external validation, in-domain error models and conformal calibration |
| `run_all.py` | Runs `review_exps.py` and `external_exps.py` unattended and zips the results |
| `test_ecgunc.py` | Unit tests (`python -m pytest test_ecgunc.py`) |
| `paper_tables/make_ext_table.py` | Builds the paper's SQI/INCART tables from the result CSVs |
| `results/` | Result CSVs used in the paper (see below) |

## Installation

Python 3.10 or newer.

    pip install -r requirements.txt

The PhysioNet databases (`mitdb`, `nstdb`, `incartdb`) are downloaded automatically by `wfdb` on the
first run into `--data-dir` (default `./data`). A GPU is optional but makes the CNNs much faster.

## Reproducing the results

    # 1. tests and a quick synthetic check of the code paths
    python -m pytest test_ecgunc.py
    python run_study.py --synthetic --out-dir out_test

    # 2. main MIT-BIH study (band-pass on, detected R-peaks) and the no-filter ablation
    python run_study.py --out-dir out_main
    python run_study.py --prefilter 0 --out-dir out_nofilter
    python analyze.py --dir out_main
    python analyze.py --dir out_nofilter --tag pf0_detected

    # 3. SQI fairness checks on MIT-BIH (the paper reports the random forest)
    python review_exps.py --models rf --main-cache out_main/cache --out out_review

    # 4. INCART external validation (all three models)
    python external_exps.py --main-cache out_main/cache --out out_external

    # 5. paper table
    python paper_tables/make_ext_table.py out_external . out_review

All runs are resumable: finished (model, regime) blocks are recorded and skipped on restart.
Random seeds are fixed (0-4 for the models; fixed offsets for noise placement).

## Result files (`results/`)

| Folder | Content |
|---|---|
| `mitbih_main/` | Classification, calibration, selective-prediction and conformal metrics per model, regime, noise, SNR and seed (band-pass on) |
| `mitbih_nofilter/` | Same without the band-pass filter |
| `mitbih_review_rf/` | SQI fairness checks for the random forest, patient-level bootstrap, SQI confounding, calibration-set class counts |
| `incart_external/` | INCART metrics, selective prediction, within-record AUROC, conformal coverage, in-domain error models, patient-level bootstrap, patient grouping, run log |

In `incart_external/ext_within.csv`, the column `within_patient` is stratified by **record**
(INCART has up to three records per patient); the paper calls it within-record.

## Data

All data are from PhysioNet (Goldberger et al., *Circulation* 101(23):e215-e220, 2000):
- MIT-BIH Arrhythmia Database (Moody and Mark, *IEEE EMB Magazine* 20(3):45-50, 2001)
- MIT-BIH Noise Stress Test Database (Moody, Muldrow and Mark, *Computers in Cardiology* 11:381-384, 1984)
- St. Petersburg INCART 12-lead Arrhythmia Database

Please follow the PhysioNet citation requirements if you use them.

## Citation

<!-- TODO (authors): add the final reference once the paper is published -->
U. H. Sathvik et al., "Confidence or Signal Quality? Choosing an Abstention Signal for ECG Beat
Classification Under Real Recorded Noise," 2026.

## License

MIT, see `LICENSE`. This is research code, not a medical device.
