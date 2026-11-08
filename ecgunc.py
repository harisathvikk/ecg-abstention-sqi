"""
ecgunc.py  —  NEW
Noise-robust, uncertainty-aware ECG heartbeat classification (study library).

Research question
-----------------
Under REAL recorded ECG noise (MIT-BIH NSTDB), which signal should trigger abstention:
the classifier's own confidence, or an independent signal-quality index (SQI)? Does combining help?

Everything is end-to-end: noise is added to the continuous record, an R-peak detector runs on
the noisy signal, beats are cut at DETECTED peaks, RR features come from DETECTED peaks.

Sections
--------
 1. constants and splits
 2. data loading (real MIT-BIH/NSTDB via wfdb, plus a synthetic generator for testing)
 3. noise, filtering, R-peak detection, matching
 4. features (morphology, RR, SQI)
 5. condition builder (+ disk cache)
 6. models (RF, 1D-CNN)
 7. calibration, selective prediction, conformal
 8. evaluation of one model/seed over all conditions
"""
from __future__ import annotations

import os
import numpy as np
from scipy import signal as ss
from scipy import stats as st
from scipy.optimize import minimize_scalar

# ─── 1. constants and splits ─────────────────────────────────────────────────────────────────
FS = 360                      # MIT-BIH sampling rate (Hz)
PRE, POST = 90, 110           # samples kept before / after the R peak (0.25 s / 0.31 s)
MATCH_TOL = int(0.15 * FS)    # 150 ms tolerance when matching detected peaks to annotations
CLASSES = ["N", "S", "V", "F"]  # AAMI EC57 classes (Q excluded)
SYM2CLS = {"N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
           "A": 1, "a": 1, "J": 1, "S": 1,
           "V": 2, "E": 2,
           "F": 3}
# de Chazal et al. inter-patient split (paced records 102, 104, 107, 217 excluded)
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122", "124",
       "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210", "212",
       "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
VAL_RECORDS = ["106", "209", "215", "223"]   # patient-disjoint CALIBRATION records (taken out of DS1)
NOISES = ["bw", "em", "ma"]                  # baseline wander, electrode motion, muscle artefact
SNRS = [24, 18, 12, 6, 0, -6]
AUG_SNRS = [24, 18, 12, 6, 0, -6]
SQI_NAMES = ["kSQI", "pSQI", "basSQI", "hfSQI", "bSQI"]


def cond_key(noise, snr):
    return "clean" if noise is None else f"{noise}{snr}"


# ─── 2. data loading ─────────────────────────────────────────────────────────────────────────
def load_mitdb(data_dir):
    """Real MIT-BIH Arrhythmia DB. Downloads from PhysioNet on first call. Uses the MLII lead."""
    import wfdb
    d = os.path.join(data_dir, "mitdb")
    if not os.path.exists(os.path.join(d, "100.dat")):
        wfdb.dl_database("mitdb", d)
    recs = {}
    for name in DS1 + DS2:
        p = os.path.join(d, name)
        hdr = wfdb.rdheader(p)
        ch = hdr.sig_name.index("MLII") if "MLII" in hdr.sig_name else 0
        rec = wfdb.rdrecord(p, channels=[ch])
        ann = wfdb.rdann(p, "atr")
        idx, cls = [], []
        for s, sym in zip(ann.sample, ann.symbol):
            if sym in SYM2CLS:
                idx.append(s)
                cls.append(SYM2CLS[sym])
        sig = np.nan_to_num(rec.p_signal[:, 0]).astype(np.float64)
        recs[name] = dict(name=name, sig=sig, idx=np.array(idx, dtype=np.int64),
                          cls=np.array(cls, dtype=np.int8))
    return recs


def load_nstdb(data_dir):
    """Real MIT-BIH Noise Stress Test DB noise records: bw, em, ma (channel 0)."""
    import wfdb
    d = os.path.join(data_dir, "nstdb")
    if not os.path.exists(os.path.join(d, "bw.dat")):
        wfdb.dl_database("nstdb", d)
    out = {}
    for n in NOISES:
        rec = wfdb.rdrecord(os.path.join(d, n), channels=[0])
        out[n] = np.nan_to_num(rec.p_signal[:, 0]).astype(np.float64)
    return out


def _template(cls, t, style):
    def g(mu, s, a):
        return a * np.exp(-0.5 * ((t - mu) / s) ** 2)
    w = style["w"]
    if cls == 0:    # normal
        return g(-0.16, 0.025, 0.12) + g(-0.025, 0.010 * w, -0.10) + g(0, 0.011 * w, 1.0) \
            + g(0.025, 0.010 * w, -0.20) + g(0.22, 0.04, 0.30)
    if cls == 1:    # supraventricular ectopic: premature, no/odd P, normal-ish QRS
        return g(-0.14, 0.03, 0.03) + g(0, 0.012 * w, 0.9) + g(0.025, 0.010 * w, -0.15) + g(0.20, 0.04, 0.22)
    if cls == 2:    # ventricular: wide, tall, inverted T
        return g(-0.01, 0.030 * w, 1.25) + g(0.07, 0.045, -0.55) + g(0.26, 0.06, -0.40)
    return g(0, 0.020 * w, 1.05) + g(0.09, 0.05, -0.30) + g(0.23, 0.05, 0.20)   # fusion


def make_synthetic(n_train=6, n_val=3, n_test=6, minutes=4.0, seed=0):
    """Synthetic records + noises: ONLY for smoke-testing the code, never for results."""
    rng = np.random.default_rng(seed)
    n_samp = int(minutes * 60 * FS)
    t = np.arange(-PRE, POST) / FS
    recs, splits = {}, {"train": [], "val": [], "test": []}
    names = [(f"s{i:02d}", "train") for i in range(n_train)] + \
            [(f"v{i:02d}", "val") for i in range(n_val)] + [(f"t{i:02d}", "test") for i in range(n_test)]
    for nm, sp in names:
        style = dict(w=rng.uniform(0.85, 1.25), amp=rng.uniform(0.7, 1.4))
        base = rng.uniform(0.7, 1.0)
        sig = rng.normal(0, 0.01, n_samp)
        pos, cls, cur, prev_ect = [], [], int(0.5 * FS), False
        while cur < n_samp - POST - 5:
            c = int(rng.choice(4, p=[0.84, 0.06, 0.08, 0.02]))
            rr = base * (0.7 if c in (1, 2) else 0.82 if c == 3 else 1.0) * (1.25 if prev_ect else 1.0)
            rr *= rng.normal(1, 0.03)
            cur += int(rr * FS)
            if cur >= n_samp - POST - 5:
                break
            b = _template(c, t, style) * style["amp"] * rng.normal(1, 0.05)
            sig[cur - PRE:cur + POST] += b
            pos.append(cur)
            cls.append(c)
            prev_ect = c in (1, 2)
        recs[nm] = dict(name=nm, sig=sig, idx=np.array(pos, dtype=np.int64), cls=np.array(cls, dtype=np.int8))
        splits[sp].append(nm)
    L = 10 * 60 * FS
    tt = np.arange(L) / FS
    bw = 0.8 * np.sin(2 * np.pi * 0.25 * tt) + np.cumsum(rng.normal(0, 0.01, L))
    sos = ss.butter(2, [2, 20], "band", fs=FS, output="sos")
    em = ss.sosfilt(sos, rng.normal(0, 1, L)) * (1 + 2 * (np.sin(2 * np.pi * tt / 40) > 0.6))
    sos2 = ss.butter(3, [20, 120], "band", fs=FS, output="sos")
    ma = ss.sosfilt(sos2, rng.normal(0, 1, L)) * (0.3 + (np.sin(2 * np.pi * tt / 25) > 0.2))
    return recs, {"bw": bw, "em": em, "ma": ma}, splits


def real_splits():
    return {"train": [r for r in DS1 if r not in VAL_RECORDS], "val": list(VAL_RECORDS), "test": list(DS2)}


# ─── 3. noise, filtering, R-peak detection, matching ─────────────────────────────────────────
def add_noise(sig, noise, snr_db, rng, half):
    """Add a segment of a real noise record at a GLOBAL SNR (dB) w.r.t. the record's power.
    half='train' uses the first half of the noise record, half='test' the second half, so test-time
    noise is never seen in training."""
    L = len(noise) // 2
    seg = noise[:L] if half == "train" else noise[L:2 * L]
    start = int(rng.integers(0, L))
    nz = seg[(np.arange(len(sig)) + start) % L]
    nz = nz - nz.mean()
    ps = np.mean((sig - sig.mean()) ** 2)
    pn = np.mean(nz ** 2) + 1e-12
    a = np.sqrt(ps / (pn * 10 ** (snr_db / 10.0)))
    return sig + a * nz


def prefilter(sig):
    """0.5-40 Hz zero-phase band-pass (standard ECG front end)."""
    sos = ss.butter(3, [0.5, 40], btype="band", fs=FS, output="sos")
    return ss.sosfiltfilt(sos, sig)


def detect_peaks(sig, method="pantompkins1985"):
    import neurokit2 as nk
    try:
        _, info = nk.ecg_peaks(np.ascontiguousarray(sig), sampling_rate=FS, method=method)
        return np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    except Exception:
        return np.zeros(0, dtype=np.int64)


def match_peaks(det, ref, tol=MATCH_TOL):
    """One-to-one nearest matching. Returns, for every detected peak, the index of its matched
    reference beat, or -1 (spurious detection)."""
    m = np.full(len(det), -1, dtype=np.int64)
    if len(det) == 0 or len(ref) == 0:
        return m
    j = np.searchsorted(ref, det)
    cl, cr = np.clip(j - 1, 0, len(ref) - 1), np.clip(j, 0, len(ref) - 1)
    dl, dr = np.abs(det - ref[cl]), np.abs(det - ref[cr])
    k = np.where(dl <= dr, cl, cr)
    d = np.minimum(dl, dr)
    ok = np.where(d <= tol)[0]
    if len(ok) == 0:
        return m
    order = ok[np.lexsort((d[ok], k[ok]))]
    first = np.ones(len(order), bool)
    first[1:] = k[order][1:] != k[order][:-1]
    sel = order[first]
    m[sel] = k[sel]
    return m


# ─── 4. features ─────────────────────────────────────────────────────────────────────────────
def cut_beats(sig, peaks):
    pad = np.pad(sig, (PRE, POST), mode="edge")
    idx = peaks[:, None] + np.arange(0, PRE + POST)[None, :]   # pad shifts by PRE
    return pad[idx]


def zscore_rows(b):
    return (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-6)


def rr_features(pk):
    """[pre-RR, post-RR, pre/local, post/local, local-mean-RR] in seconds, from ALL detected peaks."""
    n = len(pk)
    if n < 3:
        return np.tile(np.array([0.8, 0.8, 1, 1, 0.8], dtype=np.float32), (n, 1))
    t = pk / FS
    pre = np.diff(t, prepend=np.nan)
    post = np.diff(t, append=np.nan)
    med = np.nanmedian(pre)
    pre = np.where(np.isnan(pre), med, pre)
    post = np.where(np.isnan(post), med, post)
    cs = np.concatenate([[0.0], np.cumsum(pre)])
    i = np.arange(n)
    lo = np.maximum(0, i - 10)
    cnt = np.maximum(i - lo, 1)
    local = np.where(i - lo > 0, (cs[i] - cs[lo]) / cnt, med)
    local = np.clip(local, 0.2, 3.0)
    pre, post = np.clip(pre, 0.1, 3.0), np.clip(post, 0.1, 3.0)
    return np.stack([pre, post, np.clip(pre / local, 0, 5), np.clip(post / local, 0, 5), local], 1).astype(np.float32)


def morph_features(b):
    """Hand-crafted morphology features for the classical (RF) model. b: raw beats (n, 200)."""
    z = zscore_rows(b)
    ptp, mx, mn = b.max(1) - b.min(1), b.max(1), b.min(1)
    amax, amin = (b.argmax(1) - PRE) / FS, (b.argmin(1) - PRE) / FS
    area = np.abs(z[:, PRE - 20:PRE + 20]).sum(1)
    width = (np.abs(z[:, PRE - 40:PRE + 40]) > 0.5).mean(1)
    kur = np.nan_to_num(st.kurtosis(z, axis=1))
    sk = np.nan_to_num(st.skew(z, axis=1))
    P = np.abs(np.fft.rfft(z, axis=1)) ** 2
    f = np.fft.rfftfreq(z.shape[1], 1 / FS)
    bands = [(0, 5), (5, 15), (15, 30), (30, 60)]
    bp = np.stack([P[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in bands], 1)
    bp = bp / (bp.sum(1, keepdims=True) + 1e-9)
    ds = z[:, ::8]
    return np.concatenate([np.stack([ptp, mx, mn, amax, amin, area, width, kur, sk], 1), bp, ds], 1).astype(np.float32)


def sqi_features(raw_sig, peaks, peaks2, half=FS):
    """Per-beat signal-quality indices computed on the RAW (unfiltered, noisy) input in a +/-1 s window.
    kSQI  kurtosis (clean ECG is spiky/high; noise lowers it)
    pSQI  power 5-15 Hz / 5-40 Hz  (QRS band share)
    basSQI 1 - power(0-1 Hz)/power(0-40 Hz)  (baseline wander lowers it)
    hfSQI power(40-120 Hz)/power(0.5-120 Hz)  (muscle noise raises it)
    bSQI  agreement of two independent R-peak detectors around the beat (+/-4 beats)"""
    n = len(peaks)
    out = np.zeros((n, 5), dtype=np.float32)
    if n == 0:
        return out
    win = 2 * half
    pad = np.pad(raw_sig, (half, half), mode="edge")
    hann = np.hanning(win)
    f = np.fft.rfftfreq(win, 1 / FS)
    m0, m1, m2, m3, m4, m5 = (f >= 0) & (f < 1), (f >= 0) & (f < 40), (f >= 5) & (f < 15), \
                              (f >= 5) & (f < 40), (f >= 40) & (f < 120), (f >= 0.5) & (f < 120)
    for s in range(0, n, 4096):
        p = peaks[s:s + 4096]
        w = pad[p[:, None] + np.arange(win)[None, :]]
        w = w - w.mean(1, keepdims=True)
        F = np.abs(np.fft.rfft(w * hann, axis=1)) ** 2 + 1e-12
        out[s:s + 4096, 0] = np.nan_to_num(st.kurtosis(w, axis=1, fisher=False))
        out[s:s + 4096, 1] = F[:, m2].sum(1) / F[:, m3].sum(1)
        out[s:s + 4096, 2] = 1 - F[:, m0].sum(1) / F[:, m1].sum(1)
        out[s:s + 4096, 3] = F[:, m4].sum(1) / F[:, m5].sum(1)
    if len(peaks2) == 0:
        out[:, 4] = 0
    else:
        j = np.clip(np.searchsorted(peaks2, peaks), 1, len(peaks2) - 1) if len(peaks2) > 1 else np.zeros(n, int)
        dmin = np.minimum(np.abs(peaks - peaks2[np.clip(j - 1, 0, len(peaks2) - 1)]),
                          np.abs(peaks - peaks2[np.clip(j, 0, len(peaks2) - 1)]))
        agree = (dmin <= int(0.05 * FS)).astype(float)
        out[:, 4] = np.convolve(agree, np.ones(9) / 9, mode="same")
    return out


# ─── 5. condition builder (+ disk cache) ─────────────────────────────────────────────────────
def build_condition(records, names, noises, noise, snr, half, seed, use_prefilter=True, peak_source="detected"):
    """Build all beat-level arrays for one (noise type, SNR) condition over a list of records.
    snr may be a number or 'rand' (a random SNR from AUG_SNRS per record: used for augmentation)."""
    rng = np.random.default_rng(seed)
    X, M, RR, SQ, Y, R = [], [], [], [], [], []
    ref_counts = np.zeros(4, dtype=np.int64)
    n_ref = n_det = n_match = 0
    for ri, nm in enumerate(names):
        r = records[nm]
        sig = r["sig"]
        if noise is not None:
            s = AUG_SNRS[int(rng.integers(len(AUG_SNRS)))] if snr == "rand" else snr
            sig = add_noise(sig, noises[noise], s, rng, half)
        sf = prefilter(sig) if use_prefilter else sig
        ref_idx, ref_cls = r["idx"], r["cls"]
        ref_counts += np.bincount(ref_cls, minlength=4)
        n_ref += len(ref_idx)
        pk2 = detect_peaks(sf, "neurokit")
        if peak_source == "oracle":
            pk, lab = ref_idx, ref_cls.astype(np.int64)
            keep = np.ones(len(pk), bool)
            n_det += len(pk)
        else:
            pk = detect_peaks(sf, "pantompkins1985")
            mt = match_peaks(pk, ref_idx)
            keep = mt >= 0
            lab = np.where(keep, ref_cls[np.clip(mt, 0, max(len(ref_cls) - 1, 0))], -1) if len(ref_cls) else np.full(len(pk), -1)
            n_det += len(pk)
        if len(pk) == 0 or keep.sum() == 0:
            continue
        n_match += int(keep.sum())
        beats = cut_beats(sf, pk)
        rr = rr_features(pk)
        sq = sqi_features(sig, pk, pk2)
        X.append(zscore_rows(beats)[keep].astype(np.float16))
        M.append(morph_features(beats)[keep])
        RR.append(rr[keep])
        SQ.append(sq[keep])
        Y.append(lab[keep].astype(np.int8))
        R.append(np.full(int(keep.sum()), ri, dtype=np.int16))
    cat = lambda a, shp: np.concatenate(a) if a else np.zeros(shp, dtype=np.float32)
    return dict(X=cat(X, (0, PRE + POST)), M=cat(M, (0, 38)), RR=cat(RR, (0, 5)), SQ=cat(SQ, (0, 5)),
                y=cat(Y, (0,)).astype(np.int8), rec=cat(R, (0,)).astype(np.int16),
                ref_counts=ref_counts, n_ref=n_ref, n_det=n_det, n_match=n_match)


def cached_condition(cache_dir, tag, records, names, noises, noise, snr, half, seed, use_prefilter, peak_source):
    os.makedirs(cache_dir, exist_ok=True)
    fn = os.path.join(cache_dir, f"{tag}_{cond_key(noise, snr)}_{half}_pf{int(use_prefilter)}_{peak_source}.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return {k: z[k] for k in z.files}
    d = build_condition(records, names, noises, noise, snr, half, seed, use_prefilter, peak_source)
    np.savez(fn, **d)
    return d


def concat_sets(sets):
    return {k: np.concatenate([s[k] for s in sets]) for k in ["X", "M", "RR", "SQ", "y"]}


# ─── 6. models ───────────────────────────────────────────────────────────────────────────────
def class_weights(y):
    f = np.bincount(y, minlength=4).astype(float) + 1
    w = 1.0 / np.sqrt(f)
    return (w / w.mean()).astype(np.float32)


def make_cnn(use_rr):
    import torch
    import torch.nn as nn

    class BeatNet(nn.Module):
        def __init__(self):
            super().__init__()
            def blk(i, o, k):
                return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
            self.conv = nn.Sequential(blk(1, 16, 7), blk(16, 32, 5), blk(32, 64, 5), blk(64, 64, 3),
                                      nn.AdaptiveAvgPool1d(1))
            self.use_rr = use_rr
            self.rr = nn.Sequential(nn.Linear(5, 16), nn.ReLU()) if use_rr else None
            self.head = nn.Sequential(nn.Linear(64 + (16 if use_rr else 0), 64), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(64, 4))

        def forward(self, x, rr):
            h = self.conv(x.unsqueeze(1)).squeeze(-1)
            if self.use_rr:
                h = torch.cat([h, self.rr(rr)], 1)
            return self.head(h)
    return BeatNet()


def train_cnn(tr, use_rr, seed, epochs, device, bs=256):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = make_cnn(use_rr).to(device)
    X = torch.tensor(tr["X"].astype(np.float32), device=device)
    RR = torch.tensor(tr["RR"], device=device)
    y = torch.tensor(tr["y"].astype(np.int64), device=device)
    n = len(y)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * ((n + bs - 1) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(class_weights(tr["y"]), device=device))
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = lossf(net(X[b], RR[b]), y[b])
            loss.backward()
            opt.step()
            sch.step()
    return net


def predict_cnn(net, d, device, bs=4096):
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(d["y"]), bs):
            x = torch.tensor(d["X"][i:i + bs].astype(np.float32), device=device)
            rr = torch.tensor(d["RR"][i:i + bs], device=device)
            out.append(torch.softmax(net(x, rr), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 4), dtype=np.float32)


def train_predict(model, tr, seed, epochs, device):
    """Returns a predict(d) -> (n,4) probability function."""
    if model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        Xf = np.concatenate([tr["M"], tr["RR"]], 1)
        clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=seed).fit(Xf, tr["y"])

        def pred(d):
            if len(d["y"]) == 0:
                return np.zeros((0, 4), dtype=np.float32)
            p = clf.predict_proba(np.concatenate([d["M"], d["RR"]], 1))
            full = np.zeros((len(p), 4), dtype=np.float32)
            full[:, clf.classes_.astype(int)] = p
            return full
        return pred
    net = train_cnn(tr, use_rr=(model == "cnn"), seed=seed, epochs=epochs, device=device)
    return lambda d: predict_cnn(net, d, device)


# ─── 7. calibration, selective prediction, conformal ─────────────────────────────────────────
def softmax_rows(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(prob, y):
    lp = np.log(np.clip(prob, 1e-7, 1))

    def nll(logT):
        p = softmax_rows(lp / np.exp(logT))
        return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-2, 3), method="bounded").x))


def apply_temperature(prob, T):
    return softmax_rows(np.log(np.clip(prob, 1e-7, 1)) / T)


def ece(prob, y, n_bins=15):
    conf, pred = prob.max(1), prob.argmax(1)
    acc = (pred == y).astype(float)
    b = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            tot += abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return tot


def classwise_ece(prob, y, n_bins=15):
    vals = []
    for k in range(4):
        if (y == k).sum() == 0:
            continue
        p = prob[:, k]
        t = (y == k).astype(float)
        b = np.minimum((p * n_bins).astype(int), n_bins - 1)
        e = 0.0
        for j in range(n_bins):
            m = b == j
            if m.any():
                e += abs(t[m].mean() - p[m].mean()) * m.mean()
        vals.append(e)
    return float(np.mean(vals))


def brier(prob, y):
    oh = np.eye(4)[y]
    return float(((prob - oh) ** 2).sum(1).mean())


def conf_scores(prob):
    """Reject scores from confidence alone (higher = less trustworthy)."""
    s = np.sort(prob, 1)
    msp = 1 - s[:, -1]
    ent = -(prob * np.log(np.clip(prob, 1e-9, 1))).sum(1)
    margin = 1 - (s[:, -1] - s[:, -2])
    return msp, ent, margin


def fit_error_model(F, err):
    """Logistic regression predicting 'the classifier is wrong on this beat' from features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if err.min() == err.max():
        return None
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)).fit(F, err)


def err_feats(kind, prob, sq):
    msp, ent, mar = conf_scores(prob)
    conf = np.stack([msp, ent, mar], 1)
    if kind == "sqi":
        return sq
    if kind == "conf_lr":
        return conf
    return np.concatenate([sq, conf], 1)   # "combo"


def selective_rows(y, pred, score, covs=(1.0, 0.9, 0.8, 0.7, 0.5)):
    n = len(y)
    if n == 0:
        return [], np.nan, np.nan
    err = (pred != y).astype(float)
    order = np.argsort(score, kind="stable")
    aurc = float((np.cumsum(err[order]) / np.arange(1, n + 1)).mean())
    from sklearn.metrics import roc_auc_score
    auroc = float(roc_auc_score(err, score)) if 0 < err.sum() < n else np.nan
    rows = []
    for c in covs:
        kk = max(1, int(round(c * n)))
        keep = np.zeros(n, bool)
        keep[order[:kk]] = True
        silent = {}
        for cls, nm in [(1, "S"), (2, "V")]:
            m = y == cls
            silent[nm] = float(((keep & m) & (err > 0)).sum() / m.sum()) if m.any() else np.nan
        rows.append(dict(coverage=c, risk=float(err[keep].mean()), silent_S=silent["S"], silent_V=silent["V"]))
    return rows, aurc, auroc


def conformal_q(prob, y, alpha=0.1):
    q = np.ones(4)
    for k in range(4):
        s = 1 - prob[y == k, k]
        n = len(s)
        if n > 0:
            q[k] = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    return q


def conformal_eval(prob, y, q):
    sets = (1 - prob) <= q[None, :]
    out = dict(set_size=float(sets.sum(1).mean()), empty=float((sets.sum(1) == 0).mean()))
    for k, nm in enumerate(CLASSES):
        m = y == k
        out[f"cov_{nm}"] = float(sets[m, k].mean()) if m.any() else np.nan
    return out


# ─── 8. evaluation of one prediction set (one seed, or the seed-ensemble) ────────────────────
def evaluate_member(meta, cal_probs, test_probs, test_data, cal_data, test_extra_scores=None, alpha=0.1):
    """meta: dict of identifiers. cal_probs/test_probs: {cond_key: (n,4)} ; *_data: {cond_key: dict}.
    Returns (metrics_rows, selective_rows, conformal_rows)."""
    from sklearn.metrics import f1_score
    ck = list(cal_probs.keys())
    P_cal = np.concatenate([cal_probs[k] for k in ck])
    y_cal = np.concatenate([cal_data[k]["y"] for k in ck])
    SQ_cal = np.concatenate([cal_data[k]["SQ"] for k in ck])
    clean_cal = cal_probs["clean"], cal_data["clean"]["y"]
    T_clean = fit_temperature(*clean_cal)
    T_mixed = fit_temperature(P_cal, y_cal)
    err_cal = (P_cal.argmax(1) != y_cal).astype(int)
    em = {kind: fit_error_model(err_feats(kind, P_cal, SQ_cal), err_cal) for kind in ["sqi", "conf_lr", "combo"]}
    q_clean = conformal_q(*clean_cal, alpha)
    q_mixed = conformal_q(P_cal, y_cal, alpha)
    M, S, C = [], [], []
    for k, prob in test_probs.items():
        d = test_data[k]
        y, sq = d["y"], d["SQ"]
        noise = None if k == "clean" else k.rstrip("-0123456789")
        snr = np.nan if k == "clean" else int(k[len(noise):])
        base = dict(meta, noise=noise or "clean", snr=snr)
        if len(y) == 0:
            continue
        pred = prob.argmax(1)
        f1 = f1_score(y, pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
        sup = np.bincount(y, minlength=4)
        rec = np.array([((pred == c) & (y == c)).sum() / max(sup[c], 1) for c in range(4)])
        e2e = np.array([((pred == c) & (y == c)).sum() / max(d["ref_counts"][c], 1) for c in range(4)])
        row = dict(base, n=len(y), acc=float((pred == y).mean()), macroF1=float(f1[sup > 0].mean()),
                   det_sens=d["n_match"] / max(d["n_ref"], 1), det_ppv=d["n_match"] / max(d["n_det"], 1),
                   ece_raw=ece(prob, y), ece_ts_clean=ece(apply_temperature(prob, T_clean), y),
                   ece_ts_mixed=ece(apply_temperature(prob, T_mixed), y), cwece_raw=classwise_ece(prob, y),
                   brier=brier(prob, y), T_clean=T_clean, T_mixed=T_mixed)
        for c, nm in enumerate(CLASSES):
            row[f"F1_{nm}"], row[f"recall_{nm}"], row[f"e2e_recall_{nm}"] = f1[c], rec[c], e2e[c]
        M.append(row)
        msp, ent, mar = conf_scores(prob)
        scores = {"msp": msp, "entropy": ent, "margin": mar}
        for kind, mdl in em.items():
            if mdl is not None:
                scores[kind] = mdl.predict_proba(err_feats(kind, prob, sq))[:, 1]
        for j, nm in enumerate(SQI_NAMES):      # single raw SQI as reject score (sign chosen so higher = worse)
            sgn = -1 if nm in ("kSQI", "pSQI", "basSQI", "bSQI") else 1
            scores[f"raw_{nm}"] = sgn * sq[:, j]
        if test_extra_scores is not None and k in test_extra_scores:
            scores.update(test_extra_scores[k])
        scores["random"] = np.random.default_rng(0).random(len(y))
        for sname, sc in scores.items():
            rows, aurc, auroc = selective_rows(y, pred, sc)
            for r in rows:
                S.append(dict(base, score=sname, aurc=aurc, auroc_err=auroc, **r))
        for cname, q in [("clean", q_clean), ("mixed", q_mixed)]:
            C.append(dict(base, calib=cname, alpha=alpha, **conformal_eval(prob, y, q)))
    return M, S, C
