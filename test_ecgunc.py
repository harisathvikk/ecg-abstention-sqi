"""
test_ecgunc.py  —  NEW
Correctness tests for the study code. Run:  python test_ecgunc.py
Every number in the paper depends on these building blocks, so each one is checked against a
case where the right answer is known.
"""
import numpy as np
import ecgunc as E

rng = np.random.default_rng(0)
ok = lambda name: print("PASS", name)


def test_noise_snr():
    sig = np.sin(np.linspace(0, 400, 100000))
    noise = rng.normal(0, 3, 200000)
    for snr in [24, 6, -6]:
        y = E.add_noise(sig, noise, snr, rng, "train")
        n = y - sig
        got = 10 * np.log10(np.mean((sig - sig.mean()) ** 2) / np.mean(n ** 2))
        assert abs(got - snr) < 0.5, (snr, got)
    ok("add_noise hits requested SNR")


def test_noise_halves_disjoint():
    L = 20000
    t = np.arange(L) / E.FS
    # first half = 10 Hz tone, second half = 50 Hz tone -> the added noise reveals which half was used
    noise = np.concatenate([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 50 * t)])
    sig = np.sin(2 * np.pi * 1 * np.arange(8000) / E.FS)
    def dom(h):
        n = E.add_noise(sig, noise, 0, np.random.default_rng(1), h) - sig
        F = np.abs(np.fft.rfft(n * np.hanning(len(n))))
        return np.fft.rfftfreq(len(n), 1 / E.FS)[F.argmax()]
    assert abs(dom("train") - 10) < 1.0 and abs(dom("test") - 50) < 1.0
    ok("noise halves are disjoint: train uses 1st half (10 Hz), test uses 2nd half (50 Hz)")


def test_match():
    ref = np.array([100, 500, 900, 1300])
    det = np.array([105, 300, 498, 502, 905, 2000])
    m = E.match_peaks(det, ref, tol=20)
    assert m.tolist() == [0, -1, 1, -1, 2, -1], m          # 498 beats 502 for ref 500; 300/2000 spurious
    assert (m >= 0).sum() == 3                              # ref 1300 missed
    assert E.match_peaks(np.array([], dtype=int), ref).size == 0
    ok("match_peaks one-to-one, tolerance, spurious/missed")


def test_cut_beats_alignment():
    sig = np.zeros(1000)
    sig[400] = 1.0
    b = E.cut_beats(sig, np.array([400, 3]))
    assert b[0].argmax() == E.PRE and b[0].shape == (E.PRE + E.POST,)
    assert b[1].shape == (E.PRE + E.POST,)                   # edge padding works
    ok("cut_beats puts the R peak at index PRE; edges padded")


def test_rr():
    pk = np.arange(0, 360 * 20, 360)   # exactly 1 s apart
    f = E.rr_features(pk)
    assert np.allclose(f[2:-1, 0], 1.0) and np.allclose(f[2:-1, 2], 1.0)
    ok("rr_features: constant 1 s RR gives pre=1, ratio=1")


def test_ece_and_calibration():
    n = 20000
    y = rng.integers(0, 4, n)
    # perfectly calibrated: confidence 0.7, correct 70% of the time
    prob = np.full((n, 4), 0.1)
    correct = rng.random(n) < 0.7
    pred = np.where(correct, y, (y + 1) % 4)
    prob[np.arange(n), pred] = 0.7
    assert E.ece(prob, y) < 0.02, E.ece(prob, y)
    # overconfident: confidence 0.99, still 70% correct
    prob2 = np.full((n, 4), 0.01 / 3)
    prob2[np.arange(n), pred] = 0.99
    assert abs(E.ece(prob2, y) - 0.29) < 0.02
    T = E.fit_temperature(prob2, y)
    assert T > 1.5 and E.ece(E.apply_temperature(prob2, T), y) < 0.05   # TS softens + fixes
    ok("ECE correct on calibrated/overconfident cases; temperature scaling repairs overconfidence")


def test_selective():
    n = 10000
    y = rng.integers(0, 4, n)
    err = rng.random(n) < 0.2
    pred = np.where(err, (y + 1) % 4, y)
    perfect = err.astype(float) + rng.random(n) * 1e-3          # rejects exactly the errors first
    rows, aurc, auroc = E.selective_rows(y, pred, perfect)
    assert auroc > 0.999 and aurc < 0.1
    r80 = [r for r in rows if r["coverage"] == 0.8][0]
    assert r80["risk"] < 0.01                                   # all errors fall in the rejected 20%
    rows, aurc_r, auroc_r = E.selective_rows(y, pred, rng.random(n))
    assert abs(auroc_r - 0.5) < 0.03 and abs(aurc_r - 0.2) < 0.02
    # silent_V at coverage 1 == 1 - recall_V
    r1 = [r for r in rows if r["coverage"] == 1.0][0]
    m = y == 2
    assert abs(r1["silent_V"] - (pred[m] != 2).mean()) < 1e-9
    ok("selective: oracle score -> AUROC 1, risk 0 at 80%; random -> AUROC .5, AURC = error rate; silent_V consistent")


def test_conformal_coverage():
    n = 30000
    y = rng.integers(0, 4, n)
    logits = rng.normal(0, 1, (n, 4))
    logits[np.arange(n), y] += 1.5
    p = E.softmax_rows(logits)
    half = n // 2
    q = E.conformal_q(p[:half], y[:half], 0.1)
    out = E.conformal_eval(p[half:], y[half:], q)
    for c in E.CLASSES:
        assert out[f"cov_{c}"] >= 0.87, out        # class-conditional coverage ~ 90%
    ok("class-conditional conformal reaches ~90% coverage on exchangeable data")


def test_sqi_directional():
    """SQIs should move in the expected direction when noise is added."""
    recs, noises, sp = E.make_synthetic(n_train=1, n_val=1, n_test=1, minutes=2.0, seed=3)
    r = recs[sp["test"][0]]
    pk = r["idx"]
    clean = E.sqi_features(r["sig"], pk, pk)
    hf = E.add_noise(r["sig"], noises["ma"], -3, np.random.default_rng(0), "test")
    noisy = E.sqi_features(hf, pk, pk)
    assert noisy[:, 3].mean() > clean[:, 3].mean()          # muscle noise raises hfSQI
    bwn = E.add_noise(r["sig"], noises["bw"], -3, np.random.default_rng(0), "test")
    bwf = E.sqi_features(bwn, pk, pk)
    assert bwf[:, 2].mean() < clean[:, 2].mean()            # baseline wander lowers basSQI
    ok("SQIs react in the expected direction (hfSQI up under EMG, basSQI down under BW)")


def test_split_disjoint():
    s = E.real_splits()
    sets = [set(s["train"]), set(s["val"]), set(s["test"])]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert len(s["train"]) == 18 and len(s["val"]) == 4 and len(s["test"]) == 22
    assert not (set(E.DS1) & set(E.DS2))
    ok("train / calibration / test records are patient-disjoint (18 / 4 / 22)")


def test_detection_synthetic():
    recs, noises, sp = E.make_synthetic(n_train=1, n_val=1, n_test=1, minutes=3.0, seed=5)
    r = recs[sp["test"][0]]
    pk = E.detect_peaks(E.prefilter(r["sig"]))
    m = E.match_peaks(pk, r["idx"])
    sens = len(set(m[m >= 0])) / len(r["idx"])
    ppv = (m >= 0).mean()
    assert sens > 0.97 and ppv > 0.97, (sens, ppv)
    ok(f"detector on clean synthetic ECG: sens={sens:.3f} ppv={ppv:.3f}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("\nALL TESTS PASSED")
