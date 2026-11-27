"""Buoc 3: danh gia cac chien luoc chon camera, chia train/test theo segment hoac subject.

Moi nhom = (subject, seq, segment, cua so). Trong moi nhom, chien luoc chon 1 camera;
ta do MPJPE that cua camera duoc chon.

Chien luoc:
    oracle          camera co MPJPE that thap nhat (can tren, dung GT)
    random          ky vong khi chon ngau nhien
    worst           camera te nhat
    cam_prior       camera co MPJPE trung binh thap nhat tren tap train (chi dua vao id camera)
    h:<feature>     chon theo mot dac trung (huong tot/xau hoc tu tap train)
    model           GradientBoosting du doan MPJPE tuong doi tu cac dac trung *_z (can scikit-learn)

Chi so: MPJPE cua camera duoc chon, regret = MPJPE chon - MPJPE oracle, top-1 accuracy,
Spearman trung binh giua diem du doan va MPJPE that trong nhom.

Vi du:
    python -m camera_quality.evaluate --features /content/cq_features.csv --split segment
    python -m camera_quality.evaluate --features /content/cq_features.csv --split subject
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FEATURES = ["consensus_pa", "consensus_pa_min", "consensus_bone", "jitter", "bone_cv", "depth",
            "kp_conf", "bbox_scale"]


def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k, v in list(r.items()):
                if k in ("subject", "seq"):
                    continue
                try:
                    r[k] = float(v) if v not in ("", "nan") else np.nan
                except ValueError:
                    pass
            r["cam"] = int(r["cam"])
            rows.append(r)
    return rows


def rank(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    return r


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return np.nan
    ra, rb = rank(a[ok]), rank(b[ok])
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def group_key(r):
    return (r["subject"], r["seq"], int(r["seg"]), int(r["window"]))


def fold_key(r, split):
    return r["subject"] if split == "subject" else (r["subject"], r["seq"], int(r["seg"]))


def score_feature(train, feat):
    """Huong cua dac trung hoc tu tap train: +1 neu gia tri lon = MPJPE lon (xau)."""
    rhos = []
    by_g = defaultdict(list)
    for r in train:
        by_g[group_key(r)].append(r)
    for g in by_g.values():
        rhos.append(spearman([r[feat] for r in g], [r["label_mpjpe"] for r in g]))
    rho = np.nanmean(rhos) if np.any(~np.isnan(rhos)) else np.nan
    return 1.0 if np.isnan(rho) or rho >= 0 else -1.0


def fit_model(train, feats):
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        return None
    by_g = defaultdict(list)
    for r in train:
        by_g[group_key(r)].append(r)
    X, y = [], []
    for g in by_g.values():
        mu = np.mean([r["label_mpjpe"] for r in g])
        for r in g:
            X.append([r[f] for f in feats])
            y.append(r["label_mpjpe"] - mu)  # hoc MPJPE tuong doi trong nhom
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.05, min_samples_leaf=10)
    m.fit(np.array(X, float), np.array(y, float))
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="output/camera_quality/features.csv")
    ap.add_argument("--split", choices=["segment", "subject"], default="segment")
    ap.add_argument("--out", default="output/camera_quality/eval_{split}.json")
    args = ap.parse_args()

    rows = load(args.features)
    feats_avail = [f for f in FEATURES if np.any(~np.isnan([r.get(f, np.nan) for r in rows]))]
    z_feats = [f + "_z" for f in feats_avail]
    folds = sorted({fold_key(r, args.split) for r in rows}, key=str)
    print(f"{len(rows)} dong, {len({group_key(r) for r in rows})} nhom, {len(folds)} fold ({args.split}); "
          f"dac trung co du lieu: {feats_avail}")
    if len(folds) < 2:
        raise SystemExit("Can it nhat 2 fold. Voi --split subject can du lieu tu >= 2 subject.")

    strategies = ["oracle", "random", "worst", "cam_prior"] + [f"h:{f}" for f in feats_avail] + ["model"]
    res = {s: {"mpjpe": [], "regret": [], "top1": [], "rho": []} for s in strategies}
    learned_dirs = defaultdict(list)

    for fold in folds:
        train = [r for r in rows if fold_key(r, args.split) != fold]
        test = [r for r in rows if fold_key(r, args.split) == fold]
        cam_mean = defaultdict(list)
        for r in train:
            cam_mean[r["cam"]].append(r["label_mpjpe"])
        cam_prior = {c: np.mean(v) for c, v in cam_mean.items()}
        dirs = {f: score_feature(train, f) for f in feats_avail}
        for f, d in dirs.items():
            learned_dirs[f].append(d)
        model = fit_model(train, z_feats)

        by_g = defaultdict(list)
        for r in test:
            by_g[group_key(r)].append(r)
        for g in by_g.values():
            y = np.array([r["label_mpjpe"] for r in g])
            best = y.min()
            scores = {
                "oracle": y,
                "worst": -y,
                "cam_prior": np.array([cam_prior.get(r["cam"], np.mean(list(cam_prior.values()))) for r in g]),
            }
            for f in feats_avail:
                v = np.array([r[f] for r in g], float)
                scores[f"h:{f}"] = np.where(np.isnan(v), np.inf, dirs[f] * v)
            if model is not None:
                scores["model"] = model.predict(np.array([[r[f] for f in z_feats] for r in g], float))
            for s in strategies:
                if s == "random":
                    res[s]["mpjpe"].append(y.mean())
                    res[s]["regret"].append(y.mean() - best)
                    res[s]["top1"].append(1.0 / len(y))
                    continue
                if s not in scores:
                    continue
                sc = scores[s]
                pick = int(np.argmin(sc))
                res[s]["mpjpe"].append(y[pick])
                res[s]["regret"].append(y[pick] - best)
                res[s]["top1"].append(float(y[pick] == best))
                res[s]["rho"].append(spearman(np.where(np.isinf(sc), np.nan, sc), y))

    print(f"\n{'chien luoc':22s} {'MPJPE':>8s} {'regret':>8s} {'top1':>6s} {'rho':>6s}")
    summary = {}
    for s in strategies:
        if not res[s]["mpjpe"]:
            continue
        m = {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in res[s].items()}
        summary[s] = m
        print(f"{s:22s} {m['mpjpe']:8.1f} {m['regret']:8.1f} {m['top1']*100:5.0f}% {m['rho']:6.2f}")
    print("\nHuong dac trung hoc duoc (+1: lon = xau):",
          {f: float(np.mean(d)) for f, d in learned_dirs.items()})

    out = Path(args.out.format(split=args.split))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "summary": summary}, indent=1), encoding="utf-8")
    print(f"Da ghi {out}")


if __name__ == "__main__":
    main()
