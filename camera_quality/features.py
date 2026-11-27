"""Buoc 2: tinh dac trung KHONG DUNG GT cho tung (camera, cua so thoi gian) + nhan MPJPE.

Dac trung (moi camera, moi cua so):
    consensus_pa     median (qua cac camera khac) cua sai so PA giua pose camera nay va camera kia (mm).
                     Camera "lac loai" so voi so dong thuong la camera te.
    consensus_pa_min sai so PA nho nhat voi mot camera khac
    consensus_bone   median chenh lech chieu dai xuong trung binh voi cac camera khac (mm)
    jitter           gia toc trung binh cua pose can theo hong (mm/frame^2)
    bone_cv          he so bien thien chieu dai xuong theo thoi gian (xuong that khong doi)
    depth            khoang cach trung binh tu camera toi hong (m)
    kp_conf, bbox_scale  (neu pkl co)
Moi dac trung con co ban *_z = chuan hoa trong cung nhom (segment, cua so) de so sanh tuong doi.

Dong bo thoi gian: offset giua hai camera duoc tim bang cach thu cac do lech trong
[-max_offset, max_offset] va chon do lech cho sai so PA nho nhat (khong dung GT).

Nhan: label_mpjpe = MPJPE trung binh cua camera trong cua so (tu GT, chi dung de train/danh gia).

Vi du:
    python -m camera_quality.features --cache-dir /content/cq_cache --window 100 \
        --out /content/cq_features.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

P1 = ["neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow",
      "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle"]
BONES = [("neck", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
         ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("left_hip", "right_hip")]
BASE_FEATURES = ["consensus_pa", "consensus_pa_min", "consensus_bone", "jitter", "bone_cv", "depth",
                 "kp_conf", "bbox_scale"]


def batch_pa_error(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Sai so trung binh moi khop (mm) sau khi can X vao Y bang phep dong dang (Procrustes). X,Y: (N,J,3)."""
    mx, my = X.mean(1, keepdims=True), Y.mean(1, keepdims=True)
    X0, Y0 = X - mx, Y - my
    nx2 = (X0 ** 2).sum((1, 2)) + 1e-12
    H = np.einsum("nji,njk->nik", X0, Y0)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("nij,njk->nik", U, Vt)))
    Vt = Vt.copy()
    S = S.copy()
    Vt[:, -1, :] *= d[:, None]
    S[:, -1] *= d
    R = np.einsum("nij,njk->nik", U, Vt)
    scale = S.sum(1) / nx2
    Xa = scale[:, None, None] * np.einsum("nji,nik->njk", X0, R) + my
    return np.linalg.norm(Xa - Y, axis=2).mean(1) * 1000.0


class Cam:
    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.subject, self.seq = str(z["subject"]), str(z["seq"])
        self.seg, self.cam = int(z["seg"]), int(z["cam"])
        names = [str(n) for n in z["names"]]
        idx = {n: i for i, n in enumerate(names)}
        self.joints = z["joints"][:, [idx[n] for n in P1]].astype(np.float64)  # (T,13,3)
        self.fids = z["frame_ids"].astype(int)
        self.mpjpe = z["mpjpe"].astype(np.float64)
        self.kp_conf = z["kp_conf"] if "kp_conf" in z.files else None
        self.bbox = z["bbox"] if "bbox" in z.files else None
        self.row = {int(f): i for i, f in enumerate(self.fids)}
        lh, rh = P1.index("left_hip"), P1.index("right_hip")
        self.root = (self.joints[:, lh] + self.joints[:, rh]) / 2.0
        self.rel = self.joints - self.root[:, None]
        self.bone_len = np.stack([np.linalg.norm(self.joints[:, P1.index(a)] - self.joints[:, P1.index(b)], axis=1)
                                  for a, b in BONES], axis=1)  # (T,B)

    @property
    def group(self):
        return (self.subject, self.seq, self.seg)


def best_offset(a: Cam, b: Cam, max_offset: int, n_sample: int = 200, rng=None) -> int:
    """Offset o sao cho frame f cua a ung voi frame f+o cua b (khong dung GT)."""
    rng = rng or np.random.default_rng(0)
    best, best_err = 0, np.inf
    fa = np.array(sorted(a.row))
    for off in range(-max_offset, max_offset + 1):
        common = [f for f in fa if (f + off) in b.row]
        if len(common) < 20:
            continue
        if len(common) > n_sample:
            common = rng.choice(common, n_sample, replace=False)
        ia = [a.row[f] for f in common]
        ib = [b.row[f + off] for f in common]
        err = float(np.median(batch_pa_error(a.joints[ia], b.joints[ib])))
        if err < best_err:
            best, best_err = off, err
    return best


def window_features(c: Cam, rows: np.ndarray, others: list[tuple[Cam, int]]) -> dict:
    f = {}
    win_fids = c.fids[rows]
    pa_list, bone_list = [], []
    for o, off in others:
        pairs = [(c.row[fid], o.row[fid + off]) for fid in win_fids if (fid + off) in o.row]
        if len(pairs) < 5:
            continue
        ia, ib = map(list, zip(*pairs))
        pa_list.append(float(np.mean(batch_pa_error(c.joints[ia], o.joints[ib]))))
        bone_list.append(float(np.abs(c.bone_len[ia].mean(0) - o.bone_len[ib].mean(0)).mean() * 1000.0))
    f["consensus_pa"] = float(np.median(pa_list)) if pa_list else np.nan
    f["consensus_pa_min"] = float(np.min(pa_list)) if pa_list else np.nan
    f["consensus_bone"] = float(np.median(bone_list)) if bone_list else np.nan

    srt = np.sort(rows)
    contiguous = np.diff(c.fids[srt]) == 1
    rel = c.rel[srt]
    if len(srt) >= 3:
        acc = rel[2:] - 2 * rel[1:-1] + rel[:-2]
        ok = contiguous[1:] & contiguous[:-1]
        f["jitter"] = float(np.linalg.norm(acc[ok], axis=2).mean() * 1000.0) if ok.any() else np.nan
    else:
        f["jitter"] = np.nan
    bl = c.bone_len[srt]
    f["bone_cv"] = float((bl.std(0) / (bl.mean(0) + 1e-9)).mean())
    f["depth"] = float(np.median(c.root[srt, 2]))
    f["kp_conf"] = float(np.mean(c.kp_conf[srt])) if c.kp_conf is not None else np.nan
    if c.bbox is not None:
        bb = c.bbox[srt]
        f["bbox_scale"] = float(np.mean(bb[:, 2] if bb.shape[1] == 3 else
                                        np.sqrt(np.abs((bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])))))
    else:
        f["bbox_scale"] = np.nan
    return f


def zscore_within(rows: list[dict], keys: list[str]) -> None:
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        ok = ~np.isnan(vals)
        if ok.sum() < 2:
            for r in rows:
                r[k + "_z"] = np.nan
            continue
        mu, sd = vals[ok].mean(), vals[ok].std() + 1e-9
        for r, v in zip(rows, vals):
            r[k + "_z"] = (v - mu) / sd if not np.isnan(v) else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="output/camera_quality/cache")
    ap.add_argument("--out", default="output/camera_quality/features.csv")
    ap.add_argument("--window", type=int, default=100, help="so frame moi cua so; 0 = ca segment")
    ap.add_argument("--max-offset", type=int, default=30, help="do lech dong bo toi da (frame) khi tim offset")
    ap.add_argument("--min-label-frames", type=int, default=10, help="bo cua so co it frame co GT hon nguong nay")
    args = ap.parse_args()

    cams = [Cam(p) for p in sorted(Path(args.cache_dir).glob("*.npz"))]
    groups = defaultdict(list)
    for c in cams:
        groups[c.group].append(c)
    print(f"{len(cams)} camera-segment, {len(groups)} segment")

    out_rows = []
    for g, members in sorted(groups.items()):
        members.sort(key=lambda c: c.cam)
        if len(members) < 2:
            continue
        offsets = {}
        for a, b in itertools.combinations(members, 2):
            o = best_offset(a, b, args.max_offset)
            offsets[(a.cam, b.cam)], offsets[(b.cam, a.cam)] = o, -o
        print(f"  {g}: offset dong bo {sorted(set(offsets.values()))}")

        all_fids = sorted(set().union(*[set(c.row) for c in members]))
        lo, hi = all_fids[0], all_fids[-1] + 1
        W = args.window if args.window > 0 else hi - lo
        for w, start in enumerate(range(lo, hi, W)):
            win_rows = []
            for c in members:
                rows = np.array([c.row[f] for f in range(start, start + W) if f in c.row], dtype=int)
                if len(rows) < max(5, W // 4):
                    continue
                labels = c.mpjpe[rows]
                if np.sum(~np.isnan(labels)) < args.min_label_frames:
                    continue
                others = [(o, offsets[(c.cam, o.cam)]) for o in members if o.cam != c.cam]
                feats = window_features(c, rows, others)
                win_rows.append({
                    "subject": c.subject, "seq": c.seq, "seg": c.seg, "window": w, "cam": c.cam,
                    "n_frames": len(rows), **feats, "label_mpjpe": float(np.nanmean(labels)),
                })
            if len(win_rows) >= 2:
                zscore_within(win_rows, BASE_FEATURES)
                out_rows.extend(win_rows)

    if not out_rows:
        raise SystemExit("Khong co dong nao. Kiem tra cache-dir va GT.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)
    n_groups = len({(r["subject"], r["seq"], r["seg"], r["window"]) for r in out_rows})
    print(f"Da ghi {len(out_rows)} dong ({n_groups} nhom segment-cua so) -> {args.out}")


if __name__ == "__main__":
    main()
