"""Buoc 4 (huong A2): chon CAP camera va quyet dinh co fusion hay khong, khong dung GT.

Ghep hai nguon:
  - bao cao brute-force (.xlsx sheet 'Tat_ca_cap' hoac .csv): MPJPE sau fusion cua master
    (cot 'MPJPE') va MPJPE don camera cua master ('Old MPJPE') cho moi cap (master, slave).
  - features.csv (features.py): dac trung khong GT cua tung camera; gop cac cua so thanh
    1 diem/camera/segment (trung binh co trong so theo so frame).

Moi segment la 1 bai toan. Chien luoc tra ve 1 con so MPJPE that (tu bao cao) cho segment do:
  Can tren / tham chieu (dung GT):
    oracle_any       min(moi cap sau fusion, moi camera don)
    oracle_pair      cap tot nhat sau fusion
    oracle_mono      camera don tot nhat (khong fusion)
  Khong dung GT:
    random_pair      ky vong khi chon cap ngau nhien roi fusion
    random_mono      ky vong khi chon 1 camera ngau nhien, khong fusion
    prior_mono       camera co MPJPE don TB thap nhat o CAC SEGMENT KHAC (leave-one-segment-out)
    <f>_mono         camera co diem <f> tot nhat, khong fusion            (= huong A1)
    <f>_top2         master = camera tot nhat, slave = camera tot nhi, fusion
    <f>_master_rand  master = camera tot nhat, slave ngau nhien, fusion

Vi du:
    python -m camera_quality.pair_eval --report brute_force.xlsx --features features_off30.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOWER_IS_BETTER = {"consensus_pa": True, "consensus_bone": True, "consensus_pa_min": True}


def load_report(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_excel(p, sheet_name="Tat_ca_cap") if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
    out = []
    for _, r in df.iterrows():
        m = re.search(r"video_(\d+)_seg_(\d+)", str(r["Cam Master"]))
        s = re.search(r"video_(\d+)_seg_(\d+)", str(r["Cam Slave"]))
        seg = re.search(r"(S\d+)_(Seq\d+)_seg_(\d+)", str(r["Segment"]))
        if not (m and s and seg):
            continue
        try:
            fused, mono = float(r["MPJPE"]), float(r["Old MPJPE"])
        except (TypeError, ValueError):
            continue
        out.append({"subject": seg.group(1), "seq": seg.group(2), "seg": int(seg.group(3)),
                    "master": int(m.group(1)), "slave": int(s.group(1)), "fused": fused, "mono": mono})
    rep = pd.DataFrame(out).drop_duplicates(["subject", "seq", "seg", "master", "slave"])
    return rep


def camera_scores(features: str, feats: list[str]) -> pd.DataFrame:
    f = pd.read_csv(features)
    rows = []
    for key, g in f.groupby(["subject", "seq", "seg", "cam"]):
        w = g["n_frames"].to_numpy(float)
        row = dict(zip(["subject", "seq", "seg", "cam"], key))
        for c in feats:
            v = g[c].to_numpy(float)
            ok = ~np.isnan(v)
            row[c] = float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, help="bao cao brute-force .xlsx (sheet Tat_ca_cap) hoac .csv")
    ap.add_argument("--features", required=True, help="features.csv tu camera_quality.features")
    ap.add_argument("--feats", default="consensus_pa,consensus_bone", help="dac trung dung de xep hang camera")
    ap.add_argument("--out", default="output/camera_quality/pair_eval.json")
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    feats = [s.strip() for s in args.feats.split(",") if s.strip()]
    rep = load_report(args.report)
    cs = camera_scores(args.features, feats)
    segs = sorted({(r.subject, r.seq, r.seg) for r in rep.itertuples()})
    print(f"Bao cao: {len(rep)} cap, {len(segs)} segment. Dac trung: {feats}")

    # MPJPE don camera cua tung camera trong tung segment (Old MPJPE thay doi nhe theo cap do cua so dong bo)
    mono_of = rep.groupby(["subject", "seq", "seg", "master"])["mono"].mean()

    results, per_seg = {}, []
    for key in segs:
        R = rep[(rep.subject == key[0]) & (rep.seq == key[1]) & (rep.seg == key[2])]
        cams = sorted(set(R.master) | set(R.slave))
        mono = {c: float(mono_of[key + (c,)]) for c in cams if key + (c,) in mono_of}
        fused = {(r.master, r.slave): r.fused for r in R.itertuples()}
        rec = {
            "oracle_any": min(min(fused.values()), min(mono.values())),
            "oracle_pair": min(fused.values()),
            "oracle_mono": min(mono.values()),
            "random_pair": float(np.mean(list(fused.values()))),
            "random_mono": float(np.mean(list(mono.values()))),
        }
        # prior: camera tot nhat o cac segment khac (khong nhin segment dang danh gia)
        others = rep[~((rep.subject == key[0]) & (rep.seq == key[1]) & (rep.seg == key[2]))]
        prior = others.groupby("master")["mono"].mean()
        pc = min([c for c in cams if c in prior.index], key=lambda c: prior[c])
        rec["prior_mono"] = mono[pc]

        S = cs[(cs.subject == key[0]) & (cs.seq == key[1]) & (cs.seg == key[2])].set_index("cam")
        chosen = {}
        for f in feats:
            if f not in S.columns or S[f].isna().all():
                continue
            sign = 1.0 if LOWER_IS_BETTER.get(f, True) else -1.0
            ranked = sorted([c for c in cams if c in S.index and not np.isnan(S.loc[c, f])],
                            key=lambda c: sign * S.loc[c, f])
            if len(ranked) < 2:
                continue
            m, s2 = ranked[0], ranked[1]
            chosen[f] = (m, s2)
            rec[f"{f}_mono"] = mono[m]
            rec[f"{f}_top2"] = fused.get((m, s2), np.nan)
            rec[f"{f}_master_rand"] = float(np.mean([v for (a, b), v in fused.items() if a == m]))
        for k, v in rec.items():
            results.setdefault(k, []).append(v)
        per_seg.append({"segment": f"{key[0]}_{key[1]}_seg{key[2]}", **{k: round(v, 1) for k, v in rec.items()},
                        **{f"pick_{f}": f"cam{m}+cam{s}" for f, (m, s) in chosen.items()}})

    # bootstrap theo segment cho chenh lech so voi random_pair (cach brute-force mac dinh) va random_mono
    rng = np.random.default_rng(0)
    n = len(segs)
    idx = rng.integers(0, n, size=(args.n_boot, n))

    def ci(a, b):
        d = np.array(results[a]) - np.array(results[b])
        if np.isnan(d).any():
            return None
        bm = d[idx].mean(1)
        return float(d.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))

    def fmt(c):
        if c is None:
            return " " * 22
        star = "*" if c[2] < 0 or c[1] > 0 else " "
        return f"{c[0]:+6.1f} [{c[1]:+5.1f},{c[2]:+5.1f}]{star}"

    order = ["oracle_any", "oracle_pair", "oracle_mono", "random_pair", "random_mono", "prior_mono"]
    order += [k for k in results if k not in order]
    print(f"\n{'chien luoc':26s} {'MPJPE':>7s}   {'so voi random_pair':>22s}   {'so voi random_mono':>22s}")
    summary = {}
    for k in order:
        v = np.array(results[k], float)
        summary[k] = {"mpjpe": float(np.nanmean(v)), "vs_random_pair": ci(k, "random_pair"),
                      "vs_random_mono": ci(k, "random_mono")}
        print(f"{k:26s} {np.nanmean(v):7.1f}   {fmt(summary[k]['vs_random_pair'])}   {fmt(summary[k]['vs_random_mono'])}")
    print(f"(am = tot hon; * = CI95 khong chua 0; bootstrap theo {n} segment)")

    # hieu ung cua fusion khi da chon master bang dac trung: top2 (fusion) - mono (khong fusion)
    print("\nFusion co giup khi da chon master khong? (MPJPE sau fusion - khong fusion, am = fusion giup)")
    for f in feats:
        if f"{f}_top2" in results:
            d = np.array(results[f"{f}_top2"]) - np.array(results[f"{f}_mono"])
            print(f"  {f}: TB {np.nanmean(d):+.1f} mm, fusion giup o {int(np.sum(d < 0))}/{len(d)} segment")

    print("\nChi tiet tung segment:")
    print(pd.DataFrame(per_seg).to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_segment": per_seg}, indent=1, default=float), encoding="utf-8")
    print(f"\nDa ghi {out}")


if __name__ == "__main__":
    main()
