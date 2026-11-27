"""Buoc 1: trich pose 3D cua tung camera tu pkl WHAM va tinh sai so so voi GT.

Moi camera-segment duoc luu thanh mot file .npz:
    joints      (T, 21, 3)  khop 3D (met, he toa do camera) theo keypoints3D_map.yml
    frame_ids   (T,)        chi so frame goc (tracking frame_id neu co)
    mpjpe       (T,)        MPJPE tung frame (mm, can theo hong, 13 khop priority1), NaN neu khong co GT
    kp_conf     (T,)        (tuy chon) do tin cay 2D trung binh neu pkl co keypoints
    bbox        (T, k)      (tuy chon) bbox neu pkl co

GT chi dung de tao NHAN (cot mpjpe). Buoc tinh dac trung (features.py) khong doc GT.

Vi du (Colab):
    python -m camera_quality.extract --inspect \
        --seq-dir "/content/drive/.../S8/Seq1/imageSequence"
    python -m camera_quality.extract --out-dir /content/cq_cache \
        --seq-dir "/content/drive/.../S8/Seq1/imageSequence" \
        --seq-dir "/content/drive/.../S1/Seq1/imageSequence"
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compat import patch_numpy_and_inspect, load_joblib_compat  # noqa: E402

patch_numpy_and_inspect()

from keypoints_map import load_keypoints3d_map  # noqa: E402
from evaluation_pipeline.executor import _load_segment_gt, _parse_new_gt  # noqa: E402
from pose_pipeline.executor import _extract_person_payload  # noqa: E402

PKL_RE = re.compile(r"video_(\d+)_seg_(\d+)\.pkl$")


def discover(seq_dir: Path):
    """Tim tat ca pkl WHAM duoi <seq_dir>/Segments/*/video_X_seg_Y.pkl."""
    for pkl in sorted(seq_dir.glob("Segments/*/video_*_seg_*.pkl")):
        m = PKL_RE.search(pkl.name)
        if m:
            yield int(m.group(1)), int(m.group(2)), pkl


def subject_seq_from_dir(seq_dir: Path) -> tuple[str, str]:
    # .../S8/Seq1/imageSequence -> ("S8", "Seq1")
    parts = seq_dir.resolve().parts
    subj = next((p for p in reversed(parts) if re.fullmatch(r"S\d+", p)), "S?")
    seq = next((p for p in reversed(parts) if re.fullmatch(r"Seq\d+", p)), "Seq?")
    return subj, seq


def describe(obj, prefix="", depth=0, max_depth=3):
    """In cau truc pkl de biet co nhung truong nao (dung voi --inspect)."""
    if depth > max_depth:
        return
    if isinstance(obj, dict) or "defaultdict" in str(type(obj)):
        for k, v in obj.items():
            shape = getattr(v, "shape", None)
            kind = type(v).__name__
            print(f"{prefix}{k!r}: {kind}{'' if shape is None else ' ' + str(tuple(shape))}")
            if isinstance(v, dict):
                describe(v, prefix + "    ", depth + 1, max_depth)


def frame_ids_of(person: dict, total: int) -> np.ndarray:
    tracking = person.get("tracking_results_for_reproj")
    if isinstance(tracking, dict) and "frame_id" in tracking:
        ids = np.asarray(tracking["frame_id"]).astype(int).reshape(-1)
        if ids.shape[0] == total:
            return ids
    for key in ("frame_ids", "frame_id"):
        if key in person:
            ids = np.asarray(person[key]).astype(int).reshape(-1)
            if ids.shape[0] == total:
                return ids
    return np.arange(total)


def optional_2d(person: dict, total: int) -> dict:
    """Lay confidence 2D va bbox neu pkl co (ten truong co the khac nhau giua cac ban WHAM)."""
    out = {}
    sources = [person]
    tracking = person.get("tracking_results_for_reproj")
    if isinstance(tracking, dict):
        sources.append(tracking)
    for src in sources:
        for key in ("keypoints", "kp2d", "keypoints_2d", "joints2d"):
            if key in src and "kp_conf" not in out:
                kp = np.asarray(src[key])
                if kp.ndim == 3 and kp.shape[0] == total and kp.shape[-1] >= 3:
                    out["kp_conf"] = kp[..., 2].astype(np.float32).mean(axis=1)
        for key in ("bbox", "bboxes"):
            if key in src and "bbox" not in out:
                bb = np.asarray(src[key])
                if bb.ndim == 2 and bb.shape[0] == total:
                    out["bbox"] = bb.astype(np.float32)
    return out


def smpl_joints(model, person: dict, j_reg: np.ndarray, reg_idx: list[int], device, chunk: int = 256) -> np.ndarray:
    import torch

    pose = np.asarray(person["pose"], dtype=np.float32)
    trans = np.asarray(person["trans"], dtype=np.float32)
    betas = np.asarray(person["betas"], dtype=np.float32)
    total = pose.shape[0]
    if betas.ndim == 1:
        betas = np.repeat(betas[None], total, axis=0)
    betas = betas[:, :10]
    J = torch.tensor(j_reg[reg_idx], dtype=torch.float32, device=device)  # (21, 6890)
    out = []
    for s in range(0, total, chunk):
        e = min(total, s + chunk)
        p = torch.tensor(pose[s:e], device=device)
        with torch.no_grad():
            o = model(
                betas=torch.tensor(betas[s:e], device=device),
                global_orient=p[:, :3],
                body_pose=p[:, 3:72],
                transl=torch.tensor(trans[s:e], device=device),
            )
            out.append(torch.einsum("jv,bvc->bjc", J, o.vertices).cpu().numpy())
    return np.concatenate(out, axis=0)


def per_frame_mpjpe(joints: np.ndarray, frame_ids: np.ndarray, gt_map: dict, names: list[str], p1: list[str], map_data: dict) -> np.ndarray:
    """MPJPE tung frame, giong evaluation_pipeline._compute_mpjpe (can theo tam hong, mm)."""
    idx = {n: i for i, n in enumerate(names)}
    lh, rh = idx["left_hip"], idx["right_hip"]
    p1_idx = [idx[n] for n in p1]
    canonical = set(names)
    err = np.full(len(frame_ids), np.nan, dtype=np.float32)
    for t, fid in enumerate(frame_ids):
        item = gt_map.get(int(fid))
        if item is None:
            continue
        gt = _parse_new_gt(item, canonical, map_data)
        if not all(n in gt for n in p1 + ["left_hip", "right_hip"]):
            continue
        pred_root = (joints[t, lh] + joints[t, rh]) / 2.0
        gt_root = (gt["left_hip"] + gt["right_hip"]) / 2.0
        P = joints[t, p1_idx] - pred_root
        G = np.stack([gt[n] for n in p1]) - gt_root
        err[t] = float(np.linalg.norm(P - G, axis=1).mean() * 1000.0)
    return err


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-dir", action="append", required=True, help="thu muc imageSequence (lap lai cho nhieu sequence)")
    ap.add_argument("--out-dir", default="output/camera_quality/cache")
    ap.add_argument("--smpl-model", default=str(REPO / "models" / "SMPL_NEUTRAL.pkl"))
    ap.add_argument("--j-regressor", default=str(REPO / "models" / "J_regressor_body25_extended.npy"))
    ap.add_argument("--keypoints-map", default=str(REPO / "configs" / "keypoints3D_map.yml"))
    ap.add_argument("--exclude-cams", default="", help="vi du: 1,3")
    ap.add_argument("--inspect", action="store_true", help="chi in cau truc pkl dau tien roi thoat")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-new", type=int, default=0, help="xu ly toi da N file moi moi lan chay (0 = khong gioi han)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="SMPL chay CPU cung du nhanh; dung cpu de khong ton quota GPU cua Colab")
    args = ap.parse_args()

    if args.inspect:
        for sd in args.seq_dir:
            for cam, seg, pkl in discover(Path(sd)):
                print(f"== {pkl}")
                raw = load_joblib_compat(pkl)
                print("top-level:", type(raw).__name__, list(raw.keys())[:10] if hasattr(raw, "keys") else "")
                describe(_extract_person_payload(raw))
                return
        print("Khong tim thay pkl nao. Kiem tra lai --seq-dir (phai chua thu muc Segments/).")
        return

    import torch
    from pose_pipeline.smpl_runner import create_smpl_model

    excluded = {int(x) for x in args.exclude_cams.split(",") if x.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"SMPL device: {device}")
    model = create_smpl_model(args.smpl_model).to(device)
    j_reg = np.load(args.j_regressor)
    map_data = load_keypoints3d_map(args.keypoints_map)
    names = [k["name"] for k in map_data["keypoints"]]
    reg_idx = [k["regressor_index"] for k in map_data["keypoints"]]
    p1 = list(map_data["priority1"])

    manifest = []
    n_new, t_start = 0, time.perf_counter()
    for sd in args.seq_dir:
        seq_dir = Path(sd)
        subj, seq = subject_seq_from_dir(seq_dir)
        items = [x for x in discover(seq_dir) if x[0] not in excluded]
        done = sum((out_dir / f"{subj}_{seq}_seg{s}_cam{c}.npz").exists() for c, s, _ in items)
        print(f"[{subj}/{seq}] {len(items)} camera-segment, da xong truoc do: {done}")
        for cam, seg, pkl in items:
            out = out_dir / f"{subj}_{seq}_seg{seg}_cam{cam}.npz"
            if out.exists() and not args.overwrite:
                manifest.append(out.name)
                continue
            if args.max_new and n_new >= args.max_new:
                print(f"Dat gioi han --max-new {args.max_new}. Chay lai lenh de xu ly tiep.")
                break
            t_file = time.perf_counter()
            person = None
            try:
                person = _extract_person_payload(load_joblib_compat(pkl))
                total = int(np.asarray(person["pose"]).shape[0])
                joints = smpl_joints(model, person, j_reg, reg_idx, device)
                fids = frame_ids_of(person, total)
                if not np.array_equal(fids, np.arange(total)):
                    # pipeline goc ghep GT theo so thu tu 0..T-1; o day ghep theo frame_ids cua WHAM
                    gaps = int(np.sum(np.diff(fids) != 1))
                    print(f"  [chu y] seg{seg} cam{cam}: frame_ids khong phai 0..{total - 1} "
                          f"(bat dau {fids[0]}, {gaps} cho dut quang) -> ghep GT theo frame_ids")
                gt_path = seq_dir / "GT" / f"video_{cam}_seg_{seg}.json"
                if gt_path.exists():
                    mpjpe = per_frame_mpjpe(joints, fids, _load_segment_gt(gt_path), names, p1, map_data)
                else:
                    mpjpe = np.full(total, np.nan, dtype=np.float32)
                extra = optional_2d(person, total)
                # ghi ra file tam roi doi ten: neu Colab ngat giua chung se khong de lai file .npz hong
                tmp = out.with_name(out.stem + ".tmp.npz")
                np.savez_compressed(
                    tmp, joints=joints.astype(np.float32), frame_ids=fids, mpjpe=mpjpe,
                    names=np.array(names), subject=subj, seq=seq, seg=seg, cam=cam, **extra,
                )
                os.replace(tmp, out)
                manifest.append(out.name)
                n_new += 1
                print(f"  seg{seg} cam{cam}: {total} frame, MPJPE TB {np.nanmean(mpjpe):.1f} mm, "
                      f"extra={list(extra)} | {time.perf_counter() - t_file:.0f}s "
                      f"(tong {(time.perf_counter() - t_start) / 60:.1f} phut)", flush=True)
            except Exception as exc:  # mot file hong khong lam dung ca lan chay
                print(f"  [LOI] {pkl}: {exc}", flush=True)
            finally:
                del person  # pkl co mang 'verts' rat lon (T x 6890 x 3): giai phong ngay
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        else:
            continue
        break  # thoat ca vong seq khi dat --max-new
    (out_dir / "manifest.json").write_text(json.dumps(sorted(set(manifest)), indent=1), encoding="utf-8")
    print(f"Xong luot nay: {n_new} file moi, {len(set(manifest))} file trong {out_dir}")


if __name__ == "__main__":
    main()
