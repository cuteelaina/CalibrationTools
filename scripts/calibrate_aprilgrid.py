#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_aprilgrid.py —— 用 AprilGrid 6x6 板（DICT_APRILTAG_36h11）求解相机内参。

板子几何（04_目录那张，联合标定验证过）：
  6x6 个标记，tagSize=0.022m（含 1 模块黑边，标记共 8 个模块），spacing=0.3（间隙比例）
  三维点位于 Z=0 平面，单位为米。

流程:
  1. 读 images/ 下所有图，检测标记（至少 6 个才使用）
  2. 将每个标记的 4 个角点对应到三维栅格点，使用 calibrateCamera 联合多张图像标定
  3. 3 折留出验证（留出集重投影误差）
  4. 输出 calib_result.yaml + camera_openvins.yaml（radtan）+ undistort_check.png

用法:
  python3 calibrate_aprilgrid.py --images images
"""
import argparse
import glob
import os

import cv2
import numpy as np
import yaml
from calibration_config import (parse_tool_args, aprilgrid_arguments, add_detector_arguments,
                                detector_parameters, image_files, check_image_size,
                                aruco_dict, detect_markers)

def build_layout(args, rot=0, flipx=False):
    """标记 id 对应 4 个三维角点（米）。rot：布局旋转次数；flipx：是否水平翻转。
    自动扫描选均方根误差最小的几何，避免手写顺序与检测器不一致。"""
    COLS, ROWS = args.cols, args.rows
    TAG = args.tag_size / 1000.0
    GAP = TAG * args.spacing
    ids = {}
    for i in range(COLS * ROWS):
        col, row = i % COLS, i // COLS
        if rot in (1, 3):
            col, row = COLS - 1 - col, ROWS - 1 - row
        if rot in (2, 3):
            col, row = col, ROWS - 1 - row
        if flipx:
            col = COLS - 1 - col
        tl = np.array([col * (TAG + GAP), row * (TAG + GAP), 0.0])
        ids[i] = np.array([tl, tl + [TAG, 0, 0], tl + [TAG, TAG, 0], tl + [0, TAG, 0]],
                          dtype=np.float32)
    return ids


ORDER_PERMS = [
    lambda c: c,               # 角点顺序：左上、右上、右下、左下。
    lambda c: c[[1, 2, 3, 0]],
    lambda c: c[[2, 3, 0, 1]],
    lambda c: c[[3, 0, 1, 2]],
    lambda c: c[[3, 2, 1, 0]],
    lambda c: c[[1, 0, 3, 2]],
]


def collect(args, layout, order_fn=lambda c: c):
    d = aruco_dict(args.dict_name)
    p = detector_parameters(args)
    files = image_files(args.images)
    objpts, imgpts, used, size = [], [], [], None
    per_img = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        check_image_size(img, args, f)
        if size is None:
            size = (img.shape[1], img.shape[0])
        corners, ids, _ = detect_markers(img, d, p)
        if ids is None or len(ids) < args.min_markers:
            continue
        op, ip = [], []
        # 每图限用前 max_per_img 个标记（控制内存用量：72 张图 × 90 个点 × 3 折堆叠会耗尽内存）
        for c, i in zip(corners, ids.flatten()):
            if len(op) >= args.max_per_img:
                break
            if int(i) not in layout:
                continue
            op.append(layout[int(i)])
            ip.append(order_fn(c.reshape(-1, 2).astype(np.float32)))
        if len(op) < args.min_markers:
            continue
        objpts.append(np.vstack(op))
        imgpts.append(np.vstack(ip))
        per_img.append((os.path.basename(f), len(op)))
        used.append(f)
    return objpts, imgpts, used, size, per_img


def _self_test():
    """自测：校验 build_layout。"""
    import types

    args = types.SimpleNamespace(cols=6, rows=6, tag_size=22.0, spacing=0.3)
    layout = build_layout(args)
    assert len(layout) == 36
    pts = layout[0]
    assert pts.shape == (4, 3)
    tag = 22.0 / 1000.0
    gap = tag * 0.3
    dist = np.linalg.norm(layout[0][0] - layout[1][0])
    assert abs(dist - (tag + gap)) < 1e-9
    print(f"[OK] build_layout（{len(layout)} 个标记，相邻间距 {dist*1000:.2f}mm）")

    return 0


def main():
    import sys
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--test", action="store_true")
    known, _ = probe.parse_known_args()
    if known.test:
        sys.exit(_self_test())

    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--min-markers", type=int)
    ap.add_argument("--max-per-img", type=int)
    ap.add_argument("--fast", action="store_true", help="跳过 3 折验证（慢机器用）")
    ap.add_argument("--out")
    target = aprilgrid_arguments(ap)
    ap.add_argument("--cols", type=int, default=target["tagCols"])
    ap.add_argument("--rows", type=int, default=target["tagRows"])
    ap.add_argument("--tag-size", type=float, default=target["tagSize"] * 1000)
    ap.add_argument("--spacing", type=float, default=target["tagSpacing"])
    ap.add_argument("--dict", dest="dict_name")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--folds", type=int)
    add_detector_arguments(ap)
    args = parse_tool_args(ap, "calibrate_aprilgrid", bindings={"dict_name": "aprilgrid.dictionary"})
    if target["blackTagBorder"] != 1:
        ap.error("此 OpenCV AprilGrid 内参工具仅支持 blackTagBorder: 1")
    os.makedirs(args.out, exist_ok=True)

    # 自动扫描几何/角点顺序：用前 6 张图快速选均方根误差最小的组合，再用全部图精算
    probe = image_files(args.images)[:6]
    best = None
    for rot in range(4):
        for flipx in (False, True):
            lay = build_layout(args, rot, flipx)
            for oi, ofn in enumerate(ORDER_PERMS):
                # 用探测子集快速评估。
                op, ip, usedp, sz = [], [], [], None
                d = aruco_dict(args.dict_name)
                p = detector_parameters(args)
                for f in probe:
                    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    check_image_size(img, args, f)
                    if sz is None:
                        sz = (img.shape[1], img.shape[0])
                    corners, ids, _ = detect_markers(img, d, p)
                    if ids is None or len(ids) < args.min_markers:
                        continue
                    o_, i_ = [], []
                    for c, i in zip(corners, ids.flatten()):
                        if len(o_) >= max(8, args.min_markers):
                            break
                        if int(i) not in lay:
                            continue
                        o_.append(lay[int(i)])
                        i_.append(ofn(c.reshape(-1, 2).astype(np.float32)))
                    if len(o_) < args.min_markers:
                        continue
                    op.append(np.vstack(o_)); ip.append(np.vstack(i_)); usedp.append(f)
                if len(op) < 5:
                    continue
                Kf = np.zeros((3, 3)); Df = np.zeros(5)
                try:
                    rms, *_ = cv2.calibrateCamera(op, ip, sz, Kf, Df, None, None,
                                                  criteria=(cv2.TERM_CRITERIA_EPS +
                                                            cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-7))
                except cv2.error:
                    continue
                print(f"[scan] rot={rot} flipx={flipx} order={oi}: RMS={rms:.2f}")
                if best is None or rms < best[0]:
                    best = (rms, rot, flipx, oi)
    if best is None:
        raise SystemExit("扫描无有效几何组合")
    _, rot, flipx, oi = best
    print(f"[scan] 最优: rot={rot} flipx={flipx} order={oi}")
    layout = build_layout(args, rot, flipx)
    objpts, imgpts, used, size, per_img = collect(args, layout, ORDER_PERMS[oi])
    print(f"[info] 可用图片 {len(used)}/{len(glob.glob(os.path.join(args.images, '*')))} 张, 分辨率 {size}")
    if per_img:
        ns = [n for _, n in per_img]
        print(f"[info] 每图 markers: min={min(ns)} max={max(ns)} 均值={sum(ns)/len(ns):.1f}")
    if len(used) < 6:
        raise SystemExit("有效图太少 (<6)，无法稳定标定。请多采集不同角度/位置。")

    K = np.zeros((3, 3))
    D = np.zeros(5)
    flags = cv2.CALIB_FIX_K3
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-7)

    # 3 折留出验证（--fast 跳过：ARM 板上 calibrateCamera 全点云极慢，会超时）
    held = []
    if not args.fast:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(objpts))
        folds = np.array_split(idx, args.folds)
        for f in folds:
            tr = [i for i in idx if i not in set(f)]
            if len(tr) < 6:
                raise ValueError("每折训练集至少需要 6 张图，请增加图片或调整 folds")
            Kf = np.zeros((3, 3)); Df = np.zeros(5)
            cv2.calibrateCamera([objpts[i] for i in tr], [imgpts[i] for i in tr], size,
                                Kf, Df, None, None, flags=flags, criteria=crit)
            err = []
            for i in f:
                ok, rvec, tvec = cv2.solvePnP(objpts[i], imgpts[i], Kf, Df)
                if not ok:
                    continue
                proj, _ = cv2.projectPoints(objpts[i], rvec, tvec, Kf, Df)
                err.append(np.linalg.norm(proj.reshape(-1, 2) - imgpts[i], axis=1).mean())
            held.append(float(np.mean(err)))
        print(f"[heldout] {args.folds} 折重投影: {[f'{e:.3f}' for e in held]} px, 均值 {np.mean(held):.3f} px")

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, size, K, D, None, None,
                                                  flags=flags, criteria=crit)
    print(f"[result] RMS={rms:.3f} px")
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"  fu={fx:.2f} fv={fy:.2f} cu={cx:.2f} cv={cy:.2f}")
    print(f"  radtan: k1={D[0]:.6f} k2={D[1]:.6f} p1={D[2]:.6f} p2={D[3]:.6f} k3={D[4]:.6f}")

    with open(os.path.join(args.out, "calib_result.yaml"), "w") as f:
        yaml.safe_dump({
            "rms_px": float(rms),
            "heldout_px": [float(e) for e in held],
            "n_images": len(used),
            "intrinsics": {"fu": float(fx), "fv": float(fy), "cu": float(cx), "cv": float(cy)},
            "distortion_radtan": {"k1": float(D[0]), "k2": float(D[1]), "p1": float(D[2]),
                                  "p2": float(D[3]), "k3": float(D[4])},
            "resolution": list(size),
            "board": {"family": args.dict_name, "cols": args.cols, "rows": args.rows,
                      "tag_m": args.tag_size / 1000.0, "spacing": args.spacing},
        }, f, sort_keys=False)

    with open(os.path.join(args.out, "camera_openvins.yaml"), "w") as f:
        f.write(f"""# OpenVINS 相机内参，采用 Kalibr 格式（AprilGrid {args.cols}x{args.rows}，{args.dict_name}）
cam0:
  T_cam_imu: identity-placeholder  # 外参由联合标定给出，此文件只含内参
  cam_width: {size[0]}
  cam_height: {size[1]}
  intrinsics:
    rows: 1
    cols: 4
    data: [{fx:.10f}, {fy:.10f}, {cx:.10f}, {cy:.10f}]
  distortion:
    model: radtan
    rows: 1
    cols: 4
    data: [{D[0]:.10f}, {D[1]:.10f}, {D[2]:.10f}, {D[3]:.10f}]
""")
    print(f"[out] {args.out}/calib_result.yaml, {args.out}/camera_openvins.yaml")

    # 去畸变对比图
    img = cv2.imread(used[0], cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    nk, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1)
    und = cv2.undistort(img, K, D, None, nk)
    cv2.imwrite(os.path.join(args.out, "undistort_check.png"), np.hstack([img, und]))
    print(f"[out] {args.out}/undistort_check.png")


if __name__ == "__main__":
    main()
