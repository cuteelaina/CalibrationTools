#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_charuco.py —— 用采集到的标定图求解相机内参

流程:
  1. 读 images/ 下所有图，检测 ChArUco 角点
  2. calibrateCameraRO（可选用）或 calibrateCamera 求解 fu,fv,cu,cv + 畸变
  3. 输出:
       - calib_result.yaml      人可读结果 + 重投影误差
       - camera_openvins.yaml   OpenVINS 使用的内参和畸变系数（radtan）
       - undistort_check.png   去畸变前后对比图

用法:
  python3 calibrate_charuco.py --images images --square-mm 30 --marker-mm 22.5
"""

import argparse
import os

import cv2
import numpy as np
import yaml
from calibration_config import (add_detector_arguments, parse_tool_args,
                                detector_parameters, image_files, check_image_size,
                                aruco_dict, charuco_board, detect_markers, draw_board,
                                interpolate_charuco, chessboard_corners)


def build_board(args):
    dictionary = aruco_dict(args.dict_name)
    board = charuco_board(
        args.squares_x, args.squares_y,
        args.square_mm / 1000.0, args.marker_mm / 1000.0, dictionary,
    )
    params = detector_parameters(args)
    return dictionary, board, params


def detect_charuco(img, aruco_dict, board, params):
    """采集与离线求解共用相同的标记检测和棋盘角点插值。"""
    corners, ids, _ = detect_markers(img, aruco_dict, params)
    ch_corners, ch_ids = None, None
    if ids is not None and len(ids) >= 4:
        ch_corners, ch_ids = interpolate_charuco(board, img, corners, ids)
    return corners, ids, ch_corners, ch_ids


def collect(args):
    aruco_dict, board, params = build_board(args)
    files = image_files(args.images)
    all_corners, all_ids, used = [], [], []
    size = None
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        check_image_size(img, args, f)
        if size is None:
            size = (img.shape[1], img.shape[0])
        _, _, ch_corners, ch_ids = detect_charuco(img, aruco_dict, board, params)
        if ch_corners is None or ch_ids is None or len(ch_ids) < args.min_corners:
            continue
        all_corners.append(ch_corners)
        all_ids.append(ch_ids)
        used.append(os.path.basename(f))
    return board, all_corners, all_ids, used, size


def _self_test():
    """自测：校验 build_board / detect_charuco。"""
    import types

    args = types.SimpleNamespace(
        dict_name="DICT_5X5_50", squares_x=6, squares_y=9,
        square_mm=30.0, marker_mm=22.5, adaptive_window=53, refine="SUBPIX",
    )
    dictionary, board, params = build_board(args)
    assert dictionary is not None and board is not None and params is not None
    print("[OK] build_board")

    img = draw_board(board, (640, 480))
    corners, ids, ch_corners, ch_ids = detect_charuco(img, dictionary, board, params)
    print(f"[OK] detect_charuco（标记={0 if ids is None else len(ids)}，"
          f"角点={0 if ch_ids is None else len(ch_ids)}）")

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
    ap.add_argument("--out")
    ap.add_argument("--squares-x", type=int)
    ap.add_argument("--squares-y", type=int)
    ap.add_argument("--square-mm", type=float)
    ap.add_argument("--marker-mm", type=float)
    ap.add_argument("--dict", dest="dict_name")
    ap.add_argument("--min-corners", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    add_detector_arguments(ap)
    args = parse_tool_args(ap, "calibrate_charuco")
    os.makedirs(args.out, exist_ok=True)

    board, all_corners, all_ids, used, size = collect(args)
    print(f"[info] 可用图片 {len(all_corners)}/{len(image_files(args.images))} 张, 分辨率 {size}")
    if len(all_corners) < 6:
        raise SystemExit("有效图太少 (<6)，无法稳定标定。请多采集不同角度/位置。")

    W, H = size
    objp = chessboard_corners(board).astype(np.float32)   # 数组形状为 (N,1,3)，单位为米。
    objpoints, imgpoints = [], []
    for ch_c, ch_i in zip(all_corners, all_ids):
        idx = ch_i.reshape(-1)
        objpoints.append(objp[idx])
        imgpoints.append(ch_c.reshape(-1, 1, 2).astype(np.float32))

    # OpenVINS 的 CamRadtan 只读 4 个畸变系数 (k1,k2,p1,p2)，没有 k3。
    # 若解出 k3 会被静默丢弃，角点处可产生 ~20px 误差。故默认固定 k3=0。
    flags = cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (W, H), None, None, flags=flags)

    fu, fv = K[0, 0], K[1, 1]
    cu, cv_ = K[0, 2], K[1, 2]
    d = dist.reshape(-1)

    print("\n=== 标定结果 ===")
    print(f"  重投影误差 RMS : {rms:.4f} px   ({'优' if rms<0.5 else '可' if rms<1.0 else '偏高'})")
    print(f"  fu, fv         : {fu:.2f}, {fv:.2f}")
    print(f"  cu, cv         : {cu:.2f}, {cv_:.2f}   (中心应≈ {W/2:.0f}, {H/2:.0f})")
    print(f"  k1,k2,p1,p2,k3 : {', '.join(f'{x:+.5f}' for x in d)}")
    print(f"  FOV approx     : H {2*np.degrees(np.arctan(W/(2*fu))):.1f}deg  "
          f"V {2*np.degrees(np.arctan(H/(2*fv))):.1f}deg")

    # ---------------- 写结果 ----------------
    result = {
        "image_size": [W, H],
        "rms_reprojection_error_px": float(rms),
        "num_images_used": len(all_corners),
        "camera_matrix": {"fu": float(fu), "fv": float(fv), "cu": float(cu), "cv": float(cv_)},
        "K": K.tolist(),
        "distortion_coeffs_radtan": [float(x) for x in d],
        "distortion_model": "radtan",
        "fov_deg": {"horizontal": float(2*np.degrees(np.arctan(W/(2*fu)))),
                    "vertical": float(2*np.degrees(np.arctan(H/(2*fv))))},
        "board": {"squares": [args.squares_x, args.squares_y],
                  "square_mm": args.square_mm, "marker_mm": args.marker_mm,
                  "dictionary": args.dict_name},
        "images_used": used,
    }
    p_res = os.path.join(args.out, "calib_result.yaml")
    with open(p_res, "w") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    print(f"\n[out] {p_res}")

    # ---------------- OpenVINS 片段 ----------------
    ov = {
        "intrinsics": [float(fu), float(fv), float(cu), float(cv_)],
        "distortion_coeffs": [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
        "distortion_model": "radtan",
        "resolution": [W, H],
    }
    p_ov = os.path.join(args.out, "camera_openvins.yaml")
    with open(p_ov, "w") as f:
        f.write("# OpenVINS 相机内参 —— 由标定生成\n")
        yaml.safe_dump(ov, f, sort_keys=False)
    print(f"[out] {p_ov}")

    # ---------------- 去畸变对比 ----------------
    files = image_files(args.images)
    if files:
        img = cv2.imread(files[len(files)//2], cv2.IMREAD_GRAYSCALE)
        und = cv2.undistort(img, K, dist)
        h, w = img.shape
        comp = np.hstack([img, np.full((h, 4), 255, np.uint8), und])
        p_und = os.path.join(args.out, "undistort_check.png")
        cv2.imwrite(p_und, comp)
        print(f"[out] {p_und}  (左=原图 右=去畸变)")

    print("\n把上面的 intrinsics/distortion_coeffs 填进 kalibr_imucam_chain.yaml 即可。")
    return rms


if __name__ == "__main__":
    main()
