#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_calib.py —— 定量验证标定结果：去畸变是否真的让"直的变直"。

原理：平面标定板在「无畸变」针孔模型下，其角点必然被单应矩阵精确拟合；
存在畸变时，纯单应拟合会留下残差。因此：
    单应矩阵拟合残差（原图）远大于单应矩阵拟合残差（去畸变后）  ⇒ 畸变系数正确
若两者都很小 → 畸变本身很小；若去畸变后反而变大 → 标定错误。
"""
from calibration_config import (analysis_arguments, image_files, check_image_size,
                                chessboard_corners)
from calibrate_charuco import build_board, detect_charuco

import cv2
import numpy as np
import yaml

def homography_residual(im, args, dictionary, board, params, plane):
    _, _, chc, chi = detect_charuco(im, dictionary, board, params)
    if chc is None or chi is None:
        return None, 0
    n = len(chi)
    if n < args.min_corners:
        return None, 0
    idx = chi.reshape(-1)
    src = plane[idx].astype(np.float32)
    dst = chc.reshape(-1, 2).astype(np.float32)
    Hm, _ = cv2.findHomography(src, dst, 0)
    if Hm is None:
        return None, n
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1)))), n



def _self_test():
    """自测：校验 homography_residual。"""
    import types
    from calibration_config import draw_board

    args = types.SimpleNamespace(
        dict_name="DICT_5X5_50", squares_x=6, squares_y=9,
        square_mm=30.0, marker_mm=22.5, adaptive_window=53, refine="SUBPIX",
        min_corners=4,
    )
    dictionary, board, params = build_board(args)
    plane = chessboard_corners(board).astype(np.float32).reshape(-1, 3)[:, :2]
    img = draw_board(board, (640, 480))
    r, n = homography_residual(img, args, dictionary, board, params, plane)
    if r is None:
        print("[FAIL] homography_residual 未检测到角点")
        return 1
    print(f"[OK] homography_residual（残差={r:.3f}px，角点={n}）")
    return 0


def main():
    import argparse
    import sys
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--test", action="store_true")
    known, _ = probe.parse_known_args()
    if known.test:
        sys.exit(_self_test())

    args = analysis_arguments("verify_calib")
    with open(args.calib) as stream:
        res = yaml.safe_load(stream)
    cm = res['camera_matrix']
    K = np.array([[cm['fu'], 0, cm['cu']], [0, cm['fv'], cm['cv']], [0, 0, 1.0]])
    # 兼容多种键名（脚本历经迭代，键名变过）
    d = (res.get('distortion_coeffs_radtan_4')
         or res.get('distortion_coeffs_radtan')
         or res.get('distortion_coeffs_radtan_full')
         or res.get('distortion_coeffs'))
    if d is None:
        raise SystemExit(f"[err] 找不到畸变系数键，现有键: {list(res.keys())}")
    d = list(d)
    # OpenCV 需 5 项 (k1,k2,p1,p2,k3)
    D = np.array((d + [0.0] * (5 - len(d)))[:5], dtype=np.float64)[None, :]

    d, board, dp = build_board(args)
    objp = chessboard_corners(board).astype(np.float32).reshape(-1, 3)
    plane = objp[:, :2]  # 标定板平面坐标（米）。


    rows = []
    for f in image_files(args.images):
        raw = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        check_image_size(raw, args, f)
        und = cv2.undistort(raw, K, D)
        r_raw, n = homography_residual(raw, args, d, board, dp, plane)
        r_und, _ = homography_residual(und, args, d, board, dp, plane)
        if r_raw is None or r_und is None:
            continue
        rows.append((r_raw, r_und, n))

    if not rows:
        raise SystemExit("没有可用的验证图像，请检查路径、板配置和检测参数")
    a = np.array([(x[0], x[1]) for x in rows])
    print(f"图像数: {len(rows)}")
    print(f"单应拟合残差 (px):")
    print(f"  原图(raw)       : 中位 {np.median(a[:,0]):.4f}  均值 {a[:,0].mean():.4f}")
    print(f"  去畸变后(undist): 中位 {np.median(a[:,1]):.4f}  均值 {a[:,1].mean():.4f}")
    impr = (1 - np.median(a[:,1]) / max(np.median(a[:,0]), 1e-12)) * 100
    print(f"  => 残差下降 {impr:+.1f}%")
    print()
    if np.median(a[:,1]) < np.median(a[:,0]) * 0.75:
        print("PASS: 去畸变显著降低单应残差 → 畸变系数正确（直线被拉直）")
    elif np.median(a[:,1]) < 1.0:
        print("PASS(弱畸变): 原图残差本就小，去畸变后仍更小或持平 → 畸变系数合理")
    else:
        print("CHECK: 去畸变未改善，需复查")


if __name__ == "__main__":
    main()
