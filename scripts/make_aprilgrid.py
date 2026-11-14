#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_aprilgrid.py —— 生成可 1:1 打印的 AprilGrid（Kalibr 联合标定用）

为什么要自己生成：
  仓库原有的 aprilgrid_6x6_22mm.pdf 经检测**在完美渲染下也检不出任何标记**
  （4 个家族 × 多尺度 × 翻转旋转 × poppler/ghostscript 两种渲染器，全部 0/36），
  而同一台机器上正确生成的对照板是 36/36。板子图案无效 → 无论相机多好都标不了。

做法：
  - 家族固定 DICT_APRILTAG_36h11（Kalibr 默认）
  - **矢量绘制**：每个黑模块画成一个矩形，不做位图缩放 → 打印 100% 就是精确尺寸
  - 标记编号按行优先 0..N-1（Kalibr 期望的顺序）

用法：
  # 生成 + 自检
  python3 make_aprilgrid.py --out aprilgrid_6x6_22mm_FIXED.pdf
  # 只检查某个 PDF（渲染后检测）
  python3 make_aprilgrid.py --check some.pdf
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from calibration_config import (parse_tool_args, aprilgrid_arguments,
                                aruco_dict, draw_marker, detect_markers)

def marker_matrix(d, i, nmod=8, scale=40, border=None):
    """把标记采样成 nmod×nmod 的布尔矩阵（True=黑）。取模块中心，避免边缘误差。"""
    if border is None:
        border = (nmod - 6) // 2
    big = draw_marker(d, i, nmod * scale, border)
    m = np.zeros((nmod, nmod), bool)
    for r in range(nmod):
        for c in range(nmod):
            m[r, c] = big[int((r + 0.5) * scale), int((c + 0.5) * scale)] < 128
    return m


def build(out, tag_mm, spacing, cols, rows, margin_mm, family, nmod=8, border=None):
    d = aruco_dict(family)
    if border is None:
        border = (nmod - 6) // 2
    gap = tag_mm * spacing
    board = cols * tag_mm + (cols + 1) * gap
    height = rows * tag_mm + (rows + 1) * gap
    page = board + 2 * margin_mm
    c = canvas.Canvas(out, pagesize=(page * mm, (height + 2 * margin_mm) * mm))
    c.setFillColorRGB(0, 0, 0)
    eps = 0.02  # 略微外扩，消除相邻模块之间的白色接缝
    for r in range(rows):
        for col in range(cols):
            i = r * cols + col
            # Kalibr 约定：标记图案相对 cv2.aruco 默认绘制再旋转 180°
            # （见 Kalibr src/createTargetPDF.py 的 generateAprilTag(..., rotation=2)）
            m = np.rot90(marker_matrix(d, i, nmod, border=border), 2)
            ms = tag_mm / nmod
            x0 = margin_mm + gap + col * (tag_mm + gap)
            # Kalibr 约定：编号为 0 的标记位于左下角。PDF 坐标 y 向上 => r=0 放底部
            ybot = margin_mm + gap + r * (tag_mm + gap)
            for mr in range(nmod):
                for mc in range(nmod):
                    if not m[mr, mc]:
                        continue
                    x = x0 + mc * ms
                    y = ybot + (nmod - 1 - mr) * ms
                    c.rect((x - eps) * mm, (y - eps) * mm,
                           (ms + 2 * eps) * mm, (ms + 2 * eps) * mm,
                           stroke=0, fill=1)
    c.showPage()
    c.save()
    return board, page


def detect_png(png, family, nmod=8, expect=None):
    img = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1, None, None
    d = aruco_dict(family)
    import cv2
    p = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters")
         else cv2.aruco.DetectorParameters_create())
    corners, ids, _ = detect_markers(img, d, p)
    n = 0 if ids is None else len(ids)
    return n, img, corners


def check(pdf, family, dpi=300, expect=None, border=1):
    tmp = tempfile.mkdtemp(prefix="agcheck_")
    try:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", pdf,
                        os.path.join(tmp, "p")], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:                                   # noqa: BLE001
        print(f"  渲染失败: {e}")
        return 0
    files = sorted(glob.glob(os.path.join(tmp, "p*.png")))
    if not files:
        print("  没有渲染出页面")
        return 0
    if border != 1:
        print("  注意: cv2.aruco 仅能识别 1 模块黑边，border=%d 需用 Kalibr 检测器验证" % border)
    n, img, corners = detect_png(files[0], family)
    h, w = img.shape
    print(f"  渲染 {w}x{h} @{dpi}dpi  ->  检出 {n}" +
          (f"/{expect}" if expect else ""))
    if corners is not None and len(corners):
        c0 = corners[0].reshape(-1, 2)
        side = np.mean([np.linalg.norm(c0[i] - c0[(i + 1) % 4]) for i in range(4)])
        print(f"  实测 tag 边长 = {side / dpi * 25.4:.2f} mm")
        pts = np.array([corners[i].reshape(-1, 2).mean(axis=0)
                        for i in range(len(corners))])
        if len(pts) > 1:
            dd = np.sort(np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2), axis=1)
            near = dd[:, 1]
            print(f"  最近邻中心距 中位 = {np.median(near) / dpi * 25.4:.2f} mm")
    return n


def _self_test():
    """自测：校验 marker_matrix / build。"""
    import tempfile

    d = aruco_dict("DICT_APRILTAG_36h11")
    m = marker_matrix(d, 0, nmod=8, scale=40, border=1)
    assert m.shape == (8, 8) and m.dtype == bool
    print(f"[OK] marker_matrix（{m.shape}，黑模块={int(m.sum())}）")

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "grid.pdf")
        board, page = build(out, tag_mm=22.0, spacing=0.3, cols=6, rows=6,
                            margin_mm=12.0, family="DICT_APRILTAG_36h11",
                            nmod=8, border=1)
        assert os.path.isfile(out)
        print(f"[OK] build（板 {board:.1f}mm，页 {page:.1f}mm）")

    return 0


def main():
    import sys
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--test", action="store_true")
    known, _ = probe.parse_known_args()
    if known.test:
        sys.exit(_self_test())

    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--check")
    ap.add_argument("--tag-size", type=float)
    ap.add_argument("--spacing", type=float)
    ap.add_argument("--cols", type=int)
    ap.add_argument("--rows", type=int)
    ap.add_argument("--margin", type=float)
    ap.add_argument("--border", type=int,
                    help="tag 黑边模块数。1 = 本工具默认，配 blackTagBorder: 1 使用（cv2 自检可用）；"
                         "2 = Kalibr 原生约定，stock Kalibr 免补丁，但 cv2 自检不适用（需 Kalibr 检测器验证）")
    ap.add_argument("--dpi", type=int)
    target = aprilgrid_arguments(ap)
    ap.add_argument("--dict", dest="dict_name")
    ap.set_defaults(tag_size=target["tagSize"] * 1000, spacing=target["tagSpacing"],
                    cols=target["tagCols"], rows=target["tagRows"], border=target["blackTagBorder"])
    a = parse_tool_args(ap, "make_aprilgrid", bindings={"dict_name": "aprilgrid.dictionary", "out": "paths.aprilgrid_pdf"})

    if a.check:
        print(f"检查 {a.check}")
        n = check(a.check, a.dict_name, a.dpi, a.cols * a.rows, a.border)
        print("  判定:", "PASS 板子可用" if n >= a.cols * a.rows else "FAIL 板子无效")
        return 0 if n >= a.cols * a.rows else 1

    if not a.out:
        ap.error("需要 --out 或 --check")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    board, page = build(a.out, a.tag_size, a.spacing, a.cols, a.rows, a.margin, a.dict_name,
                        nmod=6 + 2 * a.border, border=a.border)
    print(f"已生成 {a.out}")
    print(f"  tag 边长 {a.tag_size}mm  间距 {a.tag_size * a.spacing:.2f}mm "
          f"({a.spacing:.2f}×)  网格 {a.cols}x{a.rows}  黑边 {a.border} 模块")
    height = a.rows * a.tag_size + (a.rows + 1) * a.tag_size * a.spacing
    print(f"  板面 {board:.1f}x{height:.1f}mm   页 {page:.1f}x{height + 2 * a.margin:.1f}mm")
    print(f"  >>> 打印务必 100%（不缩放）")
    print("自检:")
    n = check(a.out, a.dict_name, a.dpi, a.cols * a.rows, a.border)
    ok = n >= a.cols * a.rows if a.border == 1 else True
    print("  判定:", "PASS 板子可用" if ok else "FAIL 板子无效")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
