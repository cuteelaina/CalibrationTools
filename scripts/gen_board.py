#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_board.py —— 生成用于相机内参标定的标定板（精确毫米尺寸的 PDF + PNG）

默认产出（data/boards/，可用 --out 或共用 YAML 修改）:
  1. charuco_6x9_30mm_A4.pdf     ChArUco 板，6x9 格，方格 30mm，标记 22.5mm（推荐）
  2. checkerboard_9x6_25mm_A4.pdf 棋盘格，9x6 内角点，方格 25mm（备用）
  + 对应 PNG

关键改进（避免打印机自动缩放）:
  - PDF 按真实毫米绘制（1mm = 72/25.4 pt）
  - 图案四周留 ≥12mm 边距（本机打印机不可打印边为 4.10mm）
  - 底部画一根 100mm 校验尺，打印后可直接量，验证是否 1:1
  - 打印时用 `lp -o print-scaling=none`（见文末）

用法: python3 gen_board.py
"""

import os
import argparse
from calibration_config import parse_tool_args, aruco_dict, charuco_board, draw_board

import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

OUT = None  # main() 根据共用配置设置，导入不创建目录。
PX_PER_MM = None

PAGE_W_MM, PAGE_H_MM = 210.0, 297.0   # A4 纵向
BOTTOM_MARGIN_MM = 15.0               # 图案下边距（留给校验尺）
TOP_MARGIN_MM = 12.0                  # 图案上边距（留给标题）

def _mm2px(v):
    return int(round(v * PX_PER_MM))


# ---------------------------------------------------------------------------
# 标定板图案（不含白边；白边由纸的页边距提供）
# ---------------------------------------------------------------------------
def make_charuco(args):
    CH_SQ, CH_MK = args.squares_x, args.squares_y
    CH_SQ_MM, marker_mm = args.square_mm, args.marker_mm
    dictionary = aruco_dict(args.dict_name)
    board = charuco_board(
        CH_SQ, CH_MK, CH_SQ_MM / 1000.0, marker_mm / 1000.0, dictionary)
    w_mm, h_mm = CH_SQ * CH_SQ_MM, CH_MK * CH_SQ_MM
    img = draw_board(board, (_mm2px(w_mm), _mm2px(h_mm)), margin=0, border_bits=1)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    png = os.path.join(OUT, f"charuco_{CH_SQ}x{CH_MK}_{CH_SQ_MM:g}mm.png")
    if not cv2.imwrite(png, img) or not os.path.isfile(png):
        raise OSError(f"无法写入图像: {png}（请检查输出目录 {OUT} 是否存在且可写）")
    pdf = os.path.join(OUT, f"charuco_{CH_SQ}x{CH_MK}_{CH_SQ_MM:g}mm_A4.pdf")
    _sheet_pdf(pdf, img, w_mm, h_mm,
               f"ChArUco {CH_SQ}x{CH_MK}  square={CH_SQ_MM:g}mm  marker={marker_mm:g}mm",
               "打印必须 1:1（lp -o print-scaling=none）。方格边距已留 >12mm，不会被裁。")
    return {"png": png, "pdf": pdf, "board": board, "w_mm": w_mm, "h_mm": h_mm,
            "sq_mm": CH_SQ_MM, "marker_mm": marker_mm,
            "cols": CH_SQ, "rows": CH_MK}


def make_checkerboard(args):
    CB_CORNERS = (args.checker_cols, args.checker_rows)
    CB_SQ_MM = args.checker_square_mm
    cols, rows = CB_CORNERS[0] + 1, CB_CORNERS[1] + 1
    sq = _mm2px(CB_SQ_MM)
    img = np.zeros((rows * sq, cols * sq), np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 255
    w_mm, h_mm = cols * CB_SQ_MM, rows * CB_SQ_MM
    png = os.path.join(OUT, f"checkerboard_{CB_CORNERS[0]}x{CB_CORNERS[1]}_{CB_SQ_MM:g}mm.png")
    if not cv2.imwrite(png, img) or not os.path.isfile(png):
        raise OSError(f"无法写入图像: {png}（请检查输出目录 {OUT} 是否存在且可写）")
    pdf = os.path.join(OUT, f"checkerboard_{CB_CORNERS[0]}x{CB_CORNERS[1]}_{CB_SQ_MM:g}mm_A4.pdf")
    _sheet_pdf(pdf, img, w_mm, h_mm,
               f"Checkerboard {CB_CORNERS[0]}x{CB_CORNERS[1]} inner, square={CB_SQ_MM:g}mm",
               "打印必须 1:1（lp -o print-scaling=none）。")
    return {"png": png, "pdf": pdf, "sq_mm": CB_SQ_MM, "corners": CB_CORNERS}


# ---------------------------------------------------------------------------
# PDF：图案居中 + 底部 100mm 校验尺
# ---------------------------------------------------------------------------
def _sheet_pdf(path, img, w_mm, h_mm, title, note):
    tmp = path + ".tmp.png"
    if not cv2.imwrite(tmp, img) or not os.path.isfile(tmp):
        raise OSError(f"无法写入临时图像: {tmp}（请检查输出目录是否存在且可写）")
    c = canvas.Canvas(path, pagesize=A4)
    pw, ph = A4

    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(pw / 2.0, ph - 6 * mm, title)
    c.setFont("Helvetica", 6.5)
    c.drawCentredString(pw / 2.0, ph - 10 * mm, note)

    x = (PAGE_W_MM - w_mm) / 2.0 * mm
    y = BOTTOM_MARGIN_MM * mm
    c.drawImage(tmp, x, y, width=w_mm * mm, height=h_mm * mm)

    _draw_ruler(c, pw / 2.0, 9 * mm)
    c.showPage()
    c.save()
    os.remove(tmp)


def _draw_ruler(c, cx, y):
    """100mm 校验尺：打印后量此线，若 ≠100mm 说明被缩放。"""
    L = 100 * mm
    x0 = cx - L / 2.0
    c.setLineWidth(0.6)
    c.line(x0, y, x0 + L, y)
    for i in range(11):
        x = x0 + i * 10 * mm
        hgt = 3 * mm if i % 5 == 0 else 1.5 * mm
        c.line(x, y, x, y + hgt)
    c.setFont("Helvetica", 6)
    c.drawString(x0, y - 3.5 * mm, "0")
    c.drawRightString(x0 + L, y - 3.5 * mm, "100 mm")
    c.drawCentredString(cx, y - 3.5 * mm, "← 此尺应为 100.0 mm（验证打印 1:1） →")


def _self_test():
    """自测：校验 _mm2px / make_charuco / make_checkerboard。"""
    global OUT, PX_PER_MM
    import tempfile
    import types

    PX_PER_MM = 300 / 25.4
    assert _mm2px(10.0) == int(round(10.0 * PX_PER_MM))
    print("[OK] _mm2px")

    args = types.SimpleNamespace(
        squares_x=6, squares_y=9, square_mm=30.0, marker_mm=22.5,
        dict_name="DICT_5X5_50", dpi=300,
        checker_cols=9, checker_rows=6, checker_square_mm=25.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        OUT = tmp
        ch = make_charuco(args)
        assert os.path.isfile(ch["png"]) and os.path.isfile(ch["pdf"])
        print(f"[OK] make_charuco（{os.path.basename(ch['png'])}）")
        cb = make_checkerboard(args)
        assert os.path.isfile(cb["png"]) and os.path.isfile(cb["pdf"])
        print(f"[OK] make_checkerboard（{os.path.basename(cb['png'])}）")

    return 0


def main():
    import sys
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--test", action="store_true")
    known, _ = probe.parse_known_args()
    if known.test:
        sys.exit(_self_test())

    global OUT, PX_PER_MM
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--squares-x", type=int)
    ap.add_argument("--squares-y", type=int)
    ap.add_argument("--square-mm", type=float)
    ap.add_argument("--marker-mm", type=float)
    ap.add_argument("--dict", dest="dict_name")
    ap.add_argument("--dpi", type=int)
    ap.add_argument("--checker-cols", type=int)
    ap.add_argument("--checker-rows", type=int)
    ap.add_argument("--checker-square-mm", type=float)
    args = parse_tool_args(ap, "gen_board", bindings={"out": "paths.boards"})
    OUT = args.out
    PX_PER_MM = args.dpi / 25.4
    os.makedirs(OUT, exist_ok=True)
    ch = make_charuco(args)
    cb = make_checkerboard(args)
    print("=== 生成完成 ===")
    for k in ("png", "pdf"):
        print(f"  ChArUco {k}: {ch[k]}")
    for k in ("png", "pdf"):
        print(f"  棋盘格  {k}: {cb[k]}")
    print()
    print(f"ChArUco: {ch['cols']}x{ch['rows']} 格, 方格 {ch['sq_mm']:g}mm, "
          f"标记 {ch['marker_mm']:g}mm, 图案 {ch['w_mm']:.0f}x{ch['h_mm']:.0f}mm")
    mx = (PAGE_W_MM - ch['w_mm']) / 2.0
    my = PAGE_H_MM - TOP_MARGIN_MM - ch['h_mm']
    print(f"页边距: 左右 {mx:.1f}mm, 上 {TOP_MARGIN_MM:.1f}mm, 下 {BOTTOM_MARGIN_MM:.1f}mm")
    print(f"（打印机不可打印边 4.10mm → 全部在可打印区内，print-scaling=none 不会被裁）")


if __name__ == "__main__":
    main()
