#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 ROS 图像采集 ChArUco 内参照片，支持九宫格及网页空格拍照。

采集：python3 src/calibration/scripts/capture_charuco_ros.py
求解：python3 src/calibration/scripts/calibrate_charuco.py \
        --images data/calib_charuco/images --out data/calib_charuco

板参数读取 calibration.yaml 的 charuco；采集设置读取 tools.capture_charuco_ros。
min_corners 统计插值出的棋盘角点，至少检测到 4 个 ArUco 标记后才插值。
保存无叠加的灰度 PNG；网页拍照接口和自动/手动流程复用 capture_calib_ros。
"""
import argparse

import cv2

from calibrate_charuco import build_board, detect_charuco
from capture_calib_ros import AprilGridDetector, main as capture_main


class CharucoDetector(AprilGridDetector):
    name = "ChArUco"
    feature_name = "ChArUco 角点"
    dictionary_setting = "charuco.dictionary"
    extension = "png"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--squares-x", type=int)
        parser.add_argument("--squares-y", type=int)
        parser.add_argument("--square-mm", type=float)
        parser.add_argument("--marker-mm", type=float)

    def __init__(self, args):
        board_args = argparse.Namespace(**vars(args))
        board_args.dict_name = args.dict
        self.dictionary, self.board, self.params = build_board(board_args)
        if args.min_corners > (args.squares_x - 1) * (args.squares_y - 1):
            raise ValueError("min_corners 超过该 ChArUco 板的棋盘角点总数")
        self.min_markers = 4
        self.ch_corners = None
        self.ch_ids = None

    def detect(self, img):
        corners, ids, self.ch_corners, self.ch_ids = detect_charuco(
            img, self.dictionary, self.board, self.params)
        return corners, ids

    def count(self, corners, ids):
        return 0 if self.ch_ids is None else len(self.ch_ids)

    def annotate(self, display):
        if self.ch_ids is not None and self.ch_corners is not None:
            # 手动绘制角点，规避 OpenCV 5.0 的 drawDetectedCornersCharuco 断言 bug
            for x, y in self.ch_corners.reshape(-1, 2).astype(int):
                cv2.circle(display, (int(x), int(y)), 4, (0, 0, 255), -1)


def _self_test():
    """自测：校验 CharucoDetector。"""
    import types
    from calibration_config import draw_board

    args = types.SimpleNamespace(
        dict="DICT_5X5_50", squares_x=6, squares_y=9,
        square_mm=30.0, marker_mm=22.5, min_corners=8,
        adaptive_window=53, refine="SUBPIX",
    )
    det = CharucoDetector(args)
    assert det.dictionary is not None and det.board is not None and det.params is not None
    print("[OK] CharucoDetector.__init__")
    img = draw_board(det.board, (640, 480))
    corners, ids = det.detect(img)
    print(f"[OK] CharucoDetector.detect（标记={0 if ids is None else len(ids)}，"
          f"角点={det.count(corners, ids)}）")
    return 0


def main():
    import sys
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--test", action="store_true")
    known, _ = probe.parse_known_args()
    if known.test:
        sys.exit(_self_test())

    capture_main("capture_charuco_ros", CharucoDetector)


if __name__ == "__main__":
    main()
