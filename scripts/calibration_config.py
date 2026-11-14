#!/usr/bin/env python3
"""Shared YAML defaults for source scripts and installed `ros2 run` tools."""

import argparse
import json
import math
import os
from pathlib import Path

import yaml


def default_config_path():
    here = Path(__file__).resolve()
    source = here.parent.parent / "config" / "camera.yaml"
    if source.is_file():
        return source
    return here.parents[2] / "share" / "calibration" / "config" / "camera.yaml"


def _read(path):
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"配置必须是 YAML 映射: {path}")
    # aprilgrid.target_file 的相对路径解析为相对本配置文件所在目录的绝对路径。
    if "aprilgrid" in data and "target_file" in data["aprilgrid"]:
        target = Path(data["aprilgrid"]["target_file"]).expanduser()
        if not target.is_absolute():
            target = Path(path).resolve().parent / target
        data["aprilgrid"]["target_file"] = str(target)
    return data


def _merge(base, update, prefix=""):
    for key, value in update.items():
        if prefix.startswith("tools.") and prefix.count(".") == 2:
            base[key] = value
            continue
        if key not in base:
            raise ValueError(f"未知配置项: {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"配置项必须是映射: {prefix}{key}")
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value


def load_settings(path=None):
    settings = _read(default_config_path())
    if path:
        _merge(settings, _read(path))
    return settings


def _get(settings, dotted):
    value = settings
    for part in dotted.split("."):
        value = value[part]
    return value


def add_detector_arguments(parser):
    parser.add_argument("--adaptive-window", type=int)
    parser.add_argument("--refine", choices=["NONE", "SUBPIX", "CONTOUR", "APRILTAG"])


def parse_tool_args(parser, tool, argv=None, bindings=None):
    """Apply shared defaults, then tool defaults, then explicit CLI overrides."""
    parser.add_argument("--config", default=os.environ.get("CALIBRATION_CONFIG"),
                        help="共用 YAML 配置；也可设置 CALIBRATION_CONFIG")
    parser.add_argument("--print-config", action="store_true",
                        help="打印生效参数后退出，不打开相机或执行求解")
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", default=os.environ.get("CALIBRATION_CONFIG"))
    selected, _ = probe.parse_known_args(argv)
    try:
        settings = load_settings(selected.config)
        mapping = {
            "squares_x": "charuco.squares_x", "squares_y": "charuco.squares_y",
            "square_mm": "charuco.square_mm", "marker_mm": "charuco.marker_mm",
            "dict_name": "charuco.dictionary",
            "width": "camera.width", "height": "camera.height", "fps": "camera.fps",
            "topic": "camera.topic", "images": "paths.images", "out": "paths.out",
            "calib": "paths.calib", "min_corners": "analysis.min_corners",
            "folds": "analysis.folds", "adaptive_window": "detector.adaptive_window",
            "refine": "detector.refine", "target_file": "aprilgrid.target_file",
        }
        mapping.update(bindings or {})
        actions = {a.dest: a for a in parser._actions}
        defaults = {dest: _get(settings, key) for dest, key in mapping.items() if dest in actions}
        overrides = settings["tools"].get(tool, {})
        for reserved in ("config", "print_config", "help", "target_file"):
            if reserved in overrides:
                raise ValueError(f"tools 中不能覆盖 {reserved}；请使用顶层配置或命令行选择配置文件")
        defaults.update(overrides)
        for dest, value in defaults.items():
            if dest not in actions:
                raise ValueError(f"{tool} 不支持配置参数 {dest}")
            action = actions[dest]
            if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                if type(value) is not bool:
                    raise ValueError(f"{dest} 必须为 YAML 布尔值 true/false")
                continue
            if value is None or isinstance(value, (dict, list, bool)):
                raise ValueError(f"{dest} 的值类型不正确: {value!r}")
            if action.type is None and not isinstance(value, str):
                raise ValueError(f"{dest} 必须为字符串")
            if action.type:
                if action.type is int and (not isinstance(value, (str, int))):
                    raise ValueError(f"{dest} 必须是整数")
                value = action.type(value)
            if action.choices and value not in action.choices:
                raise ValueError(f"{dest} 必须取值于 {action.choices}")
            defaults[dest] = value
        parser.set_defaults(**defaults)
        args = parser.parse_args(argv)
        _validate_args(args)
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if args.print_config:
        print(json.dumps(vars(args), indent=2, ensure_ascii=False))
        parser.exit()
    return args


def _validate_args(args):
    positive = ("squares_x", "squares_y", "square_mm", "marker_mm", "width", "height",
                "fps", "target", "per_cell", "min_corners", "min_markers", "max_per_img",
                "folds", "adaptive_window", "detect_every", "win_w", "win_h", "port",
                "timeout", "save_interval", "preview_fps", "dpi", "cols", "rows", "tag_size", "border",
                "checker_cols", "checker_rows", "checker_square_mm")
    for key in positive:
        if hasattr(args, key):
            value = getattr(args, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} 必须为有限正数")
    for key in ("min_gap", "cell_sec", "min_sharp", "spacing", "margin"):
        if hasattr(args, key):
            value = getattr(args, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} 必须为有限非负数")
    if hasattr(args, "square_mm") and args.marker_mm >= args.square_mm:
        raise ValueError("marker_mm 必须小于 square_mm")
    for key in ("squares_x", "squares_y", "folds"):
        if hasattr(args, key) and getattr(args, key) < 2:
            raise ValueError(f"{key} 至少为 2")
    if hasattr(args, "adaptive_window") and args.adaptive_window < 3:
        raise ValueError("adaptive_window 至少为 3")
    if hasattr(args, "port") and args.port > 65535:
        raise ValueError("port 不得超过 65535")


def load_aprilgrid(path):
    with Path(path).open() as stream:
        target = yaml.safe_load(stream)
    if not isinstance(target, dict) or target.get("target_type") != "aprilgrid":
        raise ValueError("target_file 必须是 Kalibr AprilGrid 配置")
    for key in ("tagCols", "tagRows", "blackTagBorder"):
        if type(target.get(key)) is not int or target[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    for key in ("tagSize", "tagSpacing"):
        value = target.get(key)
        if (not isinstance(value, (float, int)) or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{key} 必须为有限正数")
    return target


def aprilgrid_arguments(parser, argv=None):
    """Add geometry defaults from the same target YAML used by Kalibr."""
    parser.add_argument("--target-file")
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", default=os.environ.get("CALIBRATION_CONFIG"))
    probe.add_argument("--target-file")
    selected, _ = probe.parse_known_args(argv)
    try:
        settings = load_settings(selected.config)
        target = load_aprilgrid(selected.target_file or settings["aprilgrid"]["target_file"])
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return target


def detector_parameters(args):
    import cv2
    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()
    params.adaptiveThreshWinSizeMax = args.adaptive_window
    params.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_" + args.refine)
    return params


def aruco_dict(family):
    """兼容 OpenCV 4.5 / 4.7+ 两套字典获取接口。"""
    import cv2
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, family))
    return cv2.aruco.Dictionary_get(getattr(cv2.aruco, family))


def charuco_board(squares_x, squares_y, square_m, marker_m, dictionary):
    """兼容 OpenCV 4.5 / 4.7+ 两套 ChArUco 板创建接口。"""
    import cv2
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard((squares_x, squares_y), square_m, marker_m, dictionary)
    return cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_m, marker_m, dictionary)


def chessboard_corners(board):
    """兼容 OpenCV 4.5 / 4.7+ 两套棋盘角点获取接口。"""
    if hasattr(board, "getChessboardCorners"):
        return board.getChessboardCorners()
    return board.chessboardCorners


def detect_markers(img, dictionary, params):
    """兼容 OpenCV 4.5 / 4.7+ 两套标记检测接口。"""
    import cv2
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(img)
    return cv2.aruco.detectMarkers(img, dictionary, parameters=params)


def interpolate_charuco(board, img, corners, ids):
    """兼容 OpenCV 4.5 / 4.7+ 两套 ChArUco 角点插值接口。"""
    import cv2
    if hasattr(cv2.aruco, "CharucoDetector"):
        ch_corners, ch_ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(img)
        return ch_corners, ch_ids
    _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, img, board)
    return ch_corners, ch_ids


def draw_marker(dictionary, marker_id, side_pixels, border_bits):
    """兼容 OpenCV 4.5 / 4.7+ 两套标记绘制接口。"""
    import cv2
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, side_pixels, borderBits=border_bits)
    return cv2.aruco.drawMarker(dictionary, marker_id, side_pixels, borderBits=border_bits)


def draw_board(board, size, margin=0, border_bits=1):
    """兼容 OpenCV 4.5 / 4.7+ 两套标定板绘制接口。"""
    if hasattr(board, "generateImage"):
        return board.generateImage(size, marginSize=margin, borderBits=border_bits)
    return board.draw(size, marginSize=margin, borderBits=border_bits)


def analysis_arguments(tool):
    parser = argparse.ArgumentParser()
    parser.add_argument("--images")
    parser.add_argument("--out")
    parser.add_argument("--calib")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--squares-x", type=int)
    parser.add_argument("--squares-y", type=int)
    parser.add_argument("--square-mm", type=float)
    parser.add_argument("--marker-mm", type=float)
    parser.add_argument("--dict", dest="dict_name")
    parser.add_argument("--min-corners", type=int)
    add_detector_arguments(parser)
    return parse_tool_args(parser, tool)


def image_files(directory):
    return sorted(str(p) for p in Path(directory).iterdir()
                  if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"))


def check_image_size(image, args, filename):
    actual = (image.shape[1], image.shape[0])
    if actual != (args.width, args.height):
        raise ValueError(f"{filename}: 图像尺寸 {actual} 与配置 {(args.width, args.height)} 不同")


def _self_test():
    """自测：验证配置读取与 OpenCV 兼容函数的核心行为。运行 `python3 calibration_config.py` 触发。"""
    import tempfile

    # 1. _read 读 dict YAML
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"camera": {"device_name": "TEST"}}, f)
        p = f.name
    assert _read(p) == {"camera": {"device_name": "TEST"}}
    print("[OK] _read 读 dict YAML")

    # 2. _read 读非 dict YAML 抛 ValueError
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("- a\n- b\n")
        p2 = f.name
    try:
        _read(p2)
        print("[FAIL] 未抛 ValueError")
        return 1
    except ValueError:
        print("[OK] _read 非 dict 抛 ValueError")

    # 3. _get 嵌套获取
    assert _get({"a": {"b": 42}}, "a.b") == 42
    print("[OK] _get 嵌套获取")

    # 4. _merge 合并与未知键报错
    base = {"camera": {"width": 640}}
    _merge(base, {"camera": {"width": 4096}})
    assert base["camera"]["width"] == 4096
    try:
        _merge({"camera": {}}, {"unknown": 1})
        print("[FAIL] _merge 未对未知键报错")
        return 1
    except ValueError:
        print("[OK] _merge 合并与未知键报错")

    # 5. OpenCV 兼容函数
    import cv2
    d = aruco_dict("DICT_5X5_50")
    b = charuco_board(6, 9, 0.03, 0.0225, d)
    m = draw_marker(d, 0, 80, 1)
    assert d is not None and b is not None and m.shape == (80, 80)
    params = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters")
              else cv2.aruco.DetectorParameters_create())
    detect_markers(m, d, params)
    print("[OK] aruco_dict / charuco_board / draw_marker / detect_markers")

    # 6. load_settings 读默认配置 + 必要字段完整性
    try:
        s = load_settings()
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[SKIP] 默认配置不可用: {exc}")
        return 0
    cam = s.get("camera")
    if not isinstance(cam, dict):
        print("[FAIL] camera 段缺失或不是映射")
        return 1
    for key in ("device_name", "frame_size", "width", "height", "fps", "topic"):
        if key not in cam:
            print(f"[FAIL] camera 缺少字段: {key}")
            return 1
    print(f"[OK] load_settings 读默认配置，device_name={cam.get('device_name')}")

    return 0


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="配置模块自测")
    ap.add_argument("--test", action="store_true", help="运行内部函数自测")
    args = ap.parse_args()
    if not args.test:
        ap.print_help()
        sys.exit(0)
    sys.exit(_self_test())
