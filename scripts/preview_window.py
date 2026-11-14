#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preview_window.py —— 在**本机桌面**上直接开一个实时预览窗口（不经过浏览器）。

适用场景：人就坐在这台机器前面（有 X 显示），想直接看到画面。
与 preview_server.py 的区别：
  - preview_server.py：通过浏览器或门户查看（远程也能看）
  - preview_window.py：本机 X 窗口（最直接，不依赖浏览器、门户或网络）

取流默认订阅 ROS /camera/image_raw —— 相机的唯一占用者仍是相机驱动，
预览与 ros2 bag record 共享同一路流，**不会跟录包抢相机**。

用法：
  DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
    python3 preview_window.py --topic /camera/image_raw
  # 按 q 或 ESC 退出
"""
import argparse
import os
import signal
import threading
import time

import cv2
import numpy as np
from calibration_config import (parse_tool_args, aprilgrid_arguments,
                                add_detector_arguments, detector_parameters)

STOP = threading.Event()


def aruco_dict(family):
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, family))
    return cv2.aruco.Dictionary_get(getattr(cv2.aruco, family))


def detect(gray, d, p):
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(d, p).detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, d, parameters=p)


def _self_test():
    """自测：校验 aruco_dict / detect。"""
    import cv2
    from calibration_config import draw_marker

    d = aruco_dict("DICT_APRILTAG_36h11")
    assert d is not None
    img = draw_marker(d, 0, 80, 1)
    p = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters")
         else cv2.aruco.DetectorParameters_create())
    corners, ids, _ = detect(img, d, p)
    print(f"[OK] aruco_dict / detect（ids={0 if ids is None else len(ids)}）")
    return 0


def main():
    import sys
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--test", action="store_true")
    known, _ = probe.parse_known_args()
    if known.test:
        sys.exit(_self_test())

    ap = argparse.ArgumentParser()
    ap.add_argument("--topic")
    ap.add_argument("--detect-every", type=int)
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--win-w", type=int)
    ap.add_argument("--win-h", type=int)
    target = aprilgrid_arguments(ap)
    ap.set_defaults(min_tags=max(target["tagRows"], target["tagCols"]) + 1)
    ap.add_argument("--dict", dest="dict_name")
    add_detector_arguments(ap)
    args = parse_tool_args(ap, "preview_window", bindings={"dict_name": "aprilgrid.dictionary"})
    if not args.no_detect and target["blackTagBorder"] != 1:
        ap.error("OpenCV 预览检测仅支持 blackTagBorder: 1，可用 --no-detect 仅看画面")
    min_tags = args.min_tags

    import rospy
    from sensor_msgs.msg import Image as RosImage
    try:
        from cv_bridge import CvBridge
    except ImportError:
        CvBridge = None

    latest = {"frame": None, "n": 0, "t": 0.0}
    lk = threading.Lock()

    rospy.init_node("preview_window", disable_signals=True)
    bridge = CvBridge() if CvBridge else None

    def cb(msg):
        try:
            if bridge is not None:
                f = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                f = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:                                  # noqa: BLE001
            return
        with lk:
            latest["frame"] = f
            latest["n"] += 1
            latest["t"] = time.time()

    rospy.Subscriber(args.topic, RosImage, cb, queue_size=1)

    def _spin():
        # rospy 的 SIGTERM 处理会关闭上下文；那是退出路径的一部分，不是故障。
        try:
            rospy.spin()
        except Exception:                                  # noqa: BLE001
            pass
    threading.Thread(target=_spin, daemon=True).start()

    # 自己接管 SIGTERM/SIGINT。rospy.init_node() 会装一个只置标志的处理器，
    # 而主线程阻塞在 cv2.waitKey 里 —— 默认 SIGTERM 能杀进程，
    # 但 rospy 的处理器会拦住它并关闭上下文，导致 spin 线程退出、
    # cv2 循环却继续跑（表现为“杀不掉的窗口进程”）。
    def _stop(signum, _frame):
        print(f"[window] signal {signum}, closing", flush=True)
        STOP.set()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)

    d = aruco_dict(args.dict_name) if not args.no_detect else None
    p = detector_parameters(args) if not args.no_detect else None
    last_corners = last_ids = None
    idx = 0
    fps = 0.0
    tprev = time.time()

    win = f"Camera preview - {args.topic}  (q/ESC quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # 显式给一个够看的大小：WINDOW_NORMAL 默认会随图像尺寸/窗管缩放，
    # 实际使用中出现过 480x240 这种看不清的尺寸。
    cv2.resizeWindow(win, args.win_w, args.win_h)
    cv2.moveWindow(win, 60, 60)

    print(f"[window] DISPLAY={os.environ.get('DISPLAY')} topic={args.topic}", flush=True)
    try:
        while not STOP.is_set():
            with lk:
                f = latest["frame"]
            if f is None:
                # 还没收到帧：先给个提示窗口，别让用户以为没启动
                ph = np.full((240, 480, 3), 30, np.uint8)
                cv2.putText(ph, f"waiting for {args.topic} ...", (24, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.imshow(win, ph)
                if cv2.waitKey(50) in (27, ord("q")):
                    break
                continue

            frame = f.copy()
            idx += 1
            tags = 0
            if d is not None:
                if idx % max(1, args.detect_every) == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = detect(gray, d, p)
                    if ids is not None and len(ids) > 0:
                        last_corners, last_ids = corners, ids
                    else:
                        last_corners = last_ids = None
                if last_ids is not None and len(last_ids) > 0:
                    tags = len(last_ids)
                    cv2.aruco.drawDetectedMarkers(frame, last_corners, last_ids)

            hh, ww = frame.shape[:2]
            for i in range(1, 3):
                cv2.line(frame, (ww * i // 3, 0), (ww * i // 3, hh), (60, 60, 60), 1)
                cv2.line(frame, (0, hh * i // 3), (ww, hh * i // 3), (60, 60, 60), 1)

            now = time.time()
            inst = 1.0 / max(now - tprev, 1e-6)
            tprev = now
            fps = 0.9 * fps + 0.1 * inst if fps else inst

            color = (0, 220, 0) if tags >= min_tags else (0, 180, 255) if tags > 0 else (0, 0, 255)
            cv2.putText(frame, f"AprilGrid tags: {tags}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"{fps:4.1f} fps  {ww}x{hh}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            if tags < min_tags:
                cv2.putText(frame, f"move so {min_tags}+ tags are visible", (10, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)

            cv2.imshow(win, frame)
            if cv2.waitKey(1) in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()
        print(f"[window] closed after {idx} frames", flush=True)


if __name__ == "__main__":
    main()
