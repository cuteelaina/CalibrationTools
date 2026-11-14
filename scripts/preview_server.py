#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preview_server.py —— 联合标定用实时预览（浏览器按需 JPEG + AprilGrid 检测叠加）

为什么用它：
  - 手持相机+IMU 对 AprilGrid 做 6 自由度运动时，必须能**实时看到画面**，
    确认标定板清晰、在视野内、覆盖不同区域；否则录包里没有可用图像，
    只能重录。
  - 叠加 AprilGrid 检测计数，直接告诉你"这一帧 Kalibr 能不能看到板子"。

两种取流方式（重要）：
  --source ros    （默认，标定时用）订阅 ROS 话题 /camera/image_raw。
                  此时相机的唯一占用者是 camera_driver —— 预览与 ros2 bag record
                  共享同一路流，互不冲突。**联合标定必须用这个模式。**
  --source v4l2   直接抓 /dev/video0（独立预览，不用 ROS）。
                  会**独占相机**，不能与相机驱动/录包同时用。

用法：
  # 标定流程（推荐）
  ros2 launch uav_bringup sensors.launch.py
  python3 preview_server.py --source ros --topic /camera/image_raw --port 8099

  # 独立看画面（不录包）
  python3 preview_server.py --source v4l2 --device /dev/video0 --port 8099

  浏览器打开 http://127.0.0.1:8099/ （或经 OpenClaw 门户代理访问）
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from calibration_config import (parse_tool_args, aprilgrid_arguments,
                                add_detector_arguments, detector_parameters)

STATE = {
    "ok": False,
    "msg": "starting",
    "src": "",
    "topic": "",
    "w": 0, "h": 0,
    "fps": 0.0,
    "tags": 0,
    "frames": 0,
    "age_ms": -1,
    "process_ms": 0.0,
}
LOCK = threading.Lock()
FRAME_READY = threading.Condition(LOCK)
JPEG = {"buf": None, "encoded_at": None}  # 最新一帧 JPEG
DETECT_ON = True


class LatestFrame:
    """单槽邮箱：处理上一帧时，新输入覆盖待处理帧，不建立历史队列。"""

    def __init__(self):
        self.ready = threading.Condition()
        self.pending = None

    def put(self, frame, stamp=None):
        with self.ready:
            self.pending = (frame, stamp, time.monotonic())
            self.ready.notify()

    def take(self):
        with self.ready:
            self.ready.wait_for(lambda: self.pending is not None)
            frame = self.pending
            self.pending = None
            return frame


def process_latest(latest, overlay, fps, convert=lambda frame: frame):
    """限速只影响预览；等待结束后才取最新帧，保持录包与采集速率。"""
    next_frame = 0.0
    while True:
        time.sleep(max(0.0, next_frame - time.monotonic()))
        frame, stamp, received_at = latest.take()
        started = time.monotonic()
        try:
            overlay.process(convert(frame), stamp=stamp, received_at=received_at,
                            started=started)
        except Exception as exc:  # 单帧转换/检测失败不能中断预览。
            with LOCK:
                STATE["msg"] = f"process error: {exc}"
        next_frame = started + 1.0 / fps


def state_snapshot():
    """调用方持有 LOCK；帧停止更新时，帧龄也继续增长。"""
    state = dict(STATE)
    if JPEG["encoded_at"] is not None and state["age_ms"] >= 0:
        state["age_ms"] += int((time.monotonic() - JPEG["encoded_at"]) * 1000)
    return state


def _aruco_dict(family):
    """兼容 OpenCV 4.5 / 4.7+ 两套接口。"""
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, family))
    return cv2.aruco.Dictionary_get(getattr(cv2.aruco, family))


def _detect(gray, dictionary, params):
    if hasattr(cv2.aruco, "ArucoDetector"):                # OpenCV 4.7 及以上版本。
        det = cv2.aruco.ArucoDetector(dictionary, params)
        return det.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


class Overlay:
    """AprilGrid 检测 + 覆盖引导 + 状态标注，输出 JPEG。

    检测开销较大（aruco 全帧搜索）；默认每 DETECT_EVERY 帧才跑一次，
    中间帧复用上次检测框。预览只用于“板子在不在、够不够”，
    降频不影响判断，但能把 CPU 让给相机驱动与录包，避免丢帧。
    """

    DETECT_EVERY = 3

    def __init__(self, args):
        self.dictionary = _aruco_dict(args.dict_name) if DETECT_ON else None
        self.params = detector_parameters(args) if DETECT_ON else None
        self.min_tags = args.min_tags
        self.target_name = getattr(args, "target_name", "AprilGrid")
        self.n = 0
        self.tprev = None
        self.fps_avg = 0.0
        self.last_corners = None
        self.last_ids = None

    def process(self, frame, stamp=None, received_at=None, started=None, detection=None):
        started = started if started is not None else time.monotonic()
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        tags = 0
        if DETECT_ON:
            # 降频：只在每 DETECT_EVERY 帧跑一次检测，其余帧复用上次结果
            if detection is not None or self.n % self.DETECT_EVERY == 0:
                if detection is None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = _detect(gray, self.dictionary, self.params)
                else:
                    # 采集工具复用本帧检测结果，避免预览再次检测。
                    corners, ids = detection
                self.last_corners = corners if (ids is not None and len(ids) > 0) else None
                self.last_ids = ids if self.last_corners is not None else None
            if self.last_ids is not None and len(self.last_ids) > 0:
                tags = len(self.last_ids)
                cv2.aruco.drawDetectedMarkers(frame, self.last_corners, self.last_ids)

        hh, ww = frame.shape[:2]
        for i in range(1, 3):                      # 3x3 覆盖引导
            cv2.line(frame, (ww * i // 3, 0), (ww * i // 3, hh), (60, 60, 60), 1)
            cv2.line(frame, (0, hh * i // 3), (ww, hh * i // 3), (60, 60, 60), 1)

        now = time.monotonic()
        inst = 1.0 / max(now - self.tprev, 1e-6) if self.tprev is not None else 0.0
        self.tprev = now
        self.fps_avg = 0.9 * self.fps_avg + 0.1 * inst if self.fps_avg else inst
        self.n += 1

        color = (0, 220, 0) if tags >= self.min_tags else (0, 180, 255) if tags > 0 else (0, 0, 255)
        cv2.putText(frame, f"{self.target_name} tags: {tags}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"{self.fps_avg:4.1f} fps   {ww}x{hh}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        if tags < self.min_tags:
            cv2.putText(frame, f"move so {self.min_tags}+ tags are visible", (10, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)

        ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        encoded_at = time.monotonic()
        with FRAME_READY:
            JPEG["buf"] = enc.tobytes()
            JPEG["encoded_at"] = encoded_at
            STATE["frames"] = self.n
            STATE["tags"] = tags
            STATE["w"], STATE["h"] = ww, hh
            STATE["fps"] = round(self.fps_avg, 1)
            STATE["ok"] = True
            STATE["msg"] = "live"
            STATE["process_ms"] = round((encoded_at - started) * 1000, 1)
            if stamp is not None:
                STATE["age_ms"] = int(max(0.0, time.time() - stamp) * 1000)
            else:
                STATE["age_ms"] = int((encoded_at - (received_at or started)) * 1000)
            FRAME_READY.notify_all()


def v4l2_loop(dev, w, h, fps, args):
    """直接抓 V4L2（独立模式；独占相机）。"""
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        with LOCK:
            STATE.update(ok=False, msg=f"无法打开 {dev}（被占用？）", src="v4l2")
        return
    # 该 UVC 只支持 MJPG，必须显式请求，否则后端协商失败
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    with LOCK:
        STATE["src"] = "v4l2"
    ov = Overlay(args)
    latest = LatestFrame()

    def capture():
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            latest.put(frame)

    threading.Thread(target=capture, daemon=True).start()
    process_latest(latest, ov, args.preview_fps)


def ros_loop(topic, args):
    """订阅 ROS 话题（标定模式；与相机驱动/录包共享流）。"""
    import rospy
    from sensor_msgs.msg import Image as RosImage
    try:
        from cv_bridge import CvBridge
    except ImportError:
        CvBridge = None

    rospy.init_node("cam_preview", disable_signals=True)
    bridge = CvBridge() if CvBridge else None
    ov = Overlay(args)
    latest = LatestFrame()

    with LOCK:
        STATE["src"] = "ros"
        STATE["topic"] = topic

    def cb(msg):
        stamp = msg.header.stamp.to_sec()
        latest.put(msg, stamp=stamp if stamp else None)

    def convert(msg):
        if bridge is not None:
            return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # sensor_msgs/Image 是原始像素，不是 imdecode 接受的压缩图片。
        channels = {"mono8": 1, "bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}
        count = channels.get(msg.encoding)
        if count is None:
            raise ValueError(f"无 cv_bridge 时不支持编码 {msg.encoding}")
        rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        frame = rows[:, :msg.width * count].copy()
        if count == 1:
            return frame.reshape(msg.height, msg.width)
        frame = frame.reshape(msg.height, msg.width, count)
        if msg.encoding == "rgb8":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if msg.encoding == "rgba8":
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return frame

    rospy.Subscriber(topic, RosImage, cb, queue_size=1)
    thread = threading.Thread(target=rospy.spin, daemon=True)
    thread.start()
    process_latest(latest, ov, args.preview_fps, convert)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>相机预览</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:14px/1.5 system-ui,sans-serif;
      display:flex;flex-direction:column;align-items:center;gap:10px;padding:12px}
 img{max-width:100%;border:1px solid #333;border-radius:6px;background:#000}
 .bar{display:flex;gap:18px;flex-wrap:wrap;justify-content:center}
 b{color:#7CFC00}
 .hint{color:#888;font-size:12px;max-width:760px;text-align:center}
</style></head>
<body>
 <h3 style="margin:4px 0">{{TITLE}}</h3>
 <div class="bar">
   <span>来源 <b id="src">-</b></span>
   <span>帧率 <b id="fps">-</b></span>
   <span>检测到 tag <b id="tags">-</b></span>
   <span>分辨率 <b id="res">-</b></span>
   <span>帧数 <b id="frames">-</b></span>
   <span>处理耗时 <b id="process">-</b> ms</span>
   <span>显示帧龄（估算） <b id="age">-</b> ms</span>
 </div>
 {{EXTRA_STATUS}}
 <img id="preview" alt="等待最新画面">
 <div class="hint">{{DETECTION_HINT}}
   让板子先后出现在画面 9 个分区、并做三轴旋转，覆盖越广标定越稳。</div>
 <script>
  const f=document.getElementById.bind(document);
  const preview=f('preview');
  let previousUrl=null;
  async function fetchFresh(path,read){
    const controller=new AbortController();
    const timer=setTimeout(()=>controller.abort(),3000);
    try{
      const response=await fetch(path,{cache:'no-store',signal:controller.signal});
      if(!response.ok) throw new Error('等待相机');
      return await read(response);
    }finally{
      clearTimeout(timer);
    }
  }
  // 一次只请求一帧，下载和显示完成后再取最新帧；慢网络不会排队播放旧图。
  async function nextFrame(){
    const started=performance.now();
    let url=null;
    try{
      const frame=await fetchFresh('frame.jpg?t='+Date.now(),async response=>({
        age:Number(response.headers.get('X-Frame-Age-Ms')),blob:await response.blob()}));
      url=URL.createObjectURL(frame.blob);
      preview.src=url;
      await preview.decode();
      f('age').textContent=Math.round(frame.age+performance.now()-started);
      if(previousUrl) URL.revokeObjectURL(previousUrl);
      previousUrl=url;
      url=null;
    }catch(e){
      f('age').textContent='等待画面 / 连接超时';
    }finally{
      if(url) URL.revokeObjectURL(url);
      setTimeout(nextFrame,Math.max(20,{{FRAME_INTERVAL}}-(performance.now()-started)));
    }
  }
  async function tick(){
    try{
      const s=await fetchFresh('state.json',response=>response.json());
      f('src').textContent=s.src+(s.topic?(' '+s.topic):'');
      f('fps').textContent=s.fps; f('tags').textContent=s.tags;
      f('res').textContent=s.w+'x'+s.h; f('frames').textContent=s.frames;
      f('process').textContent=s.process_ms;
      {{EXTRA_SCRIPT}}
    }catch(e){}
    setTimeout(tick,1000);
  }
  tick();
  nextFrame();
 </script>
</body></html>"""


def render_page(min_tags, preview_fps, title="相机预览 · AprilGrid 联合标定",
                extra_status="", extra_script="", detection_hint=None):
    """共用最新帧请求页面；采集工具可补充进度状态。"""
    if detection_hint is None:
        detection_hint = (f"绿框=检测到的 AprilGrid 标记。<b>尽量保持至少 {min_tags} 个</b>"
                          "清晰可见（OpenCV 预览仅供参考）；")
    return (PAGE.replace("{{DETECTION_HINT}}", detection_hint)
            .replace("{{FRAME_INTERVAL}}", str(round(1000 / preview_fps)))
            .replace("{{TITLE}}", title)
            .replace("{{EXTRA_STATUS}}", extra_status)
            .replace("{{EXTRA_SCRIPT}}", extra_script))


class Handler(BaseHTTPRequestHandler):
    page = None

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/state.json"):
            with LOCK:
                body = json.dumps(state_snapshot()).encode()
            self._send(200, "application/json", body)
        elif self.path.startswith("/frame.jpg"):
            with LOCK:
                buf = JPEG["buf"]
                state = state_snapshot()
            if buf is None:
                self._send(503, "text/plain; charset=utf-8", b"Waiting for camera")
            else:
                self._send(200, "image/jpeg", buf,
                           {"X-Frame-Age-Ms": str(state["age_ms"]),
                            "X-Frame-Id": str(state["frames"])})
        elif self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=FRAME")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.connection.settimeout(3.0)
            last_frame = -1
            try:
                while True:
                    with FRAME_READY:
                        ready = FRAME_READY.wait_for(
                            lambda: JPEG["buf"] is not None and STATE["frames"] != last_frame,
                            timeout=1.0)
                        if not ready:
                            continue
                        buf = JPEG["buf"]
                        last_frame = STATE["frames"]
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(buf)).encode()
                                     + b"\r\n\r\n" + buf + b"\r\n")
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return
        else:
            self._send(200, "text/html; charset=utf-8",
                       (self.page if self.page is not None else PAGE).encode())

    def _send(self, code, ctype, body, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Accel-Buffering", "no")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.connection.settimeout(3.0)
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass


def _self_test():
    """自测：校验 _aruco_dict / _detect。"""
    import cv2
    from calibration_config import draw_marker

    d = _aruco_dict("DICT_APRILTAG_36h11")
    assert d is not None
    img = draw_marker(d, 0, 80, 1)
    p = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "DetectorParameters")
         else cv2.aruco.DetectorParameters_create())
    corners, ids, _ = _detect(img, d, p)
    print(f"[OK] _aruco_dict / _detect（ids={0 if ids is None else len(ids)}）")
    return 0


def main():
    global DETECT_ON
    import signal
    import sys

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--test", action="store_true")
    known, _ = probe.parse_known_args()
    if known.test:
        sys.exit(_self_test())

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["ros", "v4l2"],
                    help="ros=订阅话题（标定用，不占相机）；v4l2=直接抓设备")
    ap.add_argument("--topic")
    ap.add_argument("--device")
    ap.add_argument("--port", type=int)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=int)
    ap.add_argument("--preview-fps", type=int, default=15,
                    help="预览处理及网页请求帧率上限（默认 15，不改变相机/录包帧率）")
    ap.add_argument("--no-detect", action="store_true")
    ap.add_argument("--detect-every", type=int,
                    help="每 N 帧跑一次 AprilGrid 检测（降 CPU，默认 3）")
    target = aprilgrid_arguments(ap)
    ap.set_defaults(min_tags=max(target["tagRows"], target["tagCols"]) + 1)
    ap.add_argument("--dict", dest="dict_name")
    add_detector_arguments(ap)
    args = parse_tool_args(ap, "preview_server", bindings={"dict_name": "aprilgrid.dictionary"})
    if not args.no_detect and target["blackTagBorder"] != 1:
        ap.error("OpenCV 预览检测仅支持 blackTagBorder: 1，可用 --no-detect 仅看画面")
    Handler.page = render_page(args.min_tags, args.preview_fps)
    DETECT_ON = not args.no_detect
    Overlay.DETECT_EVERY = max(1, args.detect_every)

    if args.source == "ros":
        target = lambda: ros_loop(args.topic, args)
    else:
        target = lambda: v4l2_loop(args.device, args.width, args.height, args.fps, args)
    threading.Thread(target=target, daemon=True).start()

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.daemon_threads = True

    # rospy 会安装 SIGTERM 处理器，但它只置“关闭请求”标志，
    # 而主线程阻塞在 serve_forever() 里永不检查 —— 结果 SIGTERM 杀不死本进程，
    # 端口被占着，下次启动报告错误码 98（地址已被占用）。所以这里显式装自己的处理器。
    def _stop(signum, _frame):
        print(f"[preview] signal {signum}, shutting down", flush=True)
        threading.Thread(target=srv.shutdown, daemon=True).start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)

    print(f"[preview] http://127.0.0.1:{args.port}/  "
          f"source={args.source} detect={DETECT_ON} every={Overlay.DETECT_EVERY}",
          flush=True)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
        print("[preview] stopped", flush=True)


if __name__ == "__main__":
    main()
