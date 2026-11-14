#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
capture_calib_ros.py —— 从 /camera/image_raw 话题采集标定图（与相机驱动共享图像）。

检测 AprilTag 标记，按 3x3 覆盖网格引导（左上/上/右上/左/中/右/左下/下/右下），
每个格子收集 min-gap 张清晰帧后进入下一格。集满 9 格自动退出。
默认提供与 preview_server 相同的浏览器预览；目标格由操作者按画面引导对齐。

用法:
  python3 capture_calib_ros.py --out images --min-gap 8 --port 8099
  本机浏览器: http://127.0.0.1:8099/
  远程预览: 在电脑运行 ssh -N -L 8099:127.0.0.1:8099 用户@机载电脑IP
  --no-preview 可关闭网页预览。
  --mode manual 在预览网页按空格或点击拍照；默认 auto 定时采集。
  终端实时显示: [格子(行,列)] 已存 N 张 | 总 M 张
  按 Ctrl+C 提前结束（已存图仍有效）。
"""
import argparse
import json
import os
import threading
import time
from http.server import ThreadingHTTPServer

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from calibration_config import add_detector_arguments, parse_tool_args, detector_parameters
import preview_server as preview


CELLS = [(r, c) for r in range(3) for c in range(3)]
CAPTURE_STATUS = """<div class="bar">
 <span>目标格 <b id="cell">-</b></span>
 <span>本格已存 <b id="cell-count">0</b></span>
 <span>总计 <b id="saved">0</b></span>
 <span>清晰度 <b id="sharpness">-</b></span>
 <span><span id="feature-name">有效特征</span> <b id="features">0</b></span>
 <span id="capture-hint">等待相机</span>
</div>
<div class="bar">
 <button id="capture-button" type="button" hidden>拍照（空格）</button>
 <span id="capture-result" role="status"></span>
</div>"""
MANUAL_SCRIPT = """<script>
document.addEventListener('DOMContentLoaded',()=>{
  const button=document.getElementById('capture-button');
  const result=document.getElementById('capture-result');
  let busy=false;
  async function capture(){
    if(button.hidden || button.disabled || busy) return;
    busy=true;
    result.textContent='正在拍照…';
    try{
      const response=await fetch('capture',{method:'POST'});
      const data=await response.json();
      result.textContent=data.message;
    }catch(e){
      result.textContent='连接中断，请检查已存张数后再试';
    }finally{busy=false;}
  }
  button.addEventListener('click',capture);
  document.addEventListener('keydown',event=>{
    if(event.code!=='Space' || button.hidden ||
       event.target.matches('input,textarea,select,[contenteditable="true"]')) return;
    event.preventDefault();
    if(!event.repeat) capture();
  });
});
</script>"""
CAPTURE_SCRIPT = """
      if(s.capture){
        const c=s.capture;
        f('cell').textContent=c.cell;
        f('cell-count').textContent=c.cell_saved+'/'+c.per_cell;
        f('saved').textContent=c.saved+'/'+c.target;
        f('sharpness').textContent=c.sharpness+'/'+c.min_sharpness;
        f('feature-name').textContent=c.feature_name;
        f('features').textContent=c.features+'/'+c.min_features;
        f('capture-hint').textContent=c.hint;
        f('capture-button').hidden=c.mode!=='manual';
        f('capture-button').disabled=c.saved>=c.target;
      }
"""


class CaptureTrigger:
    """最多一个待处理请求；超时或质量不合格不会留下延迟拍照请求。"""

    def __init__(self, timeout=2.0):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.pending = None
        self.closed = False

    def request(self):
        with self.lock:
            if self.closed:
                return 409, "采集已结束"
            if self.pending is not None:
                return 409, "正在处理上一张，请稍后再拍"
            request = {"at": time.monotonic(), "event": threading.Event(), "result": None}
            self.pending = request
        request["event"].wait(self.timeout)
        with self.lock:
            if request["result"] is None:
                request["result"] = (503, "未收到可处理的新帧，请检查相机后重新拍照")
                if self.pending is request:
                    self.pending = None
            return request["result"]

    def requested(self):
        with self.lock:
            return self.pending is not None

    def process(self, received_at, save):
        with self.lock:
            request = self.pending
            if request is None or received_at < request["at"]:
                return
            if time.monotonic() - request["at"] >= self.timeout:
                result = (503, "拍照请求已超时，请重新拍照")
            else:
                try:
                    result = save()
                except Exception as exc:
                    result = (500, f"保存失败: {exc}")
            request["result"] = result
            self.pending = None
            request["event"].set()

    def close(self):
        with self.lock:
            self.closed = True
            if self.pending is not None:
                self.pending["result"] = (409, "采集已结束")
                self.pending["event"].set()
                self.pending = None


class AprilGridDetector:
    name = "AprilGrid"
    feature_name = "标签"
    dictionary_setting = "aprilgrid.dictionary"
    extension = "jpg"

    @staticmethod
    def add_arguments(parser):
        pass

    def __init__(self, args):
        self.dictionary = preview._aruco_dict(args.dict)
        self.params = detector_parameters(args)
        self.min_markers = args.min_corners

    def detect(self, img):
        corners, ids, _ = preview._detect(img, self.dictionary, self.params)
        return corners, ids

    def count(self, corners, ids):
        return 0 if ids is None else len(ids)

    def annotate(self, display):
        pass


class CapturePreview:
    def __init__(self, args, trigger, detector):
        self.args = args
        self.detector = detector
        overlay_args = argparse.Namespace(**vars(args))
        overlay_args.dict_name = args.dict
        overlay_args.min_tags = detector.min_markers
        overlay_args.target_name = detector.name
        self.overlay = preview.Overlay(overlay_args)

        class Handler(preview.Handler):
            page = preview.render_page(
                detector.min_markers, args.preview_fps, f"相机内参采集 · {detector.name}",
                CAPTURE_STATUS + MANUAL_SCRIPT, CAPTURE_SCRIPT,
                detection_hint=f"绿框=检测到的标记。保存需至少 {args.min_corners} 个"
                               f"{detector.feature_name}，请以有效特征计数和采集提示为准；")

            def do_POST(self):
                if self.path != "/capture":
                    code, message = 404, "接口不存在"
                elif args.mode != "manual":
                    code, message = 409, "当前为 auto 模式，请在配置中选择 manual 后重启"
                else:
                    code, message = trigger.request()
                self._send(code, "application/json; charset=utf-8",
                           json.dumps({"ok": code == 200, "message": message},
                                      ensure_ascii=False).encode())

        self.server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        with preview.LOCK:
            preview.STATE.update(src="ros", topic=args.topic)
        self.thread.start()
        print(f"[preview] http://127.0.0.1:{args.port}/  "
              f"远程转发: ssh -N -L {args.port}:127.0.0.1:{args.port} 用户@机载电脑IP",
              flush=True)

    def update(self, img, corners, ids, sharp, cell_idx, per_cell, total,
               elapsed, stamp, received_at):
        args = self.args
        features = self.detector.count(corners, ids)
        done = cell_idx == len(CELLS)
        cell = CELLS[min(cell_idx, len(CELLS) - 1)]
        if done:
            hint = "采集完成"
        elif args.mode == "auto" and elapsed < args.cell_sec:
            hint = f"移动到目标格，等待 {args.cell_sec - elapsed:.1f} 秒"
        elif features < args.min_corners:
            hint = f"请让至少 {args.min_corners} 个{self.detector.feature_name}清晰可见"
        elif sharp < args.min_sharp:
            hint = "清晰度不足，请保持平稳并检查对焦"
        elif args.mode == "manual":
            hint = "移动到目标格并停稳，按空格或点击拍照"
        else:
            hint = "满足采集条件，请保持姿态并等待保存间隔"
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.detector.annotate(display)
        h, w = img.shape[:2]
        for r, c in CELLS:
            color = (0, 220, 0) if per_cell[(r, c)] >= args.min_gap else (0, 180, 255)
            cv2.putText(display, f"{per_cell[(r, c)]}/{args.min_gap}",
                        (w * c // 3 + 8, h * (r + 1) // 3 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if not done:
            r, c = cell
            cv2.rectangle(display, (w * c // 3 + 2, h * r // 3 + 2),
                          (w * (c + 1) // 3 - 2, h * (r + 1) // 3 - 2), (0, 180, 255), 2)
        with preview.LOCK:
            preview.STATE["capture"] = {
                "cell": "完成" if done else f"({cell[0] + 1},{cell[1] + 1})",
                "cell_saved": per_cell[cell], "per_cell": args.min_gap,
                "saved": total, "target": len(CELLS) * args.min_gap,
                "sharpness": round(sharp, 1), "min_sharpness": args.min_sharp,
                "hint": hint,
                "mode": args.mode,
                "features": features, "min_features": args.min_corners,
                "feature_name": self.detector.feature_name,
            }
        self.overlay.process(display, stamp=stamp, received_at=received_at,
                             detection=(corners, ids))

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


class Grab:
    def __init__(self, topic):
        self.bridge = CvBridge()
        self.frame = None
        self.sub = rospy.Subscriber(topic, Image, self.cb, queue_size=1)

    def cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "mono8")
            stamp = msg.header.stamp.to_sec()
            self.frame = (img, stamp if stamp else None, time.monotonic())
        except Exception as exc:
            rospy.logwarn(f"图像转换失败: {exc}")


def main(tool="capture_calib_ros", detector_type=AprilGridDetector):
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic")
    ap.add_argument("--out")
    ap.add_argument("--mode", choices=["auto", "manual"], default="auto",
                    help="auto 定时保存；manual 在预览网页按空格或点击拍照")
    ap.add_argument("--dict")
    ap.add_argument("--min-corners", type=int)
    ap.add_argument("--min-sharp", type=float)
    ap.add_argument("--min-gap", type=int)
    ap.add_argument("--cell-sec", type=float,
                    help="每格最短用时（防止同一位置连拍）")
    ap.add_argument("--save-interval", type=float)
    ap.add_argument("--port", type=int, help="网页预览端口（默认 8099）")
    ap.add_argument("--preview-fps", type=int, help="网页预览帧率上限（默认 15）")
    ap.add_argument("--no-preview", action="store_true", help="关闭网页，仅进行采集")
    detector_type.add_arguments(ap)
    add_detector_arguments(ap)
    args = parse_tool_args(ap, tool, bindings={
        "out": "paths.images", "dict": detector_type.dictionary_setting})
    if args.min_gap < 1:
        ap.error("--min-gap 在此工具中表示每格张数，至少为 1")
    if args.mode == "manual" and args.no_preview:
        ap.error("manual 模式需要网页拍照接口，不能使用 --no-preview")

    os.makedirs(args.out, exist_ok=True)
    detector = detector_type(args)

    per_cell = {c: 0 for c in CELLS}
    total = 0
    cell_idx = 0
    cell_t0 = time.monotonic()
    last_save = 0.0
    next_preview = 0.0
    web = None
    node = None
    trigger = CaptureTrigger()

    print(f"[capture] {detector.name} 9 格覆盖采集开始，输出 {args.out}/", flush=True)
    print(f"[capture] 模式 {args.mode}，每格需 {args.min_gap} 张。"
          "格子顺序: 左上->上->右上->左->中->右->左下->下->右下", flush=True)
    if args.mode == "manual":
        print("[capture] 在预览网页按空格或点击拍照；不合格时需重新按键。", flush=True)

    rospy.init_node("calib_capture")
    try:
        node = Grab(args.topic)
        if not args.no_preview:
            try:
                web = CapturePreview(args, trigger, detector)
            except OSError as exc:
                raise SystemExit(f"无法启动预览端口 {args.port}: {exc}；"
                                 "请更换 --port 或使用 --no-preview") from exc
        while not rospy.is_shutdown() and cell_idx < len(CELLS):
            rospy.sleep(0.1)
            sample = node.frame
            node.frame = None  # 每条 ROS 消息只处理一次，停流时不能重复保存旧帧。
            if sample is None:
                continue
            img, stamp, received_at = sample

            r, c = CELLS[cell_idx]
            now = time.monotonic()
            elapsed = now - cell_t0
            print(f"\r[格子({r+1},{c+1})] {per_cell[(r,c)]}/{args.min_gap} 张 | 总 {total} | "
                  f"{elapsed:.0f}s   ", end="", flush=True)

            save_due = (trigger.requested() if args.mode == "manual" else
                        elapsed >= args.cell_sec and now - last_save >= args.save_interval)
            preview_due = web is not None and now >= next_preview
            if not save_due and not preview_due:
                continue

            corners, ids = detector.detect(img)
            sharp = sharpness(img)

            def save_frame():
                nonlocal total, last_save, cell_idx, cell_t0, elapsed
                if detector.count(corners, ids) < args.min_corners:
                    return 422, (f"未保存：需至少 {args.min_corners} 个{detector.feature_name}，"
                                 "请调整后重新拍照")
                if sharp < args.min_sharp:
                    return 422, f"未保存：清晰度 {sharp:.1f} < {args.min_sharp}，请停稳后重新拍照"
                fn = os.path.join(args.out, f"calib_r{r}c{c}_{per_cell[(r,c)]:02d}.{detector.extension}")
                if not cv2.imwrite(fn, img):
                    raise OSError(f"无法保存采集图像: {fn}")
                per_cell[(r, c)] += 1
                total += 1
                last_save = time.monotonic()

                if per_cell[(r, c)] >= args.min_gap:
                    print(f"\n[格子({r+1},{c+1})] 完成 {per_cell[(r,c)]} 张", flush=True)
                    cell_idx += 1
                    cell_t0 = time.monotonic()
                    elapsed = 0.0
                return 200, f"已保存 {os.path.basename(fn)}（总计 {total} 张）"

            if save_due:
                if args.mode == "manual":
                    trigger.process(received_at, save_frame)
                else:
                    save_frame()

            if preview_due:
                web.update(img, corners, ids, sharp, cell_idx, per_cell, total,
                           elapsed, stamp, received_at)
                next_preview = now + 1.0 / args.preview_fps
    except KeyboardInterrupt:
        print("\n[capture] 手动结束", flush=True)
    finally:
        trigger.close()
        if web is not None:
            web.close()

    print(f"\n[capture] 总计 {total} 张，分布: "
          + " ".join(f"({r+1},{c+1})={n}" for (r, c), n in per_cell.items()), flush=True)
    ok = all(n >= args.min_gap for n in per_cell.values())
    print(f"[capture] 9 格覆盖: {'完成 ✓' if ok else '未集满（可用已存图先跑标定）'}", flush=True)


if __name__ == "__main__":
    main()
