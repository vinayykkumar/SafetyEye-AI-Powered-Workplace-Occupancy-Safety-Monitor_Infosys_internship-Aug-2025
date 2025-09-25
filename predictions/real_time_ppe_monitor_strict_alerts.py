"""
Strict PPE monitor with robust alerts (logger + optional webhook).

Features:
- YOLOv8 realtime inference
- Per-person tracking (centroid tracker)
- Helmet association: center-point inside expanded head area (better for small helmet boxes)
- Vest association: IoU with person box
- Per-person counters -> violation only after N consecutive frames
- Save violation snapshots and review snapshots
- Rotating file logger (local persistent log)
- Optional webhook (Slack/Teams/any incoming webhook) for realtime notifications

Usage (PowerShell single line):
python predictions/real_time_ppe_monitor_strict_alerts.py --weights "runs/detect/yolov8n_ppe_gpu_run_2/weights/best.pt" --source "test_images/test_video3.mp4" --no_ppe_frames_n 10 --ppe_conf_min 0.45 --ppe_low_conf 0.20 --min_person_area_ratio 0.008 --iou_threshold 0.25 --head_expand 0.40 --alert_cooldown 12.0 --webhook_url "https://hooks.slack.com/services/XXX/YYY/ZZZ"
"""

import argparse
import time
import os
import csv
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import traceback

import cv2
import numpy as np
import requests
from ultralytics import YOLO

# -------------------------
# Utility functions
# -------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interA = interW * interH
    if interA == 0: return 0.0
    aA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    aB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    return interA / float(aA + aB - interA)

def center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)//2, (y1+y2)//2)

def point_in_box(pt, box):
    x,y = pt
    x1,y1,x2,y2 = box
    return (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_snapshot(frame, tid, frame_no, reason, outpath, annotate_box=None):
    ensure_dir(os.path.dirname(outpath))
    vis = frame.copy()
    if annotate_box is not None:
        x1,y1,x2,y2 = annotate_box
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 2)
    cv2.putText(vis, f"ID:{tid} {reason}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv2.imwrite(outpath, vis)

def append_csv(path, row, header=None):
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists and header:
            w.writerow(header)
        w.writerow(row)

# -------------------------
# Simple centroid tracker
# -------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=100):
        self.next_id = 1
        self.objects = {}        # id -> (cx,cy)
        self.disappeared = {}    # id -> frames since last seen
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, centroids):
        # centroids: list of (cx,cy)
        if len(self.objects) == 0:
            for c in centroids:
                self.objects[self.next_id] = tuple(c)
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return list(self.objects.items())

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        if len(centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    del self.disappeared[oid]
                    if oid in self.objects:
                        del self.objects[oid]
            return list(self.objects.items())

        D = np.linalg.norm(np.array(object_centroids)[:,None,:] - np.array(centroids)[None,:,:], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows, assigned_cols = set(), set()
        new_objects = {}

        for r,c in zip(rows,cols):
            if r in assigned_rows or c in assigned_cols:
                continue
            if D[r,c] > self.max_distance:
                continue
            oid = object_ids[r]
            new_objects[oid] = tuple(centroids[c])
            assigned_rows.add(r); assigned_cols.add(c)

        # increase disappeared for unassigned existing
        for i, oid in enumerate(object_ids):
            if i not in rows or object_ids[i] not in new_objects:
                self.disappeared[oid] += 1

        # remove those disappeared too long
        for oid in list(self.disappeared.keys()):
            if self.disappeared[oid] > self.max_disappeared:
                del self.disappeared[oid]
                if oid in self.objects: del self.objects[oid]
                if oid in new_objects: del new_objects[oid]

        # add new centroids as new objects
        for i,c in enumerate(centroids):
            if i not in assigned_cols:
                new_objects[self.next_id] = tuple(c)
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        for oid in new_objects:
            self.disappeared[oid] = 0

        self.objects = new_objects.copy()
        return list(self.objects.items())

# -------------------------
# Alerting helpers
# -------------------------
def configure_logger(log_path, max_bytes=2_000_000, backup_count=3):
    ensure_dir(os.path.dirname(log_path))
    logger = logging.getLogger("ppe_alerts")
    logger.setLevel(logging.INFO)
    # If handlers exist (repeated imports), avoid adding duplicates
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        # Also add console handler
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

def send_webhook(webhook_url, text, image_path=None, timeout=3.0):
    """
    Simple POST to webhook. For Slack/Teams incoming webhooks, payload={"text": text} works.
    If the webhook expects a different payload, change this function accordingly.
    """
    if not webhook_url:
        return False
    payload = {"text": text}
    # If you want to upload the image to an external host, you'd need another step.
    try:
        r = requests.post(webhook_url, json=payload, timeout=timeout)
        return 200 <= r.status_code < 300
    except Exception as e:
        # don't crash on webhook failure
        return False

# -------------------------
# Main
# -------------------------
def main(args):
    logger = configure_logger(args.log_path, args.log_max_bytes, args.log_backup_count)
    try:
        model = YOLO(args.weights)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    logger.info(f"Model classes: {model.names}")
    ensure_dir(args.output_dir); ensure_dir(args.review_dir)
    violations_csv = os.path.join(args.output_dir, "violations_log.csv")
    review_csv = os.path.join(args.review_dir, "review_log.csv")
    append_csv(violations_csv, [], header=["timestamp","frame","person_id","reason","image_path","helmet_conf","vest_conf"])
    append_csv(review_csv, [], header=["timestamp","frame","person_id","reason","image_path","helmet_conf","vest_conf"])

    cap = cv2.VideoCapture(args.source if not str(args.source).isdigit() else int(args.source))
    if not cap.isOpened():
        logger.error("ERROR: cannot open source %s", args.source)
        return

    ct = CentroidTracker(max_disappeared=args.max_disappeared, max_distance=args.max_distance)

    # per-person counters
    no_helmet_counter = {}
    no_vest_counter = {}
    last_alert_time = {}

    FRAME = 0; start = time.time()
    logger.info("Starting strict PPE monitor... press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Stream ended / can't get frame.")
            break
        FRAME += 1
        FH, FW = frame.shape[:2]; frame_area = FH * FW

        results = model(frame, conf=args.conf, verbose=False)
        res = results[0]
        boxes=[]; classes=[]; confs=[]
        if res.boxes is not None and len(res.boxes)>0:
            for b in res.boxes:
                try:
                    x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                    cls = int(b.cls[0]); conf = float(b.conf[0])
                except Exception:
                    arr = b.xyxy; x1,y1,x2,y2 = map(int, arr[0]); cls=int(b.cls); conf=float(b.conf)
                boxes.append([x1,y1,x2,y2]); classes.append(cls); confs.append(conf)

        # find indices for classes
        worker_idx = [i for i,c in enumerate(classes) if model.names[c].lower() in ("worker","person","people")]
        helmet_idx = [i for i,c in enumerate(classes) if "hardhat" in model.names[c].lower() or "helmet" in model.names[c].lower()]
        vest_idx   = [i for i,c in enumerate(classes) if "vest" in model.names[c].lower()]

        centroids=[]; person_boxes=[]; person_confs=[]
        for idx in worker_idx:
            box = boxes[idx]; centroids.append(center(box)); person_boxes.append(box); person_confs.append(confs[idx])

        tracked = ct.update(centroids)
        id_map = {}
        for tid, centroid in tracked:
            if len(centroids) == 0: continue
            dists = [np.linalg.norm(np.array(centroid)-np.array(c)) for c in centroids]
            idx_near = int(np.argmin(dists))
            id_map[tid] = (person_boxes[idx_near], person_confs[idx_near])

        now = time.time()
        for tid, (pbox, pconf) in id_map.items():
            # ignore weak person detections
            if pconf < args.person_conf_min:
                no_helmet_counter[tid] = 0; no_vest_counter[tid] = 0; continue

            # ignore tiny persons (far away)
            x1,y1,x2,y2 = pbox
            area = max(0, x2-x1) * max(0, y2-y1)
            if area < args.min_person_area_ratio * frame_area:
                no_helmet_counter[tid] = 0; no_vest_counter[tid] = 0; continue

            # Prepare expanded head box for center-point matching
            ph = max(1, y2 - y1)
            head_expand = args.head_expand  # fraction to expand upwards relative to person height
            expanded_pbox = [x1, max(0, int(y1 - head_expand * ph)), x2, y2]

            # find best helmet by center-point in expanded head area
            best_h_conf = 0.0
            for hi in helmet_idx:
                hbox = boxes[hi]; hconf = confs[hi]
                hcenter = center(hbox)  # helmet center
                if point_in_box(hcenter, expanded_pbox) and hconf > best_h_conf:
                    best_h_conf = hconf

            # find best vest using IoU (vest tends to occupy torso area)
            best_v_conf = 0.0
            for vi in vest_idx:
                vbox = boxes[vi]; vconf = confs[vi]
                if iou(pbox, vbox) > args.iou_threshold and vconf > best_v_conf:
                    best_v_conf = vconf

            helmet_present = best_h_conf >= args.ppe_conf_min
            vest_present   = best_v_conf >= args.ppe_conf_min
            helmet_low = (0 < best_h_conf < args.ppe_low_conf)
            vest_low   = (0 < best_v_conf < args.ppe_low_conf)

            # low-confidence -> save to review and reset counters (no auto-violate)
            if helmet_low or vest_low:
                ts = time.strftime("%Y%m%d_%H%M%S")
                out = os.path.join(args.review_dir, f"review_{ts}_f{FRAME}_id{tid}.jpg")
                save_snapshot(frame, tid, FRAME, "low_conf_ppe", out, annotate_box=pbox)
                append_csv(review_csv, [time.strftime("%Y-%m-%d %H:%M:%S"), FRAME, tid, "low_conf_ppe", out, round(best_h_conf,3), round(best_v_conf,3)])
                logger.info(f"Saved low_conf_ppe review: id={tid} h={best_h_conf:.2f} v={best_v_conf:.2f} -> {out}")
                no_helmet_counter[tid] = 0; no_vest_counter[tid] = 0
                continue

            # update counters independently
            if not helmet_present:
                no_helmet_counter[tid] = no_helmet_counter.get(tid, 0) + 1
            else:
                no_helmet_counter[tid] = 0

            if not vest_present:
                no_vest_counter[tid] = no_vest_counter.get(tid, 0) + 1
            else:
                no_vest_counter[tid] = 0

            # decide violation if either reaches threshold
            reasons = []
            violated = False
            if no_helmet_counter.get(tid, 0) >= args.no_ppe_frames_n:
                reasons.append("NO_HARDHAT"); violated = True
            if no_vest_counter.get(tid, 0) >= args.no_ppe_frames_n:
                reasons.append("NO_VEST"); violated = True

            if violated:
                last = last_alert_time.get(tid, 0)
                if (now - last) > args.alert_cooldown:
                    reason_str = "+".join(reasons)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out = os.path.join(args.output_dir, f"violation_{ts}_f{FRAME}_id{tid}.jpg")
                    save_snapshot(frame, tid, FRAME, reason_str, out, annotate_box=pbox)
                    append_csv(violations_csv, [time.strftime("%Y-%m-%d %H:%M:%S"), FRAME, tid, reason_str, out, round(best_h_conf,3), round(best_v_conf,3)])
                    msg = f"PPE VIOLATION: {reason_str} | id={tid} frame={FRAME} helmet={best_h_conf:.2f} vest={best_v_conf:.2f} | {out}"
                    logger.info(msg)
                    # webhook (best-effort)
                    if args.webhook_url:
                        ok = send_webhook(args.webhook_url, msg)
                        if ok:
                            logger.info("Webhook delivered for id %s", tid)
                        else:
                            logger.warning("Webhook delivery failed for id %s", tid)
                    last_alert_time[tid] = now
                    # reset counters for raised reasons
                    if "NO_HARDHAT" in reasons: no_helmet_counter[tid] = 0
                    if "NO_VEST" in reasons: no_vest_counter[tid] = 0

        # Visualization
        # draw all detection boxes (workers + PPE) and color worker boxes based on PPE overlap
        for i, box in enumerate(boxes):
            cls = classes[i]; label = model.names[cls]; conf = confs[i]
            color = (0,255,0)  # default
            if i in worker_idx:
                this_box = box
                # compute expanded headbox for this worker to visualize helmet check
                # find nearest worker index to this_box to compute an expanded box for display
                overlap_h = False
                overlap_v = False
                # check helmet centers
                for hi in helmet_idx:
                    hcenter = center(boxes[hi])
                    # approximate expanded head region for this worker's box
                    x1,y1,x2,y2 = this_box
                    ph = max(1, y2-y1)
                    expanded_head = [x1, max(0, int(y1 - args.head_expand*ph)), x2, y2]
                    if point_in_box(hcenter, expanded_head) and confs[hi] >= args.ppe_conf_min:
                        overlap_h = True
                # check vest IoU
                for vi in vest_idx:
                    if iou(this_box, boxes[vi]) > args.iou_threshold and confs[vi] >= args.ppe_conf_min:
                        overlap_v = True

                if overlap_h and overlap_v:
                    color = (0,255,0)        # green = both PPE present
                elif overlap_h or overlap_v:
                    color = (0,165,255)      # orange = one PPE present (other missing)
                else:
                    color = (0,0,255)        # red = missing both

            x1,y1,x2,y2 = box
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # always draw PPE detections clearly
        for hi in helmet_idx:
            x1,y1,x2,y2 = boxes[hi]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"helmet {confs[hi]:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        for vi in vest_idx:
            x1,y1,x2,y2 = boxes[vi]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(frame, f"vest {confs[vi]:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        elapsed = time.time() - start
        fps = FRAME / elapsed if elapsed>0 else 0.0
        cv2.putText(frame, f"FPS:{fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("PPE Monitor (strict)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()
    logger.info("Finished.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="runs/detect/yolov8n_ppe_gpu_run_2/weights/best.pt")
    p.add_argument("--source", type=str, default="0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--person_conf_min", type=float, default=0.4)
    p.add_argument("--ppe_conf_min", type=float, default=0.45)
    p.add_argument("--ppe_low_conf", type=float, default=0.20)
    p.add_argument("--iou_threshold", type=float, default=0.25)
    p.add_argument("--head_expand", type=float, default=0.40, help="how much to expand head region upwards as fraction of person height")
    p.add_argument("--no_ppe_frames_n", type=int, default=10, help="consecutive frames missing before flagging (per class)")
    p.add_argument("--alert_cooldown", type=float, default=8.0)
    p.add_argument("--output_dir", type=str, default="violations")
    p.add_argument("--review_dir", type=str, default="review")
    p.add_argument("--max_disappeared", type=int, default=15)
    p.add_argument("--max_distance", type=int, default=100)
    p.add_argument("--min_person_area_ratio", type=float, default=0.003)
    p.add_argument("--webhook_url", type=str, default=None, help="optional incoming webhook URL for alerts (Slack/Teams)")
    p.add_argument("--log_path", type=str, default="alerts.log")
    p.add_argument("--log_max_bytes", type=int, default=2_000_000)
    p.add_argument("--log_backup_count", type=int, default=3)
    args = p.parse_args()
    try:
        if str(args.source).isdigit(): args.source = int(args.source)
    except: pass
    main(args)
