#!/usr/bin/env python3
# ref_call_detector.py
# Simple prototype: detect people + pose + heuristics => suggested ref calls
# Requirements:
#   pip install ultralytics mediapipe opencv-python numpy
#
# Usage:
#   python ref_call_detector.py input_video.mp4

import sys
import time
import json
from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp

VIDEO = sys.argv[1] if len(sys.argv) > 1 else 0  # path or webcam index

# Initialize models
yolo = YOLO("yolov8n.pt")  # small model for prototyping; replace with custom-trained for better accuracy
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_idx = 0

events = []

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    if boxAArea+boxBArea-interArea == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

# Very simple tracker: carry previous boxes and match by IoU (for prototype only)
prev_people = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    tstart = time.time()
    results = yolo.predict(source=frame, imgsz=640, conf=0.35, max_det=50, verbose=False)
    # results is a list; single image => results[0]
    r = results[0]
    boxes = []
    labels = []
    for det in r.boxes:
        cls = int(det.cls.cpu().numpy())
        conf = float(det.conf.cpu().numpy())
        xyxy = det.xyxy.cpu().numpy().tolist()
        # class 0 is usually 'person' in COCO; adjust per model
        boxes.append((xyxy, cls, conf))

    # find person bboxes
    person_boxes = []
    ball_boxes = []
    for xyxy, cls, conf in boxes:
        x1, y1, x2, y2 = map(int, xyxy)
        if cls == 0:
            person_boxes.append((x1, y1, x2, y2, conf))
        else:
            # naive: treat small round objects as ball candidates
            w = x2-x1; h = y2-y1
            if min(w,h) < 80 and conf>0.4:
                ball_boxes.append((x1,y1,x2,y2,conf))

    # Pose estimation for each person
    person_poses = []
    for (x1,y1,x2,y2,conf) in person_boxes:
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results_pose = pose_detector.process(crop_rgb)
        keypoints = None
        if results_pose.pose_landmarks:
            # Convert to full-frame coordinates
            kp = []
            h_crop, w_crop = crop.shape[:2]
            for lm in results_pose.pose_landmarks.landmark:
                kp_x = x1 + int(lm.x * w_crop)
                kp_y = y1 + int(lm.y * h_crop)
                kp_vis = lm.visibility
                kp.append((kp_x, kp_y, kp_vis))
            keypoints = kp
        person_poses.append({'bbox':(x1,y1,x2,y2),'conf':conf,'kp':keypoints})

    # Very simple collision/contact heuristic:
    # If two persons' bboxes overlap or are very close and relative motion is high (approx via bbox displacement), suggest "possible contact"
    suggested = []
    # match current to prev by IoU
    mapped_prev = []
    for p in person_poses:
        best_iou = 0; best_idx=-1
        for i,pp in enumerate(prev_people):
            iou_v = iou(p['bbox'], pp['bbox'])
            if iou_v > best_iou:
                best_iou = iou_v; best_idx=i
        mapped_prev.append(best_idx)

    # detect proximity pairs
    for i in range(len(person_poses)):
        for j in range(i+1, len(person_poses)):
            a = person_poses[i]['bbox']; b = person_poses[j]['bbox']
            xa1,ya1,xa2,ya2 = a; xb1,yb1,xb2,yb2 = b
            center_a = ((xa1+xa2)/2,(ya1+ya2)/2)
            center_b = ((xb1+xb2)/2,(yb1+yb2)/2)
            dist = np.hypot(center_a[0]-center_b[0], center_a[1]-center_b[1])
            avg_w = ((xa2-xa1)+(xb2-xb1))/2
            if dist < avg_w * 1.2:  # close threshold
                # estimate motion: if either matched to prev and displacement is large -> contact-like
                idx_a = mapped_prev[i]; idx_b = mapped_prev[j]
                motion_high = False
                if idx_a != -1 and idx_b != -1:
                    prev_a = prev_people[idx_a]; prev_b = prev_people[idx_b]
                    ca = ((a[0]+a[2])/2,(a[1]+a[3])/2)
                    pca = ((prev_a['bbox'][0]+prev_a['bbox'][2])/2,(prev_a['bbox'][1]+prev_a['bbox'][3])/2)
                    cb = ((b[0]+b[2])/2,(b[1]+b[3])/2)
                    pcb = ((prev_b['bbox'][0]+prev_b['bbox'][2])/2,(prev_b['bbox'][1]+prev_b['bbox'][3])/2)
                    disp_a = np.hypot(ca[0]-pca[0], ca[1]-pca[1])
                    disp_b = np.hypot(cb[0]-pcb[0], cb[1]-pcb[1])
                    if disp_a > 20 or disp_b > 20:
                        motion_high = True
                confidence = 0.5 + (0.25 if motion_high else 0.0)
                timestamp = frame_idx / fps
                suggested.append({
                    'type':'possible_contact',
                    'players': [i,j],
                    'timestamp': timestamp,
                    'confidence': round(confidence,2),
                    'bbox_a':a, 'bbox_b':b
                })

    # Append events and keep a short buffer
    for s in suggested:
        # dedupe by time proximity
        if len(events)==0 or abs(events[-1]['timestamp'] - s['timestamp']) > 0.5:
            events.append(s)
            print(f"[{s['timestamp']:.2f}s] Suggested: {s['type']} conf={s['confidence']} players={s['players']}")

    # update prev_people
    prev_people = []
    for p in person_poses:
        prev_people.append({'bbox':p['bbox'],'kp':p['kp']})

    # small display (optional)
    for p in person_poses:
        x1,y1,x2,y2 = p['bbox']
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
    for b in ball_boxes:
        x1,y1,x2,y2,_ = b
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow("proto", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# dump events
print("Detected events (JSON):")
print(json.dumps(events, indent=2))
