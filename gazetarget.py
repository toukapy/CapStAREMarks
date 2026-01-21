import cv2
import torch
import numpy as np
from collections import OrderedDict, deque
import time
import random
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from models.gazev2_3d import FrozenEncoder, GazeEstimationModel

# ============================================================
# DEVICE + SEEDS
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

YOLO_EVERY_N_FRAMES = 8

# ============================================================
# TARGET / STABILITY CONFIG
# ============================================================

OBJ_CONF_TH = 0.5
MAX_OBJECTS = 5
RAY_DIST_TH = 40

OBJECT_TTL = 10          # frames object survives without detection
MAX_STABILITY = 15       # frames to lock target

MIN_Y_RATIO = 0.01
MIN_AREA_RATIO = 0.01

# ============================================================
# LOAD YOLO
# ============================================================

object_detector = YOLO("yolo11n.pt")

# ============================================================
# GEOMETRY
# ============================================================

class GazeSmoother:
    def __init__(self, alpha=0.4):
        self.prev = None
        self.alpha = alpha

    def smooth(self, gx, gy, gz):
        current = np.array([gx, gy, gz], dtype=np.float32)
        if self.prev is None:
            self.prev = current
        else:
            self.prev = self.alpha * self.prev + (1 - self.alpha) * current
        return float(self.prev[0]), float(self.prev[1]), float(self.prev[2])


def select_target(objects, origin, gaze_end):
    ox, oy = origin
    gx, gy = gaze_end

    dx = gx - ox
    dy = gy - oy
    inv_norm = 1.0 / (dx * dx + dy * dy + 1e-6)

    best_obj = None
    best_dist = 1e12

    for obj in objects:
        cx, cy = obj["center"]

        vx = cx - ox
        vy = cy - oy
        proj = vx * dx + vy * dy

        if proj <= 0:
            continue

        dist = (vx * vx + vy * vy) - (proj * proj) * inv_norm
        if dist < best_dist:
            best_dist = dist
            best_obj = obj

    if best_dist > RAY_DIST_TH * RAY_DIST_TH:
        return None

    return best_obj

# ============================================================
# VISUALIZATION
# ============================================================

def draw_gaze_3d(frame, gaze_vec, origin, color=(120, 120, 255), scale=150):
    gx, gy, gz = gaze_vec
    if gz == 0:
        gz = 1e-6

    gx_2d = gx / gz
    gy_2d = gy / gz

    end_x = int(origin[0] - gx_2d * scale)
    end_y = int(origin[1] - gy_2d * scale)

    cv2.arrowedLine(frame, origin, (end_x, end_y), color, 2, tipLength=0.2)
    cv2.circle(frame, origin, 3, (0, 255, 255), -1)

    return (end_x, end_y)


def draw_arrow(frame, origin, end, color, thickness=2):
    cv2.arrowedLine(frame, origin, end, color, thickness, tipLength=0.2)
    cv2.circle(frame, origin, 3, (0, 255, 255), -1)

# ============================================================
# AXIS MAPPER
# ============================================================

class AxisMapper3D:
    def __init__(self, scale=200):
        self.scale = scale

    def endpoint(self, gaze, origin):
        gx, gy, gz = gaze
        gz = gz if gz != 0 else 1e-6
        ox, oy = origin
        ex = int(ox - (gx / gz) * self.scale)
        ey = int(oy - (gy / gz) * self.scale)
        return (ex, ey)

# ============================================================
# MODEL LOADING
# ============================================================

def load_model(ckpt):
    encoder = FrozenEncoder()
    model = GazeEstimationModel(
        encoder,
        output_dim=3,
        num_capsules=6
    ).to(device)

    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        feat = encoder(dummy)
        model.set_capsule_input_dim(feat.reshape(1, -1).size(1))

    state = torch.load(ckpt, map_location=device)
    state = state["state_dict"] if "state_dict" in state else state

    clean = OrderedDict()
    for k, v in state.items():
        clean[k.replace("model.", "").replace("_orig_mod.", "")] = v

    model.load_state_dict(clean, strict=False)
    model.eval()
    return model

# ============================================================
# LANDMARKS + PREPROCESS
# ============================================================

def landmarks_to_224(lmks, x1, y1, w, h):
    return np.array(
        [[(x - x1) * 224 / w, (y - y1) * 224 / h] for x, y in lmks],
        dtype=np.float32
    )

def fast_preprocess(img):
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img).permute(2, 0, 1).float()

def load_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path="face_landmarker_v2_with_blendshapes.task"
        ),
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.FaceLandmarker.create_from_options(options)

# ============================================================
# OBJECT DETECTION
# ============================================================

def detect_objects(frame):
    H, W, _ = frame.shape
    min_area = MIN_AREA_RATIO * W * H

    results = object_detector.predict(
        source=frame,
        conf=OBJ_CONF_TH,
        verbose=False,
        device=0 if torch.cuda.is_available() else "cpu"
    )[0]

    objs = []
    if results.boxes is None:
        return objs

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls.item()) == 0:
            continue

        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            continue

        cy = (y1 + y2) // 2
        if cy < MIN_Y_RATIO * H:
            continue

        cx = (x1 + x2) // 2
        objs.append({
            "center": (cx, cy),
            "bbox": (x1, y1, x2, y2),
            "area": area
        })

    objs = sorted(objs, key=lambda o: o["area"], reverse=True)
    return objs[:MAX_OBJECTS]

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    model = load_model("11012026.pth")
    landmarker = load_landmarker()
    mapper = AxisMapper3D()
    gaze_smoother = GazeSmoother(alpha=0.4)

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    faces = deque(maxlen=12)
    lmks = deque(maxlen=12)

    tracked_objects = {}
    next_oid = 0

    current_target = None
    stability = 0

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )
        ts = int((time.time() - start_time) * 1000)

        if frame_id % 3 == 0:
            res = landmarker.detect_for_video(mp_img, ts)

        if not res.face_landmarks:
            cv2.imshow("Gaze + Target", frame)
            if cv2.waitKey(1) == 27:
                break
            frame_id += 1
            continue

        pts = np.array([[p.x * W, p.y * H] for p in res.face_landmarks[0]])
        x1, y1 = pts.min(0).astype(int)
        x2, y2 = pts.max(0).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        face = frame[y1:y2, x1:x2]
        face224 = cv2.resize(face, (224, 224))

        lm_frame = [(p.x * W, p.y * H) for p in res.face_landmarks[0]]
        lm224 = landmarks_to_224(lm_frame, x1, y1, x2 - x1, y2 - y1)

        faces.append(fast_preprocess(face224))
        lmks.append(torch.from_numpy(lm224))

        # ---------- YOLO with TTL ----------
        if frame_id % YOLO_EVERY_N_FRAMES == 0:
            new_objs = detect_objects(frame)

            for oid in list(tracked_objects.keys()):
                tracked_objects[oid]["ttl"] -= 1
                if tracked_objects[oid]["ttl"] <= 0:
                    del tracked_objects[oid]

            for obj in new_objs:
                cx, cy = obj["center"]
                matched = False
                for oid, tobj in tracked_objects.items():
                    tx, ty = tobj["center"]
                    if (cx - tx)**2 + (cy - ty)**2 < 40**2:
                        tracked_objects[oid].update(obj)
                        tracked_objects[oid]["ttl"] = OBJECT_TTL
                        matched = True
                        break

                if not matched:
                    tracked_objects[next_oid] = {**obj, "ttl": OBJECT_TTL}
                    next_oid += 1

        objects = list(tracked_objects.values())

        # ---------- GAZE ----------
        if len(faces) == 12 and frame_id % 2 == 0:
            inp = torch.stack(list(faces)).unsqueeze(0).to(device)
            lmk = torch.stack(list(lmks)).unsqueeze(0).to(device)

            with torch.no_grad():
                gaze = model(inp, landmarks=lmk)[0][-1].cpu().numpy()

            # ---- SAME smoothing ----
            gx, gy, gz = gaze
            gx, gy, gz = gaze_smoother.smooth(gx, gy, gz)
            gaze_smooth = np.array([gx, gy, gz], dtype=np.float32)

            origin = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)

            # ---- SAME drawing ----
            gaze_end = draw_gaze_3d(
                frame,
                gaze_smooth,
                origin,
                color=(120, 120, 255),
                scale=150
            )

            candidate = select_target(objects, origin, gaze_end)

            if current_target is None:
                current_target = candidate
                stability = 1 if candidate is not None else 0
            else:
                if candidate is current_target:
                    stability = min(stability + 1, MAX_STABILITY)
                elif candidate is None:
                    stability -= 1
                else:
                    stability -= 2

                if stability <= 0:
                    current_target = None
                    stability = 0

        # ---------- DRAW OBJECTS ----------
        for obj in objects:
            x1o, y1o, x2o, y2o = obj["bbox"]
            if obj is current_target:
                color = (0, 255, 0)
                thickness = 3
                draw_arrow(frame, origin, obj["center"], (0, 0, 255), 3)
            else:
                color = (0, 255, 255)
                thickness = 2

            cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), color, thickness)

        cv2.imshow("Gaze + Target", frame)
        if cv2.waitKey(1) == 27:
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
