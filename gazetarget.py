import cv2
import torch
import numpy as np
from torchvision import transforms
from models.gazev2_3d import FrozenEncoder, GazeEstimationModel
from collections import OrderedDict, deque
import time
from ultralytics import YOLO
import random
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# DEVICE + SEEDS
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
YOLO_EVERY_N_FRAMES = 8   # prueba 8–10


# ============================================================
# SIMPLE TARGET ESTIMATION CONFIG
# ============================================================

OBJ_CONF_TH = 0.5
MAX_OBJECTS = 5
RAY_DIST_TH = 40  # pixels
MIN_Y_RATIO=0.01
MIN_AREA_RATIO = 0.01
# ============================================================
# LOAD OBJECT DETECTOR
# ============================================================

object_detector = YOLO("yolo11n.pt")

# ============================================================
# GEOMETRY
# ============================================================

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-6)

def point_to_ray_distance(p, o, d):
    v = p - o
    proj = np.dot(v, d)
    if proj < 0:
        return np.inf
    closest = o + proj * d
    return np.linalg.norm(p - closest)

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

    objects = []

    if results.boxes is None:
        return objects

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        cls_id = int(cls.item())

        # ---- FILTRO 0: ignorar personas ----
        if cls_id == 0:
            continue

        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)

        # ---- FILTRO 1: tamaño ----
        if area < min_area:
            continue

        # ---- FILTRO 2: región (mesa) ----
        cy = (y1 + y2) // 2
        if cy < MIN_Y_RATIO * H:
            continue

        cx = (x1 + x2) // 2

        objects.append({
            "center": (cx, cy),
            "area": area,
            "bbox": (x1, y1, x2, y2)
        })

    # ---- FILTRO 3: top-K por tamaño ----
    objects = sorted(objects, key=lambda o: o["area"], reverse=True)
    return objects[:MAX_OBJECTS]


# ============================================================
# TARGET SELECTION
# ============================================================

def select_target(objects, origin, gaze_end):
    ox, oy = origin
    gx, gy = gaze_end

    dx = gx - ox
    dy = gy - oy

    inv_norm = 1.0 / (dx*dx + dy*dy + 1e-6)

    best_obj = None
    best_dist = 1e12

    for obj in objects:
        cx, cy = obj["center"]

        vx = cx - ox
        vy = cy - oy

        proj = vx*dx + vy*dy
        if proj <= 0:
            continue

        # distancia al rayo SIN sqrt
        dist = (vx*vx + vy*vy) - (proj*proj)*inv_norm

        if dist < best_dist:
            best_dist = dist
            best_obj = obj

    if best_dist > RAY_DIST_TH * RAY_DIST_TH:
        return None

    return best_obj


# ============================================================
# VISUALIZATION
# ============================================================

def draw_arrow(frame, origin, end, color, thickness=3):
    cv2.arrowedLine(frame, origin, end, color, thickness, tipLength=0.2)
    cv2.circle(frame, origin, 3, (0, 255, 255), -1)

# ============================================================
# AXIS MAPPER
# ============================================================

class AxisMapper3D:
    def __init__(self, scale=150):
        self.scale = scale

    def endpoint(self, gaze, origin):
        gx, gy, gz = gaze
        if gz == 0:
            gz = 1e-6
        gx2d = gx / gz
        gy2d = gy / gz
        ox, oy = origin
        ex = int(ox - gx2d * self.scale)
        ey = int(oy - gy2d * self.scale)
        return (ex, ey)

# ============================================================
# LOAD MODEL
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
    if "state_dict" in state:
        state = state["state_dict"]

    clean = OrderedDict()
    for k, v in state.items():
        clean[k.replace("model.", "").replace("_orig_mod.", "")] = v

    model.load_state_dict(clean, strict=False)
    model.eval()
    return model

# ============================================================
# LANDMARKS
# ============================================================

def landmarks_to_224(lmks, x1, y1, w, h):
    lmks_224 = []
    for (x, y) in lmks:
        lx = (x - x1) * 224.0 / w
        ly = (y - y1) * 224.0 / h
        lmks_224.append([lx, ly])
    return np.array(lmks_224, dtype=np.float32)


def load_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path="face_landmarker_v2_with_blendshapes.task"
        ),
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.FaceLandmarker.create_from_options(options)

def extract_landmarks(det, size=224):
    if not det.face_landmarks:
        return None
    lm = det.face_landmarks[0]
    pts = np.zeros((len(lm), 2), np.float32)
    for i, p in enumerate(lm):
        pts[i] = [p.x * size, p.y * size]
    return pts

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    model = load_model("11012026.pth")
    landmarker = load_landmarker()
    mapper = AxisMapper3D()

    cap = cv2.VideoCapture(0)
    seq_len = 12

    faces = deque(maxlen=seq_len)
    lmks = deque(maxlen=seq_len)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W, _ = frame.shape

        # ---------- MEDIAPIPE (UNA VEZ) ----------
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        res = landmarker.detect_for_video(mp_img, int(time.time() * 1000))
        if not res.face_landmarks:
            cv2.imshow("Gaze + Target", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        # ---------- FACE BBOX ----------
        pts = np.array([[p.x * W, p.y * H] for p in res.face_landmarks[0]])
        x1, y1 = pts.min(0).astype(int)
        x2, y2 = pts.max(0).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face224 = cv2.resize(face, (224, 224))

        lm_frame = np.array([[p.x * W, p.y * H] for p in res.face_landmarks[0]])
        lm224 = landmarks_to_224(lm_frame, x1, y1, x2 - x1, y2 - y1)

        faces.append(transform(face224))
        lmks.append(torch.from_numpy(lm224))

        # ---------- YOLO (CADA FRAME) ----------
        objects = detect_objects(frame)

        # ---------- GAZE ----------
        target = None
        if len(faces) == seq_len:
            inp = torch.stack(list(faces)).unsqueeze(0).to(device)
            lmk = torch.stack(list(lmks)).unsqueeze(0).to(device)

            with torch.no_grad():
                gaze = model(inp, landmarks=lmk)[0][-1].cpu().numpy()

            origin = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
            gaze_end = mapper.endpoint(gaze, origin)

            target = select_target(objects, origin, gaze_end)

        # ---------- DRAW ALL YOLO OBJECTS ----------
        for obj in objects:
            x1o, y1o, x2o, y2o = obj["bbox"]

            if target is not None and obj is target:
                color = (0, 255, 0)    # VERDE → MIRADO
                thickness = 3
            else:
                color = (0, 255, 255)  # AMARILLO
                thickness = 2

            cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), color, thickness)

        cv2.imshow("Gaze + Target", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
