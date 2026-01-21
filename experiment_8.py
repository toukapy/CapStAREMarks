import cv2
import torch
import numpy as np
from torchvision import transforms
from models.gazev2_3d import FrozenEncoder, GazeEstimationModel
from collections import OrderedDict
import time
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from typing import List, Tuple
import random
import os
import math



import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Fijar semillas
# ==========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # si usas multi-GPU
torch.backends.cudnn.deterministic = False  # para rendimiento
torch.backends.cudnn.benchmark = True       # para rendimiento


# ==========================
# Dataset simple (no se usa online)
# ==========================
from torch.utils.data import Dataset

def color_from_id(track_id):
    np.random.seed(track_id)
    return (
        int(np.random.randint(50, 255)),
        int(np.random.randint(50, 255)),
        int(np.random.randint(50, 255)),
    )


class GazeDataset(Dataset):
    def __init__(self, images, labels, seq_len=9):
        self.images = images
        self.labels = labels  # [[gx, gy], ...]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.images) - self.seq_len + 1

    def __getitem__(self, idx):
        seq_imgs = self.images[idx:idx + self.seq_len]
        seq_labels = self.labels[idx:idx + self.seq_len]
        # Convierte a tensores
        seq_imgs = torch.stack([self._to_tensor(img) for img in seq_imgs])
        seq_labels = torch.tensor(seq_labels, dtype=torch.float32)
        return seq_imgs, seq_labels

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        # img: H x W x C en [0,255]
        img = Image.fromarray(img.astype(np.uint8))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(img)


# ==========================
# Ejes y mapeo
# ==========================
class AxisMapper:
    """
    Convierte la salida [gx, gy] del modelo a un punto final en coordenadas de imagen.
    Modos:
      - 'vector': gx,gy = vector relativo (unidades arbitrarias), se dibuja desde el centro de la cara.
      - 'abs_frame': gx,gy = coords absolutas normalizadas del frame en [-1,1].
      - 'abs_crop': gx,gy = coords absolutas normalizadas del recorte de cara en [-1,1].
    """
    def __init__(self,
                 mode:str='vector',
                 flip_x:bool=False,
                 flip_y:bool=False,
                 swap_xy:bool=False,
                 mirror_compensate:bool=True,
                 scale:int=150):
        self.mode = mode
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.swap_xy = swap_xy
        self.mirror_compensate = mirror_compensate
        self.scale = scale

    def apply_toggles(self, gx, gy):
        # Compensar espejo (si el frame se ha volcado con flip horizontal)
        if self.mirror_compensate:
            gx = -gx
        # Intercambio de ejes
        if self.swap_xy:
            gx, gy = gy, gx
        # Flips manuales
        if self.flip_x:
            gx = -gx
        if self.flip_y:
            gy = -gy
        return gx, gy

    def endpoint(self, gaze, origin, bbox, frame_shape):
        """
        gaze: np.array/list [gx, gy]
        origin: (ox, oy) en pixeles, base de la flecha
        bbox: (x, y, w, h) del rostro
        frame_shape: (H, W, C)
        return: (x_end, y_end) en pixeles dentro del frame
        """
        gx, gy = float(gaze[0]), float(gaze[1])
        H, W, _ = frame_shape

        # Aplica toggles de ejes
        gx, gy = self.apply_toggles(gx, gy)

        if self.mode == 'vector':
            # gx,gy es un vector relativo. Lo escalamos por self.scale y dibujamos desde origin.
            ox, oy = origin
            end_x = int(ox + gx * self.scale)
            end_y = int(oy - gy * self.scale)

        elif self.mode == 'abs_frame':
            # gx,gy en [-1,1] mapeados al frame completo
            x_norm = (gx + 1) / 2.0  # -> [0,1]
            y_norm = (gy + 1) / 2.0
            end_x = int(x_norm * W)
            end_y = int(y_norm * H)

        elif self.mode == 'abs_crop':
            # gx,gy en [-1,1] respecto al recorte de cara
            x, y, w, h = bbox
            x_norm = (gx + 1) / 2.0  # -> [0,1]
            y_norm = (gy + 1) / 2.0
            end_x = int(x + x_norm * w)
            end_y = int(y + y_norm * h)

        else:
            # fallback
            ox, oy = origin
            end_x = ox + int(gx * self.scale)
            end_y = oy - int(gy * self.scale)

        # Clampear dentro de la imagen
        end_x = max(0, min(W - 1, end_x))
        end_y = max(0, min(H - 1, end_y))
        return (end_x, end_y)


class AxisMapper3D(AxisMapper):
    def apply_toggles(self, gx, gy, gz):
        if self.mirror_compensate:
            gx = -gx
        if self.swap_xy:
            gx, gy = gy, gx
        if self.flip_x:
            gx = -gx
        if self.flip_y:
            gy = -gy
        return gx, gy, gz

    def endpoint(self, gaze, origin, bbox, frame_shape):
        gx, gy, gz = gaze
        gx, gy, gz = self.apply_toggles(gx, gy, gz)

        if gz == 0:
            gz = 1e-6  # evitar división por cero

        # proyectar vector 3D a 2D pantalla
        gx_2d = gx / gz
        gy_2d = gy / gz

        return super().endpoint([gx_2d, gy_2d], origin, bbox, frame_shape)


# ==========================
# Dibujo
# ==========================
def draw_gaze(frame, gaze, origin, bbox, mapper: AxisMapper, color=(0, 0, 255)):
    end_pt = mapper.endpoint(gaze, origin, bbox, frame.shape)
    cv2.arrowedLine(frame, origin, end_pt, color, 2, tipLength=0.2)
    cv2.circle(frame, origin, 3, (0, 255, 255), -1)
    return frame


# ==========================
# Calibración afín (3D)
# ==========================
class AffineCalibrator:
    """
    y = [gx, gy, gz, 1] @ THETA  (THETA: 4x3)
    """
    def __init__(self):
        self.theta = np.eye(4, 3, dtype=np.float32)  # Identity
        self.ready = False

    def apply(self, g):  # g: (3,)
        g1 = np.array([float(g[0]), float(g[1]), float(g[2]), 1.0], dtype=np.float32)
        return g1 @ self.theta  # (3,)

    def fit(self, G, T, lam=1e-3):
        """
        G: Nx3 predicciones (después de toggles, antes de invertir Y)
        T: Nx3 objetivos unitarios (direcciones en 3D)
        """
        G = np.asarray(G, np.float32)
        T = np.asarray(T, np.float32)

        # Outlier rejection por z-score
        mu = G.mean(axis=0)
        sd = G.std(axis=0) + 1e-6
        z = np.abs((G - mu) / sd)
        keep = (z < 2.5).all(axis=1)
        G = G[keep]; T = T[keep]
        if len(G) < 4:
            return

        X = np.hstack([G, np.ones((G.shape[0], 1), np.float32)])  # Nx4
        I = np.eye(4, dtype=np.float32); I[3,3] = 1e-6  # regularizar bias
        theta = np.linalg.inv(X.T @ X + lam*I) @ (X.T @ T)  # (4x3)
        self.theta = theta.astype(np.float32)
        self.ready = True

    # ==========================
    # YOLOv8 Face Detector
    # ==========================

def load_face_detector():
    try:
        model_path = hf_hub_download(
            repo_id="arnabdhar/YOLOv8-Face-Detection",
            filename="model.pt"
        )
        return YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLOv8: {e}")
        print("Falling back to Haar cascades")
        return None


def detect_faces(model, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    if model is not None:
        results = model(frame, verbose=False)
        faces = []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                det = Detections.from_ultralytics(r)
                for xyxy, conf, _ in zip(det.xyxy, det.confidence, det.class_id):
                    if conf < conf_threshold:
                        continue
                    x1, y1, x2, y2 = xyxy.astype(int)
                    w = x2 - x1
                    h = y2 - y1
                    faces.append((x1, y1, w, h))
        return faces
    else:
        # Fallback: no detector
        return []


def load_model(checkpoint_path: str):
    print(f"Loading model from {checkpoint_path}")

    # 1) Crear encoder igual que en trainv2_3d.py
    encoder = FrozenEncoder()

    # 2) Crear modelo con misma configuración que en entrenamiento
    #    (usa el mismo num_capsules que pusiste ahí: 6 u otro)
    model = GazeEstimationModel(
        encoder,
        output_dim=3,
        num_capsules=6,   # pon aquí el mismo N que usaste al entrenar
    ).to(device)

    # 3) Calcular flattened_dim y configurar cápsulas
    with torch.no_grad():
        model.encoder.eval()
        dummy = torch.randn(1, 3, 224, 224, device=device)
        feat = model.encoder(dummy)                        # (1, 1024, Hf, Wf)
        flattened_dim = feat.reshape(feat.size(0), -1).size(1)
        model.encoder.train()

    model.set_capsule_input_dim(flattened_dim)

    # 4) Cargar checkpoint (best_model_landmarks.pth o el que uses)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Si tu checkpoint tiene 'state_dict', úsalo; si no, es el propio state_dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 5) Limpiar prefijos tipo 'model.' o '_orig_mod.' si aparecen
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("model."):
            k = k[len("model."):]
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print("[INFO] Missing keys when loading:", missing)
    if unexpected:
        print("[INFO] Unexpected keys when loading:", unexpected)

    model.eval()
    return model


# ==========================
# Transform para inferencia (imagen 224x224)
# ==========================
transform_inference = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ==========================
# MediaPipe Face Landmarker utilities
# ==========================
def load_landmarker(task_path: str = 'face_landmarker_v2_with_blendshapes.task'):
    """Load MediaPipe Face Landmarker in VIDEO mode."""
    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=5,  # or more
        running_mode=vision.RunningMode.VIDEO
    )

    detector = vision.FaceLandmarker.create_from_options(options)
    return detector


def extract_landmarks_224(detection_result, img_size: int = 224):
    """Convert detection_result.face_landmarks[0] to (K,2) array in 224x224 pixel coords."""
    if not detection_result.face_landmarks:
        return None

    face_landmarks = detection_result.face_landmarks[0]
    K = len(face_landmarks)
    lmks = np.zeros((K, 2), dtype=np.float32)

    for i, lm in enumerate(face_landmarks):
        lmks[i, 0] = lm.x * img_size
        lmks[i, 1] = lm.y * img_size

    lmks[:, 0] = np.clip(lmks[:, 0], 0, img_size - 1)
    lmks[:, 1] = np.clip(lmks[:, 1], 0, img_size - 1)
    return lmks


# ==========================
# Preprocesado de recorte de cara
# ==========================
def preprocess_face(face_img):
    """
    Recibe el recorte de la cara en formato OpenCV (BGR).
    Lo convierte a RGB, aplica resize, tensor y normalización.
    """
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_tensor = transform_inference(face_rgb)
    return face_tensor


# ==========================
# Direcciones para calibración (8)
# ==========================
def unit_vectors_8():
    # Direcciones en plano XY, con Z > 0 para "frente"
    # (UP, UR, RIGHT, DR, DOWN, DL, LEFT, UL)
    dirs = {
        "UP":      np.array([0.0,  1.0,  1.0], dtype=np.float32),
        "UR":      np.array([1.0,  1.0,  1.0], dtype=np.float32),
        "RIGHT":   np.array([1.0,  0.0,  1.0], dtype=np.float32),
        "DR":      np.array([1.0, -1.0,  1.0], dtype=np.float32),
        "DOWN":    np.array([0.0, -1.0,  1.0], dtype=np.float32),
        "DL":      np.array([-1.0,-1.0,  1.0], dtype=np.float32),
        "LEFT":    np.array([-1.0, 0.0,  1.0], dtype=np.float32),
        "UL":      np.array([-1.0, 1.0,  1.0], dtype=np.float32),
    }
    out = []
    for label, v in dirs.items():
        n = np.linalg.norm(v) + 1e-6
        out.append((label, v / n))
    return out

DIRECTIONS_8 = unit_vectors_8()

REGION_NAMES = [
    "UP",
    "UP_RIGHT",
    "RIGHT",
    "BOTTOM_RIGHT",
    "BOTTOM",
    "BOTTOM_LEFT",
    "LEFT",
    "UP_LEFT",
]

def gaze_to_region(gx, gy):
    angle = math.degrees(math.atan2(gy, gx))  # [-180, 180]

    if -22.5 <= angle < 22.5:
        return "RIGHT"
    elif 22.5 <= angle < 67.5:
        return "UP_RIGHT"
    elif 67.5 <= angle < 112.5:
        return "UP"
    elif 112.5 <= angle < 157.5:
        return "UP_LEFT"
    elif angle >= 157.5 or angle < -157.5:
        return "LEFT"
    elif -157.5 <= angle < -112.5:
        return "BOTTOM_LEFT"
    elif -112.5 <= angle < -67.5:
        return "BOTTOM"
    else:
        return "BOTTOM_RIGHT"

def draw_region_grid(frame, active_region):
    H, W, _ = frame.shape
    cx, cy = W // 2, H // 2

    regions = {
        "UP": (cx, int(0.1 * H)),
        "UP_RIGHT": (int(0.9 * W), int(0.1 * H)),
        "RIGHT": (int(0.9 * W), cy),
        "BOTTOM_RIGHT": (int(0.9 * W), int(0.9 * H)),
        "BOTTOM": (cx, int(0.9 * H)),
        "BOTTOM_LEFT": (int(0.1 * W), int(0.9 * H)),
        "LEFT": (int(0.1 * W), cy),
        "UP_LEFT": (int(0.1 * W), int(0.1 * H)),
    }

    for name, (x, y) in regions.items():
        color = (0, 255, 255) if name == active_region else (180, 180, 180)
        thickness = 3 if name == active_region else 1
        cv2.circle(frame, (x, y), 18, color, thickness)
        cv2.putText(frame, name, (x - 40, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

from collections import Counter

class GazeRegionExperiment:
    def __init__(self, path="gaze_experiment.txt"):
        self.regions = REGION_NAMES
        self.idx = 0
        self.recording = False
        self.finished = False
        self.file = open(path, "w")

    def current(self):
        return self.regions[self.idx]

    def start(self):
        self.recording = True
        self.file.write(f"\nTARGET: {self.current()}\n")
        self.file.flush()
        print(f"[EXP] START {self.current()}")

    def stop(self):
        # Write marker indicating which target ended here
        self.file.write(f"NEXT: {self.current()}\n\n")
        self.file.flush()

        self.recording = False
        self.idx += 1

        if self.idx >= len(self.regions):
            self.finish()

    def log(self, region):
        if self.recording:
            self.file.write(region + "\n")

    def finish(self):
        self.finished = True
        self.file.write("\nEXPERIMENT_FINISHED\n")
        self.file.flush()
        self.file.close()
        print("[EXP] FINISHED")

    def draw_ui(self, frame):
        if self.finished:
            cv2.putText(frame, "EXPERIMENT FINISHED",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
            return

        draw_region_grid(frame, self.current())

        cv2.putText(frame, f"LOOK AT: {self.current()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)

        status = "RECORDING" if self.recording else "WAITING"
        color = (0, 255, 0) if self.recording else (0, 0, 255)

        cv2.putText(frame, f"STATUS: {status} | s=START  n=NEXT",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)






# ==========================
# Suavizado temporal (opcional, mantenemos el tuyo)
# ==========================
class GazeSmoother:
    def __init__(self, alpha=0.85):
        self.prev = None
        self.alpha = alpha

    def smooth(self, gx, gy, gz):
        current = np.array([gx, gy, gz], dtype=np.float32)
        if self.prev is None:
            self.prev = current
        else:
            self.prev = self.alpha * self.prev + (1 - self.alpha) * current
        return float(self.prev[0]), float(self.prev[1]), float(self.prev[2])


# ==========================
# Calibración guiada (8 direcciones) con MediaPipe landmarks
# ==========================
def run_calibration(cap, model, face_detector, landmarker, mapper, device, n_per_dir=60):
    """8-direction calibration using MediaPipe landmarks + model with landmarks input."""
    prompts = DIRECTIONS_8  # list of (label, unit-vector target)
    G_list, T_list = [], []

    def get_one_gaze_frame(frame):
        faces = detect_faces(face_detector, frame)
        if not faces:
            return None
        x, y, w, h = faces[0]

        # Crop face in BGR
        face_img_bgr = frame[y:y + h, x:x + w]

        # BGR -> RGB and resize to 224x224
        face_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        face_rgb_224 = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_AREA)

        # MediaPipe: wrap in mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb_224)
        timestamp_ms = int(time.time() * 1000)
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
        lmks_np = extract_landmarks_224(detection_result, img_size=224)
        if lmks_np is None:
            return None

        # Image tensor (1,1,C,H,W)
        face_tensor = transform_inference(face_rgb_224)  # (C,H,W)
        inp = face_tensor.unsqueeze(0).unsqueeze(0).to(device)

        # Landmarks tensor (1,1,K,2)
        lmk_tensor = torch.from_numpy(lmks_np).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            g_pred = model(inp, landmarks=lmk_tensor)[0].detach().cpu().numpy()  # (T,3) or (1,3)

        g = g_pred[-1]
        if g.shape[0] != 3:
            raise ValueError(f"Calibración requiere gaze 3D, got shape {g.shape}")
        gx, gy, gz = g
        gx, gy, gz = mapper.apply_toggles(gx, gy, gz)
        return np.array([gx, gy, gz], dtype=np.float32)

    for label, target in prompts:
        collected = []
        while len(collected) < n_per_dir:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Look {label}  ({len(collected)}/{n_per_dir})  - SPACE: sample, ESC: cancel",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('Gaze Estimation', frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                return None
            if k == 32:  # SPACE
                g = get_one_gaze_frame(frame)
                if g is not None:
                    collected.append(g)

        if collected:
            # robust aggregate per direction
            G_list.append(np.median(np.stack(collected, axis=0), axis=0))
            T_list.append(target)

    if not G_list:
        return None

    G = np.stack(G_list, axis=0)
    T = np.stack(T_list, axis=0)

    calib = AffineCalibrator()  # 3D calibrator
    calib.fit(G, T, lam=1e-1)
    return calib


def draw_3d_arrow_perspective(frame, gaze_vec, origin, color=(0,0,255), thickness=2, tipLength=0.2):
    """
    Dibuja una flecha 3D proyectada sobre la imagen usando perspectiva.
    gaze_vec: (gx, gy, gz) vector unitario en coordenadas de cámara
    origin: (ox, oy) pixel coordinates base de la flecha
    """
    gx, gy, gz = gaze_vec
    if gz == 0:
        gz = 1e-6

    # Proyección simple
    gx_2d = gx / gz
    gy_2d = gy / gz

    scale = 150  # longitud de flecha
    end_x = int(origin[0] - gx_2d * scale)
    end_y = int(origin[1] - gy_2d * scale)  # invertir Y para coordenadas de imagen

    cv2.arrowedLine(frame, origin, (end_x, end_y), color, thickness, tipLength=tipLength)
    cv2.circle(frame, origin, 3, (0, 255, 255), -1)
    return frame


# ==========================
# Main loop (con landmarks de MediaPipe)
# ==========================
def main():
    face_detector = load_face_detector()
    model = load_model('22122025.pth')

    # MediaPipe face landmarker for landmarks (as in landmarks_in_video.py)
    landmarker = load_landmarker()

    cap = cv2.VideoCapture(0)
    experiment = GazeRegionExperiment()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('gaze_output.avi',
                          cv2.VideoWriter_fourcc(*'XVID'),
                          30,
                          (frame_width, frame_height))

    sequence_length = 12
    prev_time = time.time()

    mapper = AxisMapper3D(
        mode='abs_crop',
        flip_x=False,
        flip_y=False,
        swap_xy=False,
        mirror_compensate=False,  # 🔴 PRUEBA: NO compensar espejo
        scale=150
    )

    smoother = GazeSmoother(alpha=0.4)
    calibrator = AffineCalibrator()

    tracks = {}  # track_id -> buffers
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # mirror for user view
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            frame_idx += 1
            continue
        H, W, _ = frame.shape

        for face_id, lmks in enumerate(result.face_landmarks):
            track_id = face_id
            color = color_from_id(track_id)

            pts = np.array([[lm.x * W, lm.y * H] for lm in lmks])
            x1, y1 = pts.min(axis=0).astype(int)
            x2, y2 = pts.max(axis=0).astype(int)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            w, h = x2 - x1, y2 - y1

            # Draw bounding box + ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Init track if new
            if track_id not in tracks:
                tracks[track_id] = {
                    "faces": deque(maxlen=sequence_length),
                    "landmarks": deque(maxlen=sequence_length),
                    "smoother": GazeSmoother(alpha=0.4),
                    "last_seen": frame_idx
                }

            track = tracks[track_id]
            track["last_seen"] = frame_idx

            # ---- FACE CROP ----
            face_img_bgr = frame[y1:y2, x1:x2]
            if face_img_bgr.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
            face_rgb_224 = cv2.resize(face_rgb, (224, 224), interpolation=cv2.INTER_AREA)

            # ---- LANDMARKS ----
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb_224)
            timestamp_ms = int(time.time() * 1000)
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            lmks_np = extract_landmarks_224(detection_result, img_size=224)
            if lmks_np is None:
                continue

            # ---- PREPROCESS ----
            face_tensor = transform_inference(face_rgb_224)
            lmk_tensor = torch.from_numpy(lmks_np).float()

            track["faces"].append(face_tensor)
            track["landmarks"].append(lmk_tensor)

            # ---- MODEL INFERENCE ----
            if len(track["faces"]) == sequence_length:
                input_tensor = torch.stack(list(track["faces"])).unsqueeze(0).to(device)
                landmarks_tensor = torch.stack(list(track["landmarks"])).unsqueeze(0).to(device)

                with torch.no_grad():
                    g_pred = model(input_tensor, landmarks=landmarks_tensor)[0]

                gaze_vec = g_pred[-1].cpu().numpy() if g_pred.ndim == 2 else g_pred.cpu().numpy()

                gx, gy, gz = gaze_vec
                gx, gy, gz = track["smoother"].smooth(gx, gy, gz)

                region = gaze_to_region(gx, gy)
                experiment.log(region)


                origin = (x1 + w // 2, y1 + h // 2)
                draw_3d_arrow_perspective(frame, gaze_vec, origin, color=color)


        # FPS computation / overlay (optional)
        curr_time = time.time()
        fps = 1.0 / max(1e-6, (curr_time - prev_time))
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow('Gaze Estimation', frame)

        experiment.draw_ui(frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('c'):
            tmp = run_calibration(cap, model, face_detector, landmarker, mapper, device, n_per_dir=60)
            if tmp is not None:
                calibrator = tmp
                print("[INFO] Calibration done.")
        elif k == ord('x'):
            mapper.flip_x = not mapper.flip_x
            print(f"[INFO] flip_x set to {mapper.flip_x}")
        elif k == ord('y'):
            mapper.flip_y = not mapper.flip_y
            print(f"[INFO] flip_y set to {mapper.flip_y}")
        elif k == ord('w'):
            mapper.swap_xy = not mapper.swap_xy
            print(f"[INFO] swap_xy set to {mapper.swap_xy}")
        elif k == ord('m'):
            mapper.mirror_compensate = not mapper.mirror_compensate
            print(f"[INFO] mirror_compensate set to {mapper.mirror_compensate}")
        elif k == ord('+'):
            mapper.scale = min(1000, mapper.scale + 10)
            print(f"[INFO] scale increased to {mapper.scale}")
        elif k == ord('-'):
            mapper.scale = max(10, mapper.scale - 10)
            print(f"[INFO] scale decreased to {mapper.scale}")
        elif k == ord('s'):
            if not experiment.recording and not experiment.finished:
                experiment.start()
        elif k == ord('n'):
            if experiment.recording:
                experiment.stop()
        if k == ord('q') or k == 27:
            if not experiment.finished:
                experiment.finish()
            break

        MAX_MISSING = 30  # frames

        for tid in list(tracks.keys()):
            if frame_idx - tracks[tid]["last_seen"] > MAX_MISSING:
                del tracks[tid]

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
