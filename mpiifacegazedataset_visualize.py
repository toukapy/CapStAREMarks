import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from models.gazev2_3d import FrozenEncoder, GazeEstimationModel


# ============================================================
# Utils
# ============================================================

def normalize(v):
    return v / (np.linalg.norm(v) + 1e-6)


def angular_error_3d(gt, pred):
    dot = np.clip(np.dot(normalize(gt), normalize(pred)), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def draw_gaze_arrow_mpii(img, gaze, color, label):
    """
    Correct projection for MPIIFaceGaze GT and aligned predictions.
    gaze: (gx, gy, gz) in camera coordinates
    """
    h, w, _ = img.shape
    cx, cy = w // 2, h // 2

    gx, gy, gz = gaze
    if gz < 1e-6:
        return

    # MPII camera projection (NO sign flips)
    x2d = gx / gz
    y2d = gy / gz

    scale = 120
    end_x = int(cx + x2d * scale)
    end_y = int(cy + y2d * scale)

    cv2.arrowedLine(
        img,
        (cx, cy),
        (end_x, end_y),
        color,
        2,
        tipLength=0.15
    )

    cv2.putText(
        img,
        label,
        (10, 25 if label == "GT" else 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def draw_gaze_arrow(img, gaze, color, label):
    """
    img: RGB uint8 image
    gaze: (3,) in camera coords
    """
    h, w, _ = img.shape
    cx, cy = w // 2, h // 2

    gx, gy, gz = gaze
    if gz == 0:
        gz = 1e-6

    # simple perspective projection
    x2d = gx / gz
    y2d = gy / gz

    scale = 120
    end_x = int(cx - x2d * scale)
    end_y = int(cy + y2d * scale)

    cv2.arrowedLine(img, (cx, cy), (end_x, end_y), color, 2, tipLength=0.15)
    cv2.putText(
        img,
        label,
        (10, 25 if label == "GT" else 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def strip_prefix(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("model."):
            k = k[len("model."):]
        out[k] = v
    return out


# ============================================================
# MediaPipe
# ============================================================

def load_landmarker(task_path):
    base = python.BaseOptions(model_asset_path=task_path)
    opts = vision.FaceLandmarkerOptions(
        base_options=base,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE
    )
    return vision.FaceLandmarker.create_from_options(opts)


def extract_landmarks(result, W, H):
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    pts = np.zeros((len(lm), 2), np.float32)
    for i, p in enumerate(lm):
        pts[i, 0] = p.x * W
        pts[i, 1] = p.y * H
    return pts


# ============================================================
# Model
# ============================================================

def load_model(ckpt_path, device):
    encoder = FrozenEncoder()
    model = GazeEstimationModel(
        encoder=encoder,
        output_dim=3,
        num_capsules=6
    ).to(device)

    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        feat = model.encoder(dummy)
        model.set_capsule_input_dim(feat.reshape(1, -1).shape[1])

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    model.load_state_dict(strip_prefix(ckpt), strict=False)
    model.eval()
    return model


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--landmarker_path", required=True)
    parser.add_argument("--out_dir", default="vis_out")
    parser.add_argument("--num_samples", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    landmarker = load_landmarker(args.landmarker_path)
    model = load_model(args.model_path, device)

    txt_path = os.path.join(args.data_dir, args.subject, f"{args.subject}.txt")
    lines = open(txt_path).readlines()

    seq_imgs = []
    seq_lmks = []

    saved = 0

    for line in lines:
        if saved >= args.num_samples:
            break

        cols = line.strip().split()
        if len(cols) < 27:
            continue

        img_path = os.path.join(args.data_dir, args.subject, cols[0])
        if not os.path.exists(img_path):
            continue

        gt = normalize(np.array(cols[24:27], dtype=np.float32))

        # --------------------------------------------------
        # Image + MediaPipe
        # --------------------------------------------------
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        result = landmarker.detect(mp_img)
        lm = extract_landmarks(result, W, H)
        if lm is None:
            continue

        # --------------------------------------------------
        # Face crop
        # --------------------------------------------------
        xs, ys = lm[:, 0], lm[:, 1]
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        pad = int(0.1 * max(x2 - x1, y2 - y1))

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad)
        y2 = min(H, y2 + pad)

        face = img[y1:y2, x1:x2]
        face = cv2.resize(face, (224, 224))

        lm[:, 0] = (lm[:, 0] - x1) * (224.0 / (x2 - x1))
        lm[:, 1] = (lm[:, 1] - y1) * (224.0 / (y2 - y1))

        # --------------------------------------------------
        # Temporal buffer
        # --------------------------------------------------
        img_t = transform(Image.fromarray(face))
        lm_t = torch.from_numpy(lm)

        seq_imgs.append(img_t)
        seq_lmks.append(lm_t)

        if len(seq_imgs) < 12:
            continue

        seq_imgs = seq_imgs[-12:]
        seq_lmks = seq_lmks[-12:]

        imgs = torch.stack(seq_imgs).unsqueeze(0).to(device)
        lmks = torch.stack(seq_lmks).unsqueeze(0).to(device)

        # --------------------------------------------------
        # Prediction
        # --------------------------------------------------
        with torch.no_grad():
            pred = model(imgs, landmarks=lmks)[0, -1].cpu().numpy()
            pred = normalize(pred)

        ae = angular_error_3d(gt, pred)

        # --------------------------------------------------
        # Draw
        # --------------------------------------------------
        vis = face.copy()
        draw_gaze_arrow_mpii(vis, gt, (255, 0, 0), "GT")
        draw_gaze_arrow_mpii(vis, pred, (0, 0, 255), "PRED")


        out_path = os.path.join(args.out_dir, f"{saved:04d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        saved += 1

    print(f"[INFO] Saved {saved} visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()
