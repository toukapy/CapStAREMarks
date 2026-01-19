import os
import re
import cv2
import torch
import numpy as np
import mediapipe as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict

from models.gazev2_3d import GazeEstimationModel, FrozenEncoder

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
SEQ_LEN = 12
BATCH_SIZE = 16
FINE_TUNE = True

RTGENE_ROOT = "RT_GENE"
CHECKPOINT = "22122025.pth"

# =========================================================
# MEDIAPIPE
# =========================================================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def infer_landmarks(img_rgb, num_landmarks=468):
    res = mp_face_mesh.process(img_rgb)

    if not res.multi_face_landmarks:
        return np.zeros((num_landmarks, 2), dtype=np.float32)

    lmks = res.multi_face_landmarks[0].landmark
    coords = np.array([[l.x, l.y] for l in lmks], dtype=np.float32)

    if coords.shape[0] >= num_landmarks:
        coords = coords[:num_landmarks]
    else:
        pad = np.zeros((num_landmarks - coords.shape[0], 2), dtype=np.float32)
        coords = np.vstack([coords, pad])

    return coords

# =========================================================
# DATASET (FIXED: SEQUENCES PER SUBJECT)
# =========================================================
class RTGeneDataset(Dataset):
    def __init__(self, subject_folders, sequence_length=SEQ_LEN):
        self.sequence_length = sequence_length
        self.subject_sequences = []

        for folder in subject_folders:
            label_file = os.path.join(folder, "label_combined.txt")
            img_dir = os.path.join(folder, "inpainted/face_after_inpainting")

            if not os.path.exists(label_file):
                continue

            subject_samples = []

            with open(label_file, "r") as f:
                for line in f:
                    match = re.match(r"(\d+), \[([^\]]+)\], \[([^\]]+)\],", line)
                    if not match:
                        continue

                    idx = int(match.group(1))
                    gaze = [float(x.strip()) for x in match.group(3).split(",")]
                    img_path = os.path.join(img_dir, f"{idx:06d}.png")

                    if os.path.exists(img_path):
                        subject_samples.append((img_path, gaze))

            # 🔑 SOLO guardamos sujetos que puedan generar secuencias
            if len(subject_samples) >= self.sequence_length:
                self.subject_sequences.append(subject_samples)

        print(f"[INFO] Subjects with valid sequences: {len(self.subject_sequences)}")

    def __len__(self):
        return sum(
            len(seq) - self.sequence_length + 1
            for seq in self.subject_sequences
        )

    def __getitem__(self, idx):
        for seq in self.subject_sequences:
            n = len(seq) - self.sequence_length + 1
            if idx < n:
                window = seq[idx:idx + self.sequence_length]
                break
            idx -= n
        else:
            raise IndexError

        imgs, lmks, gazes = [], [], []

        for img_path, gaze in window:
            img_bgr = cv2.imread(img_path)
            img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            imgs.append(
                torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            )

            lmks.append(
                torch.from_numpy(infer_landmarks(img_rgb)).float()
            )

            gazes.append(torch.tensor(gaze).float())

        return (
            torch.stack(imgs),   # (T,3,224,224)
            torch.stack(lmks),   # (T,468,2)
            torch.stack(gazes)   # (T,2)
        )

# =========================================================
# UTILS
# =========================================================
def strip_prefix(state_dict, prefix="_orig_mod."):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[len(prefix):] if k.startswith(prefix) else k] = v
    return new_state_dict

# =========================================================
# MODEL
# =========================================================
def load_model():
    encoder = FrozenEncoder().to(DEVICE)

    model = GazeEstimationModel(
        encoder=encoder,
        output_dim=3
    ).to(DEVICE)

    with torch.no_grad():
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        flat_dim = encoder(dummy).reshape(1, -1).shape[1]

    model.set_capsule_input_dim(flat_dim)

    print(f"[INFO] Loading checkpoint: {CHECKPOINT}")
    ckpt = strip_prefix(torch.load(CHECKPOINT, map_location=DEVICE))

    model_dict = model.state_dict()
    model_dict.update({
        k: v for k, v in ckpt.items()
        if k in model_dict and model_dict[k].shape == v.shape
    })

    model.load_state_dict(model_dict)
    return model

# =========================================================
# METRIC
# =========================================================
def angular_error(pred, gt):
    yaw, pitch = gt[:, 0], gt[:, 1]

    gt_vec = torch.stack([
        torch.cos(pitch) * torch.sin(yaw),
        torch.sin(pitch),
        torch.cos(pitch) * torch.cos(yaw)
    ], dim=1)

    eps = 1e-6
    pred = pred / (torch.norm(pred, dim=1, keepdim=True) + eps)
    gt_vec = gt_vec / (torch.norm(gt_vec, dim=1, keepdim=True) + eps)

    return torch.acos(
        torch.clamp((pred * gt_vec).sum(1), -1, 1)
    ) * 180.0 / np.pi

# =========================================================
# MAIN
# =========================================================
def main():
    subjects = [
        os.path.join(RTGENE_ROOT, s)
        for s in os.listdir(RTGENE_ROOT)
        if os.path.isdir(os.path.join(RTGENE_ROOT, s))
    ]

    dataset = RTGeneDataset(subjects)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = load_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)


    all_errors = []

    for epoch in range(5):
        model.train() if FINE_TUNE else model.eval()

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for imgs, lmks, gazes in pbar:
            imgs, lmks, gazes = imgs.to(DEVICE), lmks.to(DEVICE), gazes.to(DEVICE)

            optimizer.zero_grad()
            pred = model(imgs, landmarks=lmks)[:, -1]
            err = angular_error(pred, gazes[:, -1])

            if FINE_TUNE:
                err.mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

            all_errors.extend(err.detach().cpu().numpy())
            current_ae = err.mean().item()
            mean_ae = float(np.mean(all_errors))

            pbar.set_postfix({
                "AE": f"{current_ae:.2f}",
                "mean_AE": f"{mean_ae:.2f}"
            })


    all_errors = np.array(all_errors)
    print("\n================ RT-GENE RESULTS ================")
    print(f"Mean   : {all_errors.mean():.2f}")
    print(f"Median : {np.median(all_errors):.2f}")
    print(f"Std    : {all_errors.std():.2f}")
    print("================================================")

if __name__ == "__main__":
    main()
