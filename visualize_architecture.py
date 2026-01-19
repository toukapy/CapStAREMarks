import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from models.gazev2_3d import (
    FrozenEncoder,
    GazeEstimationModel,
    get_mediapipe_landmark_groups,
    build_per_capsule_heatmaps
)
from trainv2_3d_v2 import GazeDataset

# =========================================================
# CONFIG
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 12
H5_FILE = "xgaze_224/train/subject0010.h5"
CHECKPOINT = "11012026.pth"
OUT_DIR = "arch_viz2"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================
# DETERMINISTIC TRANSFORM (NO AUGMENTATION)
# =========================================================
class VizTransform:
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, frames, landmarks=None):
        T = frames.shape[0]
        out_frames = []

        for t in range(T):
            img = transforms.ToPILImage()(frames[t])
            img = img.resize(self.size)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(img)
            out_frames.append(img)

        out_frames = torch.stack(out_frames)

        if landmarks is not None:
            if isinstance(landmarks, np.ndarray):
                lmks = torch.from_numpy(landmarks).float()
            else:
                lmks = landmarks.float()

            lmks = lmks.clone()
            lmks[..., 0] *= self.size[0]
            lmks[..., 1] *= self.size[1]
        else:
            lmks = None

        return out_frames, lmks

# =========================================================
# UTILS
# =========================================================
def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3,1,1)
    img = img * std + mean
    img = img.clamp(0,1)
    img = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    return img

def draw_landmarks(img, lmks, color=(0,255,0)):
    out = img.copy()
    for (x, y) in lmks.astype(int):
        cv2.circle(out, (int(x), int(y)), 2, color, -1)
    return out

# =========================================================
# LOAD ONE SEQUENCE (12 FRAMES)
# =========================================================
files = [H5_FILE]


dataset = GazeDataset(
    h5_files=files,
    sequence_length=SEQ_LEN,
    transform=VizTransform(size=(224,224))
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)
frames, _, landmarks = next(iter(loader))

print("Loaded H5:", files)
print("Frames:", frames.shape)
print("Landmarks:", landmarks.shape)


frames = frames.to(DEVICE)
landmarks = landmarks.to(DEVICE)

B, T, C, H, W = frames.shape
last_t = T - 1

# =========================================================
# INPUT IMAGE + LANDMARKS (LAST FRAME)
# =========================================================
img_last = frames[0, last_t]
lmks_last = landmarks[0, last_t].cpu().numpy()

img_np = denormalize(img_last)
cv2.imwrite(f"{OUT_DIR}/input_face.png", img_np)

img_lmk = draw_landmarks(img_np, lmks_last)
cv2.imwrite(f"{OUT_DIR}/input_with_landmarks.png", img_lmk)

# =========================================================
# LANDMARK REGIONS
# =========================================================
groups = get_mediapipe_landmark_groups(device="cpu")
colors = [
    (255,0,0), (0,255,0), (0,0,255),
    (255,255,0), (255,0,255), (0,255,255)
]

img_regions = img_np.copy()
for g, c in zip(groups, colors):
    for idx in g:
        x, y = lmks_last[idx]
        cv2.circle(img_regions, (int(x), int(y)), 2, c, -1)

cv2.imwrite(f"{OUT_DIR}/landmark_regions.png", img_regions)

# =========================================================
# MODEL
# =========================================================
encoder = FrozenEncoder().to(DEVICE)
model = GazeEstimationModel(encoder=encoder, output_dim=3).to(DEVICE)

with torch.no_grad():
    dummy = torch.randn(1,3,224,224,device=DEVICE)
    flat_dim = encoder(dummy).reshape(1,-1).shape[1]

model.set_capsule_input_dim(flat_dim)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE), strict=False)
model.eval()

# =========================================================
# FEATURES, HEATMAPS, CAPSULES (FULL SEQUENCE)
# =========================================================
with torch.no_grad():
    x_bt = frames.view(B*T, C, H, W)
    lmk_bt = landmarks.view(B*T, landmarks.size(2), 2)

    feat_bt = model.encoder(x_bt)

    Hmaps = build_per_capsule_heatmaps(
        lmk_bt,
        model.capsule_groups,
        H=feat_bt.size(2),
        W=feat_bt.size(3),
        sigma=model.heat_sigma
    )

    tokens = model.spatial_capsules(feat_bt, Hmaps)

    feat_last = feat_bt[last_t]
    H_last = Hmaps[last_t]
    tokens_last = tokens[last_t]

# =========================================================
# REGION HEATMAPS (TOP-VALUES, PERCENTILE-BASED)
# =========================================================
H_up = F.interpolate(
    H_last.unsqueeze(1),
    size=(224,224),
    mode="bilinear",
    align_corners=False
).squeeze(1)

# GLOBAL normalization so regions compete
H_up = H_up / (H_up.max() + 1e-6)

for m in range(H_up.size(0)):
    hm = H_up[m]

    # 🔥 keep ONLY the strongest activations (top 10%)
    thr = torch.quantile(hm.view(-1), 0.90)
    hm = torch.where(hm >= thr, hm, torch.zeros_like(hm))

    hm_np = hm.cpu().numpy()

    heat = cv2.applyColorMap(
        (hm_np * 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO
    )

    overlay = cv2.addWeighted(img_np, 0.65, heat, 0.35, 0)
    cv2.imwrite(f"{OUT_DIR}/region_heatmap_strong_{m}.png", overlay)

# =========================================================
# CAPSULE TOKEN HEATMAPS (TOP-VALUES, COMPARABLE)
# =========================================================
C_feat, Hf, Wf = feat_last.shape
M, D = tokens_last.shape

# Proyectar feature map al espacio del token si hace falta
if C_feat != D:
    proj = torch.nn.Linear(C_feat, D, bias=False).to(DEVICE)
    feat_proj = proj(feat_last.permute(1,2,0)).permute(2,0,1)
else:
    feat_proj = feat_last

# 1) Proyectar TODOS los tokens primero (para normalización global)
token_maps = []

for m in range(M):
    token = tokens_last[m]                     # (D,)
    fmap = feat_proj                           # (D,Hf,Wf)

    act = (fmap * token.view(-1,1,1)).sum(dim=0)
    act = torch.relu(act)                      # (Hf,Wf)

    token_maps.append(act)

token_maps = torch.stack(token_maps)           # (M,Hf,Wf)

# 2) Upsample
token_maps_up = F.interpolate(
    token_maps.unsqueeze(1),
    size=(224,224),
    mode="bilinear",
    align_corners=False
).squeeze(1)                                   # (M,224,224)

# 3) GLOBAL normalization (capsules compete)
token_maps_up = token_maps_up / (token_maps_up.max() + 1e-6)

# 4) Percentile thresholding + visualization
for m in range(M):
    hm = token_maps_up[m]

    # 🔥 keep only strongest responses (top 10%)
    thr = torch.quantile(hm.view(-1), 0.90)
    hm = torch.where(hm >= thr, hm, torch.zeros_like(hm))

    hm_np = hm.cpu().detach().numpy()

    heat = cv2.applyColorMap(
        (hm_np * 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO
    )

    overlay = cv2.addWeighted(
        img_np, 0.6,
        heat, 0.4,
        0
    )

    cv2.imwrite(
        f"{OUT_DIR}/capsule_token_heatmap_strong_{m}.png",
        overlay
    )


print(f"[OK] Architecture visualizations saved in {OUT_DIR}/")
