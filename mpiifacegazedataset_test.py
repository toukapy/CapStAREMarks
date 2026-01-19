import os
import h5py
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.gazev2_3d import FrozenEncoder, GazeEstimationModel


# ============================================================
# Utils
# ============================================================

def angular_error_3d(gt, pred):
    gt = gt / (np.linalg.norm(gt) + 1e-6)
    pred = pred / (np.linalg.norm(pred) + 1e-6)
    dot = np.clip(np.dot(gt, pred), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def strip_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("model."):
            k = k[len("model."):]
        out[k] = v
    return out



# ============================================================
# MPIIFaceGaze NORMALIZED H5 Dataset
# ============================================================

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def gaze_2d_to_3d(pitch, yaw):
    x = -np.cos(pitch) * np.sin(yaw)
    y = -np.sin(pitch)
    z = -np.cos(pitch) * np.cos(yaw)
    return np.array([x, y, z], dtype=np.float32)


def expand_landmarks_6_to_468(lm6):
    lm468 = torch.zeros((468, 2), dtype=torch.float32)
    if isinstance(lm6, np.ndarray):
        lm6 = torch.from_numpy(lm6).float()
    lm468[:6] = lm6
    return lm468


class MPIIFaceGazeNormalizedH5(Dataset):
    def __init__(self, h5_files, seq_len=12, transform=None):
        self.h5_files = h5_files
        self.seq_len = seq_len
        self.transform = transform

        self.file_handles = []
        self.file_lengths = []

        # open files lazily
        for path in self.h5_files:
            f = h5py.File(path, "r")
            self.file_handles.append(f)
            self.file_lengths.append(f["data"].shape[3])

        # cumulative indexing
        self.cum_lengths = np.cumsum(self.file_lengths)
        self.total_len = self.cum_lengths[-1]

        if self.total_len < seq_len:
            raise RuntimeError("Not enough samples overall")

    def __len__(self):
        return self.total_len - self.seq_len + 1

    def _locate(self, global_idx):
        """
        Map global index to (file_id, local_idx)
        """
        file_id = np.searchsorted(self.cum_lengths, global_idx, side="right")
        prev = 0 if file_id == 0 else self.cum_lengths[file_id - 1]
        local_idx = global_idx - prev
        return file_id, local_idx

    def __getitem__(self, idx):
        imgs = []
        lmks = []
        gazes = []

        for t in range(self.seq_len):
            gidx = idx + t
            fid, lid = self._locate(gidx)
            f = self.file_handles[fid]

            # --------------------------------------------------
            # 1. Leer imagen del H5 (MPII normalized)
            # --------------------------------------------------
            data = f["data"]  # shape: (N, C, H, W)
            img = data[lid]  # (C, H, W)

            # --------------------------------------------------
            # 2. Normalizar a (H, W, 3)
            # --------------------------------------------------
            img = np.squeeze(img)

            if img.ndim == 3:
                # (C, H, W) -> (H, W, C)
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                else:
                    raise ValueError(f"Unexpected channel dim: {img.shape}")
            elif img.ndim == 2:
                # grayscale -> RGB
                img = np.stack([img] * 3, axis=-1)
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")

            img = img.astype(np.uint8)

            # --------------------------------------------------
            # 2. DESHACER preprocesado MATLAB / Caffe
            #    new_image = original_image(:,:,[3 2 1]);
            #    new_image = flip(new_image, 2);
            #    new_image = imrotate(new_image, 90);
            # --------------------------------------------------


            # --------------------------------------------------
            # 3. A PIL + transform
            # --------------------------------------------------
            img = Image.fromarray(img)
            img = self.transform(img)
            imgs.append(img)

            # --------------------------------------------------
            # 4. Labels
            # --------------------------------------------------
            label = f["label"][lid]


            # ---- gaze: (pitch, yaw) -> 3D
            pitch, yaw = label[0], label[1]
            gaze_3d = gaze_2d_to_3d(pitch, yaw)
            gazes.append(gaze_3d)

            # ---- landmarks: 6 -> 468 (zero padding)
            lm6 = label[4:16].reshape(6, 2)
            lm468 = torch.zeros((468, 2), dtype=torch.float32)
            lm468[:6] = torch.from_numpy(lm6).float()
            lmks.append(lm468)



        # --------------------------------------------------
        # 5. Stack temporal
        # --------------------------------------------------
        imgs = torch.stack(imgs)  # (T,3,224,224)
        lmks = torch.stack(lmks)  # (T,468,2)
        gazes = torch.tensor(gazes)  # (T,3)

        return imgs, lmks, gazes


# ============================================================
# Model loader
# ============================================================

def load_model(checkpoint_path, device):
    encoder = FrozenEncoder()
    model = GazeEstimationModel(
        encoder=encoder,
        output_dim=3,
        num_capsules=6
    ).to(device)

    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        feat = model.encoder(dummy)
        flattened_dim = feat.reshape(1, -1).shape[1]

    model.set_capsule_input_dim(flattened_dim)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    model.load_state_dict(strip_prefix(ckpt), strict=False)
    model.eval()
    return model

def angular_loss(pred, gt):
    pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-6)
    gt = gt / (gt.norm(dim=1, keepdim=True) + 1e-6)
    dot = (pred * gt).sum(dim=1).clamp(-1, 1)
    return torch.acos(dot).mean()

def angular_error_batch(pred, gt):
    """
    pred, gt: (B, 3)
    returns: scalar angular error in degrees
    """
    pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-6)
    gt = gt / (gt.norm(dim=1, keepdim=True) + 1e-6)
    dot = (pred * gt).sum(dim=1).clamp(-1, 1)
    return torch.rad2deg(torch.acos(dot)).mean().item()




# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_dir", required=True,
                        help="Directory with pXX_0.h5, pXX_1.h5 files")
    parser.add_argument("--subjects", nargs="+", required=True,
                        help="Subjects to evaluate, e.g. p00 p01")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--finetune", action="store_true",
                        help="Enable fine-tuning on MPII")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # Collect h5 files
    # --------------------------------------------------
    h5_files = []
    for s in args.subjects:
        for k in [0, 1]:
            p = os.path.join(args.h5_dir, f"{s}_{k}.h5")
            if os.path.exists(p):
                h5_files.append(p)

    if len(h5_files) == 0:
        raise RuntimeError("No .h5 files found")

    print("[INFO] Using H5 files:")
    for f in h5_files:
        print("  ", f)

    # --------------------------------------------------
    # Transform
    # --------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = MPIIFaceGazeNormalizedH5(
        h5_files=h5_files,
        seq_len=args.seq_len,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )

    model = load_model(args.model_path, device)

    if args.finetune:
        print("[INFO] Fine-tuning enabled")

        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    if args.finetune:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-4
        )

    if args.finetune:
        model.train()
        print(f"[INFO] Fine-tuning for {args.epochs} epochs")

        for epoch in range(args.epochs):
            pbar = tqdm(loader, desc=f"Fine-tune epoch {epoch + 1}/{args.epochs}")
            ae_sum = 0
            n_steps = 0

            for imgs, lmks, gazes in pbar:
                imgs = imgs.to(device)
                lmks = lmks.to(device)
                gazes = gazes.to(device)

                pred = model(imgs, landmarks=lmks)[:, -1]
                gt = gazes[:, -1]

                loss = angular_loss(pred, gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ae = angular_error_batch(pred.detach(), gt)

                if epoch == 0 and 'ae_sum' not in locals():
                    ae_sum = 0.0
                    n_steps = 0

                ae_sum += ae
                n_steps += 1
                ae_mean = ae_sum / n_steps

                pbar.set_description(
                    f"Fine-tune {epoch + 1}/{args.epochs} | "
                    f"loss: {loss.item():.4f} | "
                    f"AE: {ae:.2f}° | "
                    f"AE mean: {ae_mean:.2f}°"
                )

        print("[INFO] Fine-tuning finished")

    errors = []
    sum_err=0
    sum_sq=0
    n=0
    pbar = tqdm(loader, desc="Evaluating (normalized MPII)")

    model.eval()
    with torch.no_grad():
        for imgs, lmks, gazes in pbar:
            imgs = imgs.to(device)
            lmks = lmks.to(device)
            gazes = gazes.to(device)

            preds = model(imgs, landmarks=lmks)

            pred = preds[0, -1].cpu().numpy()
            gt = gazes[0, -1].cpu().numpy()

            err = angular_error_3d(gt, pred)
            errors.append(err)

            sum_err += err
            sum_sq += err * err
            n += 1

            mean_err = sum_err / n
            std_err = np.sqrt(sum_sq / n - mean_err ** 2)

            pbar.set_description(
                f"Evaluating (normalized MPII) | "
                f"AE mean: {mean_err:.2f}° | "
                f"AE std: {std_err:.2f}°"
            )

    errors = np.array(errors)
    print("================================")
    print(f"Mean Angular Error: {errors.mean():.2f}°")
    print(f"Median Error:       {np.median(errors):.2f}°")
    print(f"Std:                {errors.std():.2f}°")
    print("================================")


if __name__ == "__main__":
    main()
