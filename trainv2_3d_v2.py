import torch
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import numpy as np
import cv2
import torch.nn as nn
import os
from tqdm import tqdm
from torchvision import transforms
import random
import torchvision.transforms.v2 as F
from accelerate import Accelerator
from collections import OrderedDict
import wandb
from PIL import Image

# =========================
# CHG: Mejores flags CUDA
# =========================
torch.backends.cuda.matmul.allow_tf32 = True   # TF32 accel
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False     # velocidad > determinismo
torch.backends.cudnn.benchmark = True

# Fijar semillas
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Evitar que variables de entorno cambien
os.environ["PYTHONHASHSEED"] = str(seed)

# Asegurar que todo se ejecuta en GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch._dynamo
torch._dynamo.config.suppress_errors = True


def pitchyaw_to_vector_numpy(pitchyaw):
    """pitchyaw: array (..., 2) en radianes -> vector unitario (..., 3)"""
    pitch = pitchyaw[..., 0]
    yaw = pitchyaw[..., 1]
    x = np.cos(pitch) * np.sin(yaw)
    y = np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)
    vec = np.stack([x, y, z], axis=-1)
    norm = np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-8
    return vec / norm


class AngularLoss(nn.Module):
    """
    Angular loss = mean( arccos( clamp(dot(pred_u, gt_u), -1+eps, 1-eps) ) )
    Returns mean angle in RADIANS.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        # pred, gt: (..., 3)
        pred_u = torch.nn.functional.normalize(pred, dim=-1, eps=self.eps)
        gt_u = torch.nn.functional.normalize(gt, dim=-1, eps=self.eps)
        dot = torch.sum(pred_u * gt_u, dim=-1)
        dot = torch.clamp(dot, -1.0 + self.eps, 1.0 - self.eps)
        angles = torch.acos(dot)  # radians
        return angles.mean()


class SequenceTransform:
    def __init__(self, size=(224, 224), flip_prob: float = 0.0):
        """
        flip_prob:
            Probabilidad de aplicar flip horizontal a toda la secuencia.
            OJO: mientras no ajustemos las etiquetas de gaze al hacer flip,
            lo dejamos en 0.0 para evitar inconsistencias en gx.
        """
        self.size = size
        self.rotation_deg = 12  # sube de 8 a 12
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )
        self.random_erasing = transforms.RandomErasing(p=0.3, scale=(0.02, 0.25))
        self.flip_prob = flip_prob


    def _apply_to_landmarks(self, lmk, W0, H0, flip, angle):
        """
        lmk: (K,2) en [0,1] relativo a W0×H0 original.
        Devuelve (K,2) en pixeles de la imagen final (224x224) tras flip+rotate+resize.
        """
        xy = lmk.copy()
        xy[:, 0] *= W0
        xy[:, 1] *= H0

        Wt, Ht = self.size
        sx, sy = Wt / W0, Ht / H0
        xy[:, 0] *= sx
        xy[:, 1] *= sy

        if flip:
            xy[:, 0] = Wt - 1 - xy[:, 0]

        if angle != 0:
            theta = np.deg2rad(angle)
            cx, cy = Wt / 2.0, Ht / 2.0
            x0, y0 = xy[:, 0] - cx, xy[:, 1] - cy
            xr =  x0 * np.cos(theta) - y0 * np.sin(theta)
            yr =  x0 * np.sin(theta) + y0 * np.cos(theta)
            xy[:, 0] = xr + cx
            xy[:, 1] = yr + cy

        return xy

    def __call__(self, frames, landmarks=None, orig_hw=None):
        # frames: (T, C, H, W) en [0,1]
        T, _, H0, W0 = frames.shape
        pil_frames = [transforms.ToPILImage()(frames[i]) for i in range(T)]

        # mismo muestreo para toda la secuencia
        do_jitter = random.random() < 0.8
        do_erasing = random.random() < 0.5
        angle = random.uniform(-self.rotation_deg, self.rotation_deg)
        flip = random.random() < self.flip_prob

        out_frames = []
        for img in pil_frames:
            if flip:
                img = transforms.functional.hflip(img)
            if do_jitter:
                img = self.color_jitter(img)
            if angle != 0:
                img = transforms.functional.rotate(img, angle)
            img = img.resize(self.size)
            tensor_img = transforms.ToTensor()(img)
            tensor_img = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(tensor_img)
            out_frames.append(tensor_img)
        out_frames = torch.stack(out_frames)

        if do_erasing:
            out_frames = torch.stack([self.random_erasing(out_frames[i]) for i in range(T)])

        out_landmarks = None
        if landmarks is not None:
            lmks = []
            for t in range(T):
                xy = self._apply_to_landmarks(landmarks[t], W0, H0, flip, angle)
                lmks.append(torch.from_numpy(xy).float())
            out_landmarks = torch.stack(lmks)

        return out_frames, out_landmarks


class GazeDataset(Dataset):
    def __init__(self, h5_files, sequence_length=12, transform=None):
        self.h5_files = h5_files
        self.sequence_length = sequence_length
        self.transform = transform

        # Precompute number of valid sequences per file
        self.file_seq_counts = []
        for p in self.h5_files:
            with h5py.File(p, 'r') as f:
                n = max(0, f['face_patch'].shape[0] - sequence_length + 1)
                self.file_seq_counts.append(n)
        self.file_indices = np.cumsum([0] + self.file_seq_counts)
        self.num_data = int(self.file_indices[-1])

        # Cache HDF5 por worker (cada worker tendrá su propio dict)
        self._files = None

    def _ensure_open(self):
        if self._files is None:
            # SWMR=True ayuda en lectura concurrente si los h5 se crearon con SWMR; si no, puedes quitarlo.
            self._files = {p: h5py.File(p, 'r', swmr=True) for p in self.h5_files}

    def __len__(self):
        return int(self.num_data)

    def __getitem__(self, idx):
        self._ensure_open()
        file_idx = np.searchsorted(self.file_indices, idx, side='right') - 1
        local_idx = int(idx - self.file_indices[file_idx])
        h5_path = self.h5_files[file_idx]
        fid = self._files[h5_path]

        try:
            # -----------------
            # Frames (secuencia)
            # -----------------
            frames = []
            H0 = W0 = None
            for i in range(self.sequence_length):
                img = fid['face_patch'][local_idx + i]  # HxWxC (normalmente BGR)
                # Comprobaciones y saneo de canales/strides
                assert img.ndim == 3 and img.shape[2] in (3, 4), f"face_patch shape raro: {img.shape}"
                img = img[..., :3]                 # por si viene BGRA, nos quedamos en 3 canales
                img = img[:, :, ::-1].copy()       # BGR->RGB y copy() evita strides negativos

                if H0 is None:
                    H0, W0 = img.shape[:2]
                frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)

            frames = torch.stack(frames)  # (T,3,H0,W0)

            # ---- Gaze ----
            if 'face_gaze' in fid.keys():
                gazes_py = np.stack([
                    np.array(fid['face_gaze'][local_idx + i], dtype=np.float32)
                    for i in range(self.sequence_length)
                ])
                gazes_vec = pitchyaw_to_vector_numpy(gazes_py).astype(np.float32)
                gazes = torch.from_numpy(gazes_vec)  # (T,3)
            else:
                gazes = torch.zeros(self.sequence_length, 3, dtype=torch.float32)

            # ---- Landmarks ----
            if 'face_landmarks' in fid.keys():
                lmks = np.stack([
                    np.array(fid['face_landmarks'][local_idx + i], dtype=np.float32)
                    for i in range(self.sequence_length)
                ])  # (T,K,2) en [0,1] relativo a la imagen original
            else:
                lmks = None

            # ---- Transforms coherentes imagen/landmarks ----
            if self.transform:
                frames, lmks_t = self.transform(frames, landmarks=lmks)
            else:
                frames, lmks_t = frames, None

            return frames, gazes, (lmks_t if lmks_t is not None else torch.empty(0))

        except Exception as e:
            print(f"[GazeDataset] fallo en file={h5_path} idx_global={idx} local_idx={local_idx} "
                  f"seq_len={self.sequence_length}: {repr(e)}")
            # Re-lanzamos para que el error sea visible en modo num_workers=0
            raise

    def close(self):
        if getattr(self, "_files", None):
            for f in self._files.values():
                try: f.close()
                except: pass
            self._files = None

    def __del__(self):
        # por si el GC cae antes que el cierre explícito
        try: self.close()
        except: pass



def strip_prefix(state_dict, prefix="_orig_mod."):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    return new_state_dict

import atexit, random, numpy as np
from torch.utils.data import get_worker_info

def dataloader_worker_init(worker_id: int):
    """Se ejecuta en cada worker tras el spawn."""
    wi = get_worker_info()
    ds = wi.dataset
    # Semillas reproducibles por worker
    base = 42
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
    try:
        import cv2
        cv2.setNumThreads(0)  # menos interferencia
    except Exception:
        pass
    # Cierre limpio del dataset/h5 al terminar el worker
    if hasattr(ds, "close") and callable(ds.close):
        atexit.register(ds.close)

def dataloader_worker_init_debug(worker_id: int):
    """Versión con trazas más verbosas (útil si algo sigue fallando)."""
    try:
        dataloader_worker_init(worker_id)
        print(f"[worker {worker_id}] init ok")
    except Exception as e:
        import traceback
        print(f"[worker {worker_id}] init ERROR: {e}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    mp_ctx = mp.get_context("spawn")

    wandb.init(project="gaze-estimation", name="gazev2-shared-run", config={
        "batch_size": 16,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "optimizer": "AdamW",
        "epochs": 60,  # ↑ increased epochs in config
        "scheduler": "CosineAnnealingLR"
    })

    # Transformaciones
    seq_transform = SequenceTransform(size=(224, 224), flip_prob=0.0)

    # Carga de datos
    train_dir = 'xgaze_224/train'
    h5_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.h5')]
    rng = random.Random(seed)
    rng.shuffle(h5_files)

    train_size = int(0.8 * len(h5_files))
    train_files, val_files = h5_files[:train_size], h5_files[train_size:]

    train_dataset = GazeDataset(train_files, transform=seq_transform)
    val_dataset = GazeDataset(val_files, transform=seq_transform)

    # ================================
    # CHG: prefetch_factor + workers
    # ================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
        worker_init_fn=dataloader_worker_init,
        multiprocessing_context=mp_ctx,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=1,
        worker_init_fn=dataloader_worker_init,
        multiprocessing_context=mp_ctx,
    )

    # Modelo
    from models.gazev2_3d import FrozenEncoder, GazeEstimationModel

    encoder = FrozenEncoder()
    model = GazeEstimationModel(encoder, output_dim=3, num_capsules=6)

    model = model.to(device)

    # ---- calcular flattened_dim ----
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224, device=device)
        model.encoder.eval()
        feat = model.encoder(dummy)
        flattened_dim = feat.reshape(feat.size(0), -1).size(1)
        model.encoder.train()

    model.set_capsule_input_dim(flattened_dim)

    checkpoint = torch.load('01122025.pth')
    state_dict = strip_prefix(checkpoint)
    model.load_state_dict(state_dict, strict=False)

    # Optimizer (sin watch)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": model.capsule_formation.parameters(), "lr": 1e-4},
        {"params": model.routing.parameters(), "lr": 1e-4},
        {"params": model.eye_decoder.parameters(), "lr": 1e-4},
        {"params": model.face_decoder.parameters(), "lr": 1e-4},
        {"params": model.fusion.parameters(), "lr": 1e-4},
        {"params": model.lmk_fuse.parameters(), "lr": 1e-4},
        {"params": model.spatial_capsules.parameters(), "lr": 1e-4},
    ], weight_decay=1e-5)



    model = torch.compile(model)

    angular_criterion = AngularLoss(eps=1e-7)
    cosine_criterion = torch.nn.CosineEmbeddingLoss()
    alpha = 1.0
    beta = 0.2
    temporal_weight = 0.3

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    accelerator = Accelerator()
    train_loader, model, optimizer, scheduler = accelerator.prepare(
        train_loader, model, optimizer, scheduler
    )

    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5
    scaler = torch.amp.GradScaler('cuda')

    # ↑ increased epochs here
    for epoch in range(100):
        # ---------------------------
        # Training
        # ---------------------------
        model.train()
        total_train_loss = 0.0
        total_train_angular_error = 0.0

        # for per-epoch std
        train_batch_losses = []
        train_batch_errors = []

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Training]")
        for images, targets, landmarks in train_progress:
            images = images.to(device, non_blocking=True)
            if images.dim() == 4:
                images = images.to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            landmarks = (landmarks.to(device, non_blocking=True) if landmarks.numel() > 0 else None)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                predictions = model(images, landmarks=landmarks)

                predictions_seq = predictions.unsqueeze(1) if predictions.dim() == 2 else predictions
                predictions_seq = torch.nn.functional.normalize(predictions_seq, dim=-1, eps=1e-7)

                pred_flat = predictions_seq.view(-1, 3)
                tgt_flat = targets.view(-1, 3)

                loss_ang = angular_criterion(pred_flat, tgt_flat)
                pred_u = torch.nn.functional.normalize(pred_flat, dim=-1)
                tgt_u = torch.nn.functional.normalize(tgt_flat, dim=-1)
                y = torch.ones(pred_u.size(0), device=pred_u.device, dtype=pred_u.dtype)
                loss_cos = cosine_criterion(pred_u, tgt_u, y)

                if predictions_seq.size(1) > 1:
                    pred_u_seq = torch.nn.functional.normalize(predictions_seq, dim=-1)
                    dot_next = torch.sum(pred_u_seq[:, :-1, :] * pred_u_seq[:, 1:, :], dim=-1)
                    temporal_term = (1.0 - dot_next).mean()
                else:
                    temporal_term = torch.tensor(0.0, device=device)

                loss = alpha * loss_ang + beta * loss_cos + temporal_weight * temporal_term

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            last_pred = predictions_seq[:, -1, :]
            last_tgt = targets[:, -1, :]
            pred_u_last = torch.nn.functional.normalize(last_pred, dim=-1)
            tgt_u_last = torch.nn.functional.normalize(last_tgt, dim=-1)
            dot = torch.sum(pred_u_last * tgt_u_last, dim=-1).clamp(-1+1e-7, 1-1e-7)
            mean_batch_error = torch.rad2deg(torch.acos(dot)).mean().item()

            batch_loss = loss.item()
            total_train_loss += batch_loss
            total_train_angular_error += mean_batch_error

            # store for std
            train_batch_losses.append(batch_loss)
            train_batch_errors.append(mean_batch_error)

            train_progress.set_postfix(loss=f"{batch_loss:.4f}",
                                       angular_error=f"{mean_batch_error:.2f}")

        # mean ± std for training
        train_loss_arr = np.array(train_batch_losses, dtype=np.float32)
        train_err_arr = np.array(train_batch_errors, dtype=np.float32)
        avg_train_loss = float(train_loss_arr.mean())
        std_train_loss = float(train_loss_arr.std())
        avg_train_angular_error = float(train_err_arr.mean())
        std_train_angular_error = float(train_err_arr.std())

        print(
            f"Training Loss: {avg_train_loss:.4f} ± {std_train_loss:.4f}, "
            f"Angular Error: {avg_train_angular_error:.2f}° ± {std_train_angular_error:.2f}°"
        )

        # ---------------------------
        # Validation
        # ---------------------------
        model.eval()
        total_val_loss = 0.0
        total_val_angular_error = 0.0

        val_batch_losses = []
        val_batch_errors = []

        val_progress = tqdm(val_loader, desc=f"Epoch {epoch + 1} [Validation]")
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for images, targets, landmarks in val_progress:
                images = images.to(device, non_blocking=True)
                if images.dim() == 4:
                    images = images.to(memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)
                landmarks = (landmarks.to(device, non_blocking=True) if landmarks.numel() > 0 else None)

                predictions = model(images, landmarks=landmarks)
                predictions_seq = predictions.unsqueeze(1) if predictions.dim() == 2 else predictions
                predictions_seq = torch.nn.functional.normalize(predictions_seq, dim=-1, eps=1e-7)

                pred_flat = predictions_seq.view(-1, 3)
                tgt_flat = targets.view(-1, 3)

                loss_ang = angular_criterion(pred_flat, tgt_flat)
                pred_u = torch.nn.functional.normalize(pred_flat, dim=-1)
                tgt_u = torch.nn.functional.normalize(tgt_flat, dim=-1)
                y = torch.ones(pred_u.size(0), device=pred_u.device, dtype=pred_u.dtype)
                loss_cos = cosine_criterion(pred_u, tgt_u, y)

                if predictions_seq.size(1) > 1:
                    pred_u_seq = torch.nn.functional.normalize(predictions_seq, dim=-1)
                    dot_next = torch.sum(pred_u_seq[:, :-1, :] * pred_u_seq[:, 1:, :], dim=-1)
                    temporal_term = (1.0 - dot_next).mean()
                else:
                    temporal_term = torch.tensor(0.0, device=device)

                loss = alpha * loss_ang + beta * loss_cos + temporal_weight * temporal_term
                batch_loss = loss.item()
                total_val_loss += batch_loss

                last_pred = predictions_seq[:, -1, :]
                last_tgt = targets[:, -1, :]
                pred_u_last = torch.nn.functional.normalize(last_pred, dim=-1)
                tgt_u_last = torch.nn.functional.normalize(last_tgt, dim=-1)
                dot = torch.sum(pred_u_last * tgt_u_last, dim=-1).clamp(-1+1e-7, 1-1e-7)
                mean_batch_error = torch.rad2deg(torch.acos(dot)).mean().item()
                total_val_angular_error += mean_batch_error

                val_batch_losses.append(batch_loss)
                val_batch_errors.append(mean_batch_error)

                val_progress.set_postfix(loss=f"{batch_loss:.4f}",
                                         angular_error=f"{mean_batch_error:.2f}")

        val_loss_arr = np.array(val_batch_losses, dtype=np.float32)
        val_err_arr = np.array(val_batch_errors, dtype=np.float32)
        avg_val_loss = float(val_loss_arr.mean())
        std_val_loss = float(val_loss_arr.std())
        avg_val_angular_error = float(val_err_arr.mean())
        std_val_angular_error = float(val_err_arr.std())

        print(
            f"Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}, "
            f"Angular Error: {avg_val_angular_error:.2f}° ± {std_val_angular_error:.2f}°"
        )

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_landmarks.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience_limit:
            print("Early stopping triggered!")
            break

        wandb.log({
            "train_loss": avg_train_loss,
            "train_loss_std": std_train_loss,
            "train_angular_error": avg_train_angular_error,
            "train_angular_error_std": std_train_angular_error,
            "val_loss": avg_val_loss,
            "val_loss_std": std_val_loss,
            "val_angular_error": avg_val_angular_error,
            "val_angular_error_std": std_val_angular_error,
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]['lr']
        })

    print('Training finished')

