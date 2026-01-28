import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import torch._dynamo as dynamo   # <--- ya lo tienes


@dynamo.disable
def _safe_lmk_fuse(conv1x1, act, feat_btchw, hmaps_btmd):
    """
    feat_btchw: (B*T, C, H, W)
    hmaps_btmd: (B*T, M, H, W)  o (B*T, 1, H, W) o (B*T, H, W)
    Devuelve: (B*T, C, H, W) con fusión residual.
    """
    # Normaliza hmaps a (B*T, 1, H, W)
    if hmaps_btmd is None:
        return feat_btchw

    if hmaps_btmd.dim() == 3:                  # (B*T, H, W)
        hmaps_btmd = hmaps_btmd.unsqueeze(1)   # (B*T, 1, H, W)
    elif hmaps_btmd.dim() == 5:                # (B,T,M,H,W) ¡no lo queremos aquí!
        # Aplana B y T:
        BT = feat_btchw.size(0)
        H, W = feat_btchw.size(2), feat_btchw.size(3)
        hmaps_btmd = hmaps_btmd.reshape(BT, -1, H, W)  # (B*T, M, H, W)

    # Si viene (B*T, M, H, W) colapsa M
    if hmaps_btmd.dim() == 4 and hmaps_btmd.size(1) > 1:
        hmaps_btmd = hmaps_btmd.sum(dim=1, keepdim=True)  # (B*T, 1, H, W)
    elif hmaps_btmd.dim() == 4 and hmaps_btmd.size(1) == 1:
        pass
    else:
        # Cualquier otra cosa: fuerza a canal único del tamaño correcto
        BT, C, H, W = feat_btchw.size()
        hmaps_btmd = torch.zeros(BT, 1, H, W, device=feat_btchw.device, dtype=feat_btchw.dtype)

    # Evita in-place con aliasing
    hmaps_btmd = hmaps_btmd.clamp(max=1.0).contiguous()

    # Concat canal y fusiona
    x = torch.cat([feat_btchw, hmaps_btmd], dim=1)  # (B*T, C+1, H, W)
    delta = act(conv1x1(x))
    return feat_btchw + delta
def get_mediapipe_landmark_groups(device="cpu"):
    """
    Devuelve índices de landmarks agrupados por región facial (para FaceMesh de 468 puntos).
    """
    # Ojos (izquierdo y derecho)
    left_eye_idx  = torch.tensor([
        33, 133, 160, 159, 158, 157, 173, 246,
        161, 163, 144, 145, 153, 154, 155
    ], dtype=torch.long, device=device)
    right_eye_idx = torch.tensor([
        362, 263, 387, 386, 385, 384, 398, 466,
        373, 380, 381, 382, 362, 390, 249
    ], dtype=torch.long, device=device)

    # Cejas
    left_brow_idx = torch.tensor([70, 63, 105, 66, 107, 55, 65, 52], dtype=torch.long, device=device)
    right_brow_idx = torch.tensor([336, 296, 334, 293, 300, 283, 295, 285], dtype=torch.long, device=device)

    # Nariz
    nose_idx = torch.tensor([
        1, 2, 98, 327, 97, 326, 168, 6, 197, 5, 4, 45, 275, 195
    ], dtype=torch.long, device=device)

    # Boca
    mouth_idx = torch.tensor([
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415
    ], dtype=torch.long, device=device)

    # Retornar en lista de grupos
    return [left_eye_idx, right_eye_idx, left_brow_idx, right_brow_idx, nose_idx, mouth_idx]


class LandmarkGuidance(nn.Module):
    def __init__(self, feat_channels, heatmaps=1, sigma=2.0, select_every=1, select_indices=None):
        """
        - heatmaps: # de canales de mapas (dejamos 1)
        - sigma: en celdas del mapa (Hf,Wf)
        - select_every: usa 1 de cada n landmarks (submuestreo rápido)
        - select_indices: tensor/lista de índices K_sel (p.ej. solo ojos). Ignora select_every si se pasa.
        """
        super().__init__()
        assert heatmaps == 1, "Para velocidad mantenemos 1 heatmap; si quieres >1 lo añadimos luego."
        self.heatmaps = heatmaps
        self.sigma = float(sigma)
        self.select_every = max(1, int(select_every))
        self.register_buffer('_dummy', torch.ones(1))  # placeholder para el device/dtype

        # fusión ligera
        self.fuse = nn.Conv2d(feat_channels + heatmaps, feat_channels, kernel_size=1, bias=True)
        self.act = nn.GELU()

        # opcional: índices fijos (p.ej., ojos). Si es None, se hará submuestreo por select_every.
        if select_indices is not None:
            sel = torch.as_tensor(select_indices, dtype=torch.long)
        else:
            sel = None
        self.register_buffer('select_indices', sel, persistent=False)

        # rejilla (se crea perezosamente cuando sabemos Hf,Wf)
        self.register_buffer('grid_x', None, persistent=False)
        self.register_buffer('grid_y', None, persistent=False)

    def _ensure_grid(self, H, W, device, dtype):
        if (self.grid_x is None) or (self.grid_x.shape != (1, H, W)) or (self.grid_x.device != device):
            ys = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
            xs = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
            self.grid_y = ys.unsqueeze(0)  # (1,H,W)
            self.grid_x = xs.unsqueeze(0)  # (1,H,W)


    def _build_heatmap(self, landmarks_bt, H, W):
        """
        landmarks_bt: (N, K, 2) en píxeles 224×224 -> produce (N,1,H,W)
        Vectorizado y sin bucles Python.
        """
        if landmarks_bt is None or landmarks_bt.numel() == 0:
            return None

        N, K, _ = landmarks_bt.shape
        device = landmarks_bt.device
        dtype = landmarks_bt.dtype

        # Selección de puntos (opcional)
        if self.select_indices is not None:
            lm = landmarks_bt[:, self.select_indices]          # (N,Ksel,2)
        else:
            lm = landmarks_bt[:, ::self.select_every]           # (N,⌈K/n⌉,2)

        # Escala 224→(H,W)
        sx, sy = W / 224.0, H / 224.0
        x = lm[..., 0] * sx   # (N, Ks)
        y = lm[..., 1] * sy   # (N, Ks)

        # Rejilla (H,W)
        self._ensure_grid(H, W, device, dtype)
        gx = self.grid_x      # (1,H,W)
        gy = self.grid_y      # (1,H,W)

        # Distancias al cuadrado (broadcasting): (N, Ks, H, W)
        # (gx,gy) son (1,H,W); expandimos a (N, Ks, H, W)
        dx2 = (gx.unsqueeze(0).unsqueeze(1) - x.unsqueeze(-1).unsqueeze(-1)) ** 2
        dy2 = (gy.unsqueeze(0).unsqueeze(1) - y.unsqueeze(-1).unsqueeze(-1)) ** 2
        d2  = dx2 + dy2

        # d2: (N, Ks, Hf, Wf)
        var = (self.sigma ** 2)
        Hmap = torch.exp(- d2 / (2.0 * var))  # (N, Ks, Hf, Wf)

        # >>> SUMA SOBRE LOS LANDMARKS (Ks) <<<
        Hmap = Hmap.sum(dim=1)  # (N, Hf, Wf)

        # Añade canal
        Hmap = Hmap.unsqueeze(1)  # (N, 1, Hf, Wf)

        # Clamp suave opcional
        Hmap = Hmap.clamp_(max=1.0)
        return Hmap

    def forward(self, feat, landmarks_bt):
        # feat: (N, C, Hf, Wf)
        # landmarks_bt: (N, K, 2) en pix 224x224
        if landmarks_bt is None or landmarks_bt.numel() == 0:
            return feat

        N, C, Hf, Wf = feat.shape
        Hmap = self._build_heatmap(landmarks_bt, Hf, Wf)  # esperado: (N,1,Hf,Wf)

        # --- Saneador robusto (por si alguna variante deja Ks sin colapsar) ---
        if Hmap.dim() == 5:
            # p.ej. (N,1,Ks,Hf,Wf) -> colapsa Ks
            if Hmap.size(2) > 1:
                Hmap = Hmap.sum(dim=2, keepdim=False)  # (N,1,Hf,Wf)
        if Hmap.dim() == 4 and Hmap.size(1) != 1:
            # si por error Ks cayó en canales: (N,Ks,Hf,Wf)
            Hmap = Hmap.sum(dim=1, keepdim=True)  # (N,1,Hf,Wf)
        if Hmap.dim() == 3:
            Hmap = Hmap.unsqueeze(1)  # (N,1,Hf,Wf)
        # --- Asegura que el batch coincida con feat (arregla N=1 vs N=B*T) ---
        if Hmap.size(0) != N:
            Hmap = Hmap.expand(N, -1, -1, -1)  # (N,1,Hf,Wf)
        # ----------------------------------------------------------------------

        x = torch.cat([feat, Hmap], dim=1)  # (N, C+1, Hf, Wf)
        delta = self.act(self.fuse(x))
        return feat + delta


def build_per_capsule_heatmaps(lmk_bt, groups, H, W, sigma=2.0):
    """
    lmk_bt: (N, K, 2) en píxeles 224x224   (N = B*T)
    groups: lista de listas/1D-tensors con índices de landmarks por cápsula (longitud M)
    return: (N, M, H, W)  heatmaps sumados por grupo (1 canal por cápsula)
    """
    assert lmk_bt.dim() == 3 and lmk_bt.size(2) == 2, f"Esperado (N,K,2), llega {tuple(lmk_bt.shape)}"
    N, K, _ = lmk_bt.shape
    device = lmk_bt.device
    dtype = lmk_bt.dtype

    # Escala 224 -> (H,W)
    sx, sy = float(W) / 224.0, float(H) / 224.0
    x = lmk_bt[..., 0] * sx         # (N, K)
    y = lmk_bt[..., 1] * sy         # (N, K)

    # Limpieza NaN/inf
    finite_mask = torch.isfinite(x) & torch.isfinite(y)
    x = torch.where(finite_mask, x, torch.zeros_like(x))
    y = torch.where(finite_mask, y, torch.zeros_like(y))

    # Rejilla (H,W)
    gy = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)       # (H,W)
    gx = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)       # (H,W)
    gy = gy.unsqueeze(0)   # (1,H,W)
    gx = gx.unsqueeze(0)   # (1,H,W)

    var = float(sigma) ** 2
    Hmaps = []

    for g in groups:
        # Normaliza grupo -> tensor long en device y dentro de [0, K)
        if g is None:
            idx = torch.empty(0, dtype=torch.long, device=device)
        else:
            idx = torch.as_tensor(g, dtype=torch.long, device=device)
            if idx.numel() > 0:
                idx = idx[(idx >= 0) & (idx < K)]

        if idx.numel() == 0:
            # Grupo vacío: devuelve mapa nulo pero con la misma N,H,W
            Hm = torch.zeros(N, 1, H, W, device=device, dtype=dtype)
        else:
            # (N, Kg)
            xg = x[:, idx]
            yg = y[:, idx]

            # Distancias al cuadrado: (N, Kg, H, W)
            dx2 = (gx.view(1, 1, H, W) - xg.view(N, -1, 1, 1)) ** 2
            dy2 = (gy.view(1, 1, H, W) - yg.view(N, -1, 1, 1)) ** 2
            d2  = dx2 + dy2

            # Suma de gaussianas por grupo -> (N, 1, H, W)
            Hm = torch.exp(-d2 / (2.0 * var)).sum(dim=1, keepdim=True)
            # Clamp seguro (sin in-place compartido)
            Hm = Hm.clamp(max=1.0).contiguous()

        # Garantiza exactamente (N,1,H,W)
        if Hm.dim() == 3:
            Hm = Hm.unsqueeze(1)
        elif Hm.dim() == 5:
            # Por si llega (N,1,Kg,H,W)
            Hm = Hm.sum(dim=2, keepdim=True)
        Hmaps.append(Hm)

    # Concatena por cápsula -> (N, M, H, W). Todas las entradas comparten N,H,W.
    Hmaps = torch.cat(Hmaps, dim=1).contiguous()
    # Invariantes de forma
    assert Hmaps.size(0) == N and Hmaps.size(2) == H and Hmaps.size(3) == W, \
        f"Heatmaps mal formados: {tuple(Hmaps.shape)} vs N={N},H={H},W={W}"
    return Hmaps

                  # residual

class LandmarkCapsuleFormation(nn.Module):
    """
    Forma cápsulas por-región con heatmaps por cápsula y pooling ponderado.
    Entrada:
      - feat_bt: (B*T, C, Hf, Wf)
      - heatmaps_btmd: (B*T, M, Hf, Wf)
    Salida:
      - tokens: (B*T, M, D)
    """
    def __init__(self, in_channels, num_capsules, capsule_dim):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.proj = nn.Conv2d(in_channels, capsule_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(capsule_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, feat_bt, heatmaps_btmd):
        N, C, Hf, Wf = feat_bt.shape
        M = heatmaps_btmd.size(1)

        proj = self.proj(feat_bt)                   # (N, D, Hf, Wf)
        proj = proj.view(N, self.capsule_dim, Hf*Wf)   # (N, D, S)
        H    = heatmaps_btmd.view(N, M, Hf*Wf)         # (N, M, S)

        weights = torch.softmax(H, dim=-1)          # (N, M, S)
        proj_tr = proj.transpose(1, 2)              # (N, S, D)
        tokens  = torch.bmm(weights, proj_tr)       # (N, M, D)

        tokens = self.norm(tokens)
        tokens = self.act(tokens)
        tokens = self.dropout(tokens)
        return tokens                                # (N, M, D)

class FrozenEncoder(nn.Module):
    """Frozen backbone for feature extraction using ConvNeXt-Base."""

    def __init__(self, trainable_layers=0):
        super(FrozenEncoder, self).__init__()
        # Load the ConvNeXt-Base model with pretrained weights.
        base_model = convnext_base(
            weights=ConvNeXt_Base_Weights.IMAGENET1K_V1,
            stochastic_depth_prob=0.1,  # <-- enable stochastic depth (e.g. 0.1)
            layer_scale=1e-6  # keep if you already use layer scale
        )

        # Use the features from the ConvNeXt model.
        # In torchvision's implementation, `base_model.features` contains all the convolutional blocks.
        self.features = base_model.features

        # Freeze all parameters in the feature extractor.
        for param in self.features.parameters():
            param.requires_grad = False

        # Ejemplo: descongelar solo el último bloque
        for param in self.features[-1].parameters():
            param.requires_grad = True

        # Unfreeze the last few parameters (or blocks) as specified by trainable_layers.
        # Note: This simple approach unfreezes the last 'trainable_layers' parameters; depending on your needs,
        # you might want to unfreeze whole blocks instead.


        # The output channels of convnext_base are 1024.
        self.norm = nn.BatchNorm2d(1024)

    def forward(self, x):
        features = self.features(x)
        features = self.norm(features)
        return features

class CapsuleFormation(nn.Module):
    def __init__(self, input_dim, num_capsules, capsule_dim):
        super(CapsuleFormation, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.linear = nn.Linear(input_dim, num_capsules * capsule_dim)
        self.norm = nn.LayerNorm(capsule_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.5)  # Added Dropout

    def forward(self, features):
        if len(features.size()) == 3:  # Handle inputs already flattened
            B, T, flattened_dim = features.size()
            features = features.view(B * T, flattened_dim)
        elif len(features.size()) == 5:
            B, T, C, H, W = features.size()
            flattened_dim = C * H * W
            features = features.view(B * T, flattened_dim)
        else:
            raise ValueError("[CapsuleFormation] Expected 3D or 5D input, got input with size {}".format(features.size()))

        capsules = self.linear(features)
        capsules = self.dropout(capsules)  # Apply Dropout
        capsules = capsules.view(B * T, self.num_capsules, self.capsule_dim)
        capsules = self.norm(capsules)
        capsules = self.activation(capsules)
        capsules = capsules.view(B, T, self.num_capsules, self.capsule_dim)  # Restore batch and temporal structure
        return capsules

class SelfAttentionRouting(nn.Module):
    def __init__(self, num_capsules, capsule_dim, heads=4):
        super(SelfAttentionRouting, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=capsule_dim, num_heads=heads)
        self.dropout = nn.Dropout(p=0.5)  # Added Dropout

    def forward(self, capsules):
        if len(capsules.size()) == 3:
            B, N, D = capsules.size()
            T = 1
            capsules = capsules.unsqueeze(1)  # Add temporal dimension
        elif len(capsules.size()) == 4:
            B, T, N, D = capsules.size()
        else:
            raise ValueError("[SelfAttentionRouting] Expected 3D or 4D input, got input with size {}".format(capsules.size()))

        capsules = capsules.view(B * T * N, D).unsqueeze(0)  # Merge batch and temporal dimensions
        routed_capsules, _ = self.multihead_attn(capsules, capsules, capsules)
        routed_capsules = self.dropout(routed_capsules)  # Apply Dropout
        routed_capsules = routed_capsules.squeeze(0).view(B, T, N, D)  # Restore original dimensions
        return routed_capsules

class RegionDecoder(nn.Module):
    """Region-specific temporal decoder that returns per-time outputs."""
    def __init__(self, capsule_dim, num_capsules, hidden_dim, output_dim, dropout_p=0.5):
        super(RegionDecoder, self).__init__()
        # We'll combine the capsules per time step (concat along capsule dimension) -> input_size = num_capsules * capsule_dim
        self.input_size = num_capsules * capsule_dim
        self.gru = nn.GRU(self.input_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, capsules):
        """
        capsules: (B, T, N, D)
        returns: (B, T, output_dim)  -> per-frame region predictions
        """
        if len(capsules.size()) != 4:
            raise ValueError("[RegionDecoder] Expected 4D input (B,T,N,D). Got: {}".format(capsules.size()))
        B, T, N, D = capsules.size()
        # flatten capsules per time-step
        x = capsules.view(B, T, N * D)  # (B, T, N*D)
        # GRU over T
        outputs, _ = self.gru(x)  # outputs: (B, T, hidden_dim)
        outputs = self.dropout(outputs)
        # apply fc to each time step
        out = self.fc(outputs)  # (B, T, output_dim)
        return out

class GazeFusion(nn.Module):
    """Fuses region-specific outputs."""
    def __init__(self, input_dim, output_dim):
        super(GazeFusion, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=0.1)  # Added Dropout

    def forward(self, regions):
        if isinstance(regions, list):
            fused = torch.cat(regions, dim=1)
        else:
            fused = regions
        fused = self.dropout(fused)  # Apply Dropout
        return self.fc(fused)

class GazeEstimationModel(nn.Module):
    def __init__(self, encoder, capsule_dim=256, hidden_dim=512, output_dim=3,
                 num_capsules=None, capsule_groups=None):
        super().__init__()
        self.encoder = encoder

        # Si defines grupos de landmarks (recomendado), el nº de cápsulas = nº de grupos
        self.capsule_groups = capsule_groups  # lista de listas de índices (o None)
        if num_capsules is None:
            if self.capsule_groups is not None:
                num_capsules = len(self.capsule_groups)
            else:
                num_capsules = 6  # fallback

        self.capsule_dim = capsule_dim
        self.num_capsules = num_capsules

        # CapsuleFormation con placeholder de input_dim (se fijará luego con set_capsule_input_dim)
        self.capsule_formation = CapsuleFormation(input_dim=1, num_capsules=self.num_capsules,
                                                  capsule_dim=self.capsule_dim)
        self.routing = SelfAttentionRouting(num_capsules=self.num_capsules, capsule_dim=self.capsule_dim)

        # Decoders construidos con el MISMO num_capsules
        self.eye_decoder = RegionDecoder(self.capsule_dim, self.num_capsules, hidden_dim, output_dim)
        self.face_decoder = RegionDecoder(self.capsule_dim, self.num_capsules, hidden_dim, output_dim)

        self.fusion = GazeFusion(output_dim * 2, output_dim)

        # --- guiado landmarks (global simple ya lo tenías) ---
        self.lmk_guidance = LandmarkGuidance(
            feat_channels=1024,
            heatmaps=1,
            sigma=2.0,
            select_every=2,
        )

        capsule_groups = get_mediapipe_landmark_groups()
        self.capsule_groups = capsule_groups
        self.num_capsule_groups = len(capsule_groups)
        self.heat_sigma = 2.0

        # Bloque de fusión landmarks→features
        self.lmk_fuse = nn.Conv2d(1024 + 1, 1024, kernel_size=1, bias=True)
        self.lmk_act = nn.GELU()

        # NUEVO: cápsulas espaciales por-región (usa M = nº de grupos de landmarks)
        self.spatial_capsules = LandmarkCapsuleFormation(
            in_channels=1024,
            num_capsules=self.num_capsule_groups,
            capsule_dim=self.capsule_dim
        )
    def set_capsule_input_dim(self, flattened_dim: int):
        """
        Crea o reconfigura CapsuleFormation cuando ya conoces flattened_dim (C*H*W del encoder).
        Mantiene num_capsules/capsule_dim y registra correctamente el submódulo.
        """
        flattened_dim = int(flattened_dim)

        if self.capsule_formation is None:
            # Crear y registrar el módulo por primera vez
            self.capsule_formation = CapsuleFormation(
                input_dim=flattened_dim,
                num_capsules=self.num_capsules,
                capsule_dim=self.capsule_dim
            )
            # Asegurar que queda en el mismo device que el resto del modelo
            self.capsule_formation.to(next(self.parameters()).device)
        else:
            # Reconfigurar solo la capa linear si ya existía
            new_linear = nn.Linear(flattened_dim, self.num_capsules * self.capsule_dim)
            # mover al device del modelo
            new_linear.to(next(self.parameters()).device)
            self.capsule_formation.linear = new_linear
    def forward(self, x, landmarks=None):
        # Handle both sequence (5D) and single frame (4D) inputs
        was_4d = False
        if len(x.size()) == 4:
            # Single frame - add temporal dimension
            was_4d = True
            x = x.unsqueeze(1)  # Shape becomes [B, 1, C, H, W]

        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        # ... empaquetado B,T y paso por encoder
        features = self.encoder(x)  # (B*T, 1024, Hf, Wf)

        # guiado global con 1 heatmap (ya lo tenías)
        if landmarks is not None and landmarks.numel() > 0:
            lmk_bt = landmarks.view(B * T, -1, 2).to(features.device)


            # Si tienes grupos por cápsula, pásalos; si no, será 1 mapa global
            Hmaps_btmd = build_per_capsule_heatmaps(
                lmk_bt,
                self.capsule_groups,  # None -> 1 mapa
                H=features.size(2),
                W=features.size(3),
                sigma=self.heat_sigma
            )
            # Fusiona de forma segura (4D + 4D -> 4D)
            features = _safe_lmk_fuse(self.lmk_fuse, self.lmk_act, features, Hmaps_btmd)
        features = features.reshape(B, T, -1)

        # Paso por encoder -> 4D
        feat_btchw = self.encoder(x)  # (B*T, 1024, Hf, Wf)

        capsules = None
        if (landmarks is not None) and (landmarks.numel() > 0):
            # Normaliza landmarks a (B*T, K, 2)
            if landmarks.dim() == 4:  # (B, T, K, 2)
                lmk_bt = landmarks.view(B * T, landmarks.size(2), 2)
            elif landmarks.dim() == 3:  # (?, K, 2)
                N = B * T
                if landmarks.size(0) == N:
                    lmk_bt = landmarks
                elif landmarks.size(0) == B:
                    lmk_bt = landmarks.repeat_interleave(T, dim=0)
                elif landmarks.size(0) == T:
                    lmk_bt = landmarks.repeat(B, 1, 1)
                elif landmarks.size(0) == 1:
                    lmk_bt = landmarks.expand(N, -1, -1)
                else:
                    lmk_bt = landmarks[:1].expand(N, -1, -1)
            else:  # (K, 2)
                lmk_bt = landmarks.unsqueeze(0).expand(B * T, -1, -1)

            lmk_bt = lmk_bt.to(feat_btchw.device, dtype=feat_btchw.dtype)

            # Heatmaps por cápsula -> (B*T, M, Hf, Wf)
            Hmaps_btmd = build_per_capsule_heatmaps(
                lmk_bt, self.capsule_groups,
                H=feat_btchw.size(2), W=feat_btchw.size(3),
                sigma=self.heat_sigma
            )

            # Fusión global 1x1 con la suma de mapas
            Hmap_sum = Hmaps_btmd.sum(dim=1, keepdim=True)  # (B*T,1,Hf,Wf)
            feat_btchw = _safe_lmk_fuse(self.lmk_fuse, self.lmk_act, feat_btchw, Hmap_sum)  # (B*T,1024,Hf,Wf)

            # Pooling ponderado por mapa -> tokens (B*T, M, D)
            tokens_btmd = self.spatial_capsules(feat_btchw, Hmaps_btmd)

            # Reorganiza a (B, T, M, D)
            capsules = tokens_btmd.view(B, T, self.spatial_capsules.num_capsules, self.spatial_capsules.capsule_dim)

        else:
            # Fallback sin landmarks: aplanar encoder y usar cápsulas densas
            features_seq = feat_btchw.view(B, T, -1)  # (B,T, 1024*Hf*Wf)
            capsules = self.capsule_formation(features_seq)  # (B,T,M,D)


        routed_capsules = self.routing(capsules)     # (B, T, N, D)

        B, T, N, D = routed_capsules.shape
        assert N == self.num_capsules and D == self.capsule_dim, \
            f"Capsulas desajustadas: routed N={N},D={D} vs cfg N={self.num_capsules},D={self.capsule_dim}"

        eye_output = self.eye_decoder(routed_capsules)    # (B, T, out_dim)
        face_output = self.face_decoder(routed_capsules)  # (B, T, out_dim)
        combined_output = torch.cat([eye_output, face_output], dim=-1)  # (B, T, out_dim*2)
        output = self.fusion(combined_output)  # (B, T, out_dim)

        if was_4d:
            output = output.squeeze(1)  # (B, out_dim) for single-frame case

        return output