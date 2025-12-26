#!/usr/bin/env python3
"""
Precalcula facial landmarks con MediaPipe Face Landmarker v2 y los guarda en HDF5.

- Lee HDF5 con dataset 'face_patch' de forma (N, H, W, C) (BGR o RGB; detectamos y convertimos).
- Escribe:
    face_landmarks: float32 (N, 468, 2)  coords normalizadas en [0,1] respecto a (W,H) del patch
    lmk_valid:      uint8  (N,)  1 si hay detección, 0 si no

Uso:
  python precompute_landmarks_to_h5.py --input /ruta/a/carpeta_o_archivo.h5 \
      --model face_landmarker_v2_with_blendshapes.task --overwrite

Notas:
  - Si 'face_landmarks' ya existe y no pasas --overwrite, se salta el archivo.
  - Si el frame es BGR (OpenCV típico), lo convertimos a RGB antes de pasar a MediaPipe.
  - Si hay varias caras, se usa la primera.
"""

import argparse
import os
import sys
import glob
import h5py
import numpy as np
from tqdm import tqdm
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def normalize_k(xy, force_k=None, drop_iris=False):
    """
    xy: (K,2) float32 en [0,1] devuelto por MediaPipe (K=468 o 478).
    Devuelve (Kout,2) según opciones.
    """
    K = xy.shape[0]
    if force_k is None and not drop_iris:
        # AUTO: si hay 478, nos quedamos con 478; si hay 468, con 468.
        return xy

    if drop_iris and force_k is None:
        force_k = 468

    if force_k == 468:
        if K == 478:
            return xy[:468]  # quitamos 10 de iris (5 por ojo)
        elif K == 468:
            return xy
        else:
            raise ValueError(f"Landmarks inesperados: {K}, no puedo truncar a 468.")
    elif force_k == 478:
        if K == 478:
            return xy
        elif K == 468:
            # No podemos inventar los 10 del iris: rellenamos con -1 (no-datos)
            pad = np.full((10, 2), -1.0, dtype=np.float32)
            return np.concatenate([xy, pad], axis=0)
        else:
            raise ValueError(f"K inesperado: {K}")
    else:
        return xy

def probe_k_and_color(detector, patches, force_rgb, force_k, drop_iris):
    """
    Escanea unos pocos frames hasta encontrar una detección válida para inferir Kout.
    Devuelve (Kout, is_bgr).
    """
    # Asumimos BGR por defecto si viene de OpenCV; ajusta si tienes certeza de RGB.
    assumed_bgr = True if not force_rgb else False

    # Probaremos hasta 50 frames repartidos
    N = len(patches)
    idxs = np.linspace(0, max(N-1,0), num=min(50, N), dtype=int)
    for i in idxs:
        arr = patches[i]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if assumed_bgr else arr
        ok, xy = detect_landmarks_on_image(detector, img_rgb)
        if ok:
            xy = normalize_k(xy, force_k=force_k, drop_iris=drop_iris)
            return xy.shape[0], assumed_bgr

    # Si no detectamos en ninguna, elegimos Kout por force_k o por defecto 468
    fallback_k = force_k if force_k is not None else (478 if not drop_iris else 468)
    return fallback_k, assumed_bgr

def is_bgr_like(arr):
    # Heurística: si parece venir de OpenCV (BGR)
    # No es infalible, pero la mayoría de pipelines guardan BGR en HDF5.
    # Si quieres forzar RGB, añade --force-rgb y sáltate este check.
    # Aquí asumimos BGR por defecto.
    return True

def detect_landmarks_on_image(detector, img_rgb):
    """
    img_rgb: np.uint8 HxWx3 en RGB
    Devuelve:
      (valid, lmk_xy) donde lmk_xy es (468,2) en [0,1] (coords normalizadas MediaPipe)
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    res = detector.detect(mp_image)

    if not res.face_landmarks or len(res.face_landmarks) == 0:
        return False, None

    # Tomamos la primera cara
    lmks = res.face_landmarks[0]  # list of NormalizedLandmark
    # Extraer x,y normalizados [0,1]
    xy = np.array([[lm.x, lm.y] for lm in lmks], dtype=np.float32)  # (468,2)
    return True, xy

def process_h5(h5_path, detector, overwrite=False, force_rgb=False, force_k=None, drop_iris=False):
    with h5py.File(h5_path, 'a') as f:
        if 'face_patch' not in f:
            print(f"[WARN] {h5_path} no contiene 'face_patch'. Saltando.")
            return

        patches = f['face_patch']
        N = patches.shape[0]

        if 'face_landmarks' in f and not overwrite:
            print(f"[INFO] {h5_path} ya tiene 'face_landmarks'. Usa --overwrite para recalcular.")
            return

        # Averigua K (468/478) antes de crear el dataset
        Kout, assumed_bgr = probe_k_and_color(detector, patches, force_rgb, force_k, drop_iris)

        # Recrea datasets si existen
        if 'face_landmarks' in f:
            del f['face_landmarks']
        if 'lmk_valid' in f:
            del f['lmk_valid']

        dset_lm = f.create_dataset('face_landmarks', shape=(N, Kout, 2), dtype='float32', chunks=True)
        dset_ok = f.create_dataset('lmk_valid', shape=(N,), dtype='uint8', chunks=True)

        for i in tqdm(range(N), desc=os.path.basename(h5_path), unit='frm'):
            arr = patches[i]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if (not force_rgb and assumed_bgr) else arr
            ok, xy = detect_landmarks_on_image(detector, img_rgb)
            if not ok:
                dset_ok[i] = 0
                dset_lm[i, :, :] = -1.0
                continue

            xy = normalize_k(xy, force_k=force_k, drop_iris=drop_iris)
            if xy.shape[0] != Kout:
                # Puede pasar si probamos sin detección y luego sí detecta con otra K
                if force_k is None and not drop_iris:
                    # AUTO: adaptamos on-the-fly redimensionando el dataset la primera vez
                    # (HDF5 no permite cambiar la 2ª dim fácilmente; así que avisamos)
                    print(f"[WARN] {h5_path}: detectado K={xy.shape[0]} distinto a Kout={Kout}. "
                          f"Usa --overwrite con --force-k 468 o 478 para fijar. Marco este frame como no válido.")
                    dset_ok[i] = 0
                    dset_lm[i, :, :] = -1.0
                    continue
                else:
                    raise RuntimeError(f"K inesperado en frame {i}: {xy.shape[0]} vs {Kout}")

            dset_ok[i] = 1
            dset_lm[i, :, :] = xy

        print(f"[OK] Guardado 'face_landmarks' (K={Kout}) y 'lmk_valid' en {h5_path}")

def iter_h5_files(input_path):
    if os.path.isdir(input_path):
        # Busca .h5 en toda la carpeta (recursivo)
        for p in glob.glob(os.path.join(input_path, '**', '*.h5'), recursive=True):
            yield p
    elif os.path.isfile(input_path) and input_path.endswith('.h5'):
        yield input_path
    else:
        print(f"[ERROR] Ruta inválida: {input_path}")
        sys.exit(1)

def build_detector(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,                 # no necesitamos blendshapes aquí
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE          # frame a frame (datasets)
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True,
                    help='Ruta a un archivo .h5 o a una carpeta con múltiples .h5')
    ap.add_argument('--model', required=True,
                    help='Ruta a face_landmarker_v2_with_blendshapes.task (o equivalente v2)')
    ap.add_argument('--overwrite', action='store_true',
                    help='Si existe face_landmarks, lo recalcula')
    ap.add_argument('--force-rgb', action='store_true',
                    help='Asume que face_patch ya está en RGB (no convierte desde BGR)')
    ap.add_argument('--force-k', type=int, choices=[468, 478],
                    help='Fuerza a 468 (sin iris) o 478 (con iris). Si no se pasa, AUTO.')
    ap.add_argument('--drop-iris', action='store_true',
                    help='Equivalente a --force-k 468 (descarta los últimos 10 puntos de iris).')
    args = ap.parse_args()

    force_k = args.force_k
    if args.drop_iris:
        force_k = 468

    detector = build_detector(args.model)
    for h5_path in iter_h5_files(args.input):
        try:
            process_h5(h5_path, detector,
                       overwrite=args.overwrite,
                       force_rgb=args.force_rgb,
                       force_k=force_k,
                       drop_iris=args.drop_iris)
        except Exception as e:
            print(f"[ERROR] Falló {h5_path}: {e}")


if __name__ == '__main__':
    main()
