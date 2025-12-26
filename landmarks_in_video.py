from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# ---------- Dibujo de landmarks (igual que el tuyo) ----------
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image

def overlay_top_blendshapes(frame_bgr, detection_result, topk=5):
    # detection_result.face_blendshapes: List[List[Category]]
    if not detection_result.face_blendshapes:
        return frame_bgr
    first_face = detection_result.face_blendshapes[0]  # <- ya es List[Category]
    if not first_face:
        return frame_bgr

    # Ordenar por score descendente
    top = sorted(first_face, key=lambda c: c.score, reverse=True)[:topk]

    x, y0 = 10, 30
    for i, c in enumerate(top):
        name = c.category_name or getattr(c, "display_name", f"cat_{i}")
        txt = f"{name}: {c.score:.3f}"
        y = y0 + i * 22
        cv2.putText(frame_bgr, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return frame_bgr

def main():
    # Carga del modelo
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        running_mode=vision.RunningMode.VIDEO  # <--- clave para webcam (detect_for_video)
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)  # 0 = webcam por defecto
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara. Comprueba permisos / índice de cámara.")

    # (Opcional) ajusta resolución si quieres
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Empaquetar como mp.Image (SRGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Timestamp en ms (necesario para VIDEO)
        timestamp_ms = int(time.time() * 1000)

        # Inferencia
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)

        # Dibujo de landmarks en una copia RGB
        annotated_rgb = draw_landmarks_on_image(frame_rgb, detection_result)

        # Volver a BGR para OpenCV
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        # Overlay de blendshapes top-K (opcional)
        annotated_bgr = overlay_top_blendshapes(annotated_bgr, detection_result, topk=5)

        # FPS overlay (opcional)
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated_bgr, f"FPS: {fps:.1f}", (annotated_bgr.shape[1]-140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Face Landmarks (webcam)", annotated_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q o ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
