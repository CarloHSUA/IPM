
import cv2
import mediapipe as mp
from config import config


class DetectorCorporal:
    """
    Detecta la postura corporal utilizando MediaPipe Pose Landmarker.
    Permite obtener las coordenadas de las articulaciones
    y dibujar el esqueleto.
    """

    def __init__(self):
        # Configurar MediaPipe
        self.mp_pose = mp.solutions.pose

        opciones = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=config.model_path
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1
        )

        # Crear el detector
        self.detector = (
            mp.tasks.vision.PoseLandmarker.create_from_options(
                opciones
            )
        )

        self.timestamp = 0
        self.puntos = None

    def detectar(self, frame):
        """
        Detecta el cuerpo en un fotograma de OpenCV.
        Devuelve los landmarks o None.
        """

        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        imagen_mp = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Incrementar timestamp
        self.timestamp += 33

        resultado = self.detector.detect_for_video(
            imagen_mp,
            self.timestamp
        )

        if resultado.pose_landmarks:
            self.puntos = resultado.pose_landmarks[0]
        else:
            self.puntos = None

        return self.puntos

    def obtener_puntos(self):
        """
        Devuelve las coordenadas normalizadas de
        las principales articulaciones.
        """

        if self.puntos is None:
            return None

        articulaciones = {
            "hombro_izquierdo": 11,
            "hombro_derecho": 12,
            "codo_izquierdo": 13,
            "codo_derecho": 14,
            "muneca_izquierda": 15,
            "muneca_derecha": 16,
            "cadera_izquierda": 23,
            "cadera_derecha": 24,
            "rodilla_izquierda": 25,
            "rodilla_derecha": 26,
            "tobillo_izquierdo": 27,
            "tobillo_derecho": 28
        }

        puntos = {}

        for nombre, indice in articulaciones.items():
            landmark = self.puntos[indice]

            puntos[nombre] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibilidad": landmark.visibility
            }

        return puntos

    def dibujar_esqueleto(self, frame):
        """
        Dibuja el esqueleto sobre la imagen.
        """

        if self.puntos is None:
            return frame

        alto, ancho = frame.shape[:2]

        # Dibujar conexiones
        for inicio, fin in self.mp_pose.POSE_CONNECTIONS:
            punto1 = self.puntos[inicio]
            punto2 = self.puntos[fin]

            x1 = int(punto1.x * ancho)
            y1 = int(punto1.y * alto)
            x2 = int(punto2.x * ancho)
            y2 = int(punto2.y * alto)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

        # Dibujar articulaciones
        for landmark in self.puntos:
            x = int(landmark.x * ancho)
            y = int(landmark.y * alto)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

        return frame

    def cerrar(self):
        """Libera los recursos del detector."""
        self.detector.close()
