
import cv2
from vision import DetectorCorporal


# Crear detector corporal
detector = DetectorCorporal()

# Abrir webcam
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    detector.cerrar()
    raise RuntimeError("No se ha podido abrir la webcam")

try:
    while True:
        # Leer imagen de la webcam
        correcto, frame = camara.read()

        if not correcto:
            print("Error al leer la webcam")
            break

        # Voltear imagen como un espejo
        frame = cv2.flip(frame, 1)

        # Detectar cuerpo
        detector.detectar(frame)

        # Obtener coordenadas
        puntos = detector.obtener_puntos()

        if puntos is not None:
            mensaje = "Cuerpo detectado"
        else:
            mensaje = "No se detecta ninguna persona"

        # Dibujar esqueleto
        frame = detector.dibujar_esqueleto(frame)

        # Mostrar mensaje
        cv2.putText(
            frame,
            mensaje,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Mostrar ventana
        cv2.imshow(
            "MOVE & DANCE - Prueba de vision",
            frame
        )

        # Salir pulsando ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Cerrar webcam y liberar recursos
    camara.release()
    detector.cerrar()
    cv2.destroyAllWindows()
