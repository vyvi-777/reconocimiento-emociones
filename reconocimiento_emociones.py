"""
Sistema de Reconocimiento de Emociones Faciales en Tiempo Real
Librerías: OpenCV (video) + DeepFace (IA)
Controles: Q = salir | S = guardar captura
"""

import time
import cv2
from deepface import DeepFace

# ---------------------- CONFIGURACIÓN ----------------------
INDICE_CAMARA = 0           # 0 = cámara por defecto (prueba 1 si no abre)
ANALIZAR_CADA_N_FRAMES = 5  # analizar 1 de cada N frames (mejora los FPS)

# Traducción de emociones (DeepFace devuelve los nombres en inglés)
TRADUCCION = {
    "happy": "Feliz",
    "sad": "Triste",
    "angry": "Enojado",
    "surprise": "Sorpresa",
    "fear": "Miedo",
    "disgust": "Disgusto",
    "neutral": "Neutral",
}

# Colores BGR por emoción (para el recuadro y el texto)
COLORES = {
    "happy": (0, 200, 0),
    "sad": (200, 100, 0),
    "angry": (0, 0, 255),
    "surprise": (0, 215, 255),
    "fear": (180, 0, 180),
    "disgust": (0, 120, 80),
    "neutral": (200, 200, 200),
}


def analizar_emocion(frame):
    """
    Envía el frame a DeepFace y devuelve (emocion, confianza, region).
    Devuelve None si no se detecta un rostro válido.
    """
    try:
        # enforce_detection=False evita que se lance una excepción
        # cuando no hay rostro visible en el frame.
        resultado = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",  # detector rápido, ideal para tiempo real
            silent=True,
        )
        # En versiones recientes devuelve una lista (un item por rostro)
        if isinstance(resultado, list):
            resultado = resultado[0]

        # Con enforce_detection=False, si no hay rostro DeepFace devuelve
        # una región falsa; la descartamos con face_confidence.
        if resultado.get("face_confidence", 1) < 0.5:
            return None

        emocion = resultado["dominant_emotion"]
        confianza = resultado["emotion"][emocion]
        region = resultado["region"]  # dict con x, y, w, h
        return emocion, confianza, region

    except Exception as error:
        # Cualquier fallo puntual no detiene el programa
        print(f"[Aviso] Fallo temporal en el análisis: {error}")
        return None


def dibujar_resultado(frame, datos):
    """Dibuja el recuadro del rostro y la emoción traducida."""
    emocion, confianza, region = datos
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    color = COLORES.get(emocion, (255, 255, 255))
    texto = f"{TRADUCCION.get(emocion, emocion)} ({confianza:.0f}%)"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    # Fondo del texto para que se lea bien
    cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)
    cv2.putText(frame, texto, (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def main():
    # ---------------------- CAPTURA DE VIDEO ----------------------
    camara = cv2.VideoCapture(INDICE_CAMARA)
    if not camara.isOpened():
        print("Error: no se pudo acceder a la cámara.")
        return

    ultimo_resultado = None  # se reutiliza entre frames no analizados
    contador_frames = 0
    tiempo_anterior = time.time()
    capturas = 0

    print("Sistema iniciado. Presiona 'Q' para salir, 'S' para guardar captura.")

    while True:
        ok, frame = camara.read()
        if not ok:
            print("Aviso: no se pudo leer el frame de la cámara.")
            break

        # Efecto espejo (más natural para el usuario)
        frame = cv2.flip(frame, 1)

        # ---------------------- ANÁLISIS CON IA ----------------------
        # Solo se analiza 1 de cada N frames: la IA es costosa
        if contador_frames % ANALIZAR_CADA_N_FRAMES == 0:
            ultimo_resultado = analizar_emocion(frame)
        contador_frames += 1

        # ---------------------- RENDERIZADO ----------------------
        if ultimo_resultado is not None:
            dibujar_resultado(frame, ultimo_resultado)
        else:
            cv2.putText(frame, "Buscando rostro...", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Cálculo de FPS
        tiempo_actual = time.time()
        fps = 1.0 / max(tiempo_actual - tiempo_anterior, 1e-6)
        tiempo_anterior = tiempo_actual
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Reconocimiento de Emociones", frame)

        # ---------------------- CONTROLES ----------------------
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("s"):
            capturas += 1
            nombre = f"captura_{capturas}.png"
            cv2.imwrite(nombre, frame)
            print(f"Captura guardada: {nombre}")

    # Liberar recursos
    camara.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
