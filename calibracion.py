import cv2
import numpy as np

# Parámetros del tablero de ajedrez
CHESSBOARD_SIZE = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vectores para almacenar puntos 3D y 2D
objpoints = []
imgpoints = []

# Puntos 3D reales (en el mundo)
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

cap = cv2.VideoCapture(0)
count = 0

print("Presiona 'c' para capturar el tablero. Necesitas al menos 15 buenas imágenes.")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Buscar esquinas del tablero
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret)

    cv2.imshow('Calibración - Captura', frame)
    key = cv2.waitKey(1) & 0xFF

    # Captura si se presiona 'c'
    if key == ord('c') and ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        count += 1
        print(f"Imagen capturada: {count}")

    # Salir con ESC
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Calibrar si hay suficientes imágenes
if len(objpoints) > 10:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Matriz de cámara:\n", mtx)
    print("Coeficientes de distorsión:\n", dist)

    # Guardar para usar luego
    np.savez("calibracion_cam.npz", mtx=mtx, dist=dist)
    print("Calibración guardada en 'calibracion_cam.npz'")
else:
    print("No hay suficientes imágenes para calibrar.")
