import cv2
import numpy as np
import cv2.aruco as aruco
import math

# === Cargar calibración de cámara ===
calib_data = np.load('calibracion_cam.npz')
mtx = calib_data['mtx']
dist = calib_data['dist']

# === Configurar cámara y marcador ===
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

cap = cv2.VideoCapture(0)
marker_size = 5  # tamaño del marcador en cm o metros

# Variables globales para la posición del mouse
mouse_x, mouse_y = 0, 0
frame_width, frame_height = 640, 480

# Guardar intensidad anterior por cara para suavizado
prev_intensities = [0.5]*6  # inicializar con 50%

# Callback del mouse
def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y

cv2.namedWindow('AR Cube Interactive')
cv2.setMouseCallback('AR Cube Interactive', mouse_callback)

# Función para dibujar el cubo con luz suave y que siempre apunta al mouse
def draw_cube_colored_light(frame, rvec, tvec, mtx, dist, size):
    global mouse_x, mouse_y, frame_width, frame_height, prev_intensities

    half = size / 2
    pts_3d = np.float32([
        [-half, -half, -half], [-half, half, -half], [half, half, -half], [half, -half, -half],
        [-half, -half, half], [-half, half, half], [half, half, half], [half, -half, half]
    ])
    imgpts, _ = cv2.projectPoints(pts_3d, rvec, tvec, mtx, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    faces = [
        [0,1,2,3], [4,5,6,7], [0,1,5,4],
        [2,3,7,6], [1,2,6,5], [0,3,7,4]
    ]
    normals_local = [
        [0,0,-1], [0,0,1], [-1,0,0], [1,0,0], [0,1,0], [0,-1,0]
    ]
    base_colors = [
        np.array([0, 0, 255]),    # azul
        np.array([0, 255, 0]),    # verde
        np.array([255, 0, 0]),    # rojo
        np.array([255, 255, 0]),  # amarillo
        np.array([255, 0, 255]),  # magenta
        np.array([0, 255, 255])   # cyan
    ]

    # Rotación del cubo
    R, _ = cv2.Rodrigues(rvec)
    pts_cam = (R @ pts_3d.T).T + tvec
    cube_center = np.mean(pts_cam, axis=0)

    # --- Luz apuntando correctamente hacia el mouse ---
    offset = -5  # proyectamos el mouse ligeramente delante del cubo
    mouse_plane_z = cube_center[2] + offset
    mouse_dir = np.array([
        (mouse_x / frame_width - 0.5) * 2,
        (mouse_y / frame_height - 0.5) * 2,
        mouse_plane_z
    ])
    light_vec = mouse_dir - cube_center
    distance = np.linalg.norm(light_vec)
    light_dir = light_vec / distance

    # Atenuación suave según distancia
    attenuation = 1 / (1 + distance * 0.2)

    # Ordenar caras por profundidad
    face_depths = []
    for i, face in enumerate(faces):
        pts_face = pts_3d[face]
        pts_cam_face = (R @ pts_face.T).T + tvec
        z_mean = np.mean(pts_cam_face[:,2])
        face_depths.append((i, z_mean))
    face_depths.sort(key=lambda x: x[1], reverse=True)

    # Dibujar caras con transición suave
    alpha = 0.2  # factor de suavizado
    for idx, _ in face_depths:
        normal_local = np.array(normals_local[idx])
        normal_cam = (R @ normal_local.reshape(3,1)).flatten()
        intensity = np.dot(normal_cam, light_dir)
        intensity = np.clip(intensity, 0, 1)
        intensity = intensity * attenuation + 0.3  # brillo mínimo
        intensity = np.clip(intensity, 0, 1)

        # Suavizado de transición
        intensity = prev_intensities[idx] * (1 - alpha) + intensity * alpha
        prev_intensities[idx] = intensity

        color = tuple(int(c * intensity) for c in base_colors[idx])
        cv2.drawContours(frame, [imgpts[faces[idx]]], -1, color, -1)
        cv2.polylines(frame, [imgpts[faces[idx]]], True, (0,0,0), 2)

    return frame

# Rotación del cubo
def rotate_rvec_xyz(rvec, angles_deg):
    rx, ry, rz = [math.radians(a) for a in angles_deg]
    R, _ = cv2.Rodrigues(rvec)
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    R_new = R @ Rx @ Ry @ Rz
    rvec_new, _ = cv2.Rodrigues(R_new)
    return rvec_new

angle_x, angle_y, angle_z = 0, 0, 0

print("Cubo AR interactivo con luz coherente hacia el mouse. Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, mtx, dist)
        for rvec, tvec in zip(rvecs, tvecs):
            rvec_rot = rotate_rvec_xyz(rvec, (angle_x, angle_y, angle_z))
            frame = draw_cube_colored_light(frame, rvec_rot, tvec, mtx, dist, marker_size)

    cv2.imshow('AR Cube Interactive', frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

    angle_x = (angle_x + 2) % 360
    angle_y = (angle_y + 3) % 360
    angle_z = (angle_z + 1) % 360

cap.release()
cv2.destroyAllWindows()
