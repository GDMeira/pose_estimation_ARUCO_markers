'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''
#TODO: ajustar fitSin para pegar só os últimos pontos de positions e ir atualizando o gráfico

import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

positions = []

def fitSin(px, py, ax, linex, liney, linez):
    if (len(positions) < 100):
        return px, py
    
    # Função senoidal
    def senoide(x, amplitude, frequencia, fase):
        return amplitude * np.sin(2 * np.pi * frequencia * x + fase)

    # Separação das coordenadas x, y e t
    arrayData = np.array(positions)
    x = arrayData[-100:,0]
    y = arrayData[-100:,1]
    z = arrayData[-100:,2]
    t = arrayData[-100:,3]

    # Ajuste da senoide para a variação em x
    parametros_x, _ = curve_fit(senoide, t, x, p0=px, maxfev=1000)
    amplitude_x, frequencia_x, fase_x = parametros_x

    # Ajuste da senoide para a variação em y
    parametros_y, _ = curve_fit(senoide, t, y, p0=py, maxfev=1000)
    amplitude_y, frequencia_y, fase_y = parametros_y

    p0x = parametros_x
    p0y = parametros_y

    # Atualizar o gráfico
    linex.set_data(t, x)
    linex.set_label(f'X f {frequencia_x:,.2f}, fase {fase_x:,.2f}')
    liney.set_data(t, y)
    liney.set_label(f'Y f {frequencia_y:,.2f}, fase {fase_y:,.2f}')
    linez.set_data(t, z)

    ax[0].relim()  # Recalcular os limites dos eixos
    ax[0].autoscale_view()  # Redimensionar o gráfico
    ax[0].legend()
    ax[1].relim()  # Recalcular os limites dos eixos
    ax[1].autoscale_view()  # Redimensionar o gráfico
    ax[1].legend()
    ax[2].relim()  # Recalcular os limites dos eixos
    ax[2].autoscale_view()  # Redimensionar o gráfico
    plt.draw()
    plt.pause(0.001)

    return p0x, p0y

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()

    aruco_detector = cv2.aruco.ArucoDetector(cv2.aruco_dict, parameters)


    corners, ids, rejected_img_points = aruco_detector.detectMarkers(gray)

        # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
            pos = tvec[0][0][:] #posições x, y e z do qrcode [x hor, y vert e z profund]
            t = time.perf_counter() - start
            pos = np.append(pos * 100, t) #tempo da medição
            positions.append(pos)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners) 

    return frame

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]
    
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)
    start = time.perf_counter()

    p0x = [1,1,0]
    p0y = [1,1,0]

    # Configurações do gráfico
    plt.ion()  # Ativa o modo de plotagem interativo

    # Criar a figura e o objeto de plotagem
    fig1, ax = plt.subplots(nrows=1, ncols=3)
    linex, = ax[0].plot(0, 0)
    liney, = ax[1].plot(0, 0)
    linez, = ax[2].plot(0, 0)

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)

        p0x, p0y = fitSin(p0x, p0y, ax, linex, liney, linez)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()