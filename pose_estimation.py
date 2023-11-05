'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''

import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from collections import deque

positions = []
t_x = []
t_y = []

def fitSin(px, py, ax, linex, liney, linez, linex_fit, liney_fit):
    lengthOfInterval = 150
    # lengthOfInterval = 300

    if (len(positions) < lengthOfInterval):
        return px, py
    
    # Função senoidal
    def senoide(x, amplitude, frequencia, fase):
        return amplitude * np.sin(2 * np.pi * frequencia * x + fase)
    
    def moving_average(data, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(data, window, 'same')

    # Separação das coordenadas x, y e t
    arrayData = np.array(positions)
    x = arrayData[-lengthOfInterval:,0].copy()
    y = arrayData[-lengthOfInterval:,1].copy()
    z = arrayData[-lengthOfInterval:,2].copy()
    t = arrayData[-lengthOfInterval:,3].copy()

    x = x - np.mean(arrayData[:,0])
    y = y - np.mean(arrayData[:,1])

    # n = 10 # length of moving average window
    # x = moving_average(x, n)
    # x = x[n:-n]
    # y = moving_average(y, n)
    # y = y[n:-n]
    # t = t[n:-n]

    bounds = ([0.08, 0, -5*np.pi], [0.5, 0.4, 5*np.pi])

    # Ajuste da senoide para a variação em x
    parametros_x, _ = curve_fit(senoide, t, x, p0=px, bounds=bounds, maxfev=5000)
    amplitude_x, frequencia_x, fase_x = parametros_x

    # Ajuste da senoide para a variação em y
    parametros_y, _ = curve_fit(senoide, t, y, p0=py, bounds=bounds, maxfev=5000)
    amplitude_y, frequencia_y, fase_y = parametros_y

    t_x.append(1/frequencia_x)
    t_y.append(1/frequencia_y)
    interval = 30

    p0x = parametros_x
    p0y = parametros_y
    x_fit = senoide(t, *parametros_x)
    y_fit = senoide(t, *parametros_y)

    # Atualizar o gráfico
    linex.set_data(t, x)
    linex.set_label(f'Hor período {1/frequencia_x:,.2f}s, fase {fase_x*180/np.pi:,.0f} degrees \n mean: {np.mean(t_x[-interval:]):.1f} s, std: {np.std(t_x[-interval:]):.1f} s')
    liney.set_data(t, y)
    liney.set_label(f'Vert período {1/frequencia_y:,.2f}s, fase {fase_y*180/np.pi:,.0f} degrees \n mean: {np.mean(t_y[-interval:]):.1f} s, std: {np.std(t_y[-interval:]):.1f} s')
    linez.set_data(arrayData[-lengthOfInterval:,0], arrayData[-lengthOfInterval:,1])
    Ax = 22
    Ay = 9
    histereses = 4*np.sin((fase_x - fase_y) / 2)*Ax*Ay / np.sqrt(Ax**2 + Ay**2)
    linez.set_label(f'Profundidade. Histerese: {histereses:,.1f} mm')
    linex_fit.set_data(t, x_fit)
    linex_fit.set_label('x adjust')
    liney_fit.set_data(t, y_fit)
    liney_fit.set_label('y adjust')

    ax[0].relim()  # Recalcular os limites dos eixos
    ax[0].autoscale_view()  # Redimensionar o gráfico
    ax[0].legend(loc='upper right')
    ax[1].relim()  # Recalcular os limites dos eixos
    ax[1].autoscale_view()  # Redimensionar o gráfico
    ax[1].legend(loc='upper right')
    ax[2].relim()  # Recalcular os limites dos eixos
    ax[2].autoscale_view()  # Redimensionar o gráfico
    ax[2].legend(loc='upper right')
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
            pos = np.append(pos * 100, t) # tempo da medição
            # n = 6 # length of moving average window
            # if len(positions) > n - 1:
            #     sum1 = 0
            #     sum2 = 0

            #     for i in range(n - 1):
            #         sum1 = sum1 + positions[-1-i][0]
            #         sum2 = sum2 + positions[-1-i][1]

            #     pos[0] = (pos[0] + sum1) / 6
            #     pos[1] = (pos[1] + sum2) / 6  
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

    p0x = [0.2,0.2,0]
    p0y = [0.1,0.2,0]

    # Configurações do gráfico
    plt.ion()  # Ativa o modo de plotagem interativo

    # Criar a figura e o objeto de plotagem
    fig1, ax = plt.subplots(nrows=1, ncols=3)
    fig1.set_figwidth(12)
    fig1.set_figheight(5)
    linex, = ax[0].plot(0, 0)
    linex_fit, = ax[0].plot(0, 0, linestyle='--', color='orange')
    liney, = ax[1].plot(0, 0)
    liney_fit, = ax[1].plot(0, 0, linestyle='--', color='green')
    linez, = ax[2].plot(0, 0)
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)

        p0x, p0y = fitSin(p0x, p0y, ax, linex, liney, linez, linex_fit, liney_fit)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            arrayData = np.array(positions)
            x = arrayData[:,0]
            y = arrayData[:,1]

            arrayData[:,0] = x - np.mean(x)
            arrayData[:,1] = y - np.mean(y)

            string = "\n".join([" ".join(map(str, row)) for row in arrayData])

            with open('./data.txt', 'w+') as data:
                data.write(string)
            break

    video.release()
    cv2.destroyAllWindows()