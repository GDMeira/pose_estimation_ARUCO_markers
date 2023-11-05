'''
Sample Usage:-
python pose_estimation3.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
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
t_x = deque([0] * 30, maxlen=30)
t_y = deque([0] * 30, maxlen=30)

def fft_signal(ax, linex, liney, linez):
    lengthOfInterval = 200
    # lengthOfInterval = 300

    if (len(positions) < lengthOfInterval):
        return 
    
    def moving_average(data, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(data, window, 'same')

    # Separação das coordenadas x, y e t
    arrayData = np.array(positions)
    x = arrayData[-lengthOfInterval:,0].copy()
    y = arrayData[-lengthOfInterval:,1].copy()
    z = arrayData[-lengthOfInterval:,2].copy()
    t = arrayData[-lengthOfInterval:,3].copy()

    # arrayData = np.array(positions)
    # x = arrayData[:,0].copy()
    # y = arrayData[:,1].copy()
    # z = arrayData[:,2].copy()
    # t = arrayData[:,3].copy()

    x = x - np.mean(arrayData[:,0])
    y = y - np.mean(arrayData[:,1])

    # n = 10 # length of moving average window
    # x = moving_average(x, n)
    # x = x[n:-n]
    # y = moving_average(y, n)
    # y = y[n:-n]
    # t = t[n:-n]

    # FFT
    N = len(t)  # Número de pontos
    dt = t[1] - t[0]  # Intervalo de tempo entre amostras
    time_diff = t[1:] - t[:-1]

    # Calcule a média dos intervalos de tempo
    mean_dt = np.mean(time_diff)
    frequencies = np.fft.fftfreq(N, mean_dt)
    positive_frequencies_mask = frequencies > 0

    fft_result_x = np.fft.fft(x).copy()
    peak_frequency_x = np.abs(frequencies[np.argmax(np.abs(fft_result_x))])
    if (peak_frequency_x > 0): t_x.append(1 / peak_frequency_x)

    fft_result_y = np.fft.fft(y).copy()
    peak_frequency_y = np.abs(frequencies[np.argmax(np.abs(fft_result_y))])
    if (peak_frequency_y > 0): t_y.append(1 / peak_frequency_y)

    # Atualizar o gráfico
    linex.set_data(frequencies[positive_frequencies_mask], np.abs(fft_result_x[positive_frequencies_mask]))
    linex.set_label(f'Pico horizontal em {peak_frequency_x:.2f} Hz \n mean: {np.mean(t_x):.1f} s, std: {np.std(t_x):.1f} s')
    liney.set_data(frequencies[positive_frequencies_mask], np.abs(fft_result_y[positive_frequencies_mask]))
    liney.set_label(f'Pico vertical em {peak_frequency_y:.2f} Hz \n mean: {np.mean(t_y):.1f} s, std: {np.std(t_y):.1f} s')
    linez.set_data(arrayData[-lengthOfInterval:,0], arrayData[-lengthOfInterval:,1])
    Ax = 22
    Ay = 9
    linez.set_label(f'Profundidade.')

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

    return 

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, refine_markers=False):

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

    if refine_markers: #Board ?
        corners, ids, rejected_img_points = cv2.aruco.refineDetectedMarkers(
            gray, corners, ids, rejected_img_points, matrix_coefficients, distortion_coefficients
        )

    # If markers are detected
    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
            cv2.aruco.refineDetectedMarkers
            pos = tvec[0][0][:] #posições x, y e z do qrcode [x hor, y vert e z profund]
            t = time.perf_counter() - start
            pos = np.append(pos * 1000, t) # tempo da medição

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

    # Configurações do gráfico
    plt.ion()  # Ativa o modo de plotagem interativo

    # Criar a figura e o objeto de plotagem
    fig1, ax = plt.subplots(nrows=1, ncols=3)
    fig1.set_figwidth(12)
    fig1.set_figheight(5)
    linex, = ax[0].plot(0, 0)
    liney, = ax[1].plot(0, 0)
    linez, = ax[2].plot(0, 0)
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')
    ax[2].legend(loc='upper right')

    while True:
        ret, frame = video.read()

        if not ret:
            break
        
        output = pose_esitmation(frame, aruco_dict_type, k, d)

        fft_signal(ax, linex, liney, linez)

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