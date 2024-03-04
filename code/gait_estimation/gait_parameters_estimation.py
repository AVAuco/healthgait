import os
import json
import pathlib
import argparse
import sys
import cv2
import numpy as np

import pandas as pd


from scipy import fftpack
from scipy.spatial import distance
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

def parse_opt():

    parser = argparse.ArgumentParser(
        description = 'Get gait parameters for every patient'
    )

    parser.add_argument('--pose_path', type = str, required = True, help = 'folder with pose estimation')
    parser.add_argument('--sensors_path', type = str, required = True, help = 'folder with sensors location')
    parser.add_argument('--segmentation_path', type = str, required = True, help = 'folder with segmentation patient info')
    parser.add_argument('--scale', type = float, required = True, help = 'scale used to calculate real distances')
    parser.add_argument('--csv_output', type = str, required = True, help = 'csv with the gait parameters estimation')
    parser.add_argument('--fps', type = float, required = True, help = 'frames per second')

    args = parser.parse_args()

    return args



def find_intersections(segmentation_path, patient, gait_type, class_name, clip_number, sensor_data):

    # Almacenar en una lista los puntos de intersección.
    intersections = []

    prev_position = None

    # Para esta misma persona recorrer todos los fotogramas de la segmentación
    for frame_name in sorted(os.listdir(os.path.join(segmentation_path, patient, gait_type, f"{class_name}_{clip_number}_DensePose"))):
        
        # En el momento que el punto atraviese una de las rectas no tiene
        # sentido seguir comprobandola.
        if len(intersections) == 2:
            continue

        # Lectura del fotograma

        segmentation = cv2.imread(os.path.join(segmentation_path, patient, gait_type, f"{class_name}_{clip_number}_DensePose", frame_name))

        # Checking if the image is empty or not 
        if segmentation is None: 
            print("Image is empty!!")
            continue

        # Seleccionar solamente aquella porción del fotograma correspondiente
        # a los zapatos.

        # Definir una lista de colores que deseas buscar (en formato BGR)
        colores_deseados = [
            np.array([0, 255, 255], dtype=np.uint8),
            np.array([0, 0, 128], dtype=np.uint8),  
        ]

        # Crear una máscara vacía
        mascara_total = np.zeros(segmentation.shape[:2], dtype=np.uint8)

        # Iterar sobre cada color deseado y crear máscaras individuales
        for color_deseado in colores_deseados:
            mascara = cv2.inRange(segmentation, color_deseado, color_deseado)
            mascara_total = cv2.bitwise_or(mascara_total, mascara)

        #pixeles_colores_deseados = cv2.bitwise_and(segmentation, segmentation, mask=mascara_total)
        # Calcular el contorno
        contours, _ = cv2.findContours(mascara_total, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Sin en el frame actual no aparecen ninguno de los zapatos,
        # se sigue con el siguiente fotograma de segmentación
        if len(contours) == 0:

            # print("Contours error")

            continue

        point = None

        min_x = sys.maxsize

        max_x = -sys.maxsize - 1

        for contour in contours:

            for coordenates in contour:

                actual_x = coordenates[0][0]

                # Menor x
                if int(clip_number) == 2:

                    if actual_x < min_x:

                        min_x = actual_x

                        point = (min_x, coordenates[0][1])

                        
                # Mayor x
                else:

                    if actual_x > max_x:

                        max_x = actual_x

                        point = (max_x, coordenates[0][1])

                        
        #pixeles_colores_deseados = cv2.circle(pixeles_colores_deseados, point, radius=0, color=(255, 255, 255), thickness=-1)

        # Obtener las rectas
        x1, y1 = sensor_data['0']['x_min'], sensor_data['0']['y_min'] # Punto A
        x2, y2 = sensor_data['1']['x_min'], sensor_data['1']['y_min'] # Punto B

        x3, y3 = sensor_data['2']['x_max'], sensor_data['2']['y_min'] # Punto C
        x4, y4 = sensor_data['3']['x_max'], sensor_data['3']['y_min'] # Punto D

        # aux_img = np.zeros((540, 960, 3))

        # aux_img = cv2.circle(aux_img, (x1, y1), radius=0, color = (0, 0, 255), thickness = 5)
        # aux_img = cv2.circle(aux_img, (x3, y3), radius=0, color = (0, 0, 255), thickness = 5)

        # aux_img = cv2.drawContours(aux_img, contours, -1, (255,255,255), 3)

        # aux_img = cv2.circle(aux_img, point, radius=0, color = (0, 0, 255), thickness = 5)

        # aux_img = cv2.line(aux_img, (x1, y1), (x2, y2), color = (255, 0, 0), thickness = 5)

        # aux_img = cv2.line(aux_img, (x3, y3), (x4, y4), color = (255, 0, 0), thickness = 5)

        # file_aux_path = os.path.join('/opt2/data/datasets/UCA-GAIT/code/gait_parameters_estimation/aux', patient, gait_type, class_name, clip_number)

        # pathlib.Path(file_aux_path).mkdir(parents = True, exist_ok = True)

        # cv2.imwrite(os.path.join(file_aux_path, f"{pathlib.Path(frame_name).stem}.png"), aux_img)

        # Calcular producto vectorial
        if int(clip_number) == 1 and len(intersections) == 0:

            position = np.sign((x4 - x3) * (point[1] - y3) - (y4 - y3) * (point[0] - x3))

            if prev_position == -1 and (position == 0 or position == 1):

                intersections.append(int(str(pathlib.Path(frame_name).stem)))

        elif int(clip_number) == 1 and len(intersections) == 1:

            position = np.sign((x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1))

            if prev_position == -1 and (position == 0 or position == 1):

                intersections.append(int(str(pathlib.Path(frame_name).stem)))

        elif int(clip_number) == 2 and len(intersections) == 0:

            position = np.sign((x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1))

            if prev_position == 1 and (position == 0 or position == -1):

                intersections.append(int(str(pathlib.Path(frame_name).stem)))

        else:

            position = np.sign((x4 - x3) * (point[1] - y3) - (y4 - y3) * (point[0] - x3))


            if prev_position == 1 and (position == 0 or position == -1):

                intersections.append(int(str(pathlib.Path(frame_name).stem)))


        prev_position = position

    return intersections


def filter_signal(signal):

    # Se calcula la transformada rapida de fourier.
    sig_fft = fftpack.fft(signal)

    power = np.abs(sig_fft)**2

    sample_freq = fftpack.fftfreq(signal.size)

    # Calcular la frecuencia máxima
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    peak_freq = freqs[power[pos_mask].argmax()]

    # Se realiza el filtrado
    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
    filtered_sig = fftpack.ifft(high_freq_fft).real

    return filtered_sig

def calculate_step_stride(intersections, pose_data, sensor_data, real_distance, clip_number):

    distances = []

    for i in range(intersections[0], intersections[1]):

        joints = pose_data[i]["joints"]

        # ELIMINAR CUANDO CAMBIE EL CONTENIDO DE LOS FICHEROS DE POSE
        if joints != None:

            # Selecciono los joints correspondientes a los tobillos.
            l_ankle = joints["l_ankle"]
            r_ankle = joints["r_ankle"]

            # Se calcula la distancia entre ambos puntos.

            value = distance.euclidean((l_ankle["x"], l_ankle["y"]), (r_ankle["x"], r_ankle["y"]))

            distances.append(value)

        
        else: # En caso contrario, utilizar los valores de la pose predichos en el fotograma anterior.

            distances.append(np.nan)

    distances = np.array(distances)

    ok = ~np.isnan(distances)

    # En caso de haber valores perdidos de distancia, se realiza una
    # interpolación de los mismos.
    if np.sum(ok) != ok.shape[0]:

        # Posición de los valroes reales que se conocen
        xp = ok.ravel().nonzero()[0]
        # Valores reales que se conocen
        fp = distances[~np.isnan(distances)]
        # Posiciones en las que se quiere interpolar los valores
        x = np.isnan(distances).ravel().nonzero()[0]

        # Se calcula la interpolación
        distances[np.isnan(distances)] = np.interp(x, xp, fp)

    #print(distances)
        
    # Una vez calculada las distancias, es necesario realizar un filtrado
    # para poder seleccionar los puntos máximos y mínimos.

    # Se aplica un filtrado de frecuencias sobre la señal
    filtered_sig = filter_signal(np.array(distances))

    # Se obtienen los máximos de la señal (distancia máxima entre los tobillos)
    peaks, _ = find_peaks(filtered_sig)

    # Se calcula el indice real de donde se encuentran los máximos
    peaks = peaks + intersections[0]

    # plt.plot(list(range(intersections[0], intersections[1])), filtered_sig, c= 'C1')
    # plt.plot(peaks, filtered_sig[peaks - intersections[0]], "x", c='green')
    # plt.savefig("peaks.png")

    # sys.exit(-1)

    # CALCULAR ESCALA

    point_1 = (sensor_data["0"]["x_min"], sensor_data["0"]["y_min"])
    point_2 = (sensor_data["1"]["x_min"], sensor_data["1"]["y_min"])
    point_3 = (sensor_data["2"]["x_max"], sensor_data["2"]["y_min"])
    point_4 = (sensor_data["3"]["x_max"], sensor_data["3"]["y_min"])

    # Calcular matriz de homografia

    # src points
    pts_src = np.array([
        point_1, 
        point_2,
        point_3,
        point_4]
    )

    pts_dst = np.array([
        point_1,
        [point_1[0], point_2[1]],
        [point_3[0], point_1[1]],
        [point_3[0], point_2[1]]
    ])
    
    h, _ = cv2.findHomography(pts_src, pts_dst)

    new_point1 = np.array([[point_1[0]], [sensor_data["0"]["y_min"] + ((sensor_data["0"]["y_max"] - sensor_data["0"]["y_min"]) / 2)], [1]])
    new_point3 = np.array([[point_3[0]], [sensor_data["2"]["y_min"] + ((sensor_data["2"]["y_max"] - sensor_data["2"]["y_min"]) / 2)], [1]])

    new_point1 = np.matmul(h, new_point1)
    new_point3 = np.matmul(h, new_point3)

    new_point1_x = new_point1[0] / new_point1[2]
    new_point1_y = new_point1[1] / new_point1[2]

    new_point3_x = new_point3[0] / new_point3[2]
    new_point3_y = new_point3[1] / new_point3[2]

    pixel_distance = distance.euclidean((new_point1_x[0], new_point1_y[0]), (new_point3_x[0], new_point3_y[0]))

    scale = (real_distance * 100) / pixel_distance

    steps = []

    for pose in pose_data:

        frame = pose["frame"]
        joints = pose["joints"]

        # Si existe pose y es un peak, se calcula el step
        if joints != None and frame in peaks:

            l_ankle = joints["l_ankle"]
            r_ankle = joints["r_ankle"]

            new_point_l_ankle = np.array([[l_ankle["x"]], [l_ankle["y"]], [1]])
            new_point_r_ankle = np.array([[r_ankle["x"]], [r_ankle["y"]], [1]])

            new_point_l_ankle = np.matmul(h, new_point_l_ankle)
            new_point_r_ankle = np.matmul(h, new_point_r_ankle)

            new_point1_x = int((new_point_l_ankle[0] / new_point_l_ankle[2])[0])
            new_point1_y = int((new_point_l_ankle[1] / new_point_l_ankle[2])[0])

            new_point3_x = int((new_point_r_ankle[0] / new_point_r_ankle[2])[0])
            new_point3_y = int((new_point_r_ankle[1] / new_point_r_ankle[2])[0])

            # Se calcula la distancia entre ambos puntos.

            value = distance.euclidean((new_point1_x, new_point1_y), (new_point3_x, new_point3_y))

            steps.append(value)
    
    mean_step = np.array(steps).mean()

    # Nueva lista para almacenar las distancias de stride
    strides = []

    for i in range(0, len(peaks)):

        if i+1 < len(peaks):

            # Obtener los joints de los peaks actuales
            joints1 = pose_data[peaks[i]]["joints"]
            joints2 = pose_data[peaks[i+1]]["joints"]

            if joints1 is not None and joints2 is not None:

                # Hay que comprobar que pierna está detras

                # Movimiento de derecha a izquierda
                if clip_number == 1:
                    
                    #Pierna derecha está detras 
                    if joints1["l_ankle"]["x"] < joints1["r_ankle"]["x"]:

                        point_1 = joints1["r_ankle"]
                        point_2 = joints2["r_ankle"]
                    
                    else:

                        point_1 = joints1["l_ankle"]
                        point_2 = joints2["l_ankle"]


                # Izquierda a derecha
                else:

                    if joints1["l_ankle"]["x"] > joints1["r_ankle"]["x"]:

                        point_1 = joints1["r_ankle"]
                        point_2 = joints2["r_ankle"]

                    else:

                        point_1 = joints1["l_ankle"]
                        point_2 = joints2["l_ankle"]


                new_point1 = np.array([[point_1["x"]], [point_1["y"]], [1]])
                new_point2 = np.array([[point_2["x"]], [point_2["y"]], [1]])

                new_point1 = np.matmul(h, new_point1)
                new_point2 = np.matmul(h, new_point2)

                new_point1_x = int((new_point1[0] / new_point1[2])[0])
                new_point1_y = int((new_point1[1] / new_point1[2])[0])

                new_point3_x = int((new_point2[0] / new_point2[2])[0])
                new_point3_y = int((new_point2[1] / new_point2[2])[0])

                # Calcular la distancia entre los tobillos en los peaks
                value = distance.euclidean((new_point1_x, new_point1_y), (new_point3_x, new_point3_y))

                strides.append(value)

    # Calcular el promedio de stride
    mean_stride = np.array(strides).mean()

    return mean_step * scale, mean_stride * scale, len(peaks)

def calculate_parameters(pose_path, sensors_path, segmentation_path, scale, fps):

    results = pd.DataFrame()

    for index, patient in enumerate(sorted(os.listdir(pose_path))):

        results.loc[index, 'ID'] = patient

        print(f"Procesando {patient}...")

        # UGS or FGS
        for gait_type in os.listdir(os.path.join(pose_path, patient)):
            
            mean_values = {
                "mean_step": [],
                "mean_stride": [],
                "mean_cadence": [],
                "mean_velocity": []
            }
            # Se hace una media de los parámetros calculados con estos ficheros
            for pose_file in os.listdir(os.path.join(pose_path, patient, gait_type)):

                pose_filename = pathlib.Path(pose_file).stem

                class_name = pose_filename.split('_')[0]

                clip_number = pose_filename.split('_')[1]
                
                f = open(os.path.join(sensors_path, patient, gait_type, f"{class_name}.json"))
                sensor_data = json.load(f)

                # SE CALCULA LOS INSTANTES DE TIEMPO EN EL QUE EL PACIENTE ENTRA
                # DENTRO DEL PERIMETRO DE MEDICIÓN
                intersections = find_intersections(segmentation_path, patient, gait_type, class_name, clip_number, sensor_data)

                #print(f"Gait Type: {gait_type}, Class_name: {class_name}, Direction: {clip_number} -> {intersections}")

                # Deben de haber solamente dos puntos de intersección.
                if len(intersections) != 2:
                    print(f"{patient}, {gait_type}, {class_name}")
                    continue
                
                
                # UNA VEZ TENEMOS LOS INSTANTES DE TIEMPO SE PROSIGUE CON EL 
                # CALCULO DE LOS PARÁMETROS DE LA MARCHA

                # STEP

                f = open(os.path.join(pose_path, patient, gait_type, pose_file))
                pose_data = json.load(f)

                step, stride, n_steps = calculate_step_stride(intersections, pose_data, sensor_data, scale, int(clip_number))

                mean_values["mean_step"].append(step)
                mean_values["mean_stride"].append(stride)
            
                # VELOCITY

                velocity = scale / ((intersections[1] - intersections[0]) / fps)

                mean_values["mean_velocity"].append(velocity)

                # CADENCE

                time_minutes = ((intersections[1] - intersections[0]) / fps) / 60.0

                cadence = n_steps / time_minutes

                mean_values["mean_cadence"].append(cadence)

            if gait_type == 'UGS':

                results.loc[index, 'Velocity_UGS'] = round(np.array(mean_values["mean_velocity"]).mean(), 3)
                results.loc[index, 'Step_UGS'] = round(np.array(mean_values["mean_step"]).mean(), 3)
                results.loc[index, 'Stride_UGS'] = round(np.array(mean_values["mean_stride"]).mean(), 3)
                results.loc[index, 'Cadence_UGS'] = round(np.array(mean_values["mean_cadence"]).mean(), 3)
            
            else:

                results.loc[index, 'Velocity_FGS'] = round(np.array(mean_values["mean_velocity"]).mean(), 3)
                results.loc[index, 'Step_FGS'] = round(np.array(mean_values["mean_step"]).mean(), 3)
                results.loc[index, 'Stride_FGS'] = round(np.array(mean_values["mean_stride"]).mean(), 3)
                results.loc[index, 'Cadence_FGS'] = round(np.array(mean_values["mean_cadence"]).mean(), 3)
        
    return results


        

def main(args):

    POSE_PATH = args.pose_path
    SCALE = args.scale
    SENSORS_PATH = args.sensors_path
    SEGMENTATION_PATH = args.segmentation_path
    FPS = args.fps
    OUTPUT_CSV_PATH = args.csv_output


    parameters = calculate_parameters(POSE_PATH, SENSORS_PATH, SEGMENTATION_PATH, SCALE, FPS)

    # SAVE GAIT PARAMETERS

    parameters.to_csv(OUTPUT_CSV_PATH, index = False)
    
    
if __name__ == '__main__':

    args = parse_opt()

    main(args)