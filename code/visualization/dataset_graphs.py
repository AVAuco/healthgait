import os
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_opt():
    parser = argparse.ArgumentParser(
        description='Python script to visualize data set.'
    )

    parser.add_argument('--patients_measures', type=str, required=True, help='path to the csv with biometric parameters of patients.')
    parser.add_argument('--videos_path', type=str, required=True, help='path to the videos dataset.')
    parser.add_argument('--gait_parameters', type=str, required=True, help='path to the csv with gait parameters of patients.')
    parser.add_argument('--save_path', type=str, required=True, help='path where to save figures.')

    args = parser.parse_args()

    return args


def calculate_histogram(data, ranges, tags, x_label, output, save_path, fontsize=10):

    plt.figure(figsize=(13, 8))

    # Crea el histograma con el ancho personalizado de los bins
    plt.hist(data, bins=ranges, edgecolor='black', alpha=0.7, rwidth=0.90)

    # Etiquetas y título
    plt.xlabel(x_label, fontsize = fontsize)
    plt.ylabel('Frequency', fontsize = fontsize)
    #plt.title(title_name, fontweight = "bold")

    plt.yticks(fontsize=fontsize)

    # Establece las etiquetas personalizadas en el eje x y ajusta la alineación
    plt.xticks([(ranges[i] + ranges[i+1]) / 2 for i in range(len(ranges) - 1)], tags, ha='center', fontsize=fontsize)

    # Calcula y muestra el recuento encima de las barras
    counts, _, _ = plt.hist(data, bins=ranges, edgecolor='black', alpha=0)
    for i, count in enumerate(counts):
        plt.text(
            (ranges[i] + ranges[i+1]) / 2,  # Posición x centrada en el rango
            count,  # Valor del recuento
            str(int(count)),  # Formatea el recuento como cadena
            ha='center',  # Alineación horizontal centrada
            va='bottom',  # Alineación vertical en la parte superior de la barra
            fontsize=fontsize
        )
    
    # Muestra el histograma o guarda la figura
    plt.savefig(os.path.join(save_path, output))

    plt.clf()


def age_distribution(data, save_path):

    # Definición de los rangos
    ranges = [18, 35, 50, 65]
    tags = ['(18-34)', '(35-49)', '(50-64)']

    plt.figure(figsize=(13, 9))

    # Crea el histograma con el ancho personalizado de los bins
    plt.hist(data, bins=ranges, edgecolor='black', alpha=0.7, rwidth=0.90)

    # Etiquetas y título
    plt.xlabel('Age (years)', fontsize = 30)
    plt.ylabel('Frequency', fontsize = 30)
    #plt.title(title_name, fontweight = "bold")

    plt.yticks(fontsize=30)

    plt.ylim((0, 160))

    # Establece las etiquetas personalizadas en el eje x y ajusta la alineación
    plt.xticks([(ranges[i] + ranges[i+1]) / 2 for i in range(len(ranges) - 1)], tags, ha='center', fontsize=30)

    # Calcula y muestra el recuento encima de las barras
    counts, _, _ = plt.hist(data, bins=ranges, edgecolor='black', alpha=0)
    for i, count in enumerate(counts):
        plt.text(
            (ranges[i] + ranges[i+1]) / 2,  # Posición x centrada en el rango
            count,  # Valor del recuento
            str(int(count)),  # Formatea el recuento como cadena
            ha='center',  # Alineación horizontal centrada
            va='bottom',  # Alineación vertical en la parte superior de la barra
            fontsize=30
        )
    
    # Muestra el histograma o guarda la figura
    plt.savefig(os.path.join(save_path, 'Age_Histogram.svg'))

    plt.clf()

    #calculate_histogram(data, ranges, tags, 'Age Distribution', 'Age', 'Age_Histogram.svg', save_path, 30)



def sex_distribution(data, save_path):

    # Contar la cantidad de 0 y 1 en la columna 'sexo_binario'
    counts = data.value_counts()

    ax = plt.gca()

    plt.yticks(fontsize=30)

    plt.ylim((0, 225))

    ax.set_xlim([-0.5, 1.5])

    # Crear un gráfico de barras
    bars = plt.bar(counts.index, counts.values, alpha=0.7, edgecolor='black')

    for bar, label in zip(bars, counts.index):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom', fontsize = 30)

    plt.xticks([0, 1], ["Female", "Male"], fontsize = 30)

    # Asignar etiquetas a los ejes y al gráfico
    plt.xlabel('Sex', fontsize = 30)
    plt.ylabel('Frequency', fontsize = 30)
    #plt.title('Sex Distribution', fontweight = "bold")

    # Mostrar el gráfico
    plt.savefig(os.path.join(save_path, "sex_barplot.svg"))

    plt.clf()


def activity_distribution(data, save_path):

    counts = data.value_counts()

    ax = plt.gca()

    ax.set_xlim([-0.5, 1.5])

    # Crear un gráfico de barras
    bars = plt.bar(counts.index, counts.values, alpha=0.7, edgecolor='black')

    for bar, label in zip(bars, counts.index):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

    plt.xticks([0, 1], ["Non Active", "Active"])

    # Asignar etiquetas a los ejes y al gráfico
    plt.xlabel('Physical Activity Level')
    plt.ylabel('Frequency')
    plt.title('Physical Activity Distribution', fontweight = "bold")

    # Mostrar el gráfico
    plt.savefig(os.path.join(save_path, "pa_level.svg"))

    plt.clf()


def height_distribution(data, save_path):

    inicio = 140
    final = 200
    paso = 10

    lista = [valor for valor in range(int((final - inicio) / paso) + 1)]
    ranges = [inicio + i * paso for i in lista]

    tags = [f'({ranges[i]:.1f}, {ranges[i+1]:.1f}]' for i in range(len(ranges) - 1)]


    plt.figure(figsize=(13, 8))

    # Crea el histograma con el ancho personalizado de los bins
    plt.hist(data, bins=ranges, edgecolor='black', alpha=0.7, rwidth=0.90)

    # Etiquetas y título
    plt.xlabel('Height (cm)', fontsize = 30)
    plt.ylabel('Frequency', fontsize = 30)
    #plt.title(title_name, fontweight = "bold")

    plt.yticks(fontsize=30)

    plt.ylim((0, 180))

    # Establece las etiquetas personalizadas en el eje x y ajusta la alineación
    plt.xticks([(ranges[i] + ranges[i+1]) / 2 for i in range(len(ranges) - 1)], tags, ha='center', fontsize=14)

    # Calcula y muestra el recuento encima de las barras
    counts, _, _ = plt.hist(data, bins=ranges, edgecolor='black', alpha=0)
    for i, count in enumerate(counts):
        plt.text(
            (ranges[i] + ranges[i+1]) / 2,  # Posición x centrada en el rango
            count,  # Valor del recuento
            str(int(count)),  # Formatea el recuento como cadena
            ha='center',  # Alineación horizontal centrada
            va='bottom',
            fontsize=30
        )
    
    # Muestra el histograma o guarda la figura
    plt.savefig(os.path.join(save_path, "Height_Histogram.svg"))

    plt.clf()


    #calculate_histogram(data, ranges, tags, 'Height', "Height_Histogram.svg", save_path, 10)


def weight_distribution(data, save_path):

    inicio = 40
    final = 175
    paso = 15

    lista = [valor for valor in range(int((final - inicio) / paso) + 1)]
    ranges = [inicio + i * paso for i in lista]

    tags = [f'({ranges[i]:.1f}, {ranges[i+1]:.1f}]' for i in range(len(ranges) - 1)]


    plt.figure(figsize=(13, 8))

    # Crea el histograma con el ancho personalizado de los bins
    plt.hist(data, bins=ranges, edgecolor='black', alpha=0.7, rwidth=0.90)

    # Etiquetas y título
    plt.xlabel('Weight (kg)', fontsize = 30)
    plt.ylabel('Frequency', fontsize = 30)
    #plt.title(title_name, fontweight = "bold")

    plt.yticks(fontsize=30)

    plt.ylim((0, 180))

    # Establece las etiquetas personalizadas en el eje x y ajusta la alineación
    plt.xticks([(ranges[i] + ranges[i+1]) / 2 for i in range(len(ranges) - 1)], tags, ha='center', fontsize=9)

    # Calcula y muestra el recuento encima de las barras
    counts, _, _ = plt.hist(data, bins=ranges, edgecolor='black', alpha=0)
    for i, count in enumerate(counts):
        plt.text(
            (ranges[i] + ranges[i+1]) / 2,  # Posición x centrada en el rango
            count,  # Valor del recuento
            str(int(count)),  # Formatea el recuento como cadena
            ha='center',  # Alineación horizontal centrada
            va='bottom',
            fontsize=30
        )


    # Muestra el histograma o guarda la figura
    plt.savefig(os.path.join(save_path, "Weight_Histogram.svg"))

    plt.clf()


    #calculate_histogram(data, ranges, tags, 'Weight', "Weight_Histogram.svg", save_path, 10)


def number_frames_graph(videos_path, save_path):

    usual_n_frames = []
    fast_n_frames = []

    for subdir, dirs, files in os.walk(videos_path, topdown = True):

        for file in files:

            cap = cv2.VideoCapture(os.path.join(subdir, file))

            splits = file.split("_")

            if splits[0] == 'NW-WJ' or splits[0] == 'NW-WoJ':

                usual_n_frames.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            else:

                fast_n_frames.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))


    usual_n_frames = np.array(usual_n_frames)
    fast_n_frames = np.array(fast_n_frames)


    fig, ax = plt.subplots()
    
    # Crea el histograma con el ancho personalizado de los bins
    ax.hist(usual_n_frames, edgecolor='black', alpha=0.7, rwidth=0.90, label = "Usual Gait Speed")

    # Crea el histograma con el ancho personalizado de los bins
    plt.hist(fast_n_frames, edgecolor='black', alpha=0.7, rwidth=0.90, label = "Fast Gait Speed")

    plt.xlabel("Number of frames")
    plt.ylabel('Number of videos')
    plt.title("Number of Frames per Videos", fontweight = "bold")

    ax.legend()

    plt.savefig(os.path.join(save_path, "n_frames.svg"))


def bmi_plot(data, save_path):

    counts = np.zeros(4, dtype=int)

    for imc in data:

        if imc < 18.5:

            counts[0] += 1 

        elif imc >= 18.5 and imc <= 24.9:

            counts[1] += 1
        
        elif imc >= 25.0 and imc <=29.9:

            counts[2] += 1

        else:

            counts[3] += 1
    

    # Datos
    labels = ["Mild Thinness", "Normal", "Overweight", "Obese"]

    # only "explode" the 2nd slice (i.e. 'Hogs')
    explode = (0, 0, 0, 0)
    #add colors
    colors = ['#ff9999','#99ff99','#66b3ff','#fff099']
    fig1, ax1 = plt.subplots()
    ax1.pie(counts, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(save_path, "BMI_plot.svg"))



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

def bmi_velocity_plot(bmi, velocity, age_groups, sex_values, save_path, title, y_label, name):
    bmi_array = bmi.to_numpy()
    velocity_array = velocity.to_numpy()
    age_groups_array = np.array(age_groups)
    sex_values_array = np.array(sex_values)

    # Configuraciones para los colores de fondo y sus etiquetas
    background_colors = {'Mid Thinness': 'red', 'Normal': 'green', 'Overweight': 'blue', 'Obese': 'yellow'}
    background_labels = list(background_colors.keys())
    background_patches = [mpatches.Patch(color=background_colors[label], label=label) for label in background_labels]

    # Define colores y marcadores
    age_colors = {"18-34": "magenta", "35-49": "cyan", "50-64": "orange"}
    sex_markers = {1: "o", 0: "^"}  # Circulo para 1 (asumiendo masculino), triángulo para 0 (asumiendo femenino)

    fig, ax = plt.subplots()

    ax.margins(0)  # remove default margins (matplotlib version 2+)

    # Define áreas de BMI con colores de fondo
    ax.axvspan(bmi.min() - 1, 18.5, facecolor='red', alpha=0.3)
    ax.axvspan(18.5, 25.0, facecolor='green', alpha=0.3)
    ax.axvspan(25.0, 30.0, facecolor='blue', alpha=0.3)
    ax.axvspan(30.0, bmi.max() + 1, facecolor='yellow', alpha=0.3)

    # Ajusta el plot para grupos de edad y sexo
    for age_group in age_colors:
        for sex in sex_markers:
            mask = (age_groups_array == age_group) & (sex_values_array == sex)
            ax.scatter(bmi_array[mask], velocity_array[mask], color=age_colors[age_group], marker=sex_markers[sex], label=f"{age_group}, {'Male' if sex == 1 else 'Female'}")

    # Crear artistas para leyenda de grupos de edad
    age_patches = [mpatches.Patch(color=age_colors[age], label=age) for age in age_colors]

    # Crear artistas para leyenda de sexo
    sex_patches = [plt.Line2D([0], [0], marker=sex_markers[sex], color='black', label='Male' if sex == 1 else 'Female', markersize=10, linestyle="") for sex in sex_markers]

    # Combinar todas las leyendas
    legend1 = plt.legend(handles=background_patches, title="BMI Categories", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    legend2 = plt.legend(handles=age_patches, title="Age Groups", loc='upper left', bbox_to_anchor=(1.05, 0.65), borderaxespad=0.)
    legend3 = plt.legend(handles=sex_patches, title="Sex", loc='upper left', bbox_to_anchor=(1.05, 0.4), borderaxespad=0.)

    ax.add_artist(legend1)
    ax.add_artist(legend2)
    plt.gca().add_artist(legend3)

    plt.xlabel('Body Mass Index (kg/m²)')
    plt.ylabel(y_label)
    plt.title(title, fontweight="bold")

    plt.savefig(os.path.join(save_path, name), bbox_inches='tight')



def activity_age_plot(df, save_path):

    # Dividir en personas activas y no activas
    activos = df[df['PA_level'] == 1.0]
    no_activos = df[df['PA_level'] == 0.0]

    # Contar las edades en grupos para personas activas
    activos_por_grupo = {'18-34': 0, '35-49': 0, '50-64': 0}
    for edad in activos['Age']:
        if 18 <= edad <= 34:
            activos_por_grupo['18-34'] += 1
        elif 35 <= edad <= 49:
            activos_por_grupo['35-49'] += 1
        elif 50 <= edad <= 64:
            activos_por_grupo['50-64'] += 1

    # Contar las edades en grupos para personas no activas
    no_activos_por_grupo = {'18-34': 0, '35-49': 0, '50-64': 0}
    for edad in no_activos['Age']:
        if 18 <= edad <= 34:
            no_activos_por_grupo['18-34'] += 1
        elif 35 <= edad <= 49:
            no_activos_por_grupo['35-49'] += 1
        elif 50 <= edad <= 64:
            no_activos_por_grupo['50-64'] += 1

    # Crear gráfico de barras con dos grupos
    grupos_edades = ['18-34', '35-49', '50-64']
    x = np.arange(len(grupos_edades))  # Posición de los grupos en el eje x
    width = 0.35  # Ancho de las barras
    

    fig, ax = plt.subplots(figsize=(12, 9))

    # Barras para personas activas
    rects1 = ax.bar(x - width/2, [activos_por_grupo[grupo] for grupo in grupos_edades], width, label='Active', alpha=0.7, edgecolor='black')

    # Barras para personas no activas
    rects2 = ax.bar(x + width/2, [no_activos_por_grupo[grupo] for grupo in grupos_edades], width, label='No Active', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Age (years)', fontsize = 30)
    ax.set_ylabel('Number of Participants', fontsize = 30)
    ax.set_xticks(x)
    ax.set_xticklabels(grupos_edades, fontsize = 30)
    ax.legend(fontsize = 18)

    plt.yticks(fontsize=30)
    plt.ylim((0, 100))
    #ax.set_title('Number of People by Age Group and Activity Level', fontweight='bold')

    # Función para agregar recuento en la parte superior de cada barra
    def autolabel(rects, xpos='center'):
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos]*3, 3),  # 3 puntos de desplazamiento vertical
                        textcoords="offset points",
                        ha=ha[xpos], va='bottom',
                        fontsize = 30)

    autolabel(rects1, "center")
    autolabel(rects2, "center")

    plt.savefig(os.path.join(save_path, "activity_level_by_age.svg"))
    plt.close()


def get_age_groups(data):
    
    groups = []

    for age in data:

        if age < 35:

            groups.append("18-34")

        elif age >= 35 and age < 50:

            groups.append("35-49")

        else:

            groups.append("50-64")

    return groups


def main(args):

    BIOMETRIC_DATA = args.patients_measures
    GAIT_DATA = args.gait_parameters
    VIDEOS_PATH = args.videos_path
    SAVE_PATH = args.save_path
    
    # Cargar datos
    biometric_data = pd.read_csv(BIOMETRIC_DATA)
    gait_data = pd.read_csv(GAIT_DATA)

    # Identificar IDs con valores NaN en 'Velocity_UGS' y el ID específico a eliminar
    nan_velocity_ids = gait_data[gait_data['Velocity_UGS'].isna()]['ID'].tolist()
    ids_to_remove = nan_velocity_ids + ['PA001']

    age_distribution(biometric_data["Age"], SAVE_PATH)
    sex_distribution(biometric_data["Sex"], SAVE_PATH)
    activity_distribution(biometric_data["PA_level"], SAVE_PATH)
    height_distribution(biometric_data["Height"], SAVE_PATH)
    weight_distribution(biometric_data["Weight"], SAVE_PATH)
    #number_frames_graph(VIDEOS_PATH, SAVE_PATH)
    bmi_plot(biometric_data["BMI"], SAVE_PATH)

    activity_age_plot(biometric_data[["PA_level", "Age"]].dropna(), SAVE_PATH)

    # Eliminar los IDs especificados de ambos DataFrames
    biometric_data = biometric_data[~biometric_data['ID'].isin(ids_to_remove)]
    gait_data = gait_data[~gait_data['ID'].isin(ids_to_remove)]

    bmi_data = biometric_data[["ID", "BMI"]]
    velocity_data = gait_data[["ID", "Velocity_UGS"]]

    age_groups = get_age_groups(biometric_data["Age"])
    sex_values = biometric_data["Sex"]


    bmi_velocity_plot(bmi_data["BMI"], velocity_data["Velocity_UGS"], age_groups, sex_values, SAVE_PATH, 'BMI vs UGS', 'Usual Gait Speed (m/s)', 'BMI_UGS.svg')
    bmi_velocity_plot(biometric_data["BMI"], gait_data["Velocity_FGS"], age_groups, sex_values, SAVE_PATH, 'BMI vs FGS', 'Fast Gait Speed (m/s)', 'BMI_FGS.svg')

if __name__ == "__main__":

    args = parse_opt()
    main(args)
