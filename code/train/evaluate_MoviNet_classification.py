import os
import cv2
import json
import random
import pathlib
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt


from scipy.special import softmax
from sklearn.metrics import confusion_matrix


from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model


def parse_opt():

    parser = argparse.ArgumentParser(
        description = 'Evaluate MoviNet classification'
    )

    parser.add_argument('--device', type = int, default = 0, help = 'GPU used to train')
    parser.add_argument('--img_size', type = int, default = 224, help = "number of frame for clip")
    parser.add_argument('--model_id', type = str, default = "a0", help = "model ID (A0, A1, A2 ..)")
    parser.add_argument('--save_dir', type = str, required = True, help = "folder where to save the results.")
    parser.add_argument('--data_path', type = str, required = True, help = 'dataset path')
    parser.add_argument('--data_type', type = str, required = True, help = 'data type (silhouette or semantic segmentation)')
    parser.add_argument('--num_frames', type = int, default = 20, help = "number of frame for clip")
    parser.add_argument('--batch_size', type = int, default = 32, help = "batch size")
    parser.add_argument('--num_classes', type = int, default = 2, help = "number of classes")
    parser.add_argument('--id_partition', type = int, default = 0, help = "id training partition")
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = "learning rate")
    parser.add_argument('--patients_info', type = str, required = True, help = 'csv with the patients info')
    parser.add_argument('--partitions_file', type = str, required = True, help = 'partition used to train')
    parser.add_argument('--data_class', type = str, required = True, help = 'class used to train models (WoJ or WJ)')
    parser.add_argument('--optical_flow_method', type = str, help = 'what optical flow method used. TLV1 y GMFLOW')
    parser.add_argument('--target', type = str, help = 'class name')
    parser.add_argument('--checkpoint', type = str, help = 'checkpoint')
    parser.add_argument('--n_decimals', type = int, default = 3, help = "number of decimals used to round results")
    parser.add_argument('--classes_names', nargs='+', required=True, help='list with the clases names to classifier')
    parser.add_argument('--seed', type = int, default = 27, help = "seed")

    args = parser.parse_args()

    return args


def init_GPU(device):

    #Use GPU
    #CHANGE ME
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

"""
https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
"""
def set_seed(seed: int = 42) -> None:
  
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  tf.keras.utils.set_random_seed(seed)
  # When running on the CuDNN backend, two further options must be set
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"Random seed set as {seed}")


def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)

  return frame


def frames_from_video_file(video_file, num_frames, output_size = (224,224), skip_percent=0.15):
    
    frames = sorted(os.listdir(video_file))
    total_frames = len(frames)
    skip_frames = int(total_frames * skip_percent)

    # Se ajusta el intervalo de frames para no considerar los extremos.
    effective_frame_count = total_frames - (2 * skip_frames)
    frame_interval = effective_frame_count // num_frames
    
    selected_frames = []
    for i in range(num_frames):

        frame_num = skip_frames + int(i * frame_interval)
        frame = frames[frame_num]
        selected_frames.append(format_frames(cv2.imread(os.path.join(video_file, frame)), output_size))

    # Asumiendo que la conversión a BGR a RGB es necesaria
    frames = np.array(selected_frames)[..., [2, 1, 0]]
    return frames

class FrameGenerator:

  def __init__(self, videos_names, labels, n_frames, output_size = (224, 224), training = False):
    
    """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    self.videos_names = videos_names
    self.n_frames = n_frames
    self.training = training
    self.labels = labels
    self.output_size = output_size

  def __call__(self):

    pairs = list(zip(self.videos_names, self.labels))

    if self.training:
      random.shuffle(pairs)

    # yield almacena la iteración en la que se encuentra el bucle, de forma
    # que la próxima vez que se vuelva a reanudar la ejecución del bucle se 
    # continua por aquí.

    for video_name, label in pairs:

      video_frames = frames_from_video_file(video_name, self.n_frames, self.output_size) 

      yield video_frames, label, video_name


def build_classifier(batch_size, num_frames, resolution, backbone, num_classes):
  
  """Builds a classifier on top of a backbone model."""
  model = movinet_model.MovinetClassifier(
      backbone=backbone,
      num_classes=num_classes)
  model.build([batch_size, num_frames, resolution, resolution, 3])

  return model


def create_model(MODEL_ID, BATCH_SIZE, NUM_FRAMES, IMG_SIZE, NUM_CLASSES, LEARNING_RATE):


  tf.keras.backend.clear_session()

  backbone = movinet.Movinet(model_id = MODEL_ID)
  backbone.trainable = False

  # Set num_classes=600 to load the pre-trained weights from the original model
  model = movinet_model.MovinetClassifier(backbone = backbone, num_classes=600)
  model.build([None, None, None, None, 3])

  checkpoint_dir = f'/opt/data/jzafra/thesis/technical_validation/pretraining_movinet/movinet_{MODEL_ID}_base'
  checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
  checkpoint = tf.train.Checkpoint(model=model)
  status = checkpoint.restore(checkpoint_path)
  status.assert_existing_objects_matched()


  model = build_classifier(BATCH_SIZE, NUM_FRAMES, IMG_SIZE, backbone, NUM_CLASSES)


  loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

  optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE)

  model.compile(loss=loss_obj, optimizer=optimizer, metrics=['accuracy'])

  return model


def get_data(PARTITIONS_FILE, DATA_PATH, DATA_TYPE, DATA_CLASS, OPTICAL_FLOW_METHOD, PATIENTS_INFO, TARGET):

    # Lectura desde fichero de los pacientes utilizados para el entrenamiento.
    with open(PARTITIONS_FILE, 'r') as f:
    # Lee el contenido del archivo
        partitions_data = json.load(f)

    train_patients = partitions_data["train"]
    validation_patients = partitions_data["validation"]
    test_patients = partitions_data["test"]

    df = pd.read_csv(PATIENTS_INFO, sep = ',')

    # Diccionario para almacenar las rutas y clases de archivos
    data = {
        "train": {"path": [], "class": []},
        "validation": {"path": [], "class": []},
        "test": {"path": [], "class": []}
    }

    # Función para añadir datos al diccionario
    def add_data(patient, file_path):
        category = ""
        if patient in train_patients:
            category = "train"
        elif patient in validation_patients:
            category = "validation"
        elif patient in test_patients:
            category = "test"

        if category:
            data[category]["path"].append(file_path)
            data[category]["class"].append(df.loc[df['ID'] == patient][TARGET].values[0])

    # Procesar los datos
    for patient in os.listdir(DATA_PATH):
        for gait_type in os.listdir(os.path.join(DATA_PATH, patient)):
            directory = os.path.join(DATA_PATH, patient, gait_type)
            directory = directory if DATA_TYPE in ['silhouette', 'semantic_segmentation'] else os.path.join(directory, OPTICAL_FLOW_METHOD)

            for class_folder in os.listdir(directory):
                file_path = os.path.join(directory, class_folder)

                if DATA_CLASS == 'both':
                    add_data(patient, file_path)
                else:
                    first_split = class_folder.split("_")
                    second_split = first_split[0].split("-")

                    if second_split[1] == DATA_CLASS:
                        add_data(patient, file_path)

    return data

def main(args):
   
    ############################################################################
    ######################### COMMAND LINE ARGUMENTS ###########################
    ############################################################################

    SEED = args.seed
    DEVICE = args.device
    IMG_SIZE = args.img_size
    SAVE_DIR = args.save_dir
    MODEL_ID = args.model_id
    DATA_PATH = args.data_path
    DATA_TYPE = args.data_type
    DATA_CLASS = args.data_class
    NUM_FRAMES = args.num_frames
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    PATIENTS_INFO = args.patients_info
    PARTITIONS_FILE = args.partitions_file
    OPTICAL_FLOW_METHOD = args.optical_flow_method
    TARGET = args.target
    CLASSES_NAMES = args.classes_names
    CHECKPOINT = args.checkpoint
    N_DECIMALS = args.n_decimals

    DATA_PATH = os.path.join(DATA_PATH, DATA_TYPE)
    
    ############################################################################
    ############################## INICIALIZATION ##############################
    ############################################################################
    
    init_GPU(DEVICE)
    
    set_seed(SEED)

    data = get_data(PARTITIONS_FILE, DATA_PATH, DATA_TYPE, DATA_CLASS, OPTICAL_FLOW_METHOD, PATIENTS_INFO, TARGET)


    ############################################################################
    ############################# DATA GENERATORS ##############################
    ############################################################################

    output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                        tf.TensorSpec(shape = (), dtype = tf.int16),
                        tf.TensorSpec(shape = (), dtype = tf.string))

    test_ds = tf.data.Dataset.from_generator(FrameGenerator(data["test"]["path"], data["test"]["class"], NUM_FRAMES, (IMG_SIZE, IMG_SIZE)),
                output_signature=output_signature)

    test_ds = test_ds.batch(BATCH_SIZE)



    ############################################################################
    ############################# TRAINING #####################################
    ############################################################################


    # Inicialización del modelo

    model = create_model(MODEL_ID, BATCH_SIZE, NUM_FRAMES, IMG_SIZE, len(CLASSES_NAMES), LEARNING_RATE)

    model.summary()

    #model = tf.keras.models.load_model(os.path.join(SAVE_DIR, group_name, run_name, "model"))


    model.load_weights(CHECKPOINT)


    ############################################################################
    ############################ EVALUATION ####################################
    ############################################################################

    # Creación directorios para almacenar los resultados
    #results_path = os.path.join(SAVE_DIR, pathlib.Path(PARTITIONS_FILE).stem)
    results_path = SAVE_DIR
    pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

    y_trues = []
    y_preds = []
    error_paths = []  # Para almacenar las rutas de los errores

    for batch in test_ds:
        video_frames, labels, paths = batch

        preds = model.predict(video_frames)

        y_probs = softmax(preds, axis = 1)

        preds = np.array(y_probs).argmax(axis = 1)

        y_trues.extend(labels.numpy())
        y_preds.extend(preds)

        for real, pred, path in zip(labels.numpy(), preds, paths.numpy()):
            if real != pred:
                error_paths.append({"real": CLASSES_NAMES[int(real)], "pred": CLASSES_NAMES[int(pred)], "path": path.decode('utf-8')})

    # y_preds = model.predict(test_ds)

    # y_probs = softmax(y_preds, axis = 1)

    # y_preds = np.array(y_probs).argmax(axis = 1)

    # Almacenar las predicciones obtenidas y los valores reales
    np.save(os.path.join(results_path, 'preds.npy'), y_preds)
    np.save(os.path.join(results_path, 'reals.npy'), y_trues)

    metrics = {"accuray": round(accuracy_score(y_trues, y_preds), N_DECIMALS), 
               "f1 macro": round(f1_score(y_trues, y_preds), N_DECIMALS)}


    results_partition = pd.DataFrame(data = [list(metrics.values())],
    columns = list(metrics.keys()))

    # Almacenar resultados fichero csv

    results_partition.to_csv(os.path.join(results_path, 'partition_results.csv'), index = False)


    ############################################################################
    ########################### CONFUSION MATRIX ###############################
    ############################################################################

    cm = confusion_matrix(y_trues, y_preds)

    # Normalise
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10,10))

    sns.set(font_scale=3.0) # Adjust to fit

    # Crear el heatmap
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=CLASSES_NAMES, yticklabels=CLASSES_NAMES, 
                cmap='Blues', annot_kws={"size": 30})

    # Aumentar el tamaño de las etiquetas de las clases
    ax.set_xticklabels(CLASSES_NAMES, fontsize=30)  # Ajusta el tamaño de la fuente según sea necesario
    ax.set_yticklabels(CLASSES_NAMES, fontsize=30)  # Ajusta el tamaño de la fuente según sea necesario

    plt.ylabel('Actual', fontsize=30, labelpad=15)
    plt.xlabel('Predicted', fontsize=30, labelpad=15)

    plt.savefig(os.path.join(results_path, "cm.svg"))

    plt.clf()

    ############################################################################
    ######################## CLASSIFICATION FAIL ###############################
    ############################################################################

    # Definir el nombre del archivo donde se guardarán las rutas de errores
    error_paths_filename = os.path.join(results_path, "classification_fails.json")

    string_json = json.dumps(error_paths, indent = 4)

    with open(error_paths_filename, 'w') as f:

        f.write(string_json)


if __name__ == "__main__":

    args = parse_opt()

    main(args)