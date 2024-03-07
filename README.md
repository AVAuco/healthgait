# Health&Gait: a video dataset for gait-based analysis

<p align="center">
    <img src="./media/Pose.gif" width="300">
    <img src="./media/Semantic_Segmentation.gif" width="300">
    <img src="./media/Silhouette.gif" width="300">
    <img src="./media/TVL1.gif" width="300">
    <img src="./media/GMFLow.gif" width="300">
    <br>
    <sup>The differents data modalities contains in the dataset. From left to right shows pose, semantic segmentation, silhouette and optical flow.</sup>
</p>


## Description
This repository contains the **Health&Gait** dataset, the first that enables gait analysis using visual information without specific sensors, relying solely on cameras. The dataset includes multimodal features extracted from videos and gait parameters and anthropometric measurement from each participant. The dataset is intended for use in health, sports and gait analysis research.

<p align="center">
    <img src="./media/Data_Records.png" width="850">
    <br>
    <sup>Two examples of the different data types from the dataset for two participants (a) and (b).</sup>
</p>

## Dataset Contents

Health&Gait consists of 1,564 videos of 398 participants walking in a controlled closed environment, where each video has associated the following information:

- 2D pose estimation of their joints by [**AlphaPose**](https://github.com/MVIG-SJTU/AlphaPose) (JSON format files).
- Semantic segmentation by [**DensePose**](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) (PNG images).
- Optical flow by [**TVL1**](https://docs.opencv.org/3.4/dc/d47/classcv_1_1DualTVL1OpticalFlow.html) and [**GMFlow**](https://github.com/haofeixu/gmflow) (PNG images).
- Silhouette by [**YOLOV8**](https://github.com/ultralytics/ultralytics) (JPEG images).

Moreover, for each participants the following data has been recorded:

- Anthropometric measurements.
- Gait parameters obtained from OptoGait and MuscleLAB.
- Gait parameters estimated from pose information. 

<p align="center">
    <img src="./media/Health&Gait.svg" width="850">
    <br>
    <sup>Directory and file scheme of the Health&Gait database.</sup>
</p>

<details><summary> <b>Attributes in the file participants_measures.csv</b> </summary>

<center>

| Attributes          | Description                                                  | Unit                                       |
|---------------------|--------------------------------------------------------------|--------------------------------------------|
| Sex                 | Participant sex                                              | 0:female, 1:male                           |
| Age                 | Participant Age                                              | Years                                      |
| PA\_level           | Level of physical activity                                   | >=3 days: Active, < 3 days: Non active     |
| Height              | Participant height                                           | cm                                         |
| Weight              | Participant weight                                           | kg                                         |
| BMI                 | Body Mass Index                                              | kg/m<sup>2</sup>                           |
| WaistC              | Waist circumference                                          | cm                                         |
| HipC                | Hip circumference                                            | cm                                         |
| NeckC               | Neck circumference                                           | cm                                         |
| Percentage fat mass | The total mass of fat divided by total body mass             | %                                          |
| Lean mass           | The difference between total body weight and body fat weight | kg                                         |

</center>

</details>

<details><summary> <b>Atributes in the file gait_parameters_estimation.csv.</b> </summary>

<center>

| **Attributes**               | **Description**                                                                                      | **Unit**     |
|------------------------------|------------------------------------------------------------------------------------------------------|--------------|
| Step_UGS / Step_FGS          | The distance between the two toes or heels of the feet in sequence for usual/fast gait speed.       | cm           |
| Stride_UGS / Stride_FGS      | The distance between the two toes or heels of sequential strides of the same foot for usual/fast gait speed. | cm           |
| Cadence_UGS / Cadence_FGS    | The number of steps taken per unit of time for usual/fast gait speed.                                | Steps / min  |
| MonoSP_UGS / MonoSP_FGS      | Time in the swing phase where only one limb is in contact with the ground for usual/fast gait speed. | sec          |
| BiSP_UGS / BiSP_FGS          | Time that both feet are on the ground for usual/fast gait speed.                                     | sec          |
| Speed_UGS / Speed_FGS        | Participant velocity for usual/fast gait speed.                                                      | m / s        |



</center>

</details>

## Getting Started

### Download Dataset

The dataset will be accesible under request upon acceptance of the paper. A script for downloading the dataset will be provided. You will need to request my password via [email](#Contact).


### Install dependencies

The first is to create a python environment from the **requirement.txt** file. The use of [**conda**](https://docs.anaconda.com/free/miniconda/miniconda-install/) is recommended.


```
conda create --name <env> --file requirement.txt
conda activate <env>
```

Check if Tensorflow are installed correctly:

```
python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')) > 0)"
```

## Usage
The following is an indication of how to use the various scripts provided in the repository
as well as a recommendation on how to load and use the dataset. The first thing is to access th directory within the repository where the scripts are located:

```
cd scripts/
```

### Create train, validation and test partitions

The ***create_partitions.sh*** script allows the partitioning of patients into train, validation and test sets in a stratified manner for the sex classification and weight and age regression tasks.

```
bash create_partitions.sh -p <patient_measures_file> -o <output_path>
```

where:
- ***<patient_measures_file>*** is the path to the ***participants_measures.csv*** file in the dataset.
- ***<output_path>*** is the path where to save the partitions.

### Technical Validation

The following script can be used to obtained the results presented in the **technical validation** sections in the paper:

```
python technical_validation.py --config '../configfiles/train_configfile.json' --targets {sex, weight, age} --methods {MoviNet, XGBoost, MLP}
```

where ***train_configfile.json*** is a configuration file where you need to set the value of the following fields:

- **"data_path"**: path where you have downloaded the dataset.
- **"partitions_path"**: path used in the previous scripts.
- **"save_dir"**: path where to store the training results.
- **"patients_measures"**: file with the anthropometric data of the participants.
- **"gait_parameters"**: file with the information of the gait parameters.
- **"gait_parameters_estimation"**: file with the information of the estimated gait parameters.
- **"hyperparameters_search_space"**: json file where the hyperparameters search space of the XGBoost method is defined.

You can tell the script from which **target** to get the results and which **methods** to use to do so.

### Gait parameters estimation

The following script allows to obtain the gait parameters from the pose:

```
bash gait_parameters_estimation.sh -p <patient_measures_file> -s <sensor_bboxes> -e <semantic_segmentation_path> -o <output_csv_path> -k [scale] -f [fps]
```

where:
- **-p** is used to indicate the path to the ***participants_measures.csv*** file.
- **-s** is the path to the directory with the bounding boxes of OptoGait sensors (for more information [contact](#Contact) me).
- **-e** is the path to the directory with the semantic segmentation.
- **-o** is the path to the csv file where we want to store the estimates.
- **-k** is used to indicate the scale of the scene to obtain real measurements.
- **-f** is used to indicate the fps rate needed to obtain the gait parameters related to time.

### Recommendations to load and use the dataset 

<details><summary> <b>Example of DataGenerator in Tensorflow</b> </summary>

To load the multimodal features extracted from the videos (semantic segmentation, silhouette, and optical flow), the use of **DataGenerators** is recommended.

``` python
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

    for video_name, label in pairs:

      video_frames = frames_from_video_file(video_name, self.n_frames, self.output_size) 

      yield video_frames, label


def frames_from_video_file(video_file, num_frames, output_size = (224,224), skip_percent=0.15):
    
    frames = sorted(os.listdir(video_file))
    total_frames = len(frames)
    skip_frames = int(total_frames * skip_percent)

    effective_frame_count = total_frames - (2 * skip_frames)
    frame_interval = effective_frame_count // num_frames
    
    selected_frames = []
    for i in range(num_frames):

        frame_num = skip_frames + int(i * frame_interval)
        frame = frames[frame_num]
        selected_frames.append(format_frames(cv2.imread(os.path.join(video_file, frame)), output_size))

    frames = np.array(selected_frames)[..., [2, 1, 0]]
    return frames

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
```

</details>

<details><summary> <b>Get paths and targets for the DataGenerator</b> </summary>

To define the DataGenerator it is first necessary to obtain the path and target of the data type to be used:

``` python
def get_data(PARTITIONS_FILE, DATA_PATH, DATA_TYPE, DATA_CLASS, OPTICAL_FLOW_METHOD, PATIENTS_INFO, TARGET):

    with open(PARTITIONS_FILE, 'r') as f:

        partitions_data = json.load(f)

    train_patients = partitions_data["train"]
    validation_patients = partitions_data["validation"]
    test_patients = partitions_data["test"]

    df = pd.read_csv(PATIENTS_INFO, sep = ',')

    data = {
        "train": {"path": [], "class": []},
        "validation": {"path": [], "class": []},
        "test": {"path": [], "class": []}
    }

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
```

</details>


<details><summary> <b>Define and use DataGenerators</b> </summary>

The following code shows how to define the data generators for the three datasets, making used of the methods ***from_generator***:

``` python
output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
tf.TensorSpec(shape = (), dtype = tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(data["train"]["path"], data["train"]["class"], NUM_FRAMES, (IMG_SIZE, IMG_SIZE), training = True),
            output_signature=output_signature)

train_ds = train_ds.batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(data["validation"]["path"], data["validation"]["class"], NUM_FRAMES, (IMG_SIZE, IMG_SIZE)),
            output_signature=output_signature)

val_ds = val_ds.batch(BATCH_SIZE)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(data["test"]["path"], data["test"]["class"], NUM_FRAMES, (IMG_SIZE, IMG_SIZE)),
            output_signature=output_signature)

test_ds = test_ds.batch(BATCH_SIZE)
```

These generators can be used in **Tensorflow's** fit or predict methods:


``` python
results = model.fit(train_ds,
validation_data = val_ds,
epochs = EPOCHS,
validation_freq = 1,
callbacks = callbacks_list,
verbose = 1)
```

```python
y_preds = model.predict(test_ds)
``` 


</details>


## License
Health&Gait is freely available for free non-commercial use, and may be redistributed under these conditions. Please, see the [license](./LICENSE) for further details.

## Citation
```bibtex
@misc{zafra2024,
  author = {Zafra, J. et al.},
  title = {Health&Gait: a video dataset for gait-based analysis},
  year = {2024},
  publisher = {Under Review},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AVAuco/healthgait}}
}
```

## Contact
If you have any question or suggestion, contact us by **jzafra@uco.es**.

