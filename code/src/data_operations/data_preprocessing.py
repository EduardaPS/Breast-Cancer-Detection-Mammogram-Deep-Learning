import os

from imutils import paths
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical

import config


def import_minimias_dataset(data_dir: str, label_encoder) -> (np.ndarray, np.ndarray):
    """
    Import the dataset by pre-processing the images and encoding the labels.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param data_dir: Directory to the mini-MIAS images.
    :param label_encoder: The label encoder.
    :return: Two NumPy arrays, one for the processed images and one for the encoded labels.
    """
    # Initialise variables.
    images = list()
    labels = list()

    # Loop over the image paths and update the data and labels lists with the pre-processed images & labels.
    for image_path in list(paths.list_images(data_dir)):
        images.append(preprocess_image(image_path))
        labels.append(image_path.split(os.path.sep)[-2])  # Extract label from path.

    # Convert the data and labels lists to NumPy arrays.
    images = np.array(images, dtype="float32")  # Convert images to a batch.
    labels = np.array(labels)

    # Encode labels.
    labels = encode_labels(labels, label_encoder)

    return images, labels


def import_cbisddsm_training_dataset(label_encoder):
    """
    Import the dataset getting the image paths (downloaded on BigTMP) and encoding the labels.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param label_encoder: The label encoder.
    :return: Two arrays, one for the image paths and one for the encoded labels.
    """
    cbis_ddsm_path = str()
    if config.mammogram_type == "calc":
        cbis_ddsm_path = "../data/CBIS-DDSM/calc-training.csv"
    elif config.mammogram_type == "mass":
        cbis_ddsm_path = "../data/CBIS-DDSM/mass-training.csv"
    else:
        cbis_ddsm_path = "../data/CBIS-DDSM/training.csv"
    df = pd.read_csv(cbis_ddsm_path)
    list_IDs = df['img_path'].values
    labels = encode_labels(df['label'].values, label_encoder)
    return list_IDs, labels


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Pre-processing steps:
        * Load the input image in grayscale mode (1 channel),
        * resize it to 224x224 pixels for the VGG19 CNN model,
        * transform it to an array format,
        * normalise the pixel intensities.
    Originally written as a group for the common pipeline. Later amended by Adam Jaamour.
    :param image_path: The path to the image to preprocess.
    :return: The pre-processed image in NumPy array format.
    """
    target_size = (config.MINI_MIAS_IMG_SIZE['HEIGHT'], config.MINI_MIAS_IMG_SIZE["WIDTH"])
    #     if config.model == "VGG":
    #         target_size = (config.VGG_IMG_SIZE['HEIGHT'], config.VGG_IMG_SIZE["WIDTH"])
    #     elif config.model == "ResNet":
    #         target_size = (config.RESNET_IMG_SIZE['HEIGHT'], config.RESNET_IMG_SIZE["WIDTH"])
    #     elif config.model == "Inception":
    #         target_size = (config.INCEPTION_IMG_SIZE['HEIGHT'], config.INCEPTION_IMG_SIZE["WIDTH"])
    #     elif config.model == "Xception":
    #         target_size = (config.XCEPTION_IMG_SIZE['HEIGHT'], config.XCEPTION_IMG_SIZE["WIDTH"])

    image = load_img(image_path,
                     color_mode="grayscale",
                     target_size=target_size)
    image = img_to_array(image)
    image /= 255.0
    return image


def encode_labels(labels_list: np.ndarray, label_encoder) -> np.ndarray:
    """
    Encode labels using one-hot encoding.
    Originally written as a group for the common pipeline. Later ammended by Adam Jaamour.
    :param label_encoder: The label encoder.
    :param labels_list: The list of labels in NumPy array format.
    :return: The encoded list of labels in NumPy array format.
    """
    labels = label_encoder.fit_transform(labels_list)
    if label_encoder.classes_.size == 2:
        return labels
    else:
        return to_categorical(labels)


def dataset_stratified_split(split: float, dataset: np.ndarray, labels: np.ndarray) -> \
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Partition the data into training and testing splits. Stratify the split to keep the same class distribution in both
    sets and shuffle the order to avoid having imbalanced splits.
    Originally written as a group for the common pipeline. Later ammended by Adam Jaamour.
    :param split: Dataset split (e.g. if 0.2 is passed, then the dataset is split in 80%/20%).
    :param dataset: The dataset of pre-processed images.
    :param labels: The list of labels.
    :return: the training and testing sets split in input (X) and label (Y).
    """
    train_X, test_X, train_Y, test_Y = train_test_split(dataset,
                                                        labels,
                                                        test_size=split,
                                                        stratify=labels,
                                                        random_state=config.RANDOM_SEED,
                                                        shuffle=True)
    return train_X, test_X, train_Y, test_Y
