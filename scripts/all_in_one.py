import numpy as np
import os
import PIL
import PIL.Image
import random
import math

import pathlib

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics

import local_functions

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

FLOWERS_FOLDER = './flowers'
VALIDATION_SPLIT = 0.2

MODULE_PATH = "https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/feature_vector/5"
IMAGE_SIZE = (160, 160)

RANDOM_SEED = 123
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.001

LOSS = tf.keras.losses.MeanAbsoluteError()
ACTIVATION = tf.keras.activations.linear
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
METRICS = 'mae'

random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

data_dir = pathlib.Path(FLOWERS_FOLDER)
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=VALIDATION_SPLIT, subset="training", seed=RANDOM_SEED, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=VALIDATION_SPLIT, subset="validation", seed=RANDOM_SEED, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
class_names = train_ds.class_names

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    hub.KerasLayer(MODULE_PATH, trainable=False),
    tf.keras.layers.Dense(len(class_names), activation=ACTIVATION)
])

model.build([None, 160, 160, 3])  # Batch input shape.
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

"""
  PREDICTIONS
"""

PREDICTING_COUNT = 25

image_class_names, image_pathes, image_class_ids = local_functions.get_image_set(FLOWERS_FOLDER)
real_class_ids = []
predicted_class_ids = []
for i in range(0, PREDICTING_COUNT):
  image_id = math.floor(random.random()*len(image_pathes))
  image_path = image_pathes[image_id]
  image_class_id = image_class_ids[image_id]
  image = PIL.Image.open(image_path).resize((IMAGE_SIZE[0], IMAGE_SIZE[1]), PIL.Image.ANTIALIAS)
  image = np.array(image)
  x = image[None,:,:,:3]
  predicted_vector = model.predict(x)[0]
  predicted_vector_max = max(predicted_vector)
  predicted_class_id = np.where(predicted_vector == predicted_vector_max)[0][0]
  real_class_ids.append(image_class_id)
  predicted_class_ids.append(predicted_class_id)

def show_confusion_matrix(true_values, predictions, labels):
  """Compute confusion matrix and normalize."""
  confusion = sk_metrics.confusion_matrix(true_values, predictions)
  confusion_normalized = confusion.astype("float") / confusion.sum(axis=1)
  axis_labels = labels
  ax = sns.heatmap(
      confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
      cmap='Blues', annot=True, fmt='.2f', square=True)
  plt.title("Confusion matrix")
  plt.ylabel("True label")
  plt.xlabel("Predicted label")
  plt.show()

show_confusion_matrix(real_class_ids, predicted_class_ids, class_names)
