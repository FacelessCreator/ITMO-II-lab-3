
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import classification_report
import numpy as np

import math
import PIL
import random
import os
from posixpath import join


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

MODEL_FOLDER = 'build/model'
PLOTS_FOLDER = 'build/images'
FLOWERS_FOLDER = 'src/flowers'
CLASSES_COUNT = 5

VALIDATION_SPLIT = 0.2

MODULE_PATH = "https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/feature_vector/5"
IMAGE_SIZE = (160, 160)

RANDOM_SEED = 123
BATCH_SIZE = 4
EPOCHS_COUNT = 10
LEARNING_RATE = 0.001

LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
HIDDEN_LAYER_SIZE = 20
ACTIVATION = tf.keras.activations.linear
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
METRICS = 'accuracy'

CONFUSION_TEST_COUNTS = 100

random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=IMAGE_SIZE+(3,)),
    hub.KerasLayer(MODULE_PATH, trainable=False),
    tf.keras.layers.Dense(HIDDEN_LAYER_SIZE, activation=ACTIVATION),
    tf.keras.layers.Dense(CLASSES_COUNT, activation=ACTIVATION)
])

model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

train_ds = tf.keras.utils.image_dataset_from_directory(
  FLOWERS_FOLDER,
  validation_split=VALIDATION_SPLIT,
  subset="training",
  seed=RANDOM_SEED,
  image_size=IMAGE_SIZE,
  batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
  FLOWERS_FOLDER,
  validation_split=VALIDATION_SPLIT,
  subset="validation",
  seed=RANDOM_SEED,
  image_size=IMAGE_SIZE,
  batch_size=BATCH_SIZE)


history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_COUNT)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS_COUNT)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(PLOTS_FOLDER+'/train_loss.png')
plt.show()
plt.clf()

"""
def get_image_set(ROOT_DIRECTORY_PATH):
    image_class_names = []
    image_pathes = []
    image_class_ids = []
    for dirName in os.listdir(ROOT_DIRECTORY_PATH):
        dirPath = join(ROOT_DIRECTORY_PATH, dirName)
        if not os.path.isdir(dirPath):
            continue
        image_class_index = len(image_class_names)
        image_class_names.append(dirName)
        for objectName in os.listdir(dirPath):
            objectPath = join(dirPath, objectName)
            if not os.path.isfile(objectPath):
                continue
            image_pathes.append(objectPath)
            image_class_ids.append(image_class_index)
    return image_class_names, image_pathes, image_class_ids

image_class_names, image_pathes, image_class_ids = get_image_set(FLOWERS_FOLDER)
real_class_ids = []
predicted_class_ids = []
for image_id in range(0, len(image_pathes)):
  image_path = image_pathes[image_id]
  image_class_id = image_class_ids[image_id]
  image = PIL.Image.open(image_path).resize(IMAGE_SIZE, PIL.Image.ANTIALIAS)
  image = np.array(image)
  x = image[None,:,:,:3]
  predicted_vector = model.predict(x)[0]
  predicted_vector_max = max(predicted_vector)
  predicted_class_id = np.where(predicted_vector == predicted_vector_max)[0][0]
  real_class_ids.append(image_class_id)
  predicted_class_ids.append(predicted_class_id)
  if image_id % 100 == 0:
      print('tested {} of {} images'.format(image_id, len(image_pathes)))

class_names = []
for objName in os.listdir(FLOWERS_FOLDER):
    objPath = join(FLOWERS_FOLDER, objName)
    if not os.path.isdir(objPath):
        continue
    class_names.append(objName)

print("\n___ TEST RESULTS ___\n")

print(classification_report(real_class_ids, predicted_class_ids, target_names=class_names))

def plot_confusion_matrix(true_values, predictions, labels):
  confusion = sk_metrics.confusion_matrix(true_values, predictions)
  confusion_normalized = confusion.astype("float") / confusion.sum(axis=1)
  axis_labels = labels
  ax = sns.heatmap(
      confusion_normalized, xticklabels=axis_labels, yticklabels=axis_labels,
      cmap='Blues', annot=True, fmt='.2f', square=True)
  plt.title("Confusion matrix")
  plt.ylabel("True label")
  plt.xlabel("Predicted label")

plot_confusion_matrix(real_class_ids, predicted_class_ids, class_names)
plt.savefig(PLOTS_FOLDER+'/confusions.png')
plt.show()
"""