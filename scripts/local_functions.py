
import os
from posixpath import join

import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sk_metrics

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
            
def prepare_image_for_model(image_path, expected_image_size):
  image_string = tf.io.read_file(image_path)
  decoded_image = tf.io.decode_image(image_string, channels=3)
  decoded_image = tf.image.convert_image_dtype(decoded_image, tf.float32)
  return tf.image.resize(decoded_image, expected_image_size)

def plot_confusion_matrix(true_values, predictions, labels):
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

"""
def load_image_set(DIRECTORY_ROOT_PATH):
  image_datas = []
  image_classes = []
  is_root = True
  for (dirname, subdirs, filenames) in tf.gfile.Walk(DIRECTORY_ROOT_PATH):
    # The root directory gives us the classes
    if is_root:
      subdirs = sorted(subdirs)
      image_classes = collections.OrderedDict(enumerate(subdirs))
      label_to_class = dict([(x, i) for i, x in enumerate(subdirs)])
      is_root = False
    # The sub directories give us the image files for training.
    else:
      filenames.sort()
      full_filenames = [os.path.join(dirname, f) for f in filenames]
      label = dirname.split('/')[-1]
      label_class = label_to_class[label]
      # An example is the image file and it's label class.
      examples = list(zip(full_filenames, [label_class] * len(filenames)))
      num_train = int(len(filenames) * TRAIN_FRACTION)
      train_examples.extend(examples[:num_train])
      test_examples.extend(examples[num_train:])
"""
