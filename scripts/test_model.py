
import os
import math
import random
from posixpath import join

import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import local_functions
import constants as consts

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

random.seed(consts.RANDOM_SEED)

model = tf.keras.models.load_model(consts.MODEL_FOLDER)

print(model.summary())

image_class_names, image_pathes, image_class_ids = local_functions.get_image_set(consts.FLOWERS_FOLDER)
real_class_ids = []
predicted_class_ids = []
for i in range(0, consts.PREDICTING_COUNT):
  image_id = math.floor(random.random()*len(image_pathes))
  image_path = image_pathes[image_id]
  image_class_id = image_class_ids[image_id]
  image = PIL.Image.open(image_path).resize((consts.IMAGE_SIZE[0], consts.IMAGE_SIZE[1]), PIL.Image.ANTIALIAS)
  image = np.array(image)
  x = image[None,:,:,:3]
  predicted_vector = model.predict(x)[0]
  predicted_vector_max = max(predicted_vector)
  predicted_class_id = np.where(predicted_vector == predicted_vector_max)[0][0]
  real_class_ids.append(image_class_id)
  predicted_class_ids.append(predicted_class_id)

class_names = []
for objName in os.listdir(consts.FLOWERS_FOLDER):
    objPath = join(consts.FLOWERS_FOLDER, objName)
    if not os.path.isdir(objPath):
        continue
    class_names.append(objName)

local_functions.plot_confusion_matrix(real_class_ids, predicted_class_ids, class_names)
print("Saving confusion plot")
plt.savefig(consts.IMAGES_FOLDER+'/collisions.png')
print("Confusion plot saved")
plt.show()
