
MODEL_FOLDER = 'build/model'
IMAGES_FOLDER = 'build/images'
FLOWERS_FOLDER = 'src/flowers'
CLASSES_COUNT = 5

VALIDATION_SPLIT = 0.2

MODULE_PATH = "https://tfhub.dev/google/imagenet/mobilenet_v2_050_160/feature_vector/5"
IMAGE_SIZE = (160, 160)

RANDOM_SEED = 123
BATCH_SIZE = 4
EPOCHS_BETWEEN_SAVES = 20
LEARNING_RATE = 0.01

PREDICTING_COUNT = 50

import tensorflow as tf

LOSS = tf.keras.losses.MeanAbsoluteError()
HIDDEN_LAYER_SIZE = 20
ACTIVATION = tf.keras.activations.linear
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
METRICS = 'mae'
