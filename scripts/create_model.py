
import tensorflow as tf
import tensorflow_hub as hub

import constants as consts

tf.random.set_seed(consts.RANDOM_SEED)

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    hub.KerasLayer(consts.MODULE_PATH, trainable=False),
    tf.keras.layers.Dense(consts.CLASSES_COUNT, activation=consts.ACTIVATION)
])

model.build([None, consts.IMAGE_SIZE[0], consts.IMAGE_SIZE[1], 3])
model.compile(loss=consts.LOSS, optimizer=consts.OPTIMIZER, metrics=consts.METRICS)

model.save(consts.MODEL_FOLDER)
