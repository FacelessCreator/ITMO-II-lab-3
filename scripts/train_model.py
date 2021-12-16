
import pathlib

import tensorflow as tf

import constants as consts

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

tf.random.set_seed(consts.RANDOM_SEED)

model = tf.keras.models.load_model(consts.MODEL_FOLDER)

print(model.summary())

data_dir = pathlib.Path(consts.FLOWERS_FOLDER)
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=consts.VALIDATION_SPLIT, subset="training", seed=consts.RANDOM_SEED, image_size=consts.IMAGE_SIZE, batch_size=consts.BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=consts.VALIDATION_SPLIT, subset="validation", seed=consts.RANDOM_SEED, image_size=consts.IMAGE_SIZE, batch_size=consts.BATCH_SIZE)

while True:
    model.fit(train_ds, validation_data=val_ds, epochs=consts.EPOCHS_BETWEEN_SAVES)
    print("Cycle ended; saving model")
    model.save(consts.MODEL_FOLDER)
    print("Model saved")
