import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (48, 48)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale = 1.0 / 255.0,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    zoom_range = 0.1,
)

val_datagen = ImageDataGenerator(rescale = 1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    "/Users/choijiwon/expression/dataset/train",
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    color_mode = "grayscale",
    class_mode = "categorical",
)

val_generator = val_datagen.flow_from_directory(
    "/Users/choijiwon/expression/dataset/valid",
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    color_mode = "grayscale",
    class_mode = "categorical",
)