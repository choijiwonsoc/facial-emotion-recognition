import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow_addons as tfa
import numpy as np

# Set environment variables to disable multi-threading (if needed)
import os
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# Define image size and batch size
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

# Load datasets using tf.data.Dataset
train_dataset = image_dataset_from_directory(
    "/Users/choijiwon/expression/dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='grayscale',  # Use grayscale for FER2013
)
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_up_down(x), y))

def random_rotate_image(image, max_angle_degrees=36):
    # Convert max angle to radians
    max_angle_radians = np.deg2rad(max_angle_degrees)
    # Randomly sample an angle between -max_angle and +max_angle
    angle = tf.random.uniform([], minval=-max_angle_radians, maxval=max_angle_radians)
    
    # Rotate the image using the sampled angle
    image = tfa.image.rotate(image, angle)
    
    return image

# Apply the custom rotation function to the dataset
train_dataset = train_dataset.map(lambda x, y: (random_rotate_image(x), y))

val_dataset = image_dataset_from_directory(
    "/Users/choijiwon/expression/dataset/valid",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='grayscale',  # Use grayscale for FER2013
)

# Preprocess datasets
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))  # Normalize
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))  # Normalize

train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Build the Custom CNN model
def build_custom_cnn(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        # Conv Block 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Conv Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Conv Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Fully Connected Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
    return model

# Create the model
model = build_custom_cnn(input_shape=(48, 48, 1), num_classes=7)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
print(model.summary())

# Train the model
EPOCHS = 50
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping],
)

# Evaluate the model
test_generator = image_dataset_from_directory(
    "/Users/choijiwon/expression/dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='grayscale',  # Use grayscale for FER2013
)
test_generator = test_generator.map(lambda x, y: (x / 255.0, y))  # Normalize
test_generator = test_generator.prefetch(buffer_size=tf.data.AUTOTUNE)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save("emotion_detection_model_custom_cnn.h5")