import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define the paths to the dataset (Adjust these paths as per your folder locations)
# Replace these with the paths to your leapGestRecog dataset
base_data_path = r"C:\Users\sathi\OneDrive\Desktop\WORKSPACE\Prodigy\TASK-4\leapGestRecog"  # Replace with the path to the leapGestRecog folder

# Image dimensions
img_width, img_height = 128, 128

# Data augmentation to help with overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load the training and validation data
# Train directory should be inside the base folder, with subfolders for each class (gesture)
train_generator = train_datagen.flow_from_directory(
    os.path.join(base_data_path, 'train'),  # Assuming you have a 'train' folder under leapGestRecog
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

# Validation directory should also be under the base folder with subfolders for each class (gesture)
validation_generator = validation_datagen.flow_from_directory(
    os.path.join(base_data_path, 'validation'),  # Assuming you have a 'validation' folder under leapGestRecog
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical'
)

# Building the CNN Model
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the results to feed into a fully connected layer
model.add(Flatten())

# Fully Connected Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # Assuming you have 10 classes for gestures (modify if needed)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the model
model.save("hand_gesture_model.h5")

# Plot training & validation accuracy and loss
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
