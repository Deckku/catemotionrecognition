import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
CLASS_NAMES = ["angry", "beg", "disgusted", "frightened", "happy", 
               "normal", "sad", "scared", "sick", "surprised", "wonder"]
NUM_CLASSES = len(CLASS_NAMES)

# Simple CNN model
def create_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and preprocess images
def load_and_preprocess_images(image_dir):
    images = []
    labels = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(image_dir, class_name)
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(class_idx)
    return np.array(images), to_categorical(np.array(labels), num_classes=NUM_CLASSES)

# Load data (replace these with your directories)
train_dir = 'C:\\Users\\adnan\\Desktop\\IA\\projectmchach\\gooddataset\\train'
val_dir = 'C:\\Users\\adnan\\Desktop\\IA\\projectmchach\\gooddataset\\val'

X_train, Y_train = load_and_preprocess_images(train_dir)
X_val, Y_val = load_and_preprocess_images(val_dir)

# Create and train model
model = create_cnn_model()
model.fit(X_train, Y_train, epochs=15, batch_size=32, validation_data=(X_val, Y_val))

# Save the model
model.save('maybe3.h5')