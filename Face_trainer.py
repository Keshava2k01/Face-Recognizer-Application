#!/usr/bin/env python3
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import ssl

# Disable SSL certificate verification (useful for some environments, but may pose security risks)
ssl._create_default_https_context = ssl._create_unverified_context

# Path to the folder containing the images
data_dir = "/Users/keshavakarthikeyan/Desktop/ASU/Spring'24/MFG598/final project/dataset"

# Load the VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the VGG16 model
for layer in vgg_model.layers:
    layer.trainable = True

# Create a new model
last_layer = vgg_model.get_layer('block5_pool').output
#print(last_layer)
x = Flatten(name='flatten')(last_layer)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu', name='fc2')(x)
x = Dropout(0.3)(x)

# Get the list of all subdirectories (each subdirectory represents a different person)
items = os.listdir(data_dir)
persons = sum(os.path.isdir(os.path.join(data_dir, item)) for item in items)
#print(persons)

# Update the output layer to match the actual number of classes
num_classes = persons
#print(num_classes)
output = Dense(num_classes, activation='softmax', name='predictions')(x)
model = Model(vgg_model.input, output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Load images and labels
X, y = [], []
for person in items:
    person_dir = os.path.join(data_dir, person)
    if not os.path.isdir(person_dir):
        continue  # Skip if it's not a directory
    print(f"Loading images for {person}")
    for image_path in os.listdir(person_dir):
        if image_path.endswith('.DS_Store'):
            continue  # Skip .DS_Store files
        img_path = os.path.join(person_dir, image_path)
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            X.append(img)
            y.append(person)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

# Check if X is empty
if not X:
    print("No images found. Please check the data directory.")
    exit()

# Concatenate the list of arrays in X
X = np.vstack(X)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train))
# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)


# Save the model
model.save('vggface_model.h5')
# Assuming `le` is your LabelEncoder object
np.save('label_encoder_classes.npy', le.classes_)
