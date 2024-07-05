import os
from tensorflow.keras import Model
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import glob
import sys

# Model configuration (Must have the same architecture as the trained model)
base_model = tf.keras.applications.EfficientNetV2B1(weights=None, include_top=False, input_shape=(300, 300, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

predictions = Dense(4, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
model = Model(inputs=base_model.input, outputs=predictions)

weights_path = 'model_checkpoints/best_model.h5'
if os.path.exists(weights_path):
    model.load_weights(weights_path)
else:
    sys.exit("No weights file found. Please run braintumor.py to generate the model before running this script.")

# Get a list of image file paths
img_paths = glob.glob('demo/*.jpg')  # Replace this with the actual path to your images directory if needed

# Initialize a list to hold the prediction results
prediction_results = []

# Iterate over the image paths
for img_path in img_paths:
    # Load the image
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    # Preprocess the image
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_batch)

    # Apply the model to the image
    predictions = model.predict(img_preprocessed)

    # Interpret the predictions
    predicted_class_idx = np.argmax(predictions[0])
    class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']  # Replace this with your actual class names
    predicted_class = class_names[predicted_class_idx]

    # Add the result to the list of prediction results
    prediction_results.append((img_path, predicted_class))

# Print the prediction results
for img_path, predicted_class in prediction_results:
    print(f'The predicted class for the image {img_path} is: {predicted_class}')
