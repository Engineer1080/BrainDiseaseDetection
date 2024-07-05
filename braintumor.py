from idlelib import history

import numpy as np
import pandas as pd
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import datetime
warnings.filterwarnings('ignore')


if not os.path.exists('model_checkpoints'):
    os.makedirs('model_checkpoints')

train_dir = 'Training'
test_dir = 'Testing'

# Image params
batch_size = 32
img_height = 300
img_width = 300

# Create a training dataset
train_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=13,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Create a validation dataset
validation_dataset = image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=3,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Create a testing dataset
test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Get list of class names
class_names = train_dataset.class_names

# Prepare datasets
train_dataset = train_dataset.cache().shuffle(1024).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Display sample images
plt.figure(figsize=(15, 10))
for images, labels in train_dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Load the base model and add custom layers
base_model = tf.keras.applications.EfficientNetV2B1(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False

for layer in base_model.layers[:251]:
   layer.trainable = False
for layer in base_model.layers[251:]:
   layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

# Create predictions layer
predictions = Dense(4, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

# Creating a new model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Load the weights of the model you want to continue training with
weights_path = 'model_checkpoints/best_model.h5'

if os.path.exists(weights_path):
    model.load_weights(weights_path)
else:
    print(f"No weights file found at {weights_path}, training from scratch.")

best_model_path = 'model_checkpoints/best_model.h5'

# Setting up checkpoints callbacks
checkpoint_callback = ModelCheckpoint(
    filepath='model_checkpoints/model-{epoch:02d}-{val_loss:.2f}.h5',
    monitor='val_loss',
    save_freq='epoch',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

# Setting up learning rate reduction callback
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Number of training epochs
NUM_EPOCHS = 10

# Train the model and save the history
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=NUM_EPOCHS,
    verbose=1,
    callbacks=[checkpoint_callback, early_stopping, reduce_lr]
)

# After the training, save the best model separately
model.save(best_model_path)

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if not os.path.exists('plots'):
    os.makedirs('plots')

# Plot the training history
history_df = pd.DataFrame(history.history)
plt.figure(figsize = (14, 7))
sns.lineplot(data=history_df[['accuracy', 'val_accuracy']], markers = True)
plt.title('Accuracy Plot')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(f"plots/Accuracy_Plot_{now}.png")
plt.show()

# Plot the training history again but with loss and val_loss
plt.figure(figsize = (14, 7))
sns.lineplot(data=history_df[['loss', 'val_loss']], markers = True)
plt.title('Loss Plot')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(f"plots/Loss_Plot_{now}.png")
plt.show()

# Evaluate the model
print("Testing Metrics are as Follows: ")
model.evaluate(test_dataset, return_dict = True)

# Create a confusion matrix
predictions = np.argmax(model.predict(test_dataset), axis = 1)
true_labels = []

for images, labels in test_dataset:
    true_labels.extend(labels.numpy())

conf_matrix = confusion_matrix(true_labels, predictions)
conf_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(f"plots/Confusion Matrix_{now}.png")
plt.show()
