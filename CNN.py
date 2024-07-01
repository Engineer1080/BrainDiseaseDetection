from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np



# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=4, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('Training',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 color_mode='grayscale')

test_set = test_datagen.flow_from_directory('Testing',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical',
                                            color_mode='grayscale')

# Train the CNN
classifier.fit_generator(training_set,
                         steps_per_epoch=179,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=41)

# Predict the values from the test data
Y_pred = classifier.predict(test_set)

# Convert predictions classes from one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)

# Convert test observations from one hot vectors
Y_true = test_set.classes

# Compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# Get the number of classes
num_classes = len(np.unique(Y_true))

# Compute sensitivity and specificity for each class
sensitivities = []
specificities = []

for i in range(num_classes):
    true_positives = confusion_mtx[i, i]
    false_negatives = np.sum(confusion_mtx[i, :]) - true_positives
    false_positives = np.sum(confusion_mtx[:, i]) - true_positives
    true_negatives = np.sum(confusion_mtx) - true_positives - false_positives - false_negatives

    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)

    sensitivities.append(sensitivity)
    specificities.append(specificity)

# Evaluate the model
scores = classifier.evaluate(test_set)

# Print scores and metrics
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

for i in range(num_classes):
    print(f"Class {i}:")
    print(f"  Sensitivity: {sensitivities[i]:.4f}")
    print(f"  Specificity: {specificities[i]:.4f}")

# Compute and print classification report
report = classification_report(Y_true, Y_pred_classes)
print("\nClassification Report:")
print(report)

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_mtx)