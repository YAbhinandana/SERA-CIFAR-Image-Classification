
#import required libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

#loading and preprocessing of the data
# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Normalize the pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.1,
    fill_mode='nearest'
)
datagen.fit(X_train)
plt.show()

#densenet
# Load the DenseNet121 model with pretrained ImageNet weights
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # Regularization with Dropout
    layers.Dense(10, activation='softmax')
])

#compile
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#setup callbacks
# Callbacks for reducing learning rate and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

#train model
history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    epochs=100,
                    validation_data=(X_test, y_test),
                    callbacks=[reduce_lr, early_stopping])

#evaluate the model
# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#plot trainig aand validation metrices
# Plot the training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

#save model
model.save('cifar10_model.h5')


from tensorflow.keras.applications import ResNet50

# Load the ResNet50 model with pretrained ImageNet weights
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the ResNet layers
for layer in resnet_model.layers:
    layer.trainable = False

# Combine DenseNet and ResNet outputs
combined_input = layers.concatenate([base_model.output, resnet_model.output])
combined_output = layers.GlobalAveragePooling2D()(combined_input)

# Add custom layers on top
x = layers.Dense(256, activation='relu')(combined_output)
x = layers.Dropout(0.5)(x)
final_output = layers.Dense(10, activation='softmax')(x)

# Define the hybrid model
hybrid_model = models.Model(inputs=[base_model.input, resnet_model.input], outputs=final_output)

# Compile and train the hybrid model as above