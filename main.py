import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the image data to the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Load the trained model
model = load_model('cifar10_model.h5')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Plot the training and validation loss and accuracy if history is available
def plot_accuracy_loss(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Assuming you have built your model before this step
# Add this part when you train your model
history = model.fit(x_train, y_train, 
                    validation_data=(x_test, y_test), 
                    epochs=20, 
                    batch_size=64)

# Now pass the history object to the plot function to visualize accuracy and loss
plot_accuracy_loss(history)
