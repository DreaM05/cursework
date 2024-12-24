# 1. ARCHITECTURE MODIFICATION AND COMPARISON
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Helper function: Rescale images
def rescale(data):
    return np.reshape(data, data.shape + (1,))

# Helper function: Plot results
def plot_results(history, title):
    loss_values = history.history['loss']
    accuracy_values = history.history['accuracy']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    ax1.plot(range(len(loss_values)), loss_values, label='Loss')
    ax1.set_title(f'{title} - Loss')
    ax2.plot(range(len(accuracy_values)), accuracy_values, label='Accuracy')
    ax2.set_title(f'{title} - Accuracy')

    ax1.legend(), ax2.legend()
    plt.tight_layout()
    plt.show()

# Load MNIST from TensorFlow
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Preprocessing: Normalize and reshape images
training_images = rescale(training_images / 255.0)
test_images = rescale(test_images / 255.0)

# Different model architectures to test
architectures = [
    {"name": "Baseline (Dense)", "layers": [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]},
    {"name": "Deep Dense", "layers": [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ]},
    {"name": "Dropout Dense", "layers": [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ]}
]

# Loop through architectures and train models
for arch in architectures:
    print(f"\nTraining model: {arch['name']}")

    # Define model
    model = tf.keras.Sequential(arch['layers'])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train model
    history = model.fit(training_images, training_labels, epochs=5, batch_size=256, verbose=1)

    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot results
    plot_results(history, arch['name'])

# 2. MNIST AND NPZ DATASET ANALYSIS
import os
import numpy as np
import matplotlib.pyplot as plt

# Define path to MNIST npz file
current_dir = os.getcwd()
data_path = os.path.join(current_dir, "./sample_data/mnist.npz")

# Load MNIST npz file
with np.load(data_path) as data:
    npz_training_images = data['x_train']
    npz_training_labels = data['y_train']
    npz_test_images = data['x_test']
    npz_test_labels = data['y_test']

# Analyze dataset: Print shapes
print("\nNPZ Dataset Shapes:")
print(f"Training Images: {npz_training_images.shape}, Training Labels: {npz_training_labels.shape}")
print(f"Test Images: {npz_test_images.shape}, Test Labels: {npz_test_labels.shape}")

# Plot histogram of labels
plt.figure(figsize=(8, 4))
plt.hist(npz_training_labels, bins=range(11), rwidth=0.8, align='left')
plt.title("Label Distribution in Training Set")
plt.xlabel("Digit")
plt.ylabel("Frequency")
plt.show()

# Save and reload npz file as a test
save_path = os.path.join(current_dir, "mnist_test_save.npz")
np.savez(save_path, x_train=npz_training_images, y_train=npz_training_labels, x_test=npz_test_images, y_test=npz_test_labels)

# Reload and verify
with np.load(save_path) as reloaded_data:
    print("\nReloaded NPZ Data Keys:", reloaded_data.files)
    print(f"Reloaded Training Images Shape: {reloaded_data['x_train'].shape}")
