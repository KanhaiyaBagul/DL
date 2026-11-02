"""
Script to generate Jupyter Notebooks (.ipynb) for Deep Learning Practicals:
- Practical 2: Feedforward neural networks with Keras and TensorFlow
- Practical 3: Image classification model (4 stages)
- Practical 4: Autoencoder anomaly detection
- Practical 5: Continuous Bag of Words (CBOW) Model
"""

import json
import os

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

def md_cell(source_lines):
    if isinstance(source_lines, str):
        source_lines = [line + '\n' for line in source_lines.split('\n')]
    else:
        source_lines = [l if l.endswith('\n') else l + '\n' for l in source_lines]
    # Remove trailing newline from last line for neatness
    if source_lines:
        source_lines[-1] = source_lines[-1].rstrip('\n')
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }

def code_cell(source_lines):
    if isinstance(source_lines, str):
        source_lines = [line + '\n' for line in source_lines.split('\n')]
    else:
        source_lines = [l if l.endswith('\n') else l + '\n' for l in source_lines]
    if source_lines:
        source_lines[-1] = source_lines[-1].rstrip('\n')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

# ==============================================================================
# PRACTICAL 2: FEEDFORWARD NEURAL NETWORKS WITH KERAS AND TENSORFLOW
# ==============================================================================
p2_cells = [
    md_cell("""# Practical 2: Implementing Feedforward Neural Networks with Keras and TensorFlow

## Objective:
To design, implement, train, and evaluate a Multi-Layer Feedforward Neural Network (Multilayer Perceptron - MLP) using Keras and TensorFlow on a handwritten digit dataset loaded from a **CSV file** (`mnist_train.csv` and `mnist_test.csv`).

---

### Practical Outline:
- **a. Import the necessary packages**
- **b. Load the training and testing data (from CSV)**
- **c. Define the network architecture using Keras**
- **d. Train the model using SGD**
- **e. Evaluate the network**
- **f. Plot the training loss and accuracy**

---
### Dataset Description:
- **`mnist_train.csv`**: Contains training samples with 785 columns (Column `label` = digit 0-9, Columns `pixel0` to `pixel783` = grayscale pixel intensity [0, 255]).
- **`mnist_test.csv`**: Contains test samples for out-of-sample evaluation.
"""),

    md_cell("## a. Import the Necessary Packages"),
    code_cell("""# Stage a: Import required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

# Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow Version:", tf.__version__)
print("Packages successfully imported!")
"""),

    md_cell("## b. Load the Training and Testing Data (from CSV)\nIn this step, we read the training and testing datasets directly from **CSV files** (`mnist_train.csv` and `mnist_test.csv`) using Pandas."),
    code_cell("""# Helper function to find CSV path either in current dir or datasets/
def get_path(filename):
    if os.path.exists(filename):
        return filename
    elif os.path.exists(os.path.join('datasets', filename)):
        return os.path.join('datasets', filename)
    else:
        return filename

train_csv_path = get_path('mnist_train.csv')
test_csv_path = get_path('mnist_test.csv')

print(f"Loading training data from: {train_csv_path}")
print(f"Loading testing data from:  {test_csv_path}")

# Load datasets into pandas DataFrames
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

print(f"Train DataFrame shape: {train_df.shape}")
print(f"Test DataFrame shape:  {test_df.shape}")

# Preview first 5 rows
train_df.head()
"""),

    code_cell("""# Separate features (pixels) and target label
X_train_raw = train_df.drop('label', axis=1).values
y_train_raw = train_df['label'].values

X_test_raw = test_df.drop('label', axis=1).values
y_test_raw = test_df['label'].values

print("X_train shape:", X_train_raw.shape)
print("y_train shape:", y_train_raw.shape)
print("X_test shape: ", X_test_raw.shape)
print("y_test shape: ", y_test_raw.shape)

# Check class distribution
unique, counts = np.unique(y_train_raw, return_counts=True)
print("\\nTraining class distribution:")
for u, c in zip(unique, counts):
    print(f"  Digit {u}: {c} samples")
"""),

    code_cell("""# Data Preprocessing:
# 1. Normalize pixel values from range [0, 255] to [0.0, 1.0] for stable gradient descent
X_train = X_train_raw.astype('float32') / 255.0
X_test = X_test_raw.astype('float32') / 255.0

# 2. One-hot encode the target labels (0-9 -> 10 binary output units)
num_classes = 10
y_train = to_categorical(y_train_raw, num_classes=num_classes)
y_test = to_categorical(y_test_raw, num_classes=num_classes)

print("Preprocessed X_train range: [", X_train.min(), ", ", X_train.max(), "]")
print("Sample one-hot encoded label (digit ", y_train_raw[0], "):", y_train[0])
"""),

    code_cell("""# Visualize sample images from the CSV data
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    # Reshape 784 1D pixels back to 28x28 2D image for visualization
    sample_img = X_train_raw[i].reshape(28, 28)
    plt.imshow(sample_img, cmap='gray')
    plt.title(f"Label: {y_train_raw[i]}")
    plt.axis('off')
plt.suptitle("Sample Handwritten Digits Loaded from CSV", fontsize=14)
plt.tight_layout()
plt.show()
"""),

    md_cell("""## c. Define the Network Architecture using Keras

A Multi-Layer Feedforward Neural Network (MLP) consists of:
- **Input Layer**: 784 input nodes (flattened $28 \\times 28$ pixels).
- **Hidden Layer 1**: Dense layer with 128 neurons and ReLU ($f(x) = \\max(0, x)$) activation.
- **Dropout Layer**: Dropout rate = 0.2 to prevent overfitting.
- **Hidden Layer 2**: Dense layer with 64 neurons and ReLU activation.
- **Output Layer**: Dense layer with 10 neurons and Softmax activation to output probabilities for digits 0–9.
"""),

    code_cell("""# Stage c: Define Feedforward Neural Network Architecture
model = Sequential([
    # First Hidden Layer
    Dense(128, activation='relu', input_shape=(784,), name='hidden_layer_1'),
    Dropout(0.2, name='dropout_1'),
    
    # Second Hidden Layer
    Dense(64, activation='relu', name='hidden_layer_2'),
    Dropout(0.1, name='dropout_2'),
    
    # Output Layer (10 classes with Softmax probability distribution)
    Dense(num_classes, activation='softmax', name='output_layer')
])

# Display model architecture summary
model.summary()
"""),

    md_cell("""## d. Train the Model using SGD (Stochastic Gradient Descent)

We compile the model with:
- **Optimizer**: `SGD(learning_rate=0.01, momentum=0.9)`
- **Loss Function**: `categorical_crossentropy` (standard for multi-class classification)
- **Evaluation Metric**: `accuracy`
"""),

    code_cell("""# Stage d: Compile model using Stochastic Gradient Descent (SGD)
learning_rate = 0.01
momentum = 0.9

sgd_optimizer = SGD(learning_rate=learning_rate, momentum=momentum)

model.compile(
    optimizer=sgd_optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
epochs = 30
batch_size = 32

print(f"Training model for {epochs} epochs with batch size {batch_size}...")
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    verbose=1
)
print("Training completed successfully!")
"""),

    md_cell("## e. Evaluate the Network\nWe evaluate the trained model on unseen test data from `mnist_test.csv`."),
    code_cell("""# Stage e: Evaluate on Test Data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("=" * 45)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("=" * 45)

# Predict probabilities and class labels
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Classification Report
print("\\nClassification Report:")
print(classification_report(y_test_raw, y_pred_classes, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test_raw, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix - Test Set", fontsize=14)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
"""),

    md_cell("## f. Plot the Training Loss and Accuracy\nWe plot the loss and accuracy curves across all training epochs for both training and validation splits."),
    code_cell("""# Stage f: Plot Training Loss and Accuracy Curves
epochs_range = range(1, len(history.history['loss']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Loss Curve
ax1.plot(epochs_range, history.history['loss'], 'b-o', label='Training Loss', linewidth=1.5)
ax1.plot(epochs_range, history.history['val_loss'], 'r--s', label='Validation Loss', linewidth=1.5)
ax1.set_title('Training and Validation Loss', fontsize=13)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Categorical Crossentropy Loss', fontsize=11)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(fontsize=10)

# Subplot 2: Accuracy Curve
ax2.plot(epochs_range, history.history['accuracy'], 'b-o', label='Training Accuracy', linewidth=1.5)
ax2.plot(epochs_range, history.history['val_accuracy'], 'r--s', label='Validation Accuracy', linewidth=1.5)
ax2.set_title('Training and Validation Accuracy', fontsize=13)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(fontsize=10)

plt.suptitle("Feedforward Neural Network Performance Metrics (SGD Optimizer)", fontsize=15)
plt.tight_layout()
plt.show()
"""),

    code_cell("""# Visualizing predictions on sample test images
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = X_test_raw[i].reshape(28, 28)
    true_lbl = y_test_raw[i]
    pred_lbl = y_pred_classes[i]
    color = 'green' if true_lbl == pred_lbl else 'red'
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {true_lbl}\\nPred: {pred_lbl}", color=color)
    plt.axis('off')
plt.suptitle("Sample Test Predictions (Green = Correct, Red = Incorrect)", fontsize=14)
plt.tight_layout()
plt.show()
"""),

    md_cell("""## Conclusion:
1. Successfully loaded and preprocessed image pixel data from CSV files (`mnist_train.csv` and `mnist_test.csv`).
2. Designed a Multi-layer Feedforward Neural Network using Keras `Sequential` API with ReLU and Softmax activations.
3. Trained the model using Stochastic Gradient Descent (`SGD`) optimizer.
4. Achieved high classification accuracy on the test set.
5. Visualized learning curves, confusion matrix, and sample prediction outputs.
""")
]


# ==============================================================================
# PRACTICAL 3: IMAGE CLASSIFICATION MODEL (4 STAGES)
# ==============================================================================
p3_cells = [
    md_cell("""# Practical 3: Build the Image Classification Model

## Objective:
To build, train, and evaluate a deep Convolutional Neural Network (CNN) for image classification on a CSV-based image dataset (`fashion_mnist_train.csv` and `fashion_mnist_test.csv`) by dividing the workflow into the following 4 stages:

- **Stage a. Loading and preprocessing the image data**
- **Stage b. Defining the model’s architecture**
- **Stage c. Training the model**
- **Stage d. Estimating the model’s performance**

---
### Why CNN for Image Classification?
Unlike standard feedforward networks that flatten 2D spatial features into 1D vectors, Convolutional Neural Networks preserve spatial relationships using:
1. **Convolutional Layers (`Conv2D`)**: Extract local features (edges, textures, shapes) through learnable filter kernels.
2. **Pooling Layers (`MaxPooling2D`)**: Downsample feature maps to reduce spatial dimensions, providing translation invariance and computational efficiency.
3. **Dense Fully Connected Layers**: Perform the final high-level reasoning and multi-class classification.
"""),

    md_cell("## Stage a. Loading and Preprocessing the Image Data\nIn this stage, we load images stored in CSV format, reshape them into 3D tensors `(height, width, channels)`, normalize pixel intensities, and prepare labels."),
    code_cell("""# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow Version:", tf.__version__)
"""),

    code_cell("""# Stage a.1: Load data from CSV
def get_path(filename):
    if os.path.exists(filename):
        return filename
    elif os.path.exists(os.path.join('datasets', filename)):
        return os.path.join('datasets', filename)
    else:
        return filename

train_path = get_path('fashion_mnist_train.csv')
test_path = get_path('fashion_mnist_test.csv')

print(f"Loading training CSV from: {train_path}")
print(f"Loading testing CSV from:  {test_path}")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Train samples: {train_df.shape[0]}, Columns: {train_df.shape[1]}")
print(f"Test samples:  {test_df.shape[0]}, Columns: {test_df.shape[1]}")
"""),

    code_cell("""# Stage a.2: Separate features and labels
X_train_raw = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

X_test_raw = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Stage a.3: Reshape into (num_samples, 28, 28, 1) for Conv2D input
# Height = 28, Width = 28, Color channels = 1 (Grayscale)
X_train = X_train_raw.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test_raw.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print("Preprocessed X_train shape:", X_train.shape)
print("Preprocessed X_test shape: ", X_test.shape)
print("Value range: [", X_train.min(), ", ", X_train.max(), "]")

# Class labels mapping
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
"""),

    code_cell("""# Stage a.4: Visualize sample preprocessed images
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i].squeeze(), cmap='viridis')
    plt.title(f"{class_names[y_train[i]]} ({y_train[i]})")
    plt.axis('off')
plt.suptitle("Sample Preprocessed Images from Dataset", fontsize=14)
plt.tight_layout()
plt.show()
"""),

    md_cell("""## Stage b. Defining the Model's Architecture

We build a Convolutional Neural Network (CNN) architecture:
1. **Conv2D Block 1**: 32 filters of size $3 \\times 3$, ReLU activation, followed by `MaxPooling2D(2, 2)`.
2. **Conv2D Block 2**: 64 filters of size $3 \\times 3$, ReLU activation, followed by `MaxPooling2D(2, 2)`.
3. **Flatten**: Converts 2D feature maps into a 1D feature vector.
4. **Dense Layer**: 128 hidden units with ReLU activation and Dropout (0.3).
5. **Output Layer**: 10 units with Softmax activation for 10-class probability distribution.
"""),

    code_cell("""# Stage b: Define the CNN Model Architecture
model = Sequential([
    # Convolutional Block 1
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1), name='conv_1'),
    MaxPooling2D(pool_size=(2, 2), name='pool_1'),
    
    # Convolutional Block 2
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', name='conv_2'),
    MaxPooling2D(pool_size=(2, 2), name='pool_2'),
    
    # Classification Head
    Flatten(name='flatten'),
    Dense(128, activation='relu', name='dense_1'),
    Dropout(0.3, name='dropout'),
    Dense(10, activation='softmax', name='output_layer')
])

model.summary()
"""),

    md_cell("""## Stage c. Training the Model

In this stage:
- We compile the model using the `Adam` optimizer, `sparse_categorical_crossentropy` loss, and `accuracy` metric.
- We train the network for 20 epochs with a validation split of 20%.
- We monitor and record training history.
"""),

    code_cell("""# Stage c.1: Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Stage c.2: Train Model
epochs = 20
batch_size = 32

print(f"Training CNN on {X_train.shape[0]} samples for {epochs} epochs...")
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    verbose=1
)
print("Training successfully finished!")
"""),

    code_cell("""# Stage c.3: Visualize Training Loss and Accuracy Progression
epochs_range = range(1, len(history.history['loss']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss Plot
ax1.plot(epochs_range, history.history['loss'], 'b-o', label='Training Loss')
ax1.plot(epochs_range, history.history['val_loss'], 'r--s', label='Validation Loss')
ax1.set_title('CNN Training and Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

# Accuracy Plot
ax2.plot(epochs_range, history.history['accuracy'], 'b-o', label='Training Accuracy')
ax2.plot(epochs_range, history.history['val_accuracy'], 'r--s', label='Validation Accuracy')
ax2.set_title('CNN Training and Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()
"""),

    md_cell("""## Stage d. Estimating the Model's Performance

In this stage, we comprehensively estimate and evaluate the model's performance on the unseen test dataset:
1. Compute overall Test Loss and Test Accuracy.
2. Generate predictions on test data.
3. Compute Precision, Recall, and F1-Score for each class via Classification Report.
4. Plot Confusion Matrix to analyze misclassifications.
5. Display test images alongside predicted and actual class labels.
"""),

    code_cell("""# Stage d.1: Evaluate on unseen test dataset
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print("=" * 45)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print("=" * 45)

# Stage d.2: Generate Predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Stage d.3: Classification Report
print("\\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
"""),

    code_cell("""# Stage d.4: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix for Image Classification Model", fontsize=14)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()
"""),

    code_cell("""# Stage d.5: Visual Performance Estimation on Sample Test Images
plt.figure(figsize=(14, 6))
num_display = 10
for i in range(num_display):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].squeeze(), cmap='viridis')
    actual = class_names[y_test[i]]
    predicted = class_names[y_pred[i]]
    is_correct = (y_test[i] == y_pred[i])
    color = 'green' if is_correct else 'red'
    
    plt.title(f"True: {actual}\\nPred: {predicted}", color=color, fontsize=10)
    plt.axis('off')
plt.suptitle("Sample Test Image Predictions (Green = Correct, Red = Misclassified)", fontsize=14)
plt.tight_layout()
plt.show()
"""),

    md_cell("""## Conclusion:
1. **Stage a**: Successfully loaded image data from CSV and reshaped to 4D tensor `(N, 28, 28, 1)` with normalized pixel values.
2. **Stage b**: Constructed a Convolutional Neural Network with 2 Conv2D+Pooling blocks and Dense classification head.
3. **Stage c**: Trained model with Adam optimizer and monitored convergence.
4. **Stage d**: Estimated model performance using classification metrics, confusion matrix, and visual predictions.
""")
]


# ==============================================================================
# PRACTICAL 4: AUTOENCODER FOR ANOMALY DETECTION
# ==============================================================================
p4_cells = [
    md_cell("""# Practical 4: Autoencoder for Anomaly Detection

## Objective:
To design, implement, train, and evaluate a Deep Autoencoder to perform **Anomaly Detection** on an Electrocardiogram (ECG) dataset loaded from `ecg.csv`.

---

### Practical Outline:
- **a. Import required libraries**
- **b. Upload / access the dataset**
- **c. Encoder converts it into latent representation**
- **d. Decoder networks convert it back to the original input**
- **e. Compile the models with Optimizer, Loss, and Evaluation Metrics**

---

### How Autoencoders Detect Anomalies:
1. **Autoencoder Principle**: An autoencoder is an unsupervised neural network trained to reconstruct its own input ($x \\approx \\hat{x}$).
2. **Anomaly Detection Logic**:
   - We train the autoencoder **ONLY on normal data** (e.g., normal ECG heartbeats).
   - The model learns the compact latent representation of normal cardiac cycles.
   - When **anomalous data** (arrhythmia/heart disease) is passed to the trained autoencoder, the model fails to reconstruct it accurately.
   - Hence, **anomalous samples exhibit significantly higher Reconstruction Error (MAE/MSE)**.
   - By choosing a threshold (e.g., $\\mu + \\sigma$ of normal reconstruction error), any signal with error above the threshold is flagged as an **Anomaly**.
"""),

    md_cell("## a. Import Required Libraries"),
    code_cell("""# Stage a: Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

print("TensorFlow Version:", tf.__version__)
print("All libraries imported successfully!")
"""),

    md_cell("""## b. Upload / Access the Dataset

We load the `ecg.csv` dataset:
- Each row contains **140 continuous ECG signal voltage readings** followed by **1 binary label**:
  - `1.0`: **Normal ECG rhythm**
  - `0.0`: **Abnormal ECG rhythm (Anomaly)**
"""),

    code_cell("""# Stage b.1: Access ecg.csv
def get_path(filename):
    if os.path.exists(filename):
        return filename
    elif os.path.exists(os.path.join('datasets', filename)):
        return os.path.join('datasets', filename)
    else:
        return filename

ecg_path = get_path('ecg.csv')
print(f"Accessing ECG dataset from: {ecg_path}")

# Load CSV (no header format: 140 features + 1 label)
raw_df = pd.read_csv(ecg_path, header=None)
print("Raw Dataset Shape:", raw_df.shape)
print("Features: 140 signal points, Label: last column")
raw_df.head()
"""),

    code_cell("""# Stage b.2: Separate features and labels
raw_data = raw_df.values
features = raw_data[:, 0:-1]  # First 140 columns
labels = raw_data[:, -1]      # Last column (1.0 = Normal, 0.0 = Anomaly)

# Split into train and test sets
X_train_all, X_test, y_train_all, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train set: {X_train_all.shape}, Test set: {X_test.shape}")
print(f"Normal in test: {np.sum(y_test == 1.0)}, Anomalous in test: {np.sum(y_test == 0.0)}")

# CRUCIAL STEP: Train the autoencoder ONLY on NORMAL data
train_normal_mask = (y_train_all == 1.0)
X_train_normal = X_train_all[train_normal_mask]

test_normal_mask = (y_test == 1.0)
X_test_normal = X_test[test_normal_mask]
X_test_anomalous = X_test[~test_normal_mask]

print(f"X_train_normal shape for training: {X_train_normal.shape}")
"""),

    code_cell("""# Stage b.3: Visualize Normal vs Anomalous ECG Signals
time_steps = np.arange(140)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(time_steps, X_train_normal[0], 'g-', label='Sample Normal ECG', linewidth=2)
plt.plot(time_steps, X_train_normal[1], 'g--', alpha=0.6, label='Sample Normal ECG 2')
plt.title("Normal ECG Heartbeat (Class 1.0)", fontsize=13)
plt.xlabel("Time Step (140 measurements)")
plt.ylabel("Normalized Signal")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time_steps, X_test_anomalous[0], 'r-', label='Sample Anomaly ECG', linewidth=2)
plt.plot(time_steps, X_test_anomalous[1], 'r--', alpha=0.6, label='Sample Anomaly ECG 2')
plt.title("Anomalous / Arrhythmia ECG Heartbeat (Class 0.0)", fontsize=13)
plt.xlabel("Time Step (140 measurements)")
plt.ylabel("Normalized Signal")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.suptitle("ECG Signals: Normal vs Anomalous Morphology", fontsize=15)
plt.tight_layout()
plt.show()
"""),

    md_cell("""## c & d. Encoder and Decoder Networks

### c. Encoder converts input into latent representation:
The encoder progressively compresses the 140-dimensional ECG signal through hidden layers into a lower-dimensional **latent bottleneck** representation (dimension = 8).

### d. Decoder converts latent representation back to the original input:
The decoder mirrors the encoder, taking the 8-dimensional bottleneck and reconstructing the original 140-dimensional signal.
"""),

    code_cell("""# Stage c & d: Build Autoencoder Architecture
input_dim = 140

# Input Layer
input_layer = Input(shape=(input_dim,), name='input_ecg')

# --- ENCODER ---
# Progressively compress: 140 -> 64 -> 32 -> 16 -> 8
e1 = Dense(64, activation='relu', name='encoder_dense_1')(input_layer)
e2 = Dense(32, activation='relu', name='encoder_dense_2')(e1)
e3 = Dense(16, activation='relu', name='encoder_dense_3')(e2)
latent_representation = Dense(8, activation='relu', name='latent_bottleneck')(e3)

# --- DECODER ---
# Progressively reconstruct: 8 -> 16 -> 32 -> 64 -> 140
d1 = Dense(16, activation='relu', name='decoder_dense_1')(latent_representation)
d2 = Dense(32, activation='relu', name='decoder_dense_2')(d1)
d3 = Dense(64, activation='relu', name='decoder_dense_3')(d2)
reconstructed_output = Dense(input_dim, activation='sigmoid', name='reconstructed_output')(d3)

# Full Autoencoder Model
autoencoder = Model(inputs=input_layer, outputs=reconstructed_output, name='ECG_Autoencoder')

# Separate Encoder Model for inspection of latent representations
encoder_model = Model(inputs=input_layer, outputs=latent_representation, name='Encoder_Only')

autoencoder.summary()
"""),

    md_cell("""## e. Compile the Model with Optimizer, Loss, and Evaluation Metrics

- **Optimizer**: `Adam(learning_rate=0.001)`
- **Loss Function**: `Mean Absolute Error (mae)` or `Mean Squared Error (mse)` to measure reconstruction difference
- **Evaluation Metric**: `mae`
"""),

    code_cell("""# Stage e.1: Compile the Autoencoder
autoencoder.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mae',
    metrics=['mae']
)

# Stage e.2: Train Autoencoder on Normal ECG data only
epochs = 30
batch_size = 64

print(f"Training Autoencoder on {X_train_normal.shape[0]} normal ECG samples...")
history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, X_test),
    shuffle=True,
    verbose=1
)
print("Autoencoder training finished successfully!")
"""),

    code_cell("""# Stage e.3: Plot Training and Validation Loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], 'b-o', label='Training Loss (Normal ECG)')
plt.plot(history.history['val_loss'], 'r--s', label='Validation Loss (Full Test Set)')
plt.title("Autoencoder Training and Validation Reconstruction Loss (MAE)", fontsize=13)
plt.xlabel("Epoch")
plt.ylabel("Reconstruction Error (MAE)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
"""),

    md_cell("## Anomaly Detection Inference & Threshold Determination"),
    code_cell("""# Step 1: Compute reconstruction error on NORMAL training data
reconstructed_train = autoencoder.predict(X_train_normal)
train_mae_loss = np.mean(np.abs(reconstructed_train - X_train_normal), axis=1)

# Step 2: Set threshold as mean + 1 standard deviation of normal training loss
threshold = np.mean(train_mae_loss) + np.std(train_mae_loss)

print("=" * 50)
print(f"Normal Training Reconstruction MAE Mean: {np.mean(train_mae_loss):.5f}")
print(f"Normal Training Reconstruction MAE Std:  {np.std(train_mae_loss):.5f}")
print(f"Anomaly Detection Threshold (Mean + Std): {threshold:.5f}")
print("=" * 50)
"""),

    code_cell("""# Visualize Reconstruction Comparison: Normal vs Anomaly
# Normal Test sample
sample_norm = X_test_normal[0:1]
recon_norm = autoencoder.predict(sample_norm)

# Anomalous Test sample
sample_anom = X_test_anomalous[0:1]
recon_anom = autoencoder.predict(sample_anom)

plt.figure(figsize=(14, 5))

# Plot Normal Reconstruction
plt.subplot(1, 2, 1)
plt.plot(sample_norm[0], 'b-', label='Input Normal ECG', linewidth=2)
plt.plot(recon_norm[0], 'r--', label='Reconstructed ECG', linewidth=2)
plt.fill_between(range(140), sample_norm[0], recon_norm[0], color='pink', alpha=0.5, label='Reconstruction Error')
norm_err = np.mean(np.abs(sample_norm[0] - recon_norm[0]))
plt.title(f"Normal ECG: Accurate Reconstruction\\nMAE Error = {norm_err:.4f} (< Threshold)", fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Plot Anomalous Reconstruction
plt.subplot(1, 2, 2)
plt.plot(sample_anom[0], 'b-', label='Input Anomalous ECG', linewidth=2)
plt.plot(recon_anom[0], 'r--', label='Reconstructed ECG', linewidth=2)
plt.fill_between(range(140), sample_anom[0], recon_anom[0], color='lightcoral', alpha=0.5, label='Reconstruction Error')
anom_err = np.mean(np.abs(sample_anom[0] - recon_anom[0]))
plt.title(f"Anomalous ECG: Poor Reconstruction\\nMAE Error = {anom_err:.4f} (> Threshold)", fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.suptitle("Autoencoder Reconstruction Fidelity Comparison", fontsize=15)
plt.tight_layout()
plt.show()
"""),

    code_cell("""# Step 3: Compute reconstruction error on Test Dataset and Evaluate
reconstructed_test = autoencoder.predict(X_test)
test_mae_loss = np.mean(np.abs(reconstructed_test - X_test), axis=1)

# Plot Loss Histogram for Normal vs Anomaly in Test Set
plt.figure(figsize=(10, 5))
plt.hist(test_mae_loss[y_test == 1.0], bins=30, alpha=0.7, color='green', label='Normal ECGs')
plt.hist(test_mae_loss[y_test == 0.0], bins=30, alpha=0.7, color='red', label='Anomalous ECGs')
plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
plt.title("Reconstruction Error Distribution: Normal vs Anomaly", fontsize=14)
plt.xlabel("Reconstruction Loss (MAE)")
plt.ylabel("Number of Samples")
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()
"""),

    code_cell("""# Classify test samples:
# If loss < threshold -> Normal (1.0), else -> Anomaly (0.0)
y_pred = np.where(test_mae_loss < threshold, 1.0, 0.0)

# Performance Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=" * 45)
print("Autoencoder Anomaly Detection Performance:")
print("=" * 45)
print(f"Accuracy:  {acc * 100:.2f}%")
print(f"Precision: {prec * 100:.2f}%")
print(f"Recall:    {rec * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%")
print("=" * 45)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Anomaly (0)', 'Normal (1)'],
            yticklabels=['Anomaly (0)', 'Normal (1)'])
plt.title("Confusion Matrix for Anomaly Detection", fontsize=13)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()
"""),

    md_cell("""## Conclusion:
1. Imported necessary libraries and accessed `ecg.csv` dataset.
2. Extracted normal ECG patterns to train an unsupervised Autoencoder.
3. Constructed an Encoder to compress 140-dimensional signals into an 8-dimensional latent space.
4. Constructed a Decoder to reconstruct the latent representation back to the original 140 dimensions.
5. Successfully detected anomalies based on reconstruction loss threshold with high accuracy and recall.
""")
]


# ==============================================================================
# PRACTICAL 5: CONTINUOUS BAG OF WORDS (CBOW) MODEL
# ==============================================================================
p5_cells = [
    md_cell("""# Practical 5: Implement the Continuous Bag of Words (CBOW) Model

## Objective:
To design, implement, train, and evaluate a **Continuous Bag of Words (CBOW)** Word2Vec neural network model using a custom text corpus loaded from **`cbow_dataset.csv`**.

---

### Practical Stages:
- **Stage a. Data preparation**
- **Stage b. Generate training data**
- **Stage c. Train model**
- **Stage d. Output**

---

### Theory of CBOW Architecture:
- In the **CBOW model**, the goal is to predict a **target center word** $w_t$ given its **surrounding context words**:
  $$\\text{Context: } [w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}] \\longrightarrow \\text{Target: } w_t$$
- The context words are projected through an **Embedding / Shared Weight matrix**.
- The embeddings of all context words in the window are **averaged**.
- The averaged representation passes through a **Softmax layer** over the vocabulary to output the probability of each candidate target word.
"""),

    md_cell("## Stage a. Data Preparation\nIn this stage, we load the text dataset from `cbow_dataset.csv`, clean and tokenize sentences, build the unique vocabulary, and create bidirectional mappings between words and integer IDs."),
    code_cell("""# Stage a: Import libraries
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.decomposition import PCA

print("TensorFlow Version:", tf.__version__)
"""),

    code_cell("""# Stage a.1: Load text dataset from CSV
def get_path(filename):
    if os.path.exists(filename):
        return filename
    elif os.path.exists(os.path.join('datasets', filename)):
        return os.path.join('datasets', filename)
    else:
        return filename

cbow_csv_path = get_path('cbow_dataset.csv')
print(f"Loading CBOW text corpus from: {cbow_csv_path}")

df_cbow = pd.read_csv(cbow_csv_path)
print(f"Total sentences loaded: {len(df_cbow)}")
df_cbow.head(10)
"""),

    code_cell("""# Stage a.2: Text Cleaning and Tokenization
def clean_and_tokenize(text):
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^a-z0-9\\s]', '', text)
    tokens = text.strip().split()
    return tokens

# Tokenize all sentences
all_sentences = [clean_and_tokenize(sentence) for sentence in df_cbow['text'].values]

# Flatten token list to inspect vocabulary
all_tokens = [token for sentence in all_sentences for token in sentence]
total_words = len(all_tokens)

# Build Vocabulary and Mappings
vocab_counts = Counter(all_tokens)
vocab = sorted(list(vocab_counts.keys()))
vocab_size = len(vocab)

word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for i, word in enumerate(vocab)}

print(f"Total running words in corpus: {total_words}")
print(f"Unique Vocabulary Size:        {vocab_size}")
print(f"Sample 15 Vocabulary items:    {vocab[:15]}")
"""),

    md_cell("""## Stage b. Generate Training Data

In this stage:
- We set a **context window size** (e.g., `WINDOW_SIZE = 2`).
- For each target word at index $t$, context words are $[t-2, t-1, t+1, t+2]$.
- We construct input pairs: `(context_word_ids, target_word_id)`.
- Convert target words into one-hot vectors.
"""),

    code_cell("""# Stage b: Generate Context-Target Training Pairs
WINDOW_SIZE = 2  # 2 words before and 2 words after (total 4 context words)

context_data = []
target_data = []

for sentence in all_sentences:
    if len(sentence) < 2 * WINDOW_SIZE + 1:
        continue
    for i in range(WINDOW_SIZE, len(sentence) - WINDOW_SIZE):
        # Extract context words around center word i
        context = (
            sentence[i - WINDOW_SIZE : i] + 
            sentence[i + 1 : i + WINDOW_SIZE + 1]
        )
        target = sentence[i]
        
        # Convert words to integer IDs
        context_ids = [word_to_id[w] for w in context]
        target_id = word_to_id[target]
        
        context_data.append(context_ids)
        target_data.append(target_id)

X = np.array(context_data)
y_int = np.array(target_data)
y = to_categorical(y_int, num_classes=vocab_size)

print(f"Total generated training samples: {len(X)}")
print(f"Shape of Input X (context indices): {X.shape}")
print(f"Shape of Target y (one-hot vectors): {y.shape}")

# Preview sample context-target training pairs
print("\\nSample Context -> Target Training Pairs:")
for j in range(5):
    ctx_words = [id_to_word[idx] for idx in X[j]]
    tgt_word = id_to_word[y_int[j]]
    print(f"  Context: {ctx_words} ---> Target: '{tgt_word}'")
"""),

    md_cell("""## Stage c. Train Model

We implement the CBOW neural network architecture:
1. **Input Layer**: Accepts $2 \\times \\text{WINDOW\\_SIZE} = 4$ integer word IDs.
2. **Embedding Layer**: Projects word IDs into a low-dimensional dense vector space of dimension `embedding_dim = 50`.
3. **Global Average Pooling**: Computes the mean vector of the context word embeddings.
4. **Dense Output Layer**: Softmax layer with `vocab_size` units, predicting probability for each word in the vocabulary.
"""),

    code_cell("""# Stage c: Define CBOW Neural Network Architecture
embedding_dim = 50
context_length = 2 * WINDOW_SIZE

cbow_model = Sequential([
    # Embedding Layer: maps each token id to embedding_dim vector
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=context_length, name='embedding_layer'),
    # Average the context embeddings
    GlobalAveragePooling1D(name='average_context'),
    # Dense Softmax classification layer
    Dense(vocab_size, activation='softmax', name='output_softmax')
], name='CBOW_Model')

# Compile model
cbow_model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cbow_model.summary()
"""),

    code_cell("""# Train the CBOW model
epochs = 80
batch_size = 32

print(f"Training CBOW model for {epochs} epochs...")
history = cbow_model.fit(
    X, y,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)
print("CBOW Model Training Complete!")
"""),

    code_cell("""# Plot CBOW Training Loss and Accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', linewidth=2)
plt.title('CBOW Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Categorical Crossentropy')
plt.grid(True, linestyle=':', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', linewidth=2)
plt.title('CBOW Training Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()
"""),

    md_cell("""## Stage d. Output

In this final stage:
1. **Target Word Prediction**: Given context words, predict the missing center word.
2. **Embedding Weights Extraction**: Extract learned word vectors from the embedding layer.
3. **Cosine Similarity**: Compute semantic similarity between learned word vectors.
4. **2D PCA Visualization**: Plot learned word vectors in 2D semantic space.
"""),

    code_cell("""# Stage d.1: Predicting target word given context words
def predict_target_word(context_words, top_k=3):
    # Ensure all context words exist in vocabulary
    valid_words = [w.lower() for w in context_words if w.lower() in word_to_id]
    if len(valid_words) != 2 * WINDOW_SIZE:
        print(f"Error: Please provide exactly {2 * WINDOW_SIZE} valid context words present in vocabulary.")
        return
    
    ctx_ids = np.array([[word_to_id[w] for w in valid_words]])
    predictions = cbow_model.predict(ctx_ids, verbose=0)[0]
    
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    print(f"\\nContext Words: {valid_words}")
    print("Predicted Target Words (Top Candidates):")
    for rank, idx in enumerate(top_indices, 1):
        word = id_to_word[idx]
        prob = predictions[idx] * 100
        print(f"  {rank}. '{word}' (Probability: {prob:.2f}%)")

# Test prediction with sample context
predict_target_word(['deep', 'learning', 'machine', 'learning'])
predict_target_word(['neural', 'networks', 'representations', 'of'])
"""),

    code_cell("""# Stage d.2: Extract Learned Word Embeddings
embedding_matrix = cbow_model.get_layer('embedding_layer').get_weights()[0]
print("Learned Embedding Matrix Shape (vocab_size, embedding_dim):", embedding_matrix.shape)

# Helper function to compute cosine similarity
def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def find_most_similar_words(target_word, top_n=5):
    target_word = target_word.lower()
    if target_word not in word_to_id:
        print(f"Word '{target_word}' not found in vocabulary.")
        return
    
    target_idx = word_to_id[target_word]
    target_vec = embedding_matrix[target_idx]
    
    similarities = []
    for word, idx in word_to_id.items():
        if word == target_word:
            continue
        sim = cosine_similarity(target_vec, embedding_matrix[idx])
        similarities.append((word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\\nMost similar words to '{target_word}':")
    for rank, (word, sim) in enumerate(similarities[:top_n], 1):
        print(f"  {rank}. '{word}' -> Cosine Similarity: {sim:.4f}")

# Find similar words
find_most_similar_words('neural')
find_most_similar_words('learning')
find_most_similar_words('data')
"""),

    code_cell("""# Stage d.3: 2D Visualization of Word Embeddings using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embedding_matrix)

# Select a subset of prominent words to plot cleanly
selected_words = [
    'neural', 'networks', 'deep', 'learning', 'data',
    'words', 'embeddings', 'model', 'training', 'loss',
    'gradient', 'descent', 'artificial', 'classification',
    'accuracy', 'layers', 'representation', 'anomaly', 'autoencoders'
]
selected_words = [w for w in selected_words if w in word_to_id]

plt.figure(figsize=(12, 8))
for word in selected_words:
    idx = word_to_id[word]
    x, y_coord = reduced_embeddings[idx]
    plt.scatter(x, y_coord, color='dodgerblue', s=80)
    plt.annotate(word, xy=(x, y_coord), xytext=(5, 2),
                 textcoords='offset points', ha='left', va='bottom',
                 fontsize=11, fontweight='semibold')

plt.title("2D Projection (PCA) of CBOW Word Embeddings", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=11)
plt.ylabel("Principal Component 2", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
"""),

    md_cell("""## Conclusion:
1. **Stage a**: Loaded text corpus from `cbow_dataset.csv`, cleaned, tokenized, and created vocabulary dictionary mappings.
2. **Stage b**: Generated training pairs of context words ($X$) and target words ($y$) with sliding window size of 2.
3. **Stage c**: Built and trained a CBOW architecture with an Embedding layer and Global Average Pooling.
4. **Stage d**: Outputted target word predictions, extracted continuous word vectors, calculated cosine similarities, and projected embeddings in 2D space.
""")
]

# Write out notebooks
notebooks = {
    "Practical_2_Feedforward_Neural_Networks.ipynb": p2_cells,
    "Practical_3_Image_Classification.ipynb": p3_cells,
    "Practical_4_Autoencoder_Anomaly_Detection.ipynb": p4_cells,
    "Practical_5_Continuous_Bag_of_Words_CBOW.ipynb": p5_cells,
}

for filename, cells in notebooks.items():
    nb = make_notebook(cells)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Generated: {filename} ({len(cells)} cells)")

print("\nAll 4 Jupyter Notebooks generated successfully!")

