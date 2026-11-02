"""
Dataset Generator Script for Deep Learning Practicals
Generates CSV datasets for:
- Practical 2: mnist_train.csv, mnist_test.csv
- Practical 3: fashion_mnist_train.csv, fashion_mnist_test.csv
- Practical 4: ecg.csv (for Autoencoder Anomaly Detection)
- Practical 5: cbow_dataset.csv (for Continuous Bag of Words)
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
import scipy.ndimage as ndimage

os.makedirs('datasets', exist_ok=True)

print("1. Generating MNIST CSV datasets (mnist_train.csv, mnist_test.csv)...")
digits = load_digits()
X_orig = digits.images  # (1797, 8, 8)
y_orig = digits.target  # (1797,)

# Upsample 8x8 to 28x28 images
num_samples = len(X_orig)
X_resized = np.zeros((num_samples, 28, 28), dtype=np.float32)

for i in range(num_samples):
    # Zoom from 8x8 to 28x28 (factor 28/8 = 3.5)
    zoomed = ndimage.zoom(X_orig[i], 28.0 / 8.0, order=1)
    # Clip and scale to 0-255
    zoomed = np.clip(zoomed, 0, 16)
    zoomed = (zoomed / 16.0 * 255.0).astype(np.uint8)
    X_resized[i] = zoomed

# Flatten to (1797, 784)
X_flat = X_resized.reshape(num_samples, 784).astype(int)

# Split into train (1400) and test (397)
X_train, X_test = X_flat[:1400], X_flat[1400:]
y_train, y_test = y_orig[:1400], y_orig[1400:]

pixel_cols = [f'pixel{i}' for i in range(784)]

df_mnist_train = pd.DataFrame(X_train, columns=pixel_cols)
df_mnist_train.insert(0, 'label', y_train)
df_mnist_train.to_csv('datasets/mnist_train.csv', index=False)

df_mnist_test = pd.DataFrame(X_test, columns=pixel_cols)
df_mnist_test.insert(0, 'label', y_test)
df_mnist_test.to_csv('datasets/mnist_test.csv', index=False)

print(f"   Saved mnist_train.csv: {df_mnist_train.shape}")
print(f"   Saved mnist_test.csv:  {df_mnist_test.shape}")


print("\n2. Generating Fashion MNIST CSV datasets (fashion_mnist_train.csv, fashion_mnist_test.csv)...")
# Generate synthetic 28x28 patterns for 10 fashion classes:
# 0: T-shirt, 1: Trouser, 2: Pullover, 3: Dress, 4: Coat,
# 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot
np.random.seed(42)

def generate_fashion_image(category):
    img = np.zeros((28, 28), dtype=np.float32)
    noise = np.random.normal(0, 10, (28, 28))
    
    if category == 0:  # T-shirt: torso box + short sleeves
        img[6:24, 7:21] = 180
        img[6:14, 3:7] = 160
        img[6:14, 21:25] = 160
    elif category == 1:  # Trouser: two vertical legs
        img[4:26, 8:12] = 200
        img[4:26, 16:20] = 200
        img[4:10, 8:20] = 200
    elif category == 2:  # Pullover: body + long sleeves
        img[6:24, 8:20] = 190
        img[6:22, 4:8] = 170
        img[6:22, 20:24] = 170
    elif category == 3:  # Dress: narrow top + wide flared skirt
        img[4:12, 10:18] = 210
        for r in range(12, 26):
            w = int(4 + (r - 12) * 0.7)
            img[r, max(0, 14-w):min(28, 14+w)] = 210
    elif category == 4:  # Coat: long body + collar + sleeves
        img[4:26, 9:19] = 175
        img[6:24, 4:9] = 160
        img[6:24, 19:24] = 160
        img[4:10, 13:15] = 40  # collar gap
    elif category == 5:  # Sandal: thin sole + straps
        img[22:25, 4:24] = 220
        img[14:22, 8:11] = 180
        img[14:22, 17:20] = 180
    elif category == 6:  # Shirt: collar + body + buttons
        img[6:24, 8:20] = 185
        img[6:18, 4:8] = 160
        img[6:18, 20:24] = 160
        img[6:24, 14] = 50  # button line
    elif category == 7:  # Sneaker: low cut shoe shape
        img[18:25, 3:25] = 220
        img[13:18, 5:16] = 200
    elif category == 8:  # Bag: square body + strap loop
        img[12:24, 6:22] = 210
        img[6:12, 10:18] = 150
        img[8:12, 12:16] = 0  # hollow handle
    elif category == 9:  # Ankle boot: high ankle shoe shape
        img[18:25, 4:25] = 230
        img[8:18, 5:14] = 210
    
    img = np.clip(img + noise, 0, 255)
    return img.astype(int).flatten()

fashion_data = []
fashion_labels = []
samples_per_class_train = 120
samples_per_class_test = 30

for cat in range(10):
    for _ in range(samples_per_class_train):
        fashion_data.append(generate_fashion_image(cat))
        fashion_labels.append(cat)

df_fashion_train = pd.DataFrame(fashion_data, columns=pixel_cols)
df_fashion_train.insert(0, 'label', fashion_labels)
# Shuffle train
df_fashion_train = df_fashion_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
df_fashion_train.to_csv('datasets/fashion_mnist_train.csv', index=False)

fashion_test_data = []
fashion_test_labels = []
for cat in range(10):
    for _ in range(samples_per_class_test):
        fashion_test_data.append(generate_fashion_image(cat))
        fashion_test_labels.append(cat)

df_fashion_test = pd.DataFrame(fashion_test_data, columns=pixel_cols)
df_fashion_test.insert(0, 'label', fashion_test_labels)
df_fashion_test = df_fashion_test.sample(frac=1.0, random_state=42).reset_index(drop=True)
df_fashion_test.to_csv('datasets/fashion_mnist_test.csv', index=False)

print(f"   Saved fashion_mnist_train.csv: {df_fashion_train.shape}")
print(f"   Saved fashion_mnist_test.csv:  {df_fashion_test.shape}")


print("\n3. Generating ECG dataset for Anomaly Detection (ecg.csv)...")
def generate_ecg_signal(is_normal=True, length=140):
    t = np.linspace(0, 1, length)
    if is_normal:
        p = 0.25 * np.exp(-((t - 0.20) ** 2) / (2 * 0.03 ** 2))
        q = -0.15 * np.exp(-((t - 0.35) ** 2) / (2 * 0.015 ** 2))
        r = (2.0 + np.random.normal(0, 0.1)) * np.exp(-((t - 0.40) ** 2) / (2 * 0.02 ** 2))
        s = -0.4 * np.exp(-((t - 0.45) ** 2) / (2 * 0.02 ** 2))
        t_wave = 0.45 * np.exp(-((t - 0.70) ** 2) / (2 * 0.06 ** 2))
        signal = p + q + r + s + t_wave + np.random.normal(0, 0.03, length)
    else:
        anomaly_type = np.random.choice([0, 1, 2])
        if anomaly_type == 0:
            signal = -1.2 * np.exp(-((t - 0.45) ** 2) / (2 * 0.08 ** 2)) + 0.3 * np.sin(10 * np.pi * t)
        elif anomaly_type == 1:
            signal = 0.8 * np.sin(14 * np.pi * t) + np.random.normal(0, 0.25, length)
        else:
            signal = 1.0 * np.exp(-((t - 0.25) ** 2) / (2 * 0.09 ** 2)) - 0.9 * np.exp(-((t - 0.65) ** 2) / (2 * 0.07 ** 2))
        signal += np.random.normal(0, 0.08, length)
    
    # Normalize to [0, 1] range like standard min-max scaled ECG
    norm_sig = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    return norm_sig

ecg_records = []
# 1000 normal ECGs (label 1.0)
for _ in range(1000):
    sig = generate_ecg_signal(is_normal=True)
    ecg_records.append(np.append(sig, 1.0))

# 300 abnormal ECGs (label 0.0)
for _ in range(300):
    sig = generate_ecg_signal(is_normal=False)
    ecg_records.append(np.append(sig, 0.0))

ecg_array = np.array(ecg_records)
# Shuffle rows
np.random.shuffle(ecg_array)

# Standard format: 140 signal columns + 1 label column (no header or with header)
# We save without header as standard in tf.data ECG tutorials, and also save a version with headers
ecg_df = pd.DataFrame(ecg_array)
ecg_df.to_csv('datasets/ecg.csv', index=False, header=False)
print(f"   Saved ecg.csv: {ecg_df.shape} (1000 Normal beats [1.0], 300 Anomalous beats [0.0])")


print("\n4. Generating CBOW NLP Dataset (cbow_dataset.csv)...")
cbow_sentences = [
    "deep learning is a subset of machine learning based on artificial neural networks",
    "neural networks learn representations of data through multiple layers of abstraction",
    "natural language processing enables computers to understand and process human language",
    "continuous bag of words model predicts the target word from surrounding context words",
    "the skip gram model predicts context words given a target word",
    "word embeddings represent words as continuous vectors in a low dimensional space",
    "similar words are mapped to nearby points in the vector embedding space",
    "feedforward neural networks pass information in one direction from input to output",
    "gradient descent optimizes the weights of the neural network by minimizing the loss function",
    "stochastic gradient descent updates model parameters using a small batch of training data",
    "convolutional neural networks are widely used for image classification and computer vision",
    "recurrent neural networks are designed to process sequential data like text and time series",
    "autoencoders learn a compressed representation of input data for reconstruction and anomaly detection",
    "an anomaly detection system identifies data points that deviate significantly from normal patterns",
    "activation functions introduce non linearity into artificial neural network architectures",
    "rectified linear unit is a popular activation function for deep neural networks",
    "softmax activation function converts logits into probability distributions across classes",
    "cross entropy loss measures the performance of a classification model whose output is probability",
    "mean squared error measures the average squared difference between estimated values and actual values",
    "backpropagation calculates the gradient of the loss function with respect to each weight",
    "hyperparameter tuning optimizes learning rate batch size and number of hidden layers",
    "transfer learning allows models trained on large datasets to solve related tasks",
    "tokenization splits text into smaller units such as words or subword tokens",
    "vocabulary size determines the dimensionality of the one hot encoded vectors",
    "context window size controls how many surrounding words are considered during training",
    "word vectors capture semantic and syntactic relationships between different words",
    "cosine similarity measures the angle between two word vectors in vector space",
    "vector representation allows mathematical operations on words such as king minus man plus woman equals queen",
    "unsupervised learning discovers hidden patterns in data without using explicit target labels",
    "supervised learning trains a predictive model on labeled pairs of input and output data",
    "dense layers connect every neuron in one layer to every neuron in the subsequent layer",
    "dropout regularization prevents overfitting by randomly deactivating neurons during training",
    "batch normalization stabilizes the learning process and reduces training time in deep networks",
    "the learning rate is a crucial hyperparameter that determines the step size during optimization",
    "loss curves visualize how model error decreases across successive training epochs",
    "evaluation metrics such as accuracy precision recall and f1 score assess model performance",
    "confusion matrix summarizes classification results showing true positives and false positives",
    "latent space captures the essential underlying features of complex high dimensional data",
    "reconstruction error measures how accurately an autoencoder reproduces its original input",
    "anomalous samples exhibit significantly higher reconstruction error compared to normal samples",
    "word2vec is an efficient framework for learning continuous distributed vector representations of words",
    "shallow two layer neural networks can learn surprisingly rich semantic word embeddings",
    "matrix factorization and neural network methods are popular approaches for representation learning",
    "language models predict the likelihood of a sequence of words based on linguistic context",
    "text preprocessing includes lowercasing punctuation removal tokenization and stopword filtering",
    "target words are predicted from the average vector of context words in cbow architecture",
    "embedding layer acts as a lookup table that maps integer indices to dense vectors",
    "softmax layer computes the conditional probability distribution over the entire vocabulary",
    "training deep learning models requires substantial computational resources including graphics processing units",
    "artificial intelligence systems transform industries through automated decision making and pattern recognition"
]

cbow_df = pd.DataFrame({
    'id': range(1, len(cbow_sentences) + 1),
    'text': cbow_sentences
})
cbow_df.to_csv('datasets/cbow_dataset.csv', index=False)
print(f"   Saved cbow_dataset.csv: {cbow_df.shape} ({len(cbow_sentences)} sentences)")

print("\nAll datasets generated successfully!")

