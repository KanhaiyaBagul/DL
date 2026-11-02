# Deep Learning Practicals (Laboratory Submissions)

This repository contains complete implementations of **Deep Learning Practicals (Practicals 2, 3, 4, and 5)** using **TensorFlow, Keras, and Pandas**, provided as individual Jupyter Notebooks (`.ipynb`) and using **CSV files** for all datasets.

---

## 📂 Project Structure

```text
Deep Learning Practical/
│
├── Practical_2_Feedforward_Neural_Networks.ipynb   # Practical 2 Notebook
├── Practical_3_Image_Classification.ipynb          # Practical 3 Notebook
├── Practical_4_Autoencoder_Anomaly_Detection.ipynb # Practical 4 Notebook
├── Practical_5_Continuous_Bag_of_Words_CBOW.ipynb  # Practical 5 Notebook
│
├── datasets/                                       # CSV Datasets
│   ├── mnist_train.csv                             # MNIST training data (1400 samples, 785 cols)
│   ├── mnist_test.csv                              # MNIST testing data (397 samples, 785 cols)
│   ├── fashion_mnist_train.csv                     # Fashion-MNIST training data (1200 samples)
│   ├── fashion_mnist_test.csv                      # Fashion-MNIST testing data (300 samples)
│   ├── ecg.csv                                     # ECG 5000 signals (1300 samples, 141 cols)
│   └── cbow_dataset.csv                            # Text corpus for CBOW (50 sentences)
│
├── generate_datasets.py                            # Python script to regenerate all CSV datasets
├── create_notebooks.py                             # Python script that builds the notebooks
└── README.md                                       # Comprehensive documentation & manual
```

> **Note on CSV File Paths**: Each notebook automatically looks for CSV files in both the current directory (`.`) and the `datasets/` folder, ensuring zero configuration whether run locally or in Google Colab.

---

## 📘 Practical Breakdown & Specifications

### Practical 2: Implementing Feedforward Neural Networks with Keras & TensorFlow
- **File**: `Practical_2_Feedforward_Neural_Networks.ipynb`
- **Dataset**: `mnist_train.csv` and `mnist_test.csv`
- **Sub-tasks Covered**:
  - **a. Import the necessary packages**: TensorFlow, Keras Sequential, Dense, Dropout, SGD, Pandas, NumPy, Matplotlib, Scikit-learn.
  - **b. Load the training and testing data**: Loaded via `pd.read_csv()`, extracted `label` and 784 pixel columns, normalized to $[0.0, 1.0]$, one-hot encoded labels with `to_categorical(num_classes=10)`. Visualized sample digits.
  - **c. Define network architecture using Keras**: Sequential MLP with Input(784) $\rightarrow$ Dense(128, ReLU) $\rightarrow$ Dropout(0.2) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dropout(0.1) $\rightarrow$ Dense(10, Softmax).
  - **d. Train the model using SGD**: Compiled with `SGD(learning_rate=0.01, momentum=0.9)` and `categorical_crossentropy` loss. Trained for 30 epochs with validation split.
  - **e. Evaluate the network**: Evaluated test accuracy and loss, generated classification report (Precision, Recall, F1), and plotted Confusion Matrix.
  - **f. Plot training loss and accuracy**: Plotted training vs. validation loss and accuracy curves side-by-side, plus sample image predictions (green for correct, red for incorrect).

---

### Practical 3: Build the Image Classification Model (4 Stages)
- **File**: `Practical_3_Image_Classification.ipynb`
- **Dataset**: `fashion_mnist_train.csv` and `fashion_mnist_test.csv` (10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **4 Stages Covered**:
  - **Stage a. Loading and preprocessing the image data**: Loaded CSVs with pandas, extracted features and labels, reshaped 784 1D vectors into 4D image tensors `(batch_size, 28, 28, 1)` for 2D convolution, normalized pixel intensities, and plotted sample images with class names.
  - **Stage b. Defining the model’s architecture**: Designed a Convolutional Neural Network (CNN) featuring `Conv2D(32, 3x3)` $\rightarrow$ `MaxPooling2D(2x2)` $\rightarrow$ `Conv2D(64, 3x3)` $\rightarrow$ `MaxPooling2D(2x2)` $\rightarrow$ `Flatten()` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(10, Softmax)`.
  - **Stage c. Training the model**: Compiled with `Adam(lr=0.001)` and `sparse_categorical_crossentropy`. Trained for 20 epochs with validation curves recorded and plotted.
  - **Stage d. Estimating the model’s performance**: Evaluated test loss and accuracy, printed per-class classification report, plotted confusion matrix heatmap, and visualized predictions on test images with confidence.

---

### Practical 4: Autoencoder for Anomaly Detection
- **File**: `Practical_4_Autoencoder_Anomaly_Detection.ipynb`
- **Dataset**: `ecg.csv` (140 signal points per row + 1 target column: `1.0` = Normal heartbeat, `0.0` = Arrhythmia / Anomaly)
- **Sub-tasks Covered**:
  - **a. Import required libraries**: TensorFlow, Keras Model API, Dense, Adam, train_test_split, Pandas, Matplotlib, Scikit-learn metrics.
  - **b. Upload / access the dataset**: Loaded `ecg.csv`, separated signal features and labels, split into train and test sets. Isolated normal ECGs (`label == 1.0`) for training. Plotted normal vs. anomalous ECG waveforms.
  - **c. Encoder converts it into latent representation**: Progressively compressed input dimension: `140` $\rightarrow$ `Dense(64, relu)` $\rightarrow$ `Dense(32, relu)` $\rightarrow$ `Dense(16, relu)` $\rightarrow$ `Dense(8, relu)` (bottleneck latent code).
  - **d. Decoder networks convert it back to original input**: Progressively reconstructed signal: `Dense(16, relu)` $\rightarrow$ `Dense(32, relu)` $\rightarrow$ `Dense(64, relu)` $\rightarrow$ `Dense(140, sigmoid)`.
  - **e. Compile the models with Optimizer, Loss, and Evaluation Metrics**: Compiled with `Adam(lr=0.001)` and `mae` (Mean Absolute Error). Trained exclusively on normal heartbeats.
  - **Anomaly Detection Evaluation**: Computed reconstruction error threshold ($\mu + \sigma$ of normal training loss), plotted reconstruction difference between normal and anomalous signals, plotted error distribution histograms, and evaluated classification using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

### Practical 5: Continuous Bag of Words (CBOW) Model
- **File**: `Practical_5_Continuous_Bag_of_Words_CBOW.ipynb`
- **Dataset**: `cbow_dataset.csv` (50 rich AI & Deep Learning sentences)
- **4 Stages Covered**:
  - **Stage a. Data preparation**: Loaded text from `cbow_dataset.csv`, performed cleaning, lowercasing, punctuation stripping, tokenization, vocabulary creation, and built bidirectional lookup mappings (`word_to_id` and `id_to_word`).
  - **Stage b. Generate training data**: Implemented sliding context window (`WINDOW_SIZE = 2`, total 4 context words per target). Extracted context-target pairs `([w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}], w_t)` and one-hot encoded targets.
  - **Stage c. Train model**: Built Keras CBOW neural network: `Embedding(vocab_size, embedding_dim=50)` $\rightarrow$ `GlobalAveragePooling1D()` $\rightarrow$ `Dense(vocab_size, Softmax)`. Compiled with `Adam(lr=0.01)` and `categorical_crossentropy`. Trained for 80 epochs.
  - **Stage d. Output**:
    1. Target word prediction function given context words (prints top-3 candidates with probabilities).
    2. Extracted learned dense word embedding matrix.
    3. Implemented Cosine Similarity function to find most semantically similar words.
    4. Plotted 2D projection of learned word vectors using Principal Component Analysis (PCA).

---

## 🚀 How to Run the Practicals

### Option 1: In VS Code or Local Jupyter Notebook
1. Ensure Python 3.9+ is installed along with:
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
   ```
2. Open the workspace folder in VS Code or launch Jupyter Lab:
   ```bash
   jupyter lab
   ```
3. Open any of the `.ipynb` files and run the cells sequentially from top to bottom.

### Option 2: In Google Colab
1. Upload the `.ipynb` file to Google Colab.
2. In Colab's file manager on the left pane, upload the corresponding CSV file (e.g. `mnist_train.csv`, `ecg.csv`, etc.).
3. Run all cells (`Ctrl + F9` or `Runtime -> Run all`).

