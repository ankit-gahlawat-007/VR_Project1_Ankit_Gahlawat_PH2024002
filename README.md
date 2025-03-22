# Face Mask Detection, Classification, and Segmentation

## 🧠 Introduction
This project aims to build computer vision solutions that can detect, classify, and segment face masks in images. It involves both classical machine learning techniques using handcrafted features and deep learning approaches like CNNs and U-Net to perform classification and segmentation tasks. At the end we see a comparison of both the approaches highlighting the supremacy of the deep learning techniques.

## 📂 Dataset

### 1. **Face Mask Classification Dataset**
- **Source**: [Face Mask Detection Dataset](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)
- **Structure**: Contains labeled images of people **with** and **without** masks organized into two folders: `with_mask` and `without_mask`.

### 2. **Masked Face Segmentation Dataset**
- **Source**: [MFSD - Masked Face Segmentation Dataset](https://github.com/sadjadrz/MFSD)
- **Structure**: Includes images with masks along with corresponding ground truth mask segmentation maps.

## 🛠 Methodology

### A. Binary Classification using Handcrafted Features

1. **Preprocessing**:
   - Images were loaded from the dataset and resized to a fixed shape (100×100).
   - Converted to grayscale for feature extraction.

2. **Feature Extraction**:
   - **HOG (Histogram of Oriented Gradients)** features were extracted from each image using `skimage.feature.hog`.
   - These features help capture the structural patterns of the face with or without a mask.

3. **Model Training**:
   - Two classifiers were trained using the HOG features:
     - **Support Vector Machine (SVM)** with a linear kernel.
     - **Multi-Layer Perceptron (MLP)** neural network using `MLPClassifier` from `sklearn`.

4. **Evaluation**:
   - Models were evaluated using accuracy on a test split.
   - Accuracy was compared between SVM and MLP to assess which classifier performs better on handcrafted features.


### B. Binary Classification using CNN

1. **Dataset Preparation**:
   - The dataset was manually split into **train (70%)**, **validation (15%)**, and **test (15%)** sets using `train_test_split`.
   - Images were resized to **128×128** pixels and normalized using mean and standard deviation `[0.5]`.

2. **CNN Architecture**:
   - A custom CNN (`MaskCNN`) was designed with three convolutional blocks:
     - Each block consists of a `Conv2d` layer followed by `ReLU` and `MaxPool2d`.
   - The fully connected head includes:
     - A `Flatten` layer, a hidden `Linear` layer with `ReLU`, `Dropout`, and a final `Linear` layer with a `Sigmoid` activation for binary output.

3. **Training Setup**:
   - Loss Function: **Binary Cross Entropy Loss (BCELoss)**
   - Optimizer: **Adam** with a learning rate of **0.001**
   - Batch Size: **32**
   - Epochs: **10**
   - Training was performed on CPU/GPU depending on availability.

4. **Evaluation**:
   - Model performance was evaluated on the test set using:
     - **Accuracy Score**
     - **Confusion Matrix**
     - **Classification Report** (precision, recall, F1-score)
   - Results were visualized using a seaborn confusion matrix heatmap.

5. **Model Saving**:
   - The trained CNN model was saved using `torch.save()` and later reloaded for testing with the `model_test.py` script.


### C. Region Segmentation using Traditional Techniques

1. **Preprocessing**:
   - The input images were read and resized to a consistent shape for uniformity.
   - Images were converted to grayscale to simplify mask region detection.

2. **Segmentation Techniques**:
   - **Gaussian Blur** was applied to smooth the image and reduce noise.
   - **Canny Edge Detection** was used to highlight edges, helping to identify the contours of mask regions.

3. **Post-Processing and Visualization**:
   - The segmented regions were visualized using contours drawn over the original images.
   - Each processed image displayed both the original and its segmented mask region for comparison.


### D. Mask Segmentation using U-Net

1. **Dataset Preparation**:
   - Used the **Masked Face Segmentation Dataset (MFSD)** consisting of facial images and corresponding ground truth mask segmentation maps.
   - Input images and masks were resized to a fixed resolution to match U-Net's input size requirements.
   - Pixel values were normalized and converted into tensors suitable for PyTorch training.

2. **U-Net Architecture**:
   - A standard **U-Net** architecture was implemented with an encoder-decoder structure:
     - The encoder path used convolutional layers followed by ReLU and max pooling for downsampling.
     - The decoder path used up-convolutions and skip connections to restore spatial resolution.
   - The final output layer used a `Sigmoid` activation function to produce a binary segmentation mask.

3. **Training Setup**:
   - **Loss Function**: Combination of **Binary Cross-Entropy (BCE)** and **Dice Loss** to balance foreground-background class distribution.
   - **Optimizer**: **Adam** optimizer with a learning rate of **0.0001**
   - **Epochs**: Trained for **50 epochs**
   - **Batch Size**: 16

4. **Evaluation**:
   - The model was evaluated using:
     - **Intersection over Union (IoU)**
     - **Dice Coefficient**
   - Performance was visualized by overlaying predicted masks on input images for qualitative inspection.

5. **Results**:
   - The U-Net model achieved significantly higher accuracy and segmentation quality compared to traditional techniques.


## ⚙️ Hyperparameters and Experiments 

"chaneg this according to our experiments"

### CNN Model:
| Parameter       | Values Tried              |
|----------------|---------------------------|
| Learning Rate  | 0.001, 0.0001              |
| Batch Size     | 32, 64                     |
| Optimizer      | Adam, SGD                  |
| Activation     | ReLU, Softmax (final layer)|

### U-Net Model:
| Parameter       | Values Tried                    |
|----------------|----------------------------------|
| Epochs         | 20, 50                           |
| Batch Size     | 16, 32                           |
| Learning Rate  | 0.0001                           |
| Loss Function  | Binary Cross-Entropy + Dice Loss |

## 📊 Results

| Task                                | Method            | Metric      | Score    |
|-------------------------------------|-------------------|-------------|----------|
| Binary Classification               | SVM               | Accuracy    | 91%      |
|                                     | MLP               | Accuracy    | 89%      |
|                                     | CNN               | Accuracy    | **95%**  |
| Region Segmentation                                     | Canny Edge Detection    | IoU         | 0.55     |
| Mask Segmentation                   | U-Net             | IoU         | **0.84** |

"Provide images here too"

## 🔍 Observations and Analysis
- CNN outperformed classical ML classifiers significantly due to its ability to learn complex features.
- Traditional segmentation techniques provided rough masks but lacked precision.
- U-Net performed extremely well in mask segmentation, showcasing its strength in pixel-level tasks.
- Challenges:

## ▶️ How to Run the Code

### 1. Binary Classification (Handcrafted + CNN):

### 2. Segmentation (Handcrafted + CNN):

