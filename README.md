# Image Classification using CNN (CIFAR-10 Dataset)

This project demonstrates a custom-built Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model is trained to classify images into 10 different categories, including airplanes, cars, cats, dogs, and more.

---

## Dataset: CIFAR-10
- 60,000 32x32 color images
- 10 categories (6,000 images per class)
- Preloaded from `tensorflow.keras.datasets`

---

##  Key Features

###  Data Preprocessing
- Loaded dataset using TensorFlow
- Applied normalization (pixel values scaled to 0–1)
- Split dataset into training, validation, and test sets

###  CNN Architecture
- Multiple Convolutional + MaxPooling layers
- Flatten and Dense layers
- ReLU activation and Softmax output layer

###  Model Training
- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy, Precision, Recall, F1-Score

###  Evaluation & Visualization
- Training & validation accuracy/loss curves
- Confusion matrix using Seaborn heatmap
- Classification report using `sklearn`

---

##  Results

- Model shows strong performance in identifying multiple object categories
- Visualizations help in identifying misclassifications and performance trends

---

##  Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn

---

##  How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
