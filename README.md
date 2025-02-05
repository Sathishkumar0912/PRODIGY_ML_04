# Hand Gesture Recognition with CNN

This repository contains code for building and training a Convolutional Neural Network (CNN) model for recognizing hand gestures using the **Leap Gesture Recognition** dataset. The model is built using TensorFlow and Keras, and it utilizes data augmentation techniques to improve model generalization.


## Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.x
- TensorFlow (with Keras)
- NumPy
- OpenCV
- Matplotlib

You can install the required dependencies using `pip`:


pip install tensorflow numpy opencv-python matplotlib


## Dataset

The code assumes that you have the **Leap Gesture Recognition** dataset, which contains images of hand gestures. The dataset should be organized into two main directories:

- `train/`: Contains training images with subfolders for each class (gesture).
- `validation/`: Contains validation images with subfolders for each class (gesture).

Ensure the dataset is placed in the following structure:

```
leapGestRecog/
│
├── train/
│   ├── gesture_1/
│   ├── gesture_2/
│   └── ...
│
└── validation/
    ├── gesture_1/
    ├── gesture_2/
    └── ...
```

## Usage

1. Clone this repository to your local machine:

```
git clone https://github.com/yourusername/hand-gesture-recognition.git
```

2. Place the **Leap Gesture Recognition** dataset in the appropriate directory (e.g., `leapGestRecog/`).
3. Modify the `base_data_path` in the code to reflect the correct path to the dataset folder.

4. Run the Python script:

```
python train_gesture_model.py
```

This will train the CNN model on the dataset and save the model as `hand_gesture_model.h5`.

## Model Architecture

The model is a Convolutional Neural Network (CNN) with the following architecture:

1. **Conv2D Layer** with 32 filters of size (3x3) followed by MaxPooling (2x2)
2. **Conv2D Layer** with 64 filters of size (3x3) followed by MaxPooling (2x2)
3. **Conv2D Layer** with 128 filters of size (3x3) followed by MaxPooling (2x2)
4. **Flatten Layer** to convert the 3D outputs to 1D
5. **Dense Layer** with 128 units and ReLU activation
6. **Dropout Layer** with 50% dropout
7. **Final Dense Layer** with 10 units (one for each gesture class) and Softmax activation for classification

## Training the Model

To train the model:

1. Set the correct `base_data_path` to your dataset directory.
2. Run the model training script (`train_gesture_model.py`). The model will:
   - Use data augmentation on the training dataset to avoid overfitting.
   - Train the model for 10 epochs.
   - Save the trained model as `hand_gesture_model.h5` in the current directory.
   - Plot the training and validation accuracy/loss graphs to monitor the training process.

## Results

After training, the model's performance is visualized using Matplotlib. The accuracy and loss curves for both training and validation datasets are displayed to show how the model performed across epochs.
