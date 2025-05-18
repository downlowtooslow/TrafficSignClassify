# Traffic Sign Recognition with Deep Learning

This project implements a deep learning model to classify traffic signs from images. It covers data loading, preprocessing, model training with CNNs, data augmentation, hyperparameter tuning, and evaluation on both a dataset and independent images.

---

## Table of Contents

- [Dataset](#dataset)
- [Setup & Dependencies](#setup--dependencies)
- [Data Loading](#data-loading)
- [Data Preprocessing & Augmentation](#data-preprocessing--augmentation)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Inference on Random Images](#inference-on-random-images)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Testing on New Data](#testing-on-new-data)
- [Results & Visualizations](#results--visualizations)
- [License](#license)

---

## Dataset

- The dataset consists of traffic sign images stored in the `trafficsigns_dataset` directory.
- The dataset includes images categorized into different classes, organized in class-named subfolders.
- Additional sign types are stored in a ZIP file (`types.zip`) with further classification.

---

## Setup & Dependencies

Make sure you have the following installed:

- Python 3.6+
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://www.scipy.org/)

Install the required Python packages:

```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib scipy
```

---

## Data Loading & Preprocessing

- The dataset images are loaded using `ImageDataGenerator`.
- Data is split into training and validation sets (`80/20`).
- Images are resized to `(28, 28)` and converted to grayscale for consistency.
- Data augmentation techniques are used to improve model robustness.

---

## Model Architecture

- Convolutional Neural Network (CNN) with multiple Conv2D, MaxPooling, Dropout, and Dense layers.
- Can be trained with or without augmentation.
- Hyperparameter tuning is implemented via Keras Tuner for optimal performance.

---

## Training & Evaluation

- Models are trained over 20 epochs by default.
- Loss and accuracy curves are plotted to monitor overfitting/underfitting.
- Best models are saved (`.keras` files) for later inference.

---

## Inference on Random Images

- Random images from the dataset are loaded.
- The model predicts their class and displays the images with predicted labels.

---

## Hyperparameter Tuning

- (Commented out) code included for hyperparameter tuning to optimize filters, kernel sizes, dense units, and optimizer choice.
- Can be enabled to find the best model configuration.

---

## Testing on New Data

- The model can evaluate individual images or entire folders.
- Supports predictions on unseen images stored in `evaluation_images/`.
- Reports accuracy on an independent test set.

---

## Results & Visualizations

- Loss and accuracy over epochs are visualized for both training and validation.
- Model performance is visualized with confusion matrices, classification reports, and sample predictions.

---

## Usage

1. **Prepare your dataset**: Organize images per class in `trafficsigns_dataset/`.
2. **Run training**:

```python
# Run the main training scripts provided
```

3. **Evaluate on new images**:

```python
# Select images from your folder and predict
```

4. **Optional**: Perform hyperparameter tuning for better performance.

---

## License

This project is for educational purposes. Feel free to adapt, extend, or contribute.

---

## Notes

- Adjust paths as needed.
- Use virtual environments for dependency management.
- Ensure dataset directories are correctly structured.
