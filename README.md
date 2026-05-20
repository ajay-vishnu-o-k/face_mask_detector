# Face Mask Detection

A deep learning project to detect whether a person is wearing a face mask or not using computer vision and transfer learning.

## Overview

This project implements a binary classification model to identify if individuals are wearing face masks. It uses **MobileNetV2**, a lightweight pre-trained convolutional neural network, fine-tuned for this specific task.

## Features

- **Transfer Learning**: Leverages MobileNetV2 pre-trained on ImageNet
- **Two-Stage Training**:
  - Stage 1: Train classification head only (5 epochs)
  - Stage 2: Fine-tune last 30 base layers (10 epochs)
- **Data Augmentation**: Includes rotation, zoom, brightness adjustment, and flip augmentations
- **Class Balancing**: Uses computed class weights to handle imbalanced datasets
- **Early Stopping**: Prevents overfitting with early stopping callback
- **Learning Rate Scheduling**: Reduces learning rate on plateau for better convergence

## Dataset

The project uses the **Face Mask Dataset** from Kaggle:
- Source: [omkargurav/face-mask-dataset](https://www.kaggle.com/omkargurav/face-mask-dataset)
- Downloaded automatically using `kagglehub`
- Classes: `with_mask` and `without_mask`

## Project Structure

```
├── facemaske (2).ipynb    # Main notebook with the complete pipeline
├── README.md              # This file
└── data/                  # Dataset directory (downloaded from Kaggle)
    ├── with_mask/         # Images of people with masks
    └── without_mask/      # Images of people without masks
```

## Requirements

```
tensorflow>=2.0
keras
scikit-learn
numpy
matplotlib
seaborn
kagglehub
```

## Installation & Setup

1. **Install dependencies**:
   ```bash
   pip install tensorflow scikit-learn numpy matplotlib seaborn kagglehub
   ```

2. **Configure Kaggle API** (if needed):
   - Place your Kaggle API key at `~/.kaggle/kaggle.json`
   - Download link: https://www.kaggle.com/account

3. **Run the notebook**:
   - Open `facemaske (2).ipynb` in Jupyter or JupyterLab
   - Execute cells sequentially to train and evaluate the model

## Configuration

Key hyperparameters (defined in Cell 2):

```python
IMG_SIZE   = (224, 224)      # Input image size
BS         = 32               # Batch size
SEED       = 42               # Random seed for reproducibility

# Stage 1: Train head only
S1_EPOCHS  = 5
S1_LR      = 1e-3

# Stage 2: Fine-tune
S2_EPOCHS  = 10
S2_LR      = 1e-5
UNFREEZE_LAYERS = 30
```

## Training Pipeline

### Stage 1: Head Training
- Freeze all MobileNetV2 layers
- Train only the classification head
- Uses higher learning rate (1e-3) for quick initial training

### Stage 2: Fine-tuning
- Unfreeze last 30 layers of MobileNetV2
- Fine-tune with low learning rate (1e-5)
- Apply early stopping and learning rate reduction

### Data Augmentation

```python
- Rotation: ±20°
- Zoom: 20%
- Width/Height shift: 20%
- Shear: 15%
- Horizontal flip: Yes
- Brightness: 0.6-1.2x
- Validation split: 20%
```

## Evaluation

The model is evaluated using:
- **Classification Report**: Precision, recall, F1-score
- **Confusion Matrix**: Visualization of predictions vs. actual
- **Loss & Accuracy Curves**: Training and validation metrics over epochs

## Model Architecture

```
Input (224, 224, 3)
    ↓
MobileNetV2 (pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
Dense (128) + BatchNormalization + ReLU
    ↓
Dropout (0.5)
    ↓
Dense (2, softmax)  [with_mask, without_mask]
```

## Callbacks Used

- **EarlyStopping**: Monitors validation loss, patience of 5 epochs
- **ReduceLROnPlateau**: Reduces learning rate when loss plateaus
- **ModelCheckpoint**: Saves best model weights during training

## Usage

1. **Train the model**:
   - Run all cells in the notebook sequentially
   - Model weights are automatically saved during training

2. **Make predictions** (after training):
   ```python
   # Single image prediction
   img = keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(224, 224))
   img_array = keras.preprocessing.image.img_to_array(img)
   img_array = np.expand_dims(img_array, axis=0)
   prediction = model.predict(img_array)
   ```

## Performance Metrics

The model produces:
- **Accuracy**: Classification accuracy on validation set
- **Precision & Recall**: Per-class performance metrics
- **Confusion Matrix**: Visual representation of model predictions

## Future Improvements

- [ ] Add real-time face mask detection from webcam
- [ ] Deploy model using TensorFlow Lite for mobile devices
- [ ] Implement model quantization for faster inference
- [ ] Add SHAP explanations for model interpretability
- [ ] Extend to multi-class classification (proper mask, improper mask, no mask)
- [ ] Deploy as a REST API using Flask/FastAPI

## Author

Ajay Vishnu

## License

This project is for educational purposes.

## References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Kaggle Face Mask Dataset](https://www.kaggle.com/omkargurav/face-mask-dataset)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

---

**Note**: Ensure you have sufficient disk space (~200MB) for the dataset download and GPU/TPU available for faster training.
