# Image Classifier with Deep Learning

A command-line application for training and using deep learning models to classify images. Built with PyTorch and supports popular architectures like VGG and ResNet for transfer learning.

## Features

- **Multiple Architecture Support**: Choose from VGG13, VGG19, or ResNet18
- **Transfer Learning**: Leverages pre-trained models for faster training and better accuracy
- **Customizable Hyperparameters**: Configure learning rate, hidden units, dropout, and epochs
- **GPU Support**: Optional GPU acceleration for faster training
- **Model Persistence**: Save and load trained models with checkpoints
- **Top-K Predictions**: Get the top K most likely classes for any image
- **Category Mapping**: Map class indices to human-readable category names

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- NumPy
- Pillow
- Matplotlib

Install dependencies:
```bash
pip install torch torchvision numpy pillow matplotlib
```

## Project Structure

```
aipnd_project/
├── train.py              # Training script
├── predict.py            # Prediction script
├── workspace_utils.py    # Helper functions and utilities
├── cat_to_name.json      # Category name mappings
└── flowers/              # Dataset directory (example)
    ├── train/
    ├── valid/
    └── test/
```

## Usage

### Training a Model

Train a new model on your dataset:

```bash
python train.py <data_directory> [options]
```

**Basic Example:**
```bash
python train.py flowers/ --save_dir checkpoints/model.pth --epochs 10 --gpu
```

**Training Options:**
- `data_dir` (required): Path to the dataset directory containing `train/`, `valid/`, and `test/` subdirectories
- `--save_dir`: Directory to save the trained model checkpoint (default: current directory)
- `--arch`: Model architecture - `vgg13`, `vgg19`, or `resnet18` (default: `vgg19`)
- `--learning_rate`: Learning rate for optimizer (default: `0.001`)
- `--hidden_units`: Number of hidden units in classifier (default: `512`)
- `--dropout`: Dropout rate for regularization (default: `0.3`)
- `--epochs`: Number of training epochs (default: `20`)
- `--gpu`: Enable GPU acceleration
- `--verbose`: Print detailed training information
- `--load_dir`: Load a pre-trained checkpoint to continue training

**Advanced Example:**
```bash
python train.py flowers/ \
  --save_dir models/vgg13_custom.pth \
  --arch vgg13 \
  --learning_rate 0.003 \
  --hidden_units 1024 \
  --dropout 0.5 \
  --epochs 25 \
  --gpu \
  --verbose
```

### Making Predictions

Use a trained model to classify images:

```bash
python predict.py <path_to_image> <checkpoint> [options]
```

**Basic Example:**
```bash
python predict.py flowers/test/1/image_06743.jpg checkpoints/model.pth --gpu
```

**Prediction Options:**
- `input` (required): Path to the image file to classify
- `checkpoint` (required): Path to the model checkpoint
- `--top_k`: Return top K most likely classes (default: `5`)
- `--category_names`: JSON file mapping categories to names (default: `cat_to_name.json`)
- `--gpu`: Use GPU for inference
- `--data_dir`: Data directory for validation (optional)

**Example with Custom Categories:**
```bash
python predict.py flowers/test/1/image_06743.jpg models/vgg13_custom.pth \
  --top_k 3 \
  --category_names custom_categories.json \
  --gpu
```

## Dataset Format

The dataset should be organized in the following structure:

```
data_directory/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class_2/
│       └── image3.jpg
├── valid/
│   └── ...
└── test/
    └── ...
```

Each subdirectory represents a class, and images within that directory belong to that class.

## Model Architecture

The classifier uses transfer learning with frozen feature extractors from pre-trained models:

**VGG Architectures (VGG13, VGG19):**
- Pre-trained feature extractor (frozen)
- Custom classifier:
  - Fully connected layer (25088 → hidden_units)
  - ReLU activation
  - Dropout
  - Output layer (hidden_units → num_classes)
  - LogSoftmax

**ResNet18:**
- Pre-trained feature extractor (frozen)
- Custom classifier with the same structure adapted for ResNet's feature dimensions

## Example Output

**Training:**
```
Setting up new model with architecture 'vgg19'...
Epoch 1/10.. Train loss: 3.892.. Test loss: 2.456.. Test accuracy: 0.423
Epoch 2/10.. Train loss: 2.234.. Test loss: 1.876.. Test accuracy: 0.564
...
Training complete.

==================================================
Evaluating on Test Set...
==================================================

Test Results:
  Test Loss: 0.687
  Test Accuracy: 0.821 (82.1%)
==================================================

Model saved to checkpoints/model.pth
```

**Prediction:**
```
Loading model from checkpoint checkpoints/model.pth...
Loading image 'flowers/test/1/image_06743.jpg'

Top 5 Predictions:
  1. pink primrose      : 95.23%
  2. hard-leaved pocket : 2.34%
  3. canterbury bells   : 1.12%
  4. sweet pea          : 0.67%
  5. english marigold   : 0.45%
```

## Technical Details

### Image Preprocessing

All images undergo the following preprocessing:
- **Training**: Random rotation (±30°), random resized crop (224×224), random horizontal flip
- **Validation/Testing**: Resize to 256px (shorter side), center crop to 224×224
- **Normalization**: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]

### Training Process

1. Loads pre-trained model with frozen feature extractor
2. Replaces classifier with custom architecture
3. Trains only the classifier weights using Adam optimizer
4. Validates on validation set after each epoch
5. Evaluates final model on test set
6. Saves checkpoint with model state, optimizer state, and class mappings

### Checkpoint Format

Saved checkpoints include:
- Model architecture name
- Classifier structure
- Model state dictionary
- Optimizer state
- Class-to-index mapping

## Tips for Best Results

1. **Start Small**: Use fewer epochs (5-10) for initial experiments
2. **GPU Recommended**: Training is significantly faster with `--gpu` flag
3. **Adjust Learning Rate**: If loss plateaus, try reducing learning rate (e.g., 0.0001)
4. **Monitor Overfitting**: If validation accuracy is much lower than training, increase dropout or reduce hidden units
5. **Experiment with Architectures**: Different datasets may work better with different architectures

## Troubleshooting

**CUDA Out of Memory:**
- Reduce batch size in `workspace_utils.py` (default: 64)
- Use a smaller architecture (e.g., ResNet18 instead of VGG19)

**Poor Accuracy:**
- Train for more epochs
- Increase hidden units
- Try different architectures
- Verify dataset quality and organization

**Slow Training:**
- Enable GPU with `--gpu` flag
- Reduce image size or batch size
- Use a smaller model architecture

## License

This project is part of the Udacity AI Programming with Python Nanodegree.

## Acknowledgments

- Pre-trained models from PyTorch's torchvision library
- Flower dataset example structure
