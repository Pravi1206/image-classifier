# Imports here
from xml.parsers.expat import model
from collections import OrderedDict

import argparse
import json
import numpy as np
import os

from PIL import Image

from matplotlib import pyplot as plt

# torch imports
from torch import nn, optim
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models


MODEL_CONFIGURATION: dict = {
    'vgg13': {
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
        'crop_size': 224
    },
    'vgg19': {
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
        'crop_size': 224
    },
    'resnet18': {
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
        'crop_size': 224
    }
}


def _get_number_of_classes(json_classifier_file: str="cat_to_name.json") -> int:
    """ Gets the number of classes in the dataset.
        Args:
            json_classifier_file (str): Path to the JSON file mapping class indices to names.
        Returns:
            int: Number of classes.
    """
    if not os.path.exists(json_classifier_file):
        raise FileNotFoundError(f"The specified JSON file '{json_classifier_file}' does not exist.")

    with open(json_classifier_file, 'r') as f:
        class_to_name = json.load(f)

    return len(class_to_name)


def setup_dataloader( data_dir: str, architecture: str, train: bool ) -> torch.utils.data.DataLoader:
    """ Sets up the dataloader for the given data directory.
        Args:
            data_dir (str): Directory containing the data.
        Returns:
            torch.utils.data.DataLoader: DataLoader for the dataset.
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The specified data directory '{data_dir}' does not exist.")

    # Define your transforms for the training, validation, and testing sets
    if train:
        data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(MODEL_CONFIGURATION[architecture]['crop_size']),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(MODEL_CONFIGURATION[architecture]['normalize_mean'],
                                                             MODEL_CONFIGURATION[architecture]['normalize_std'])])
    else:
        data_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(MODEL_CONFIGURATION[architecture]['crop_size']),
                                        transforms.ToTensor(),
                                        transforms.Normalize(MODEL_CONFIGURATION[architecture]['normalize_mean'],
                                                             MODEL_CONFIGURATION[architecture]['normalize_std'])])
    # Using the image datasets and the trainforms, define the dataloaders
    data_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)
    # Shuffle training data for better generalization, keep validation/test data ordered
    dataloader = torch.utils.data.DataLoader(data_datasets, batch_size=64, shuffle=train)

    return data_datasets, dataloader   


def _setup_model_architecture(architecture: str, hidden_units: int, dropout: float, verbose: bool) -> nn.Module:
    """Sets up a transfer learning model for VGG or ResNet.

    Args:
        architecture (str): Model architecture ('vgg13', 'resnet18', etc.)
        hidden_units (int): Number of units in the first hidden layer.
        dropout (float): Dropout rate for regularization.
    Returns:
        nn.Module: Configured model ready for training.
    """

    # Load pretrained model
    model = models.__dict__[architecture](pretrained=True)

    if verbose:
        print("[DEBUG] Pretrained model loaded:", model.__class__.__name__)

    # Freeze feature extractor
    for param in model.parameters():
        param.requires_grad = False

    # Get output feature size of the base model
    if "vgg" in architecture:
        in_features = model.classifier[0].in_features
    if "resnet" in architecture:
        in_features = model.fc.in_features

    if verbose:
        print(f"[DEBUG] Base model '{architecture}' has {in_features} input features to the classifier.")
        print(f"[DEBUG] Classifier will have {hidden_units} hidden units and {dropout} dropout.")

    # Sanity check: hidden_units should not exceed input size
    if hidden_units > in_features:
        raise ValueError(
            f"hidden_units ({hidden_units}) should not be larger than "
            f"in_features ({in_features}) for {architecture}."
        )

    # Get the target number of output classes
    class_count: int = _get_number_of_classes()

    # --- Build classifier dynamically ---
    classifier_layers = OrderedDict(
            [
                ("fc1", nn.Linear(in_features, hidden_units)),
                ("relu1", nn.ReLU()),
                ("drop1", nn.Dropout(dropout)),
                ("fc2", nn.Linear(hidden_units, class_count)),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    classifier = nn.Sequential(classifier_layers)

    # Replace classifier/fc layer depending on architecture
    if "vgg" in architecture:
        model.classifier = classifier
    elif "resnet" in architecture:
        model.fc = classifier

    return model


def get_model(args: argparse.Namespace, device: torch.device, dropout: float, verbose: bool):
    """ Gets the model based on the command-line arguments.
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
            device (torch.device): Device to load the model onto.
            dropout (float): Dropout rate for the model.
            verbose (bool): Whether to print detailed information.
        Returns:
            nn.Module: The initialized or loaded model.
    """

    optimizer = None

    if args.load_dir:
        print(f"Loading model from checkpoint {args.load_dir}...")
        model, optimizer = _load(
            filepath=args.load_dir,
            device=device,
            verbose=verbose
        )
    else:
        print(f"Setting up new model with architecture '{args.arch}'...")
        model: torch.nn.Module = _setup_model_architecture(
            architecture=args.arch, hidden_units=args.hidden_units, dropout=dropout, verbose=verbose
        )

    return model.to(device), optimizer


def _load(filepath: str, device: torch.device, verbose: bool) -> torch.nn.Module:
    """
    Loads a model checkpoint and rebuilds the model.

    Args:
        filepath (str): Path to the checkpoint file (e.g., 'checkpoint.pth').
        device (str, optional): 'cpu' or 'cuda'. Automatically detects if None.

    Returns:
        torch.nn.Module: The loaded and ready-to-use model.
    """

    # Load the checkpoint
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    if verbose:
        print(f"Checkpoint keys: {checkpoint.keys()}")

    # Rebuild the pretrained model architecture
    model_name = checkpoint.get('architecture')

    if model_name not in models.__dict__:
        raise ValueError(f"Unknown architecture '{model_name}'. Available models: {list(models.__dict__.keys())}")

    model = models.__dict__[model_name](pretrained=True)

    print("Our model: \n\n", model, '\n')
    if verbose:
        print("The state dict keys: \n\n", model.state_dict().keys())

    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False

    # Attach saved classifier
    model.classifier = checkpoint['classifier']

    # Load model state
    model.load_state_dict(checkpoint['state_dict'])

    # Restore class-to-index mapping
    model.class_to_idx = checkpoint.get('class_to_idx', None)

    if "optimizer_state" in checkpoint:
        optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint.get("learning_rate", 0.001))
        optimizer.load_state_dict(checkpoint["optimizer_state"])
 
    # Move to the appropriate device
    model.to(device)

    print("Model loaded successfully and moved to", device)

    return model, optimizer


def save(filename: str, model: nn.Module, architecture: str, optimizer: optim.Optimizer, train_datasets: datasets.ImageFolder):
    """ Saves the model checkpoint.
        Args:
            filename (str): The path to save the checkpoint.
            model (nn.Module): The trained model.
            architecture (str): The model architecture.
            optimizer (optim.Optimizer): The optimizer used during training.
            train_datasets (datasets.ImageFolder): The training dataset.
    """

    checkpoint = {
        'architecture': architecture,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'class_to_idx': train_datasets.class_to_idx
    }

    print(f"Saving checkpoint to {filename}...")
    torch.save(checkpoint, filename)


def load_and_rebuild_model(checkpoint_path: str, device: torch.device, dataloader: torch.utils.data.DataLoader, rebuild: bool, verbose: bool) -> nn.Module:
    """ Loads a model checkpoint and rebuilds the model.
        Args:
            checkpoint_path (str): Path to the checkpoint file.
            device (torch.device): Device to load the model onto.
            dataloader (torch.utils.data.DataLoader): Validation dataloader for sanity check.
            verbose (bool): Whether to print detailed information.
        Returns:
            nn.Module: The loaded and ready-to-use model.
    """

    # Load the checkpoint
    model, _ = _load(
        filepath=checkpoint_path,
        device=device,
        verbose=verbose
    )

    # Optional sanity check: forward pass on one batch
    if rebuild and dataloader is not None:
        model.eval()
        with torch.no_grad():
            inputs, labels = next(iter(dataloader))
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model(inputs)

            if verbose:
                print("Output shape:", logps.shape)

    return model


def image_show(image, architecture: str, ax=None, title=None):

    """ Displays a processed image.
        Args:
            image (torch.Tensor): The processed image tensor.
            architecture (str): Model architecture to determine normalization parameters.
            ax (matplotlib.axes.Axes, optional): Axes object to plot on. Creates new one if None.
            title (str, optional): Title for the plot.
        Returns:
            matplotlib.axes.Axes: The Axes object with the image.
    """
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array(MODEL_CONFIGURATION[architecture]['normalize_mean'])
    std = np.array(MODEL_CONFIGURATION[architecture]['normalize_std'])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def process_image(path_to_image: str, architecture: str) -> np.ndarray:
    '''Scales, crops, and normalizes a PIL image for a PyTorch model.
        Args:
            path_to_image (str): Path to the image file.
            architecture (str): Model architecture to determine normalization parameters.
        Returns:
            np.ndarray: Processed image as a NumPy array.
    '''

    with Image.open(path_to_image) as im:

        # Maintain aspect ratio: shortest side = 256 px
        width, height = im.size
        if width < height:
            new_width = 256
            new_height = int(256 * height / width)
        else:
            new_height = 256
            new_width = int(256 * width / height)
        im = im.resize((new_width, new_height))

        # Center crop to 224x224
        left   = (new_width - MODEL_CONFIGURATION[architecture]['crop_size']) / 2
        top    = (new_height - MODEL_CONFIGURATION[architecture]['crop_size']) / 2
        right  = left + MODEL_CONFIGURATION[architecture]['crop_size']
        bottom = top + MODEL_CONFIGURATION[architecture]['crop_size']
        im = im.crop((left, top, right, bottom))

        # Convert to numpy array and scale to [0, 1]
        np_image = np.array(im) / 255.0

        # Normalize each color channel
        mean = np.array(MODEL_CONFIGURATION[architecture]['normalize_mean'])
        std = np.array(MODEL_CONFIGURATION[architecture]['normalize_std'])
        np_image = (np_image - mean) / std

        # Reorder dimensions to match PyTorch: [C, H, W]
        np_image = np_image.transpose((2, 0, 1))
        
    return np_image