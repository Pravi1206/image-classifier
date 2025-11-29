import argparse
import json
import os 

from torch import nn
import torch 

from matplotlib import pyplot as plt
from PIL import Image

# Get script directory for default data path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, 'flowers')

# Import helper functions from workspace_utils in the same directory
from workspace_utils import load_and_rebuild_model, process_image, image_show, setup_dataloader


def predict(image_path: str, model: nn.Module, architecture: str, top_k: int, device: torch.device) -> tuple:
    """ Predict the class (or classes) of an image using a trained deep learning model.
        Args:
            image_path (str): Path to the image file.
            model (nn.Module): The trained model.
            top_k (int): Number of top most likely classes to return.
            device (torch.device): Device to perform computation on.
        Returns:
            tuple: (probabilities, classes) where probabilities is a list of the top K probabilities and classes is a list of the corresponding class labels.
    """

    print(f"Loading image '{image_path}'")
    
    # load and process image
    image = process_image(path_to_image=image_path, architecture=architecture)
 
    # convert to tensor
    image = torch.from_numpy(image).float()  # ensure float
 
    # add batch dimension and move to device
    image = image.unsqueeze(0).to(device)

    # Disable gradient computation
    with torch.no_grad():
        model.eval()
        logits = model.forward(image)
        ps = torch.exp(logits)

    # Get top-K results
    top_p, top_class_idx = ps.topk(top_k, dim=1)

    # Convert to CPU numpy arrays
    top_p = top_p.cpu().numpy()[0]
    top_class_idx = top_class_idx.cpu().numpy()[0]

    # Invert the class_to_idx dictionary
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_class_idx]
    
    return top_p, top_classes


def parse_args():
    """
    Parse command-line arguments for training a model.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a deep learning model on a dataset.")

    # Required positional argument
    parser.add_argument('image', type=str, help='Path to the image file.')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint file.')

    # Optional arguments
    parser.add_argument('--arch', choices=['vgg13', 'vgg19', 'resnet18'], default="vgg19", help='Model architecture (e.g., vgg19, resnet50).')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help='Directory containing the training data.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to the category names JSON file.')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for training.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print detailed training information.')

    return parser.parse_args()


def main(args):
    """
    Main prediction logic.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    
    # load the category names
    with open(args.category_names, 'r') as f:
        flower_to_name = json.load(f)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    
    _, validloader = setup_dataloader(data_dir=os.path.join(args.data_dir, 'valid'), architecture=args.arch, train=False)

    # setup the model
    model = load_and_rebuild_model(
        checkpoint_path=args.checkpoint,
        device=device,
        dataloader=validloader,
        rebuild=True,
        verbose=args.verbose
    )

    # get predictions
    print("Predicting...")
    probs, classes = predict(
        image_path=args.image,
        model=model,
        architecture=args.arch,
        top_k=args.top_k,
        device=device
    )

    # print results
    class_names = [flower_to_name[c] for c in classes]
    
    # Load the image for display
    with Image.open(args.image) as pil_image:
        # Process image for PyTorch (for tensor display)
        image_tensor = torch.from_numpy(process_image(path_to_image=args.image, architecture=args.arch)).float()
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), nrows=2)
    image_show(image_tensor, architecture=args.arch, ax=ax1)
    ax1.set_title(class_names[0])  # predicted class at top
    
    # Bar chart of top probabilities
    ax2.barh(range(len(probs)), probs)
    ax2.set_yticks(range(len(probs)))
    ax2.set_yticklabels(class_names)
    ax2.invert_yaxis()  # highest probability on top
    ax2.set_xlabel("Probability")

    # Save instead of show
    plt.tight_layout()

    image_filename = os.path.basename(args.image)
    output_file    = os.path.join(SCRIPT_DIR, f"{image_filename}_prediction.png")

    if os.path.exists(output_file):
        print(f"Output file '{output_file}' already exists and will be overwritten.")
        os.remove(output_file)

    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Prediction plot saved to: {output_file}")

    # Try to show if interactive backend is available
    try:
        plt.show()
    except:
        pass 


if __name__ == "__main__":
    main(args=parse_args())
