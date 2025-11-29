import argparse
import os
import sys

# torch imports
from torch import nn, optim
import torch

# Add the script directory to Python path to find workspace_utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import helper functions from workspace_utils in the same directory
from workspace_utils import get_model, setup_dataloader, save


def _get_hyperparameters(args) -> dict:
    """ Extracts hyperparameters from command-line arguments.
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        Returns:
            dict: Hyperparameters for training.
    """

    return {
        'learning_rate': args.learning_rate,
        'hidden_units': args.hidden_units,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'arch': args.arch
    }


def parse_args():
    """
    Parse command-line arguments for training a model.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a deep learning model on a dataset.")

    # Required positional argument
    parser.add_argument('data_dir', type=str, help='Directory containing the training data.')

    # Optional arguments
    parser.add_argument('--load_dir', type=str, default=None, help='Directory to load a pre-trained model checkpoint.')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the trained model.')
    parser.add_argument('--arch', choices=['vgg13', 'vgg19', 'resnet18'], default='vgg19', help='Model architecture (e.g., "vgg13", "resnet18").')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier.')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for the classifier.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training.')
    parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for training.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print detailed training information.')

    return parser.parse_args()


def test(model: nn.Module, data_dir: str, architecture: str, device: torch.device) -> float:
    """
    Evaluate the trained model on the test dataset.
    
    Args:
        model (nn.Module): Trained model to evaluate
        data_dir (str): Root data directory
        architecture (str): Model architecture name
        device (torch.device): Device to run evaluation on
    
    Returns:
        float: Test accuracy
    """
    # Load test dataset
    _, testloader = setup_dataloader(
        data_dir=os.path.join(data_dir, 'test'), 
        architecture=architecture, 
        train=False
    )
    
    criterion = nn.NLLLoss()
    test_loss = 0.0
    accuracy = 0.0
    
    # Set model to evaluation mode
    model.eval()
    
    print("\n" + "="*50)
    print("Evaluating on Test Set...")
    print("="*50)
    
    # Disable gradient computation
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += equals.float().mean().item()
    
    # Calculate final metrics
    avg_test_loss = test_loss / len(testloader)
    avg_accuracy = accuracy / len(testloader)
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {avg_test_loss:.3f}")
    print(f"  Test Accuracy: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
    print("="*50 + "\n")
    
    return avg_accuracy


def train(model: nn.Module, hyperparameters: dict, device: torch.device, optimizer: optim.Optimizer, data_dir: str, save_dir: str, verbose: bool) -> nn.Module:
    """ Trains the model.
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        Returns:
            nn.Module: The trained model.
    """
    
    # check if the save dir exists and throw an error
    if os.path.exists(save_dir):
        raise FileExistsError(f"Save directory '{save_dir}' already exists. Please choose a different directory to save the model.")
    
    # set to cuda if gpu is available and gpu argument is set to true
    train_dataset, trainloader = setup_dataloader(data_dir=os.path.join(data_dir, 'train'), architecture=hyperparameters.get("arch"), train=True)
    valid_dataset, validloader = setup_dataloader(data_dir=os.path.join(data_dir, 'valid'), architecture=hyperparameters.get("arch"), train=False)

    # set parameters
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    if optimizer is None:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=hyperparameters.get("learning_rate"))
 
    model.to(device)

    steps = 0
    running_loss = 0.0
    print_every = len(trainloader)

    # hyperparameters
    for epoch in range(hyperparameters.get("epochs")):

        model.train()
        running_loss = 0.0

        if verbose:
            print("Start epoch", epoch+1, " from ", hyperparameters.get("epochs"))
        
        for inputs, labels in trainloader:
            steps += 1
            
            # Print progress every 50 steps
            if verbose and steps % 50 == 0:
                print(f"Step {steps}...")

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # get log probabilities and loss
            logps = model.forward(inputs)
            loss  = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # drop out the traning loop and get in the validation loop
        test_loss, accuracy = 0.0, 0.0

        # turns out dropout and makes it ready for validation
        model.eval()

        # use torch.no_grad() during validation — avoids wasting memory on gradients.
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()

                # Calculate accuracy -> get the real probabilities from softmax
                ps = torch.exp(logps)

                # gets the top answer along the columns 
                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                # Convert to float on GPU, then get scalar to avoid device mismatch
                accuracy += equals.float().mean().item()


        # Track the loss and accuracy on the validation set to determine the best hyperparameters
        print(f"Epoch {epoch+1}/{hyperparameters.get('epochs')}.. "
            f"Train loss: {running_loss/print_every:.3f}.. "
            f"Test loss: {test_loss/len(validloader):.3f}.. "
            f"Test accuracy: {accuracy/len(validloader):.3f}")


    print("Training complete.")

    #  UPDATE: Evaluate on test set after training
    test_accuracy = test(
        model=model,
        data_dir=data_dir,
        architecture=hyperparameters.get("arch"),
        device=device
    )

    # Save the model checkpoint
    save(
        filename=save_dir, 
        model=model, 
        architecture=hyperparameters.get("arch"), 
        optimizer=optimizer, 
        train_datasets=train_dataset
    )

    print("Model saved to ", save_dir)
    return model


def main(args):
    """
    Main training logic.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    
    hyperparameters: dict = _get_hyperparameters(args)

    try:
        # setup the model
        model, optimizer = get_model(
            args=args, 
            device=device,
            dropout=hyperparameters.get('dropout'),
            verbose=args.verbose
        )

        # train the model
        train(
            model=model, 
            hyperparameters=_get_hyperparameters(args), 
            device=device,
            optimizer=optimizer,
            data_dir=args.data_dir, 
            save_dir=args.save_dir,
            verbose=args.verbose
        )

        print("Training complete.")

        return 0

    except Exception as err:
        print("ERROR:", err)
        return 1
    
if __name__ == "__main__":
    sys.exit(main(args=parse_args()))
