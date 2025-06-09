"""
Main entry point for running an experiment.

Parses configuration options from the command line, such as which model
and dataset to use, then starts the experiment by calling the ExperimentRunner.
"""
import argparse
import json
from ExperimentRunner import ExperimentRunner

def load_dataset_config():
    """Load dataset paths from the JSON config file."""
    with open('./datasets_config.json', 'r') as f:
        return json.load(f)

def main():
    """Parse command-line arguments and run the experiment."""
    # Set up the argument parser to read command-line options.
    parser = argparse.ArgumentParser(description="Run experiments with configurable parameters")
    parser.add_argument('--model', type=str, required=True, help='Model name, e.g., "resnet", "vgg"')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of epochs to train')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to use')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate for training')
    parser.add_argument('--balanced', action='store_true', help='Use class weighting to balance the dataset')
    parser.add_argument('--horizontal_flip', action='store_true', help='Apply horizontal flip augmentation')
    parser.add_argument('--vertical_flip', action='store_true', help='Apply vertical flip augmentation')
    parser.add_argument('--rotation', type=int, default=0, help='Degrees of rotation for augmentation')
    parser.add_argument('--crop', nargs=3, type=float, help='Random resized crop params: [enabled, min_scale, max_scale]')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on (e.g., "cuda:0", "cpu")')
    parser.add_argument('--color_jitter', nargs=4, type=float, metavar=('BRIGHTNESS', 'CONTRAST', 'SATURATION', 'HUE'), help='Color jitter parameters for augmentation')
    args = parser.parse_args()

    # Load the configurations for all available datasets.
    dataset_configs = load_dataset_config()
    dataset_info = dataset_configs.get(args.dataset)

    if not dataset_info:
        print(f"Dataset {args.dataset} not found in datasets_config.json.")
    else:
        # If the dataset is found, create an ExperimentRunner and start it.
        experiment = ExperimentRunner(args, dataset_info)
        experiment.run()

# This makes the script runnable from the command line.
if __name__ == '__main__':
    main()