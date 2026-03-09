"""
scripts/download_datasets.py
Download and cache common datasets used across PyTorch Mastery Hub notebooks.
Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --datasets mnist cifar10
    python scripts/download_datasets.py --list
"""

import os
import sys
import argparse

# Ensure src/ is importable when run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
import torchvision.transforms as transforms

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

AVAILABLE_DATASETS = {
    "mnist": "MNIST handwritten digits (60k train, 10k test, ~12 MB)",
    "fashion_mnist": "Fashion-MNIST clothing items (60k train, 10k test, ~30 MB)",
    "cifar10": "CIFAR-10 tiny images (50k train, 10k test, ~170 MB)",
    "cifar100": "CIFAR-100 fine-grained (50k train, 10k test, ~170 MB)",
}


def download_mnist(data_dir: str):
    print("  Downloading MNIST...")
    torchvision.datasets.MNIST(root=data_dir, train=True, download=True,
                               transform=transforms.ToTensor())
    torchvision.datasets.MNIST(root=data_dir, train=False, download=True,
                               transform=transforms.ToTensor())
    print("  ✓ MNIST downloaded")


def download_fashion_mnist(data_dir: str):
    print("  Downloading Fashion-MNIST...")
    torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True,
                                      transform=transforms.ToTensor())
    torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True,
                                      transform=transforms.ToTensor())
    print("  ✓ Fashion-MNIST downloaded")


def download_cifar10(data_dir: str):
    print("  Downloading CIFAR-10...")
    torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                 transform=transforms.ToTensor())
    torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                 transform=transforms.ToTensor())
    print("  ✓ CIFAR-10 downloaded")


def download_cifar100(data_dir: str):
    print("  Downloading CIFAR-100...")
    torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True,
                                  transform=transforms.ToTensor())
    torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True,
                                  transform=transforms.ToTensor())
    print("  ✓ CIFAR-100 downloaded")


DOWNLOAD_FUNCTIONS = {
    "mnist": download_mnist,
    "fashion_mnist": download_fashion_mnist,
    "cifar10": download_cifar10,
    "cifar100": download_cifar100,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download datasets for PyTorch Mastery Hub"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(AVAILABLE_DATASETS.keys()),
        default=["mnist", "fashion_mnist", "cifar10"],
        help="Datasets to download (default: mnist fashion_mnist cifar10)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help=f"Directory to store datasets (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("\nAvailable datasets:")
        for name, desc in AVAILABLE_DATASETS.items():
            print(f"  {name:<20} {desc}")
        return

    os.makedirs(args.data_dir, exist_ok=True)
    print(f"\nDownloading to: {args.data_dir}")
    print(f"Datasets: {', '.join(args.datasets)}\n")

    for dataset_name in args.datasets:
        try:
            DOWNLOAD_FUNCTIONS[dataset_name](args.data_dir)
        except Exception as e:
            print(f"  ✗ Failed to download {dataset_name}: {e}")

    print("\n✓ All requested datasets downloaded!")
    print(f"  Location: {args.data_dir}")


if __name__ == "__main__":
    main()
