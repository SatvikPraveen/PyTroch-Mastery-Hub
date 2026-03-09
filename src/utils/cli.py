"""
Command-line interface for PyTorch Mastery Hub.
Provides quick access to common operations.
Usage:
    pytorch-hub --help
    pytorch-hub info
    pytorch-hub test [--module MODULE]
    pytorch-hub download-data [--datasets mnist cifar10]
"""

import argparse
import sys


def cmd_info(_args):
    """Print environment and package information."""
    try:
        import torch
        import torchvision
        import numpy
        import matplotlib

        print("PyTorch Mastery Hub — Environment Info")
        print("─" * 40)
        print(f"  PyTorch     : {torch.__version__}")
        print(f"  TorchVision : {torchvision.__version__}")
        print(f"  NumPy       : {numpy.__version__}")
        print(f"  Matplotlib  : {matplotlib.__version__}")
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        print(f"  Best device : {device}")
        if torch.cuda.is_available():
            print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_test(args):
    """Run the test suite."""
    import subprocess
    cmd = ["python", "-m", "pytest", "tests/", "-v"]
    if args.module:
        cmd += [f"tests/test_{args.module}/"]
    sys.exit(subprocess.call(cmd))


def cmd_download_data(args):
    """Download datasets."""
    import subprocess
    cmd = ["python", "scripts/download_datasets.py"]
    if args.datasets:
        cmd += ["--datasets"] + args.datasets
    sys.exit(subprocess.call(cmd))


def main():
    parser = argparse.ArgumentParser(
        prog="pytorch-hub",
        description="PyTorch Mastery Hub command-line interface",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    subparsers.add_parser("info", help="Show environment information")

    # test
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--module", type=str, default=None,
                             help="Module to test (e.g., 'fundamentals', 'nlp')")

    # download-data
    dl_parser = subparsers.add_parser("download-data", help="Download datasets")
    dl_parser.add_argument(
        "--datasets", nargs="+",
        choices=["mnist", "fashion_mnist", "cifar10", "cifar100"],
        default=["mnist", "cifar10"],
    )

    args = parser.parse_args()

    dispatch = {
        "info": cmd_info,
        "test": cmd_test,
        "download-data": cmd_download_data,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
