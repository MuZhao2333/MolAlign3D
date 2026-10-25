"""Evaluate reconstruction from either released ZINC checkpoint."""
import sys

from common.cli import main


if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
