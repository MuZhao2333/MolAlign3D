"""Train either released ZINC model with its own configuration."""
import sys

from common.cli import main


if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
