"""Run the original ZINC latent protocols with either released checkpoint."""
import sys

from common.cli import main


if __name__ == "__main__":
    main(["experiments", *sys.argv[1:]])
