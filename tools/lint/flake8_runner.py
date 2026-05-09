"""A runner for flake8 integration in Bazel."""

import sys

from flake8.main.cli import main

if __name__ == "__main__":
    sys.exit(main())
