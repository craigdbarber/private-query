"""A runner for bandit security linting in Bazel."""

import sys

from bandit.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
