"""A runner for pylint integration in Bazel."""

import sys

from pylint import lint


def main() -> None:
    """Run the pylint runner."""
    lint.Run(sys.argv[1:])


if __name__ == "__main__":
    main()
