import sys
import os
from pylint import lint

if __name__ == "__main__":
    lint.Run(sys.argv[1:])