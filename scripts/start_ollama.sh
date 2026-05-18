#!/bin/bash
# This script runs the ollama service

set -e

print_usage() {
    echo "Usage: $0 [-h|--host HOST] [-o|-ollama_home_dir] [-u|--usage]"
    echo ""
    echo "Options:"
    echo "  -h, --host HOST             Specify the host URL for ollama to serve from."
    echo "  -o, --ollama_home_dir DIR   Specify the ollama home directory."
    echo "  -u, --usage                 Display this help message."
}

## parse command line options
if [ $# = 0 ]; then
    echo "Failed to parse options!!" >&2
    print_usage
    exit 1
fi

HOST_URL=""
OLLAMA_HOME_DIR=""
USAGE=false

while [[ $# -gt 0 ]]; do
    case "$1" in 
        -h | --host)
            HOST_URL="$2"
            shift 2 ;;
        -o | --ollama_home_dir)
            OLLAMA_HOME_DIR="$2"
            shift 2 ;;
        -u | --usage)
            USAGE=true
            shift ;;
        *)
            echo "Uknown option: $1" >&2
            print_usage
            exit 1 ;;
    esac
done

if [ "$USAGE" = true ]; then
    print_usage
    exit 0
fi

if [ "$HOST_URL" = "" ]; then
    echo "Failed to parse host." >&2
    print_usage
    exit 1
fi

if [ "$OLLAMA_HOME_DIR" = "" ]; then
    echo "Failed to parse ollama home dir." >&2
    print_usage
    exit 1
fi

export OLLAMA_HOST="$HOST_URL"
export OLLAMA_KEEP_ALIVE="-1"
export HOME="$OLLAMA_HOME_DIR"

ollama serve > /dev/null &

# To test the service is running properly run this command
# curl http://<HOST_URL>/api/chat -d '{
#  "model": "llama3.2:latest",
#  "messages": [{ "role": "user", "content": "Hello!" }]
#}'
