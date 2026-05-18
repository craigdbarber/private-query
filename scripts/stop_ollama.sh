#!/bin/bash

# This script stops the locally running ollama service

set -e
pkill -9 ollama
