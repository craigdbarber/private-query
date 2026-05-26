# private-query

## Table of Contents
* [Project Overview](#project-overview)
* [Dependencies](#dependencies)
* [Setup](#setup)
* [Usage](#usage)
* [Configuration](#configuration)
* [Testing and Linting](#testing-and-linting)

## Project Overview

`private-query` is an AI solution offering [Retrieval Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) [Large Language Model (LLM)](https://en.wikipedia.org/wiki/Large_language_model) querying functionality isolated in a private environment. It caters to users and organizations who cannot expose their data to public/commercially offered LLM solutions due to: privacy, policy, trade secrets, copyright, etc. concerns. It provides a set of commands for embedding a local collection of documents into a privately hosted ChromaDB database, and then using that embedded data as context for RAG queries against a privately hosted open source LLM running within Ollama. Configuration provides options for targeting both ChromaDB and Ollama instances running on the user's local machine, or instances running within a private network. `private-query` has been programmed to only provide answers exclusively sourced from information embedded within ChromaDB in order to prevent hallucinations. Additionally, each answer is accompanied by a list of references showing which documents information was sourced from.

## Dependencies
- [Ollama](https://ollama.com): Used for hosting a privately running open source LLM.
- [ChromaDB](https://www.trychroma.com/products/chromadb): Used for privately storing and querying data.
- [Bazel](https://bazel.build/docs): Used for building and testing.
- [uv](https://docs.astral.sh/uv/): Used for package and Python environment management.
- [Python](https://www.python.org): The software development language used.

## Setup

This section provides instructions for downloading, installing, and configuring `private-query`. Use of [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) is highly encouraged for Windows users.

* Checkout the repo:
```bash
mkdir private-query
cd private-query
git clone https://github.com/craigdbarber/private-query.git .
```
* Initialize the local `uv` environment:
```bash
uv venv && uv sync && source .venv/bin/activate
```

* To install `private-query` into `~/.local/bin/` using `uv`:
```bash
uv tool install .
```

* To build a stand-alone [PEX](https://github.com/benley/bazel_rules_pex) binary using `bazel`:
```bash
bazel build //:private_query
cp bazel-bin/private-query .
```

* Setup the configuration file, see [Configuration](#configuration):
```bash
cp config.yaml.example config.yaml
```

* Running `private-query` with `uv`:
```bash
uv run private-query [args]
```

* Running `private-query` with `bazel`:
```bash
bazel run //:private_query_cli -- [args]
```

* Ensure Ollama is running either locally or on a hosted environment within your private network, see:  [Ollama Quickstart](https://docs.ollama.com/quickstart).


## Usage

This section provides usage information for the `private-query` tool.

* Display help and usage information:
```bash
private-query help
```

* Loading documents into the ChromaDB:
```bash
# loading an entire directory
private-query load [path/dir]

# loading an invidivual file
private-query load [path/file]
```

* Executing a RAG query against the LLM hosted in Ollama:
```bash
private-query query "your query"
```


## Configuration

This section contains information for modifying the project's configuration file to suit the user's needs, see: [Example Configuration](config.yaml.example).

* `collection_name`: This is the unique name for the ChromaDB collection documents will be embedded into and queried against. Providing this option allows for users to configure isolated collections of embedded data.
* `chroma`: Configuration settings for ChromaDB.
    * `embedding_model`: The open source embedding model used to transform and embed user documents. Pulled from [Hugging Face](https://huggingface.co).
    * `embedding_model_revision`: The revision of the embedding model to be utilized.
    * `model_cache_directory`: The directory utilized to cache downloaded models.
    * `database`: The configuration section for the database to be utilized.
        * `type`: Either `local` or `remote` 
        * `persist_directory`: For `local` type, the local directory the database will use to save data.
        * `host`: For `remote` type, the host URL of the ChromaDB server to be utilized.
        * `database_name`: For `remote` type, the name of the database to be utilized.
        * `auth_token`: For `remote` type (Optional), the token to be used for authenticating with the ChromaDB server.
* `ollama`: Configuration settings for Ollama.
    * `host`: The host URL of the Ollama service to be utilized, `"localhost:11434"` for local.
    * `model`: The open source LLM to be used for querying. Pulled from [Hugging Face](https://huggingface.co)
    * `api_key`: (Optional) used for an authorization against Ollama.

## Testing and Linting

This section describes how to run tests and linting for this project.

* Executing Tests:
```bash
bazel test //...
```

* Executing linters:
```bash
bazel build --config=lint //...
```