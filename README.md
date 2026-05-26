# private-query

## Table of Contents
* [Project Overview](#project-overview)
* [Dependencies](#dependencies)
* [Setup](#setup)
* [Usage](#usage)
* [Configuration](#configuration)
* [Testing and Linting](#testing-and-linting)
* [Design & Architecture](design_architecture.md)

## Project Overview

`private-query` is a secure, locally-hosted Retrieval-Augmented Generation (RAG) tool designed for querying private datasets. It is ideal for users and organizations with strict data privacy, policy, or intellectual property concerns who cannot use public LLM services.

**Key Features:**
* **100% Private:** Operates entirely within your local environment or private network, keeping your data secure.
* **Local Ingestion & Search:** Embeds local documents into a self-hosted ChromaDB database for fast context retrieval.
* **Open-Source LLMs:** Uses local models running on Ollama to execute queries against your embedded data.
* **Hallucination Prevention:** Strictly scoped to provide answers sourced exclusively from your provided documents.
* **Source Citations:** Every answer includes clear references to the specific documents used to generate it.

## Dependencies
- [Ollama](https://ollama.com): Used for hosting a privately running open source LLM.
- [ChromaDB](https://www.trychroma.com/products/chromadb): Used for privately storing and querying data.
- [Bazel](https://bazel.build/docs): Used for building and testing.
- [uv](https://docs.astral.sh/uv/): Used for package and Python environment management.
- [Python](https://www.python.org): Version `>= 3.12` is required (for `torch` compatibility).

## Setup

This section provides instructions for downloading, installing, and configuring `private-query`. Use of [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) is highly encouraged for Windows users.

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/craigdbarber/private-query.git
cd private-query
```

2. **Initialize the local `uv` environment:**
```bash
uv venv && uv sync && source .venv/bin/activate
```

3. **Configure the application, ([Configuration](#configuration)):**
```bash
cp config.yaml.example config.yaml
```

### Execution Options

* **As a CLI Tool (via uv):**
```bash
uv tool install .
private-query [args]
```

* **Via Bazel (PEX binary):**
```bash
bazel build //:private_query
cp bazel-bin/private-query .
./private-query [args]
```

* **Development (direct uv):**
```bash
uv run private-query [args]
```

* **Development (direct bazel):**
```bash
bazel run //:private_query_cli -- [args]
```

* **Ollama Service:** Ensure Ollama is running either locally or on a hosted environment within your private network. See the [Ollama Quickstart](https://docs.ollama.com/quickstart) for more details.


## Usage

This section provides usage information for the `private-query` tool.

* Display help and usage information:
```bash
private-query help
```

* Loading documents into ChromaDB:
```bash
# loading an entire directory
private-query load [path/dir]

# loading an individual file
private-query load [path/file]
```

* Executing a RAG query against the LLM hosted in Ollama:
```bash
private-query query "your query"
```


## Configuration

This section contains information for modifying the project's configuration file to suit your needs. See [config.yaml.example](config.yaml.example) for a complete template.

* `collection_name`: The unique name for the ChromaDB collection. This allows for isolated datasets.
* `chroma`: Configuration for the ChromaDB vector database.
    * `embedding_model`: The open-source embedding model used (via [Hugging Face](https://huggingface.co)).
    * `embedding_model_revision`: The specific revision of the model to be utilized.
    * `model_cache_directory`: The local directory used to cache downloaded models.
    * `database`: The database backend configuration.
        * `type`: Either `local` or `remote`.
        * `persist_directory`: (For `local` type) The directory where the database saves data.
        * `host`: (For `remote` type) The host URL of the ChromaDB server.
        * `database_name`: (For `remote` type) The name of the database.
        * `auth_token`: (For `remote` type, Optional) The authentication token for the server.
* `ollama`: Configuration for the Ollama LLM service.
    * `host`: The host URL of the Ollama service (e.g., `localhost:11434`).
    * `model`: The open-source LLM to be used for querying (via [Hugging Face](https://huggingface.co)).
    * `api_key`: (Optional) The API key for authorization against the Ollama service.

## Testing and Linting

This section describes how to run tests and linting for this project.

* **Unit Tests**:
```bash
bazel test //...
```
* **Integration Tests** (Requires local Ollama server):
```bash
# Queries and runs all tests tagged with 'integration'
bazel test $(bazel query "attr(tags, 'integration', //...)")
```

* Executing linters:
```bash
bazel build --config=lint //...
```

* Executing formatters:
```bash
bazel run //tools/format
```
## Design & Architecture

For a detailed overview of the system design, architectural principles, and component relationships, please refer to the [Design & Architecture](design_architecture.md) document.
