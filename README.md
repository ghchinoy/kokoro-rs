# Kokoro TTS (Rust CLI)

This is a native Rust command-line tool for running the [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) v1.0 model locally.

This CLI is designed to integrate **`misaki-rs`** (for exact Python pipeline 1:1 phonetic parity and heteronym disambiguation) and **`ort`** (for direct ONNX bindings). This provides a highly optimized, cross-platform local inference experience while maintaining feature parity with the official Python implementation.

## Installation

You can easily install the CLI globally using `cargo` from crates.io:

```bash
cargo install kokoro-cli
```

## Prerequisites

This CLI adheres strictly to XDG Base Directory specifications. The application expects the models to be located in `~/.local/share/kokoro/models/v1.0`.

### 1. Setup the Kokoro v1.0 (Multi-Language) Model

The CLI includes a built-in `setup` command that will automatically download, verify, and extract the model to your XDG data directory, as well as install the required `voices.json` metadata.

```bash
kokoro-cli setup
```

## Usage

This tool provides a highly structured command-line interface built on `clap`, making it predictable, easily discoverable, and self-documenting.

### Build the binary

```bash
cargo build --release
```
*The compiled binary will be located in `target/release/kokoro-cli`.*

### Generate speech (`speak`)

Generate audio from a given text string using the Kokoro 1.0 model. This uses the native `ort` ONNX runtime and `misaki-rs` for phonetic parity.

```bash
./target/release/kokoro-cli speak "Hello, world! This is a test of the CLI." --voice 0 --out test.wav
```

### Discovering Voices and Languages (`voices`, `languages`)

The `voices` and `languages` commands support machine-readable `--json` flags for easy integration with automation scripts and AI agents.

```bash
# List all available languages
./target/release/kokoro-cli languages

# Filter voices by a specific language
./target/release/kokoro-cli voices --language "Spanish"

# Output in JSON format for scripting
./target/release/kokoro-cli voices --language "English (US)" --json
```

## Why Rust?

By building in Rust, we gain access to `misaki-rs`—a native port of the official phonetic engine—allowing us to achieve 100% parity with the Python generation scripts. This means features like inline phoneme overrides (e.g., `[Kokoro](/kˈOkəɹO/)`) are supported out-of-the-box. At the same time, we maintain the ease of distribution of a fast, natively compiled binary that does not require the end user to manage complex Python environments or PyTorch installations.
