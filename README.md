# Kokoro TTS (Rust CLI)

[![Crates.io Version](https://img.shields.io/crates/v/kokoro-cli.svg)](https://crates.io/crates/kokoro-cli)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A native Rust command-line tool for running the [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) v1.0 model locally. 

This CLI integrates **`misaki-rs`** (for exact Python pipeline 1:1 phonetic parity and heteronym disambiguation) and **`ort`** (for direct ONNX bindings). It provides a highly optimized, cross-platform local inference experience while maintaining feature parity with the official Python implementation, entirely without requiring Python or PyTorch on the host machine.

---

## Table of Contents

- [1. Installation](#1-installation)
  - [Prerequisites (CMake)](#prerequisites)
  - [Global Install](#global-install)
  - [Apple Silicon Hardware Acceleration](#apple-silicon-hardware-acceleration)
- [2. Quick Start / Usage](#2-quick-start--usage)
  - [Model Setup](#1-model-setup)
  - [Generate Speech](#2-generate-speech)
  - [Inline Phonetic Overrides](#3-inline-phonetic-overrides)
- [3. Local Development Setup](#3-local-development-setup)
  - [Building from Source](#building-from-source)
- [4. Maintainer Publish Process](#4-maintainer-publish-process)
- [5. Contributing](#5-contributing)
- [6. Additional Documentation](#6-additional-documentation)
- [7. License](#7-license)

---

## 1. Installation

### Prerequisites

Building the CLI (including via `cargo install`) requires **CMake** installed on your system. Under the hood, `misaki-rs` compiles the `espeak-ng` C library from source to act as a fallback phonetic G2P engine.

* **macOS**: `brew install cmake`
* **Debian/Ubuntu**: `sudo apt-get install cmake`
* **Arch Linux**: `sudo pacman -S cmake`

### Global Install

Install the stable release of the CLI globally using `cargo`:

```bash
cargo install kokoro-cli
```

### Apple Silicon Hardware Acceleration

If you are on an Apple Silicon (M1/M2/M3/M4) Mac, you can enable native CoreML hardware acceleration for dramatic speedups by compiling with the `mac-acceleration` feature flag:

```bash
cargo install kokoro-cli --features mac-acceleration
```

---

## 2. Quick Start / Usage

### 1. Model Setup

The CLI expects models and voice embeddings in your local XDG data directory. Automate downloading, hash verification, and metadata setup with the built-in `setup` command:

```bash
kokoro-cli setup
```

### 2. Generate Speech

Generate a high-quality speech `.wav` file in a single command. You can select voices by ID or Name:

```bash
# Using a Voice ID (e.g. 0 for Bella)
kokoro-cli speak "Hello, world! This is a test of the CLI." --voice 0 --out hello.wav

# Or using a Voice Name substring (e.g. af_bella)
kokoro-cli speak "Hello, world! Auto-playing this audio." --voice af_bella --play
```

* **Play Directly**: Pass the `--play` flag to automatically stream the audio to your system speakers immediately upon generation!
* **Fast G2P Preview**: Pass `--dry-run` (or `--phonemes-only`) to instantaneously preview the resulting phonetic IPA layout on your console without running the ONNX neural network session.

### 3. Inline Phonetic Overrides

Pronounce rare names or non-English words by specifying their exact IPA pronunciation inline using Markdown link style `[word](/ipa/)` or plain slash style `word /ipa/`:

```bash
# Markdown Link Style
kokoro-cli speak "The [Kokoro](/kˈOkəɹO/) model is fast." --play

# Plain Slash Style (with percent-encoded Unicode)
kokoro-cli speak "The Kokoro /k%CB%88Ok%C9%99%C9%B9O/ model is fast." --play
```

---

## 3. Local Development Setup

To clone the repository and build or run the CLI during local development:

```bash
# 1. Clone the repository
git clone https://github.com/ghchinoy/kokoro-rs.git
cd kokoro-rs

# 2. Build the binary locally in release mode
cargo build --release

# (Optional) Build with Apple CoreML acceleration:
cargo build --release --features mac-acceleration

# 3. The local binary is placed in target/release/
./target/release/kokoro-cli speak "Testing local build." --play
```

To run the unit test suite verifying G2P parsing and inline overrides:
```bash
cargo test
```

---

## 4. Maintainer Publish Process

For project maintainers publishing updates to [crates.io](https://crates.io):

```bash
# 1. Bump version in Cargo.toml
# 2. Publish to crates.io
cargo publish
```
Refer to [docs/releasing_to_crates_io.md](docs/releasing_to_crates_io.md) for full release details.

---

## 5. Contributing

Pull requests are very welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 6. Additional Documentation

For more in-depth configuration, scripting, and evaluation guides, explore our auxiliary documentation:

* [Full CLI User Guide](docs/user_guide.md) — Subcommand parameter reference, JSON pipeline scripting, advanced override examples, and OS directories.
* [Prompting & Expression Guide](docs/prompting.md) — Techniques for configuring expressiveness and speech patterns.
* [Custom Voices & Lexicons](docs/customizing_voices_and_lexicons.md) — Creating, adding, and loading custom voice embeddings.
* [TTS Quality Evaluation Guide](docs/evaluating_tts.md) — Methodology and metrics for evaluating synthesized speech outputs.

---

## 7. License

Distributed under either the **MIT License** or **Apache License, Version 2.0**, at your option. See [LICENSE](LICENSE) for details.
