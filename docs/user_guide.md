# Kokoro TTS (Rust CLI) User Guide

Welcome to the comprehensive User Guide for `kokoro-cli`, a native Rust command-line tool for running the Kokoro TTS v1.0 model locally. This guide details installation, setup, phonetic overrides, detailed command-line options, architecture, hardware acceleration, troubleshooting, and scripting.

---

## Table of Contents

- [1. Installation & Setup](#1-installation--setup)
  - [Global Installation](#global-installation)
  - [Hardware Acceleration (Apple Silicon/CoreML)](#hardware-acceleration-apple-siliconcoreml)
  - [Prerequisites (CMake)](#prerequisites-cmake)
- [2. Command-Line Reference](#2-command-line-reference)
  - [kokoro-cli setup](#kokoro-cli-setup)
  - [kokoro-cli speak](#kokoro-cli-speak)
  - [kokoro-cli voices](#kokoro-cli-voices)
  - [kokoro-cli languages](#kokoro-cli-languages)
- [3. Phonetic Parsing & Inline Overrides](#3-phonetic-parsing--inline-overrides)
  - [Markdown-Style Overrides](#markdown-style-overrides)
  - [Plain Slash-Style Overrides](#plain-slash-style-overrides)
  - [Percent-Encoded IPA Inputs](#percent-encoded-ipa-inputs)
- [4. Architecture & Technical Details](#4-architecture--the-voice-model)
  - [How Kokoro Works (ORT & Misaki)](#how-kokoro-works-ort--misaki)
  - [Vocabulary & Voice Embeddings](#vocabulary--voice-embeddings)
  - [XDG Directory Layout](#xdg-directory-layout)
- [5. Advanced Integration & Scripting](#5-advanced-integration--scripting)
  - [JSON Pipeline Automation](#json-pipeline-automation)
  - [LLM Piping & Streaming Output](#llm-piping--streaming-output)
- [6. Troubleshooting & FAQ](#6-troubleshooting--faq)

---

## 1. Installation & Setup

### Global Installation

To install `kokoro-cli` globally, use the standard `cargo` tool:

```bash
cargo install kokoro-cli
```

The compiled binary will be placed directly in your `~/cargo/bin` folder.

### Hardware Acceleration (Apple Silicon/CoreML)

Apple Silicon (M1/M2/M3/M4) Mac users can gain significant inference speedups and power savings by utilizing Apple's Neural Engine via the `mac-acceleration` feature flag. This compiles native `ort` with Apple CoreML bindings:

```bash
cargo install kokoro-cli --features mac-acceleration
```

### Prerequisites (CMake)

Building this project from source (or via `cargo install`) requires **CMake** installed on your host machine.
* **Why?** Under the hood, the phonetic engine (`misaki-rs`) depends on compiling the `espeak-ng` C library from source to act as a fallback Grapheme-to-Phoneme (G2P) engine.
* **If it fails**: If you see compilation errors like `os error 2: No such file or directory`, make sure CMake is installed:
  - **macOS**: `brew install cmake`
  - **Debian/Ubuntu**: `sudo apt-get install cmake`
  - **Arch Linux**: `sudo pacman -S cmake`

---

## 2. Command-Line Reference

The CLI is structured into three main subcommands: `setup`, `speak`, `voices`, and `languages`.

### kokoro-cli setup

Before running the model for the first time, you must download the 310MB ONNX model and 26MB voice configurations. The built-in `setup` command automates downloading, hash verification, folder structuring, and metadata placement.

```bash
kokoro-cli setup
```

**Flags:**
* `-f`, `--force`: Force-redownloads and overwrites existing models in your local storage.

---

### kokoro-cli speak

Translates input text into speech.

```bash
kokoro-cli speak "Hello from Kokoro-rs!" --voice af_bella --out output.wav
```

**Arguments and Flags:**
* `text` (positional): The string to translate into speech.
* `-v`, `--voice <id_or_name>`: The voice profile to use (default: `0` / `af_bella`). You can pass the exact numeric ID, the exact voice name, or a unique substring.
* `-s`, `--speed <factor>`: Control speech speed (e.g. `1.0` is normal, `1.2` is faster).
* `-o`, `--out <filename>`: Output filename (default: `output.wav`).
* `-p`, `--phonemes`: Interpret the input text as raw phonemes directly, bypassing the G2P engine entirely.
* `--verbose`: Show full initialization execution timings and hardware telemetry.
* `--play`: Automatically play the generated audio directly to your system speakers using the native `rodio` playback library, in addition to writing the `.wav` file.
* `--dry-run` / `--phonemes-only`: Skip ONNX model loading and audio generation completely. Instantaneously runs only the lightweight G2P translation on the text and prints the resulting phonemes to the terminal.

---

### kokoro-cli voices

Query and discover available voice profiles.

```bash
kokoro-cli voices --language "Spanish"
```

**Flags:**
* `-l`, `--language <name>`: Filter voice profiles by language.
* `--json`: Output result in machine-readable JSON format.

---

### kokoro-cli languages

List all languages supported by the voice profiles.

```bash
kokoro-cli languages --json
```

**Flags:**
* `--json`: Output result in machine-readable JSON format.

---

## 3. Phonetic Parsing & Inline Overrides

A common limitation of TTS systems is pronouncing rare names, acronyms, or non-English words. `kokoro-cli` supports **inline G2P phoneme overrides**, allowing you to manually specify the IPA phonetic spelling for a specific word inline within otherwise normal text.

The system supports two syntaxes:

### Markdown-Style Overrides

Follows standard Markdown link syntax: `[word](/phonemes/)`. The system strips leading/trailing slashes around the phonemes.

```bash
kokoro-cli speak "The [Kokoro](/kˈOkəɹO/) model is fast." --out test.wav
```

### Plain Slash-Style Overrides

Uses trailing slashes directly after the target word: `word /phonemes/`.

```bash
kokoro-cli speak "The Kokoro /kˈOkəɹO/ model is fast." --out test.wav
```

### Percent-Encoded IPA Inputs

Because terminal emulators can sometimes corrupt raw Unicode IPA characters, you can pass percent-encoded IPA characters. The CLI will automatically decode them during parsing:

* `%CB%88` -> `ˈ` (stress)
* `%C9%99` -> `ə` (schwa)
* `%C9%B9` -> `ɹ` (turned r)

```bash
kokoro-cli speak "The Kokoro /k%CB%88Ok%C9%99%C9%B9O/ model is fast." --out test.wav
```

---

## 4. Architecture & Technical Details

### How Kokoro Works (ORT & Misaki)

1. **Grapheme-to-Phoneme (G2P)**: Input text is parsed by `misaki-rs`, which translates plain letters into IPA phonemes (like `Hello` to `həˈloʊ`). It matches words against its internal golden/silver dictionary data (~15MB JSON, compiled in the binary) and uses `espeak-ng` fallback for unknown words.
2. **Phoneme Overrides**: If inline overrides are present, our parser bypasses G2P for those specific segments and splices your hand-specified phonemes into the phoneme stream.
3. **Token Mapping**: Phoneme characters are mapped to integer vocab IDs using the Model's `tokens.txt`.
4. **Style Retrieval**: Based on the token length, a 256-float voice embedding style vector is extracted from `voices.bin`.
5. **ONNX Graph Execution**: The `ort` (ONNX Runtime) session executes the neural network graph (`model.onnx`) with the `tokens`, `style`, and `speed` tensors.
6. **WAV Encoding / Playback**: Float32 audio samples are converted to 16-bit PCM and saved as standard WAV or piped to `rodio` for playback.

### XDG Directory Layout

On all systems, files are stored strictly adhering to modern operating system storage specifications:

| Platform | Layout Path |
|----------|-------------|
| **macOS** | `~/Library/Application Support/kokoro/models/v1.0/` |
| **Linux** | `~/.local/share/kokoro/models/v1.0/` |
| **Windows** | `%APPDATA%\Roaming\kokoro\models\v1.0\` |

Within this folder, the system expects:
* `model.onnx` — The 310MB inference graph.
* `voices.bin` — The 26MB compiled voice embeddings.
* `voices.json` — Voice metadata.
* `tokens.txt` — Phonemic vocab mappings.

---

## 5. Advanced Integration & Scripting

### JSON Pipeline Automation

AI agents and automated pipelines can query voices in JSON and pipe the voice IDs directly:

```bash
# Get the first Spanish voice ID using jq
VOICE_ID=$(kokoro-cli voices --language "Spanish" --json | jq '.[0].id')

# Generate speech using that voice ID
kokoro-cli speak "Hola mundo!" --voice "$VOICE_ID" --out hola.wav
```

### LLM Piping & Dry Run Previews

To quickly preview how a long AI-generated text will sound phonetically before spending any compute or power on audio generation, pipe the output through the `--dry-run` flag:

```bash
echo "Welcome to the future of speech!" | kokoro-cli speak - --dry-run
```

---

## 6. Troubleshooting & FAQ

### Q: Why does the compile fail with an `espeak-ng` error?
**A:** You are missing `cmake` on your path. Install CMake on your system (e.g. `brew install cmake` or `sudo apt-get install cmake`) and try again.

### Q: How do I resolve `unsupported voice` or `ambiguous voice` errors?
**A:** Use the `kokoro-cli voices` command to view exact spelling. If you provide a substring like `bella`, ensure it matches exactly one voice. If it matches multiple, the CLI will output the matched names and ask you to be more specific.

### Q: Why do I hear static or noise during playback?
**A:** Your default system output sound device may have an incompatible sample rate, or your speakers are already exclusively locked. Check your OS audio configuration. `kokoro-cli` outputs mono audio at 24000Hz.
