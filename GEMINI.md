# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## Build & Test

```bash
# Build the CLI
cargo build --release

# Run the CLI
cargo run --release -- [COMMAND]
```

## Architecture Overview

**Kokoro-CLI (Rust)** is a standalone executable utilizing:
* **`clap`**: For structured, self-documenting CLI commands.
* **`directories`**: To strictly adhere to XDG Base Directory standards (models are managed in `~/.local/share/kokoro/models/v1.0`).
* **`misaki-rs`**: For 1:1 parity with the Python G2P pipeline (including heteronym disambiguation and `[word](/phoneme/)` inline phonetic tags).
* **`ort` (v2.0+)**: To execute the `model.onnx` graph directly using native tensor math instead of C++ wrappers.

## Conventions & Patterns

### 1. Dependency Considerations & Build Chain
* **`cmake` is required:** Building the project requires `cmake` installed on the host machine because `misaki-rs` depends on compiling the `espeak-ng` C library from source as a fallback G2P engine. If a build fails with `os error 2: No such file or directory`, check that `cmake` is installed (`brew install cmake`).
* **`ort` v2 API:** Be aware that `ort` 2.0+ uses a newer API than 1.16 (e.g., `ort::inputs!` macros and `Tensor::from_array`). DO NOT downgrade to `1.16` as those releases have been yanked from crates.io.

### 2. Working with Tensors (`ort` and Kokoro)
* **Zero-Copy Loading:** The `voices.bin` file is 27MB. Do NOT load the entire file into memory. Calculate the byte offset using `(voice_id * 522_240) + (style_index * 256 * 4)` and seek exactly 256 floats (1024 bytes) into memory. Kokoro weights are Little-Endian.
* **Tensor Shapes:** The Kokoro ONNX model requires the following inputs:
  - `tokens`: Int64 Tensor of shape `[1, sequence_length]`. Must be padded with `0` (BOS/EOS).
  - `style`: Float32 Tensor of shape `[1, 256]`.
  - `speed`: Float32 Tensor of shape `[1]`.

### 3. Error Handling and Mutability
* Use `anyhow::Result` to propagate errors idiomatically rather than unwrapping or throwing panics.
* `ort` sessions require a mutable reference (`&mut self`) to execute the `run()` graph to ensure thread safety across the ONNX runtime. Pay close attention to borrow checker constraints when working within the `KokoroEngine` struct.

### 4. Hardware Acceleration (Apple Silicon)
* **CoreML Execution Provider:** ONNX Runtime supports CoreML natively. We gate this behind a `mac-acceleration` cargo feature to avoid breaking Linux/Windows builds that may not have the required frameworks.
* **Silencing Telemetry:** macOS CoreAnalytics frameworks dump verbose "Context leak detected" warnings to `stderr` when using CoreML via ONNX C libraries. We use the `shh` crate to temporarily capture and suppress `stderr` during `Session::run()` and `commit_from_file()` unless a `--verbose` flag is passed.

### 5. CLI UX Best Practices
* **Flexible Identifiers:** Instead of forcing users to remember numeric IDs for models or voices, the CLI parses the `voices.json` metadata to allow exact and partial string matches for voice names (e.g., `af_bella` or `bella`). Always prioritize human-readable inputs with graceful fallbacks and disambiguation errors.
