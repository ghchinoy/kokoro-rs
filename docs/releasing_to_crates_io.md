# Releasing to crates.io

This guide explains how to publish the `kokoro-cli` CLI crate to `crates.io`, the official Rust package registry. This makes it easy for other Rust developers and users to install your CLI globally using `cargo install kokoro-cli`.

## 1. Prerequisites

Before you can publish a package, you need to set up an account and authenticate your local environment.

1. **Create an account on crates.io:**
   * Go to [crates.io](https://crates.io/) and log in using your GitHub account.
   * Navigate to your Account Settings.

2. **Generate an API token:**
   * In your Account Settings, go to the "API Tokens" section.
   * Click "New Token", give it a name (e.g., "macbook-pro"), and copy the token.

3. **Authenticate your local Cargo:**
   Run the following command in your terminal, and paste the token you just copied:
   ```bash
   cargo login
   ```
   Then paste your token.

   *This saves the token to `~/.cargo/credentials` so Cargo can authenticate future publish requests.*

## 2. Preparing your `Cargo.toml`

Before publishing, your `Cargo.toml` file needs some essential metadata. `crates.io` requires fields like `description` and `license` to be present.

Open `Cargo.toml` and ensure the following fields are filled out under the `[package]` section:

```toml
[package]
name = "kokoro" # Must be unique on crates.io! (You may need to use "kokoro-cli" if "kokoro" is taken)
version = "0.1.0" # Semantic versioning (Major.Minor.Patch)
edition = "2021"
description = "A native Rust CLI for Kokoro TTS using XDG standards and agent-first design"
license = "MIT OR Apache-2.0" # Strongly recommended for open-source Rust projects
repository = "https://github.com/ghchinoy/kokoro-rs" # URL to the source code
readme = "README.md" # Points to your README so crates.io can render it
keywords = ["tts", "cli", "kokoro", "ai", "audio"] # Max 5 keywords
categories = ["command-line-utilities", "multimedia::audio"] # From https://crates.io/category_slugs
```

## 3. Testing and Verification

Before publishing, it is crucial to ensure everything compiles cleanly and your working tree is clean.

1. **Run tests and build:**
   ```bash
   cargo test
   cargo build --release
   ```

2. **Check for uncommitted changes:**
   Cargo will refuse to publish if you have uncommitted changes in your Git repository. Ensure you have committed everything:
   ```bash
   git status
   git commit -am "Prepare for release 0.1.0"
   ```

3. **Perform a dry run:**
   This command packages the crate and verifies that it *can* be published, without actually uploading it to the registry. It's a great way to catch missing files or metadata errors.
   ```bash
   cargo publish --dry-run
   ```

## 4. Publishing the Crate

Once the dry run succeeds, you are ready to publish!

Run the following command:
```bash
cargo publish
```

*Note: Publishing is **permanent**. You cannot overwrite a version once it has been published. If you make a mistake, you must increment the version number in `Cargo.toml` and publish again.*

## 5. Installing the Published Crate

After publishing, anyone can install your CLI globally on their system using:

```bash
cargo install kokoro-cli
```
*(Note: The binary installed will be named `kokoro-cli` unless explicitly renamed in Cargo.toml).*

## Post-Release: Yanking a Version (Emergency Only)

If you accidentally publish a version with a critical bug or security flaw, you can "yank" it. Yanking prevents new projects from depending on that version, but allows existing projects that already downloaded it to continue compiling.

```bash
cargo yank --vers 0.1.0
```
To undo a yank:
```bash
cargo yank --vers 0.1.0 --undo
```
```
