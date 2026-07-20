//! cli/mod.rs
//! This module defines the command-line interface structure using the `clap` crate.
//! `clap` uses Rust macros (the `#[derive(...)]` attributes) to automatically generate
//! parsing logic and help menus based on the shape of our structs.

use clap::{Args, Parser, Subcommand};
use serde::{Deserialize, Serialize}; // Serde is Rust's standard serialization/deserialization framework.
use std::fs; // The standard filesystem module.
use std::path::PathBuf; // PathBuf is a dynamically sized, mutable path type (like String, but for file paths).

/// A native Rust CLI tool for running Kokoro TTS locally.
///
/// Designed to provide 1:1 parity with the Python pipeline using misaki-rs
/// and ort, without requiring Python or PyTorch on the host machine.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Setup the Kokoro v1.0 model
    ///
    /// Downloads and extracts the Kokoro model and voices.json to the XDG data directory.
    /// This command replaces the need for the setup_models.sh script.
    Setup(SetupArgs),

    /// Generate speech from text (start here)
    ///
    /// Generates audio from a given text string using the Kokoro 1.0 model.
    /// Ensure you have the Kokoro ONNX model files downloaded to the XDG data directory.
    Speak(SpeakArgs),

    /// List available voices and supported languages
    Voices(VoicesArgs),

    /// List available languages
    Languages(LanguagesArgs),
}

#[derive(Args, Debug)]
pub struct SetupArgs {
    /// Force redownload even if the model already exists
    #[arg(short, long)]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct SpeakArgs {
    /// Text to convert to speech.
    pub text: String,

    #[arg(short, long, env = "KOKORO_MODEL_DIR")]
    pub model_dir: Option<String>,

    #[arg(short, long, default_value = "0")]
    pub voice: String,

    #[arg(short, long, default_value_t = 1.0)]
    pub speed: f32,

    #[arg(short, long, default_value = "output.wav")]
    pub out: String,

    /// Treat the input text as a raw phoneme string, bypassing G2P conversion
    #[arg(short, long, default_value_t = false)]
    pub phonemes: bool,

    /// Show verbose output (includes native hardware execution warnings)
    #[arg(long, default_value_t = false)]
    pub verbose: bool,

    /// Play the generated audio directly to the system speakers
    #[arg(long, default_value_t = false)]
    pub play: bool,
}

#[derive(Args, Debug)]
pub struct VoicesArgs {
    #[arg(short, long, env = "KOKORO_MODEL_DIR")]
    pub model_dir: Option<String>,

    #[arg(short, long)]
    pub language: Option<String>,

    #[arg(long, default_value_t = false)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct LanguagesArgs {
    #[arg(short, long, env = "KOKORO_MODEL_DIR")]
    pub model_dir: Option<String>,

    #[arg(long, default_value_t = false)]
    pub json: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VoiceMeta {
    pub id: u32,
    pub name: String,
    pub language: String,
    pub gender: String,
}

pub fn get_default_model_dir(override_dir: &Option<String>) -> PathBuf {
    if let Some(dir) = override_dir {
        return PathBuf::from(dir);
    }
    
    if let Ok(data_home) = std::env::var("XDG_DATA_HOME") {
        return PathBuf::from(data_home).join("kokoro").join("models").join("v1.0");
    }

    if let Some(base_dirs) = directories::BaseDirs::new() {
        return base_dirs.home_dir().join(".local").join("share").join("kokoro").join("models").join("v1.0");
    }

    PathBuf::from(".")
}

pub fn handle_setup(args: &SetupArgs) -> Result<(), String> {
    let dir = get_default_model_dir(&None);
    if dir.exists() {
        if args.force {
            println!("Force flag provided. Removing existing model directory...");
            fs::remove_dir_all(&dir).map_err(|e| format!("Failed to remove existing directory: {}", e))?;
        } else {
            println!("Model directory already exists at {:?}", dir);
            println!("Use --force to redownload and replace it.");
            return Ok(());
        }
    }

    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create model directory: {}", e))?;

    let url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2";
    let archive_name = "kokoro-multi-lang-v1_0.tar.bz2";
    let expected_hash = "c133d26353d776da730870dac7da07dbfc9a5e3bc80cc5e8e83ab6e823be7046";

    println!("Downloading Kokoro v1.0 from {}...", url);
    let curl_status = std::process::Command::new("curl")
        .args(&["-SL", "-o", archive_name, url])
        .status()
        .map_err(|e| format!("Failed to execute curl: {}", e))?;

    if !curl_status.success() {
        return Err("Failed to download model archive.".to_string());
    }

    println!("Verifying checksum...");
    let shasum_status = std::process::Command::new("shasum")
        .args(&["-a", "256", archive_name])
        .output()
        .map_err(|e| format!("Failed to execute shasum: {}", e))?;

    if shasum_status.status.success() {
        let output_str = String::from_utf8_lossy(&shasum_status.stdout);
        if !output_str.starts_with(expected_hash) {
            fs::remove_file(archive_name).ok();
            fs::remove_dir_all(&dir).ok();
            return Err(format!("Checksum mismatch!\nExpected: {}\nActual output: {}", expected_hash, output_str));
        }
        println!("Checksum passed!");
    } else {
        println!("\x1b[33mWarning: shasum command failed or not found, skipping checksum validation.\x1b[0m");
    }

    println!("Extracting {}...", archive_name);
    let tar_status = std::process::Command::new("tar")
        .args(&["xvf", archive_name, "-C", dir.to_str().unwrap(), "--strip-components=1"])
        .status()
        .map_err(|e| format!("Failed to execute tar: {}", e))?;

    if !tar_status.success() {
        return Err("Failed to extract model archive.".to_string());
    }

    fs::remove_file(archive_name).ok();

    println!("Writing voices.json for v1.0...");
    let voices_json = include_str!("../../assets/voices.json");
    fs::write(dir.join("voices.json"), voices_json)
        .map_err(|e| format!("Failed to write voices.json: {}", e))?;

    println!("\n\x1b[32mSetup complete!\x1b[0m Model installed to {:?}", dir);
    Ok(())
}

pub fn handle_speak(args: &SpeakArgs) -> Result<(), String> {
    let dir = get_default_model_dir(&args.model_dir);
    let voices_json = dir.join("voices.json");

    if !voices_json.exists() {
        eprintln!("\x1b[36mHint: Run 'kokoro-cli setup' to download models to {:?}\x1b[0m", dir);
        return Err(format!("Model directory not found: {:?}", dir));
    }

    let data = fs::read_to_string(&voices_json)
        .unwrap_or_else(|_| include_str!("../../assets/voices.json").to_string());
    
    let voices: Vec<VoiceMeta> = serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse voices.json: {}", e))?;

    let voice_id: u32 = match args.voice.parse::<u32>() {
        Ok(id) => id,
        Err(_) => {
            let query = args.voice.to_lowercase();
            // First try an exact match (case-insensitive)
            let exact_matches: Vec<&VoiceMeta> = voices.iter()
                .filter(|v| v.name.to_lowercase() == query)
                .collect();

            if exact_matches.len() == 1 {
                exact_matches[0].id
            } else if exact_matches.is_empty() {
                // If no exact match, try a substring match
                let partial_matches: Vec<&VoiceMeta> = voices.iter()
                    .filter(|v| v.name.to_lowercase().contains(&query))
                    .collect();

                if partial_matches.is_empty() {
                    return Err(format!("No voice found matching '{}'", args.voice));
                } else if partial_matches.len() > 1 {
                    let matched_names: Vec<String> = partial_matches.iter().map(|v| v.name.clone()).collect();
                    return Err(format!("Ambiguous voice '{}'. Matched multiple: {:?}", args.voice, matched_names));
                } else {
                    partial_matches[0].id
                }
            } else {
                return Err(format!("Ambiguous voice '{}'. Matched multiple exact names.", args.voice));
            }
        }
    };

    println!("\x1b[34mInitializing TTS with model from {:?}...\x1b[0m", dir);
    
    // Wire up the new TTS Engine
    let mut engine = crate::tts::KokoroEngine::new(&dir, args.verbose)
        .map_err(|e| format!("Failed to initialize TTS engine: {}", e))?;

    println!("\x1b[34mGenerating audio for voice ID {}...\x1b[0m", voice_id);
    
    let audio_samples = if args.phonemes {
        engine.generate_audio_from_phonemes(&args.text, voice_id, args.speed, args.verbose)
    } else {
        engine.generate_audio(&args.text, voice_id, args.speed, args.verbose)
    }.map_err(|e| format!("Failed to generate audio: {}", e))?;

    // Create a valid WAV file using `hound`
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    
    let mut writer = hound::WavWriter::create(&args.out, spec)
        .map_err(|e| format!("Failed to create dummy WAV file: {}", e))?;
        
    // Write the raw float samples as 16-bit PCM
    for &sample in &audio_samples {
        let amplitude = (sample * std::i16::MAX as f32) as i16;
        writer.write_sample(amplitude).unwrap();
    }
    writer.finalize().map_err(|e| format!("Failed to finalize WAV file: {}", e))?;

    if args.play {
        println!("\x1b[34mPlaying generated audio...\x1b[0m");
        use rodio::{OutputStream, Sink};
        use rodio::buffer::SamplesBuffer;
        
        let (_stream, stream_handle) = OutputStream::try_default()
            .map_err(|e| format!("Failed to open audio output stream: {}", e))?;
        let sink = Sink::try_new(&stream_handle)
            .map_err(|e| format!("Failed to create audio sink: {}", e))?;
        let source = SamplesBuffer::new(1, 24000, audio_samples.clone());
        sink.append(source);
        sink.sleep_until_end();
    }

    println!("\x1b[32mSuccess!\x1b[0m Audio saved to {}", args.out);
    Ok(())
}

pub fn handle_voices(args: &VoicesArgs) -> Result<(), String> {
    let dir = get_default_model_dir(&args.model_dir);
    let json_path = dir.join("voices.json");

    let data = fs::read_to_string(&json_path).unwrap_or_else(|_| include_str!("../../assets/voices.json").to_string());

    let voices: Vec<VoiceMeta> = serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse voices.json: {}", e))?;

    let mut filtered = voices;
    if let Some(lang) = &args.language {
        filtered.retain(|v| v.language == *lang);
    }

    if args.json {
        let out = serde_json::to_string_pretty(&filtered).unwrap();
        println!("{}", out);
        return Ok(());
    }

    if filtered.is_empty() {
        println!("No voices found matching language: {:?}", args.language);
        return Ok(());
    }

    println!("{:<4} | {:<15} | {:<15} | {:<10}", "ID", "NAME", "LANGUAGE", "GENDER");
    println!("{:-<4}-+-{:-<15}-+-{:-<15}-+-{:-<10}", "", "", "", "");
    for v in filtered {
        println!("{:<4} | {:<15} | {:<15} | {:<10}", v.id, v.name, v.language, v.gender);
    }

    Ok(())
}

pub fn handle_languages(args: &LanguagesArgs) -> Result<(), String> {
    let dir = get_default_model_dir(&args.model_dir);
    let json_path = dir.join("voices.json");

    let data = fs::read_to_string(&json_path).unwrap_or_else(|_| include_str!("../../assets/voices.json").to_string());

    let voices: Vec<VoiceMeta> = serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse voices.json: {}", e))?;

    let mut langs = std::collections::HashSet::new();
    for v in voices {
        langs.insert(v.language);
    }

    let mut sorted_langs: Vec<String> = langs.into_iter().collect();
    sorted_langs.sort();

    if args.json {
        let out = serde_json::to_string_pretty(&sorted_langs).unwrap();
        println!("{}", out);
        return Ok(());
    }

    println!("Available Languages:");
    for l in sorted_langs {
        println!("  - {}", l);
    }

    Ok(())
}
