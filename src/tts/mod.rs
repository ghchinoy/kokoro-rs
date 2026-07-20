use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use misaki_rs::{G2P, Language};

/// The default sample rate (24000Hz) of the Kokoro model audio output.
pub const SAMPLE_RATE: u32 = 24000;
/// The number of channels (1 = mono) of the Kokoro model audio output.
pub const CHANNELS: u16 = 1;

/// Represents the initialized TTS Engine containing the ONNX session
/// and the Misaki G2P parser.
pub struct KokoroEngine {
    onnx_session: Session,
    g2p: G2P,
    vocab: HashMap<char, i64>,
    voices_bin_path: PathBuf,
}

impl KokoroEngine {
    /// Initializes the TTS Engine by loading the Kokoro ONNX model
    /// and the Misaki G2P phonetic dictionaries.
    pub fn new(model_dir: &Path, verbose: bool) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        let voices_bin_path = model_dir.join("voices.bin");
        let tokens_path = model_dir.join("tokens.txt");

        // WIRING STEP 1: Load the Vocabulary mapping
        println!("  -> [Vocab] Loading tokens.txt...");
        let vocab = Self::load_vocab(&tokens_path)?;

        // WIRING STEP 2: Initialize the Misaki G2P engine
        println!("  -> [Misaki-rs] Initializing G2P engine...");
        let g2p = G2P::new(Language::EnglishUS); // EnglishUS = American English

        // WIRING STEP 3: Initialize the ORT ONNX Session
        println!("  -> [Ort] Loading ONNX model from {:?}...", model_path);
        
        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("Ort builder error: {:?}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Ort optimization error: {:?}", e))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Ort thread error: {:?}", e))?;

        #[cfg(feature = "mac-acceleration")]
        {
            println!("  -> [Ort] Registering CoreML Execution Provider...");
            builder = builder
                .with_execution_providers([ort::execution_providers::CoreMLExecutionProvider::default().build()])
                .map_err(|e| anyhow::anyhow!("Failed to register CoreML: {:?}", e))?;
        }

        let onnx_session = {
            // Silence stderr from C libraries (like CoreML CoreAnalytics leaks) unless verbose
            let _silencer = if !verbose { shh::stderr().ok() } else { None };
            builder
                .commit_from_file(&model_path)
                .map_err(|e| anyhow::anyhow!("Failed to load model.onnx: {:?}", e))?
        };

        Ok(Self {
            onnx_session,
            g2p,
            vocab,
            voices_bin_path,
        })
    }

    /// Translates input text into phonemes, taking inline overrides into account,
    /// utilizing the pre-loaded G2P engine.
    pub fn text_to_phonemes(&self, text: &str) -> Result<String> {
        process_g2p_with_overrides(&self.g2p, text)
    }

    /// Generates raw float32 audio samples from input text.
    pub fn generate_audio(&mut self, text: &str, voice_id: u32, speed: f32, verbose: bool) -> Result<Vec<f32>> {
        let full_phonemes = self.text_to_phonemes(text)?;
        println!("  -> [Misaki-rs] Phonemes: {}", full_phonemes);
        
        self.generate_audio_from_phonemes(&full_phonemes, voice_id, speed, verbose)
    }

    /// Generates raw float32 audio samples directly from raw phonemes.
    pub fn generate_audio_from_phonemes(&mut self, phonemes: &str, voice_id: u32, speed: f32, verbose: bool) -> Result<Vec<f32>> {
        // STEP 2: Map phonemes to integer tokens using our vocabulary
        // Kokoro sequences must begin and end with 0 (which acts as BOS/EOS/PAD)
        let mut token_ids: Vec<i64> = Vec::with_capacity(phonemes.chars().count() + 2);
        token_ids.push(0); // BOS
        
        let mut unmapped = Vec::new();
        for c in phonemes.chars() {
            if let Some(&id) = self.vocab.get(&c) {
                token_ids.push(id);
            } else if !unmapped.contains(&c) {
                unmapped.push(c);
                eprintln!(
                    "\x1b[33mWarning: Phoneme character '{}' ({}) is not present in the model's vocabulary (tokens.txt) and was skipped.\x1b[0m",
                    c,
                    c.escape_unicode()
                );
            }
        }
        token_ids.push(0); // EOS

        self.generate_audio_from_tokens(token_ids, voice_id, speed, verbose)
    }

    /// Generates raw float32 audio samples from token IDs.
    pub fn generate_audio_from_tokens(&mut self, token_ids: Vec<i64>, voice_id: u32, speed: f32, verbose: bool) -> Result<Vec<f32>> {
        let token_len = token_ids.len();

        // STEP 3: Extract the specific Voice Embedding tensor from voices.bin
        // Each voice consists of 510 styles (based on token length) of 256 floats each.
        // Total floats per voice = 510 * 256 = 130,560 floats = 522,240 bytes.
        println!("  -> [Ort] Extracting voice tensor for ID {}...", voice_id);
        let style_index = std::cmp::min(token_len, 509); 
        let voice_byte_offset = (voice_id as u64) * 522_240; // 510 * 256 * 4 bytes
        let style_byte_offset = voice_byte_offset + ((style_index as u64) * 256 * 4);

        let mut f = File::open(&self.voices_bin_path)?;
        f.seek(SeekFrom::Start(style_byte_offset))?;
        
        let mut style_bytes = vec![0u8; 256 * 4];
        f.read_exact(&mut style_bytes)?;

        // Convert the raw bytes back into f32s (Kokoro weights are little-endian)
        let mut style_vector: Vec<f32> = Vec::with_capacity(256);
        for chunk in style_bytes.chunks_exact(4) {
            let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            style_vector.push(val);
        }

        // STEP 4: Prepare Input Tensors for ONNX
        // tokens: [1, sequence_length], Int64
        let tokens_tensor = ort::value::Tensor::from_array(([1, token_len], token_ids))
            .map_err(|e| anyhow::anyhow!("Failed to create tokens tensor: {}", e))?;
            
        // style: [1, 256], Float32
        let style_tensor = ort::value::Tensor::from_array(([1, 256], style_vector))
            .map_err(|e| anyhow::anyhow!("Failed to create style tensor: {}", e))?;
            
        // speed: [1], Float32
        let speed_tensor = ort::value::Tensor::from_array(([1], vec![speed]))
            .map_err(|e| anyhow::anyhow!("Failed to create speed tensor: {}", e))?;

        // STEP 5: Execute the ONNX Graph
        println!("  -> [Ort] Executing ONNX Graph...");
        
        let outputs = {
            let _silencer = if !verbose { shh::stderr().ok() } else { None };
            self.onnx_session.run(ort::inputs![
                "tokens" => tokens_tensor,
                "style" => style_tensor,
                "speed" => speed_tensor,
            ])?
        };

        // STEP 6: Extract the audio float array
        let audio_tensor = outputs["audio"].try_extract_tensor::<f32>()?;
        let audio_samples = audio_tensor.1.to_vec();

        println!("  -> [Ort] Audio generated successfully! ({} samples)", audio_samples.len());
        Ok(audio_samples)
    }

    /// Loads the tokens.txt mapping (e.g., `a 43`) into a HashMap
    fn load_vocab(path: &Path) -> Result<HashMap<char, i64>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut vocab = HashMap::new();

        use std::io::BufRead;
        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                // The character is the first part, the integer ID is the second
                let c = parts[0].chars().next().unwrap();
                if let Ok(id) = parts[1].parse::<i64>() {
                    vocab.insert(c, id);
                }
            }
        }
        Ok(vocab)
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Segment {
    Normal(String),
    Phonemes(String),
}

fn percent_decode(s: &str) -> String {
    let mut bytes = Vec::with_capacity(s.len());
    let mut chars = s.as_bytes().iter();
    while let Some(&b) = chars.next() {
        if b == b'%' {
            if let (Some(&h), Some(&l)) = (chars.next(), chars.next()) {
                if let Some(decoded) = hex_to_byte(h, l) {
                    bytes.push(decoded);
                    continue;
                }
                bytes.push(b'%');
                bytes.push(h);
                bytes.push(l);
            } else {
                bytes.push(b'%');
            }
        } else {
            bytes.push(b);
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn hex_to_byte(h: u8, l: u8) -> Option<u8> {
    let h_val = (h as char).to_digit(16)?;
    let l_val = (l as char).to_digit(16)?;
    Some((h_val << 4 | l_val) as u8)
}

fn find_markdown_link(chars: &[char], start: usize) -> Option<(usize, usize, usize)> {
    let mut bracket_depth = 0;
    let mut word_end = None;
    for (idx, &c) in chars.iter().enumerate().skip(start) {
        if c == '[' {
            bracket_depth += 1;
        } else if c == ']' {
            bracket_depth -= 1;
            if bracket_depth == 0 {
                word_end = Some(idx);
                break;
            }
        }
    }
    
    let word_end = word_end?;
    if word_end + 2 < chars.len() && chars[word_end + 1] == '(' {
        let ipa_start = word_end + 2;
        for (idx, &c) in chars.iter().enumerate().skip(ipa_start) {
            if c == ')' {
                return Some((word_end, ipa_start, idx));
            }
        }
    }
    None
}

fn find_closing_slash(chars: &[char], start: usize) -> Option<usize> {
    chars.iter().skip(start + 1).position(|&c| c == '/').map(|pos| start + 1 + pos)
}

fn find_last_word_start(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let mut idx = bytes.len();
    while idx > 0 {
        let c = bytes[idx - 1] as char;
        if c.is_alphanumeric() || c == '_' || c == '-' {
            idx -= 1;
        } else {
            break;
        }
    }
    if idx < bytes.len() {
        Some(idx)
    } else {
        None
    }
}

fn parse_inline_overrides(text: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut current_normal = String::new();
    let mut i = 0;
    let chars: Vec<char> = text.chars().collect();
    
    while i < chars.len() {
        if chars[i] == '[' {
            if let Some((_word_end, ipa_start, ipa_end)) = find_markdown_link(&chars, i) {
                if !current_normal.is_empty() {
                    segments.push(Segment::Normal(current_normal));
                    current_normal = String::new();
                }
                let mut ipa: String = chars[ipa_start..ipa_end].iter().collect();
                if ipa.starts_with('/') {
                    ipa.remove(0);
                }
                if ipa.ends_with('/') {
                    ipa.pop();
                }
                let decoded_ipa = percent_decode(&ipa);
                segments.push(Segment::Phonemes(decoded_ipa));
                i = ipa_end + 1; // move past ')'
                continue;
            }
        }
        
        if chars[i] == '/' {
            if let Some(closing_idx) = find_closing_slash(&chars, i) {
                let preceding_trimmed = current_normal.trim_end();
                if !preceding_trimmed.is_empty() && preceding_trimmed.chars().last().unwrap().is_alphanumeric() {
                    if let Some(word_start) = find_last_word_start(preceding_trimmed) {
                        let normal_prefix = &preceding_trimmed[..word_start];
                        if !normal_prefix.is_empty() {
                            segments.push(Segment::Normal(normal_prefix.to_string()));
                        }
                        current_normal = String::new();
                        
                        let ipa: String = chars[i+1..closing_idx].iter().collect();
                        let decoded_ipa = percent_decode(&ipa);
                        segments.push(Segment::Phonemes(decoded_ipa));
                        i = closing_idx + 1;
                        continue;
                    }
                }
            }
        }
        
        current_normal.push(chars[i]);
        i += 1;
    }
    
    if !current_normal.is_empty() {
        segments.push(Segment::Normal(current_normal));
    }
    
    segments
}

/// Shared helper to convert text to phonemes with inline overrides using a provided G2P instance.
fn process_g2p_with_overrides(g2p: &G2P, text: &str) -> Result<String> {
    let segments = parse_inline_overrides(text);
    
    let mut full_phonemes = String::new();
    for segment in segments {
        match segment {
            Segment::Normal(chunk) => {
                if chunk.trim().is_empty() {
                    full_phonemes.push_str(&chunk);
                } else {
                    let (p, _) = g2p.g2p(&chunk).map_err(|e| anyhow::anyhow!("G2P error: {:?}", e))?;
                    let mut p_cleaned = p;
                    
                    if chunk.starts_with(char::is_whitespace) && !p_cleaned.starts_with(char::is_whitespace) && !full_phonemes.ends_with(char::is_whitespace) && !full_phonemes.is_empty() {
                        full_phonemes.push(' ');
                    }
                    
                    if full_phonemes.ends_with(char::is_whitespace) {
                        p_cleaned = p_cleaned.trim_start().to_string();
                    }
                    
                    full_phonemes.push_str(&p_cleaned);
                }
            }
            Segment::Phonemes(ipa) => {
                if !full_phonemes.is_empty() && !full_phonemes.ends_with(char::is_whitespace) {
                    full_phonemes.push(' ');
                }
                full_phonemes.push_str(&ipa);
            }
        }
    }
    Ok(full_phonemes)
}

/// Helper to convert text to phonemes using the default US English G2P with inline overrides.
/// NOTE: Language::EnglishUS is also used in KokoroEngine::new. If multi-language speech
/// is implemented later, these call sites must stay in sync (e.g. by passing Language to this helper).
pub fn g2p_text_to_phonemes(text: &str) -> Result<String> {
    let g2p = G2P::new(Language::EnglishUS);
    process_g2p_with_overrides(&g2p, text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_inline_overrides_markdown() {
        let text = "The [Kokoro](/kˈOkəɹO/) model...";
        let segments = parse_inline_overrides(text);
        assert_eq!(
            segments,
            vec![
                Segment::Normal("The ".to_string()),
                Segment::Phonemes("kˈOkəɹO".to_string()),
                Segment::Normal(" model...".to_string()),
            ]
        );
    }

    #[test]
    fn test_parse_inline_overrides_plain() {
        let text = "The Kokoro /k%CB%88Ok%C9%99%C9%B9O/ model...";
        let segments = parse_inline_overrides(text);
        assert_eq!(
            segments,
            vec![
                Segment::Normal("The ".to_string()),
                Segment::Phonemes("kˈOkəɹO".to_string()),
                Segment::Normal(" model...".to_string()),
            ]
        );
    }

    #[test]
    fn test_percent_decode() {
        assert_eq!(percent_decode("hello%20world"), "hello world");
        assert_eq!(percent_decode("%CB%88"), "ˈ");
        assert_eq!(percent_decode("%C9%99"), "ə");
        assert_eq!(percent_decode("%C9%B9"), "ɹ");
    }

    #[test]
    fn test_g2p_text_to_phonemes() {
        let text = "The [Kokoro](/kˈOkəɹO/) model is fast.";
        let phonemes = g2p_text_to_phonemes(text).unwrap();
        assert!(phonemes.contains("kˈOkəɹO"));
        assert!(phonemes.contains("ðə"));
        assert!(phonemes.contains("fˈæst"));
    }

    #[test]
    fn test_cli_dry_run_interactions() {
        use crate::cli::{SpeakArgs, handle_speak};

        let args = SpeakArgs {
            text: "The [Kokoro](/kˈOkəɹO/) model is fast.".to_string(),
            model_dir: None,
            voice: "0".to_string(),
            speed: 1.0,
            out: "output.wav".to_string(),
            phonemes: false,
            verbose: false,
            play: false,
            dry_run: true,
        };
        let result = handle_speak(&args);
        assert!(result.is_ok());

        let args_raw = SpeakArgs {
            text: "kˈOkəɹO".to_string(),
            model_dir: None,
            voice: "0".to_string(),
            speed: 1.0,
            out: "output.wav".to_string(),
            phonemes: true,
            verbose: false,
            play: false,
            dry_run: true,
        };
        let result_raw = handle_speak(&args_raw);
        assert!(result_raw.is_ok());
    }
}
