# Kokoro Prompting Guide

Kokoro is a lightweight (82M parameter) Text-to-Speech model. Because it is highly optimized for speed and naturalness, it lacks some of the heavy orchestration layers found in commercial APIs. 

This guide explains how to effectively format your text to get the best audio out of Kokoro.

---

## 1. Pacing & Pauses (Punctuation is King)

Kokoro does **not** natively support SSML tags like `<break time="2s"/>`. Instead, the model's pacing is inferred entirely from standard punctuation.

* **Commas (`,`):** Creates a short, natural breath pause.
* **Periods (`.`) and Semicolons (`;`):** Creates longer, sentence-ending pauses with a natural pitch drop.
* **Ellipses (`...`):** Creates a trailing, thoughtful pause. 
  * *Warning:* Overusing dots (e.g., `......`) can confuse the model and cause it to generate hallucinated whispering or breathy static.
* **Quotation Marks (`" "`):** Wrapping dialogue in quotes forces a slightly sharper boundary and pitch reset, making the speech sound more deliberate.

### Example:
```text
# Rushed and monotonous
The quick brown fox jumped over the lazy dog and then it ran away into the forest.

# Better pacing with commas
The quick brown fox, jumped over the lazy dog, and then, it ran away into the forest.
```

---

## 2. Pronunciation & Lexicons

Kokoro uses **Grapheme-to-Phoneme (G2P)** processing. When you pass it text, it looks up words in a massive dictionary (the `lexicon-*.txt` files) to find their exact phonetic spelling.

### The Kokoro Phonetic System
The phonetic system used by Kokoro is **IPA-like**, but not standard IPA or X-SAMPA. It relies heavily on `espeak-ng` roots with specific proprietary mappings.
* `A` represents the `/eɪ/` sound (as in "g**a**te").
* `T` represents the American alveolar tap `/ɾ/` (as in "wa**t**er").
* The primary stress mark (`ˈ`) is placed **immediately before the vowel**, not at the beginning of the syllable.

If a word is pronounced incorrectly, Kokoro's engine either doesn't have it in the lexicon or its rules-based fallback is guessing incorrectly. 

*(Note: While the official Python pipeline allows forcing pronunciations using a Markdown syntax like `[Kokoro](/kˈOkəɹO/)`, this CLI wrapper currently processes raw strings directly to the C++ engine. If you need custom pronunciations, you must spell the word out phonetically in plain English, e.g., "Coe core oh").*

---

## 3. Emotion, Tone, and Audio Tags

Kokoro is trained on clean, high-quality speech data. It is **not** trained on conversational acoustic events or paralinguistic sounds.

### What You Cannot Do
* **No Audio Tags:** You cannot use tags like `[laugh]`, `[sigh]`, `[whisper]`, or `[cough]`. The model will likely read the literal brackets.
* **No Forced Laughter:** Attempting to force a laugh phonetically (e.g., `"ha ha ha"`) will sound robotic, as the model reads the literal syllables.
* **No Emotion Tags:** You cannot pass a tag like `(angry)` or `(sad)`.

### How to Change Emotion
Emotion and tone are entirely driven by the **Voice ID** you select. 

For example, in the English (US) voices:
* `af_bella` (Voice ID: 2) is trained to sound energetic, warm, and expressive.
* `af_nicole` (Voice ID: 6) is trained to sound muted, clear, and professional (like an audiobook narrator).

If your script requires a shift from calm to excited, you should split your text and generate two separate audio files using different Voice IDs.

---

## 4. References & Sources

The information in this guide is derived from the following sources and community discussions regarding Kokoro's capabilities:
* [Kokoro TTS Official Model Card (hexgrad/Kokoro-82M)](https://huggingface.co/hexgrad/Kokoro-82M)
* [hexgrad/misaki on GitHub](https://github.com/hexgrad/misaki) (The Grapheme-to-Phoneme converter library and custom phonetic mappings used by Kokoro)
* [Kokoro Web Interface and Examples](https://kokoroweb.app/)
* [Voxta AI (Handling Stage Cues and Integrations)](https://voxta.ai/)
* General Community Discussions on Reddit (e.g., r/LocalLLaMA) regarding Kokoro's phonetic system and pause handling.
