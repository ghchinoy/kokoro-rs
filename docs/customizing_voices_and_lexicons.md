# Customizing Kokoro: Voices, Training, and Lexicons

Kokoro is an incredibly efficient 82M parameter text-to-speech model built on the StyleTTS2 architecture. Because its weights are completely open (Apache 2.0), the community has developed several ways to customize voices and pronunciation.

This guide outlines how to modify lexicons, blend voices, and approach training.

---

## 1. Modifying Lexicons & Pronunciation

If Kokoro mispronounces a word (like a unique name or industry jargon), you have a few ways to fix it depending on how you are running the model.

### Modifying Lexicons in the Go CLI (Sherpa-ONNX)
Because this Go CLI wraps the `k2-fsa/sherpa-onnx` C++ engine, the Grapheme-to-Phoneme (G2P) process is handled via static text files rather than Python logic.

If you need to permanently fix a pronunciation:
1. Navigate to your XDG model directory: `~/.local/share/kokoro/models/v1.0/`
2. Open the relevant lexicon file (e.g., `lexicon-us-en.txt`).
3. Add a new line with the word and its phonetic spelling using the Kokoro/espeak-ng phonetic alphabet.
   * Format: `word w ˈ ɜ ɹ d`
4. The next time you run the CLI, our `getMergedLexicon` function will pick up the change.

### Lexicons in the Official Python Pipeline
If you are using the official Python implementation (which uses the `Misaki` G2P library), you have more dynamic options:
* **Inline Overrides:** You can specify the exact IPA-like pronunciation of a word directly in your text string using the syntax: `[word](/phonemes/)`.
  * *Example:* `Hello [Kokoro](/kˈOkəɹO/)!`
* **Custom Lexicon JSON:** Misaki allows loading custom JSON dictionaries into the pipeline to override default system (`espeak-ng`) fallbacks.

---

## 2. Voice Cloning & "New" Voices

The core official Kokoro repository **does not** support zero-shot voice cloning (e.g., uploading a 3-second `.wav` file to instantly generate a matching voice). Instead, it relies on pre-computed **Voice Tensors**.

However, you can create "new" voices using the following methods:

### Voice Blending (The Official Method)
Because voices are just 512-dimensional mathematical embeddings (tensors), you can create unique hybrid voices by averaging or interpolating them together.
* **How it works (in Python):** You can load two `.pt` voice files and use `torch.lerp` (linear interpolation) to mix them. For example, mixing 70% of `af_heart` with 30% of `af_bella` generates a completely new sounding voice that is a blend of both characteristics.
* *Note:* To use blended voices in this Go CLI, you would need to export the resulting blended tensor back into the `voices.bin` binary format expected by `sherpa-onnx`.

### Third-Party Cloning Tools
The open-source community has built extensions on top of Kokoro to enable zero-shot cloning:
* **KokoClone:** A popular community project that adds zero-shot cloning capabilities by integrating a "Kanade" model, allowing you to upload reference audio to extract a new voice tensor.

---

## 3. Training & Fine-Tuning

While the official repository is geared toward inference, you can train or fine-tune Kokoro yourself because the underlying architecture is public.

* **Architecture:** Kokoro is based on the **StyleTTS2** architecture.
* **Efficiency:** The original model was famously trained on less than 100 hours of high-quality audio for approximately $1,000. It is highly efficient to fine-tune if you have the technical expertise.
* **Process:** Fine-tuning requires a clean dataset of audio (typically 1–10 hours for a highly accurate custom voice) and perfectly matching transcriptions. You would utilize the standard StyleTTS2 training framework to update the model weights.

---

## 4. References & Sources

The information in this guide is aggregated from official documentation and community developments:
* **Hugging Face Official Model Card:** [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Details on voice blending and the Misaki G2P library).
* **Misaki G2P Repository:** [hexgrad/misaki on GitHub](https://github.com/hexgrad/misaki) (The Grapheme-to-Phoneme converter library and custom phonetic mappings used by Kokoro).
* **KokoClone Repository:** [KokoClone on GitHub](https://github.com/KokoClone) (Community zero-shot voice cloning).
* **DigitalOcean Tutorials:** [Introduction to Kokoro-82M](https://www.digitalocean.com/community/tutorials/kokoro-82m-text-to-speech) (Explaining voice tensors and architecture).
* **k2-fsa/sherpa-onnx:** [Sherpa-ONNX Documentation](https://k2-fsa.github.io/sherpa/) (Understanding the C++ implementation of lexicons).
* **Community Guide:** [Training Kokoro (semidark/kokoro-deutsch)](https://github.com/semidark/kokoro-deutsch/discussions/8) (Recent insights and reports on training custom models).