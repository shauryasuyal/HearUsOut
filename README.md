# HearUsOut

**An AI-Powered Multi Speaker Voice Separation & Targeted Extraction System**

HearUsOut is a state-of-the-art speech separation system capable of untangling up to 5 concurrent speakers from a single audio mixture. It features a completely decoupled architecture with a stunning vanilla frontend and a heavy-duty PyTorch backend powered by a fine-tuned `SR_CorrNet` model.

Website Link: https://hearusout.netlify.app/

HuggingFace Link: https://huggingface.co/spaces/shauryasuyal/HearUsOut_AIMS

Video Demo: https://drive.google.com/file/d/1K5Eu96mO2H1naJfL6k6zxleB4FN-voIN/view?usp=sharing

---

## Key Features & Novelties

### Native AI Auto-Detect (No External VAD Required)
Instead of relying on clunky, heavily-gated external Voice Activity Detection (VAD) models like Pyannote, HearUsOut leverages the native `is_var_spks=True` property of the `SR_CorrNet_SS` model. By forcing a maximum 5-channel separation and dynamically computing RMS energy thresholding on the output tracks, the system flawlessly filters out silent tracks and auto-detects the true number of active speakers.

### Zero-Shot Targeted Voice Extraction
Instead of just separating everyone blindly, the system integrates a state-of-the-art Speaker Verification network (`ECAPA-TDNN`). By uploading a short reference clip of one speaker, the system generates dense feature embeddings, compares cosine similarity against the separated tracks, and automatically extracts **only** that specific person. This is one of the novelties of our project.

### Heuristic Adaptive Preprocessing Engine
Before audio even hits the separation model, our engine automatically detects and fixes corrupted inputs:
- **Declipping:** Applies a soft-clip reconstruction heuristic (`tanh`) to smoothly roll off digital clipping.
- **Loudness Normalization:** Applies integrated LUFS normalization to seamlessly scale broadcast loudness to exactly -23 LUFS.
- **Silence Trimming:** Trims dead-air via decibel-thresholding to prevent the model from hallucinating artifacts.

### Dynamic Mixing Fine-Tuning (Zero-Storage RAM Mixing)
The underlying `SR_CorrNet` checkpoint was fine-tuned onto Libri5Mix using a custom dynamic mixing pipeline. Only clean, isolated speech is stored locally (~90% storage savings). Mixtures are summed live in RAM right before hitting the GPU, ensuring the model sees a fresh, never-repeated mixture every single training step with randomized gain, scaling, and cropping.

---

## Architecture

The system is intentionally decoupled to allow heavy GPU computation to run on free cloud hardware while keeping the frontend lightning fast.

1. **Frontend (`/static/`):** A premium, glassmorphism UI built with 100% Vanilla HTML, CSS, and JS. Deploys seamlessly to Netlify or Vercel. 
2. **Backend (`hf_app.py`):** A Python API built on Gradio & PyTorch. Designed specifically to run on Hugging Face Spaces (ZeroGPU) with intelligent GPU memory management.

---

## Quick Start / Deployment

### 1. Deploying the Backend (Hugging Face)
1. Create a new **Gradio** Space on Hugging Face.
2. Upload the following files to the Space:
   - `hf_app.py` (Rename this to `app.py` upon upload)
   - `requirements.txt`
   - `model_config.yaml`
   - `finetuned_model.pt` (Your fine-tuned weights)
3. The Space will automatically build the Docker image and start the API.

### 2. Deploying the Frontend (Netlify / Vercel)
1. Open `static/app.js` and update line 173 to point to your new Hugging Face Space:
   ```javascript
   const client = await Client.connect("your-username/your-space-name");
   ```
2. Simply deploy the `static` folder to Netlify, Vercel, or GitHub Pages.


