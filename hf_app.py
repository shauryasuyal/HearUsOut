import os
import uuid
import torch
import torchaudio
import gradio as gr
import spaces
import soundfile as sf
import numpy as np
import pyloudnorm as pyln
import torch.nn.functional as F
from sr_corrnet import SSInference

os.makedirs("pretrained_models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("Loading finetuned SR-CorrNet model on CPU...")
try:
    sep_model = SSInference.from_pretrained(
        config="model_config.yaml",
        checkpoint_path="finetuned_model.pt",
        device="cpu"
    )
except Exception as e:
    print(f"Error loading SR-CorrNet (expected if dependencies aren't installed locally): {e}")

# Pyannote removed in favor of native SR-CorrNet RMS energy auto-detect

DF_AVAILABLE = False
try:
    from df.enhance import enhance, init_df, load_audio
    df_model, df_state, _ = init_df()
    DF_AVAILABLE = True
    print("DeepFilterNet loaded!")
except Exception as e:
    print(f"DeepFilterNet not available: {e}")

ECAPA_AVAILABLE = False
try:
    from speechbrain.inference.speaker import EncoderClassifier
    speaker_verifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="pretrained_models/spkrec",
        run_opts={"device": "cpu"}
    )
    ECAPA_AVAILABLE = True
    print("ECAPA-TDNN loaded!")
except Exception as e:
    print(f"ECAPA-TDNN not available: {e}")

def save_wav(path, tensor, sr=8000):
    audio = tensor.squeeze().detach().cpu().numpy()
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = (audio / peak) * 0.95
    sf.write(path, audio, sr)

def adaptive_preprocess(audio_path):
    try:
        y, sr = torchaudio.load(audio_path)
    except Exception:
        import librosa
        y_np, sr = librosa.load(audio_path, sr=None, mono=False)
        y = torch.from_numpy(y_np if y_np.ndim == 2 else y_np[np.newaxis, :])

    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
    if sr != 8000:
        y = torchaudio.transforms.Resample(sr, 8000)(y)

    y_np = y.squeeze().numpy()
    
    # 1. Declipping (soft clip reconstruction heuristic)
    peak = np.max(np.abs(y_np))
    if peak >= 0.99:
        y_np = np.tanh(y_np)

    # 2. Loudness Normalization (LUFS)
    meter = pyln.Meter(8000)
    try:
        loudness = meter.integrated_loudness(y_np)
        if loudness < -30 or loudness > -14:
            y_np = pyln.normalize.loudness(y_np, loudness, -23.0)
    except Exception:
        y_np = y_np / (np.max(np.abs(y_np)) + 1e-9) * 0.95

    # 3. Silence Trimming
    try:
        import librosa
        non_silent = librosa.effects.split(y_np, top_db=40)
        if len(non_silent) > 0:
            start = non_silent[0][0]
            end = non_silent[-1][1]
            y_np = y_np[start:end]
    except Exception:
        pass

    out_path = f"outputs/_pre_{uuid.uuid4().hex[:6]}.wav"
    sf.write(out_path, y_np, 8000)
    return out_path, y_np

def get_speaker_embedding(wav_path):
    if not ECAPA_AVAILABLE:
        return None
    signal, fs = torchaudio.load(wav_path)
    if fs != 16000:
        signal = torchaudio.transforms.Resample(fs, 16000)(signal)
    with torch.no_grad():
        embeddings = speaker_verifier.encode_batch(signal)
    return embeddings.squeeze()

@spaces.GPU(duration=60)
def process_audio(audio_path, requested_speakers, target_voice_path=None):
    try:
        if not audio_path:
            return "Please upload an audio file.", None, None, None, None, None
            
        pre_path, y_np = adaptive_preprocess(audio_path)
        waveform = torch.from_numpy(y_np).unsqueeze(0).float() # (1, T)
        
        auto_detect = requested_speakers == "Auto-detect"
        num_speakers = 5 if auto_detect else int(requested_speakers)
        
        msg = "Auto-detecting speakers..." if auto_detect else f"Using {num_speakers} speakers."
        
        job_dir = f"outputs/{uuid.uuid4().hex[:8]}"
        os.makedirs(job_dir, exist_ok=True)
        
        # Move internal components to CUDA
        sep_model.engine.model.to("cuda")
        sep_model.engine.device = torch.device("cuda")
        try:
            sep_model.engine.stft.to("cuda")
            sep_model.engine.istft.to("cuda")
        except:
            pass
            
        waveform = waveform.to("cuda")
        with torch.no_grad():
            res = sep_model.process_waveform(waveform, n_spks=torch.tensor(num_speakers, device="cuda"))
        
        # Filter active speakers based on energy
        files = []
        energies = []
        for spk_wav in res['waveforms']:
            audio = spk_wav.squeeze().detach().cpu().numpy()
            energy = np.mean(audio**2)
            energies.append(audio)
            
        if auto_detect:
            # Find the loudest track's energy
            energies_vals = [np.mean(a**2) for a in energies]
            max_energy = max(energies_vals)
            
            # 8% energy threshold (approx -11dB) to filter out model leakage/noise
            threshold = max_energy * 0.08 
            
            valid_audios = []
            # Sort from loudest to quietest
            sorted_indices = np.argsort([-e for e in energies_vals])
            
            for idx in sorted_indices:
                a = energies[idx]
                e = energies_vals[idx]
                # Always keep at least the top 2 loudest tracks, then apply threshold for the rest
                if len(valid_audios) < 2 or e > threshold:
                    valid_audios.append(a)
                    
            msg = f"Auto-detected {len(valid_audios)} active speakers."
        else:
            valid_audios = energies
            
        for i, audio in enumerate(valid_audios):
            p = os.path.join(job_dir, f"spk_{i+1}.wav")
            # Peak normalize
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = (audio / peak) * 0.95
            sf.write(p, audio, 8000)
            files.append(p)

        # --- FEATURE 2: GENERATIVE REFINEMENT (DeepFilterNet) ---
        if DF_AVAILABLE:
            msg += " | ✨ Enhanced with DeepFilterNet"
            refined_files = []
            for f in files:
                # DF requires 48kHz usually, load_audio handles resampling if needed
                audio_df, sr_df = load_audio(f, sr=df_state.sr())
                with torch.no_grad():
                    enhanced = enhance(df_model, df_state, audio_df)
                refined_path = f.replace(".wav", "_refined.wav")
                torchaudio.save(refined_path, enhanced.cpu(), df_state.sr())
                refined_files.append(refined_path)
            files = refined_files

        # --- FEATURE 1: TARGETED VOICE EXTRACTION ---
        if target_voice_path and ECAPA_AVAILABLE:
            target_emb = get_speaker_embedding(target_voice_path)
            if target_emb is not None:
                best_sim = -1.0
                best_file = None
                
                for f in files:
                    track_emb = get_speaker_embedding(f)
                    if track_emb is not None:
                        sim = F.cosine_similarity(target_emb.unsqueeze(0), track_emb.unsqueeze(0)).item()
                        if sim > best_sim:
                            best_sim = sim
                            best_file = f
                
                if best_file:
                    msg = f"Target Voice Matched! (Similarity: {best_sim*100:.1f}%)" + msg
                    files = [best_file] # Discard the rest

        outputs = files + [None] * (5 - len(files))
        return msg, outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]

    except Exception as e:
        import traceback
        return f"ERROR: {str(e)}\n\n{traceback.format_exc()}", None, None, None, None, None

with gr.Blocks(title="🎙️ HearUsOut V2: Multi-Speaker Separator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ HearUsOut V2: Multi-Speaker Separator")
    with gr.Row():
        audio_in = gr.Audio(type="filepath", label="Upload Mixed Audio (WAV/MP3/FLAC)")
        target_voice_in = gr.Audio(type="filepath", label="Target Voice Sample (Optional)")
        speaker_count = gr.Radio(["Auto-detect", "2", "3", "4", "5"], value="Auto-detect", label="Number of Speakers")
    
    btn = gr.Button("Separate Voices", variant="primary")
    status = gr.Textbox(label="Status")
    with gr.Row():
        out1 = gr.Audio(label="Speaker 1")
        out2 = gr.Audio(label="Speaker 2")
        out3 = gr.Audio(label="Speaker 3")
        out4 = gr.Audio(label="Speaker 4")
        out5 = gr.Audio(label="Speaker 5")
    
    btn.click(
        fn=process_audio,
        inputs=[audio_in, speaker_count, target_voice_in],
        outputs=[status, out1, out2, out3, out4, out5],
        api_name="predict"
    )

if __name__ == "__main__":
    demo.queue().launch()
