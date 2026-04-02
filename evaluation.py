# Install required packages
# !pip install pesq pystoi mir_eval

import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import os
from pesq import pesq
from mir_eval.separation import bss_eval_sources

def match_rms(reference, signal):
    ref_rms = np.sqrt(np.mean(reference ** 2) + 1e-8)
    sig_rms = np.sqrt(np.mean(signal ** 2) + 1e-8)
    return signal * (ref_rms / sig_rms)

# ─────────────────────────────────────────
# Model Definition
# ─────────────────────────────────────────
class AVConcatBLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=250, num_layers=3, freq_bins=257):
        super(AVConcatBLSTM, self).__init__()
        self.blstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc      = nn.Linear(hidden_size * 2, freq_bins)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.blstm(x)
        out    = self.fc(out)
        out    = self.sigmoid(out) * 10
        return out

# ─────────────────────────────────────────
# Reconstruct waveform from a single file
# ─────────────────────────────────────────
def reconstruct_waveform(speaker, filename, base_dir, model, device, stats_cache):
    # Load stats (cache to avoid reloading for every file)
    if speaker not in stats_cache:
        stats_path = f"{base_dir}/{speaker}/norm_stats.npy"
        stats      = np.load(stats_path, allow_pickle=True).item()
        stats_cache[speaker] = stats
    mean = stats_cache[speaker]["mean"]
    std  = stats_cache[speaker]["std"]

    # Load inputs
    av    = np.load(f"{base_dir}/{speaker}/concatenated_features/{filename}.npy")
    mixed = np.load(f"{base_dir}/{speaker}/audio_preprocessed/{filename}.npy")

    # Run model
    av_tensor = torch.tensor(av, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_iam = model(av_tensor).squeeze(0).cpu().numpy()

    # Undo normalization
    mixed_unnorm = (mixed * std) + mean

    # Undo power law
    mixed_mag = np.power(np.abs(mixed_unnorm), 1/0.3)

    # Apply IAM
    pred_clean   = pred_iam * mixed_mag       # (time_frames, 257)
    pred_clean_T = pred_clean.T               # (257, time_frames)

    # Get phase from raw mixed wav
    raw_wav_path  = f"{base_dir}/{speaker}/audio_mixed/{filename}.wav"
    mixed_wav, _  = librosa.load(raw_wav_path, sr=16000)
    stft_mixed    = librosa.stft(mixed_wav, n_fft=512, hop_length=160, win_length=400)
    phase         = np.angle(stft_mixed)

    # Trim to same length
    min_len      = min(pred_clean_T.shape[1], phase.shape[1])
    pred_clean_T = pred_clean_T[:, :min_len]
    phase        = phase[:, :min_len]

    # Reconstruct
    reconstructed = pred_clean_T * np.exp(1j * phase)
    enhanced_wav  = librosa.istft(reconstructed, hop_length=160, win_length=400)

    return enhanced_wav, mixed_wav

def load_clean_wav(speaker, filename, base_dir):
    clean_path = f"{base_dir}/{speaker}/audio/{filename}.wav"
    clean_wav, _ = librosa.load(clean_path, sr=16000)
    return clean_wav

# ─────────────────────────────────────────
# Compute metrics
# ─────────────────────────────────────────
def compute_sdr(clean, enhanced):
    # Trim to same length
    min_len   = min(len(clean), len(enhanced))
    clean     = clean[:min_len]
    enhanced  = enhanced[:min_len]
    sdr, _, _, _ = bss_eval_sources(
        clean[np.newaxis, :],
        enhanced[np.newaxis, :]
    )
    return sdr[0]

def compute_pesq_score(clean, enhanced, sr=16000):
    min_len  = min(len(clean), len(enhanced))
    clean    = clean[:min_len]
    enhanced = enhanced[:min_len]
    mode = 'wb' if sr == 16000 else 'nb'
    score = pesq(sr, clean, enhanced, mode)
    return score

def compute_sdr_noisy(clean, mixed):
    min_len = min(len(clean), len(mixed))
    clean   = clean[:min_len]
    mixed   = mixed[:min_len]
    sdr, _, _, _ = bss_eval_sources(
        clean[np.newaxis, :],
        mixed[np.newaxis, :]
    )
    return sdr[0]

# ─────────────────────────────────────────
# Evaluate on test speakers
# ─────────────────────────────────────────
def evaluate(test_speakers, base_dir, model, device):
    stats_cache = {}

    sdr_noisy_list    = []
    sdr_enhanced_list = []
    pesq_noisy_list   = []
    pesq_enhanced_list= []

    for speaker in test_speakers:
        av_dir = f"{base_dir}/{speaker}/concatenated_features"
        files  = sorted([f.replace(".npy", "") for f in os.listdir(av_dir) if f.endswith(".npy")])

        print(f"\nEvaluating speaker: {speaker} ({len(files)} files)")

        for filename in files:
            try:
                # Reconstruct enhanced waveform
                enhanced_wav, mixed_wav = reconstruct_waveform(
                    speaker, filename, base_dir, model, device, stats_cache
                )

                # Load clean reference
                clean_wav = load_clean_wav(speaker, filename, base_dir)

                # Trim all to same length
                min_len      = min(len(clean_wav), len(enhanced_wav), len(mixed_wav))
                clean_wav    = clean_wav[:min_len]
                enhanced_wav = enhanced_wav[:min_len]
                mixed_wav    = mixed_wav[:min_len]
                enhanced_wav = match_rms(clean_wav, enhanced_wav)
                mixed_wav    = match_rms(clean_wav, mixed_wav)

                # Compute metrics
                sdr_n  = compute_sdr_noisy(clean_wav, mixed_wav)
                sdr_e  = compute_sdr(clean_wav, enhanced_wav)
                pesq_n = compute_pesq_score(clean_wav, mixed_wav)
                pesq_e = compute_pesq_score(clean_wav, enhanced_wav)

                sdr_noisy_list.append(sdr_n)
                sdr_enhanced_list.append(sdr_e)
                pesq_noisy_list.append(pesq_n)
                pesq_enhanced_list.append(pesq_e)

            except Exception as e:
                print(f"  Skipping {filename}: {e}")
                continue

    # ── Print results ──
    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"{'Metric':<15} {'Noisy':>10} {'Enhanced':>10}")
    print(f"{'-'*35}")
    print(f"{'SDR':<15} {np.mean(sdr_noisy_list):>10.2f} {np.mean(sdr_enhanced_list):>10.2f}")
    print(f"{'PESQ':<15} {np.mean(pesq_noisy_list):>10.2f} {np.mean(pesq_enhanced_list):>10.2f}")
    print(f"{'='*50}")

    return {
        "sdr_noisy":     np.mean(sdr_noisy_list),
        "sdr_enhanced":  np.mean(sdr_enhanced_list),
        "pesq_noisy":    np.mean(pesq_noisy_list),
        "pesq_enhanced": np.mean(pesq_enhanced_list),
    }

# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────
base_dir = "/Users/hrishikeshbingewar/Downloads/av_project/GRID"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = AVConcatBLSTM(input_size=463, hidden_size=250, num_layers=3, freq_bins=257)
model.load_state_dict(torch.load(
    "/Users/hrishikeshbingewar/Downloads/best_av_concat_model (1).pth",
    map_location=device
))
model.eval()
model.to(device)
print("Model loaded!")

results = evaluate(
    test_speakers=["s9", "s10"],  # your test speakers
    base_dir=base_dir,
    model=model,
    device=device
)
