import os
import numpy as np
import librosa
from pesq import pesq
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import os
import scipy.ndimage
# ─────────────────────────────────────────\
model_path  = '/Users/hrishikeshbingewar/Downloads/best_av_concat_model (7).pth'
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = AVConcatBLSTM(input_size=463, hidden_size=250, num_layers=3, freq_bins=257)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)
def reconstruct_audio(spk, filename, model, device, base_dir):
    import numpy as np
    import torch
    import librosa

    # Paths
    av_path      = f"{base_dir}/{spk}/concatenated_features/{filename}.npy"
    mixed_npy    = f"{base_dir}/{spk}/audio_preprocessed/{filename}.npy"
    raw_wav_path = f"{base_dir}/{spk}/audio_mixed/{filename}.wav"
    stats_path   = f"{base_dir}/{spk}/norm_stats.npy"

    # Load data
    av    = np.load(av_path)
    mixed = np.load(mixed_npy)
    stats = np.load(stats_path, allow_pickle=True).item()
    mean, std = stats["mean"], stats["std"]

    # Model inference
    av_tensor = torch.tensor(av, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_iam = model(av_tensor).squeeze(0).cpu().numpy()

    # 🔥 mask sharpening (your improvement)
    pred_iam = pred_iam ** 1.5   # tweak 1.3–2.0 if needed

    # Undo normalization
    mixed_unnorm = (mixed * std) + mean

    # Undo power compression (safe)
    mixed_mag = np.power(np.maximum(mixed_unnorm, 1e-8), 1/0.3)

    # Apply mask
    pred_clean = pred_iam * mixed_mag
    pred_clean = pred_clean.T  # (freq, time)

    # Phase from mixture
    mixed_wav, _ = librosa.load(raw_wav_path, sr=16000)
    stft_mixed = librosa.stft(mixed_wav, n_fft=512, hop_length=160, win_length=400)
    phase = np.angle(stft_mixed)

    # Align
    min_len = min(pred_clean.shape[1], phase.shape[1])
    pred_clean = pred_clean[:, :min_len]
    phase = phase[:, :min_len]

    # Reconstruct
    reconstructed = pred_clean * np.exp(1j * phase)
    waveform = librosa.istft(reconstructed, hop_length=160, win_length=400)

    # Normalize
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)

    return waveform

# speakers
speakers = ["s31", "s32", "s33", "s34"]

base_dir = "/Users/hrishikeshbingewar/Downloads/av_project/GRID"

def compute_metrics(clean, enhanced):
    import numpy as np

    # remove NaNs
    clean = np.nan_to_num(clean)
    enhanced = np.nan_to_num(enhanced)

    # align
    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    # normalize
    clean = clean / (np.max(np.abs(clean)) + 1e-8)
    enhanced = enhanced / (np.max(np.abs(enhanced)) + 1e-8)

    try:
        pesq_score = pesq(16000, clean, enhanced, 'nb')
    except:
        pesq_score = np.nan

    try:
        sdr, _, _, _ = bss_eval_sources(clean[np.newaxis,:], enhanced[np.newaxis,:])
        sdr_score = sdr[0]
    except:
        sdr_score = np.nan

    return pesq_score, sdr_score


all_pesq = []
all_sdr = []



for spk in ["s31","s32","s33","s34"]:
    clean_dir = f"{base_dir}/{spk}/audio"

    for f in os.listdir(clean_dir):
        try:
            clean_path = os.path.join(clean_dir, f)
            clean, _ = librosa.load(clean_path, sr=16000)

            enhanced = reconstruct_audio(spk, f.replace(".wav",""), model, device, base_dir)

            pesq_score, sdr_score = compute_metrics(clean, enhanced)
            all_pesq.append(pesq_score)
            all_sdr.append(sdr_score)
            print(f"{spk} {f}: PESQ={pesq_score:.3f}, SDR={sdr_score:.2f}")

        except FileNotFoundError:
            print(f"Skipping {spk}/{f} (missing file)")
            continue


print("\nFINAL RESULTS:")
print("Avg PESQ:", np.nanmean(all_pesq))
print("Avg SDR:", np.nanmean(all_sdr))
