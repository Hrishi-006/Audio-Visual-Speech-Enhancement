import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import os

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
# Load Model
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = AVConcatBLSTM(input_size=463, hidden_size=250, num_layers=3, freq_bins=257)
model.load_state_dict(torch.load(
    "/Users/hrishikeshbingewar/Downloads/best_av_concat_model_2.pth",
    map_location=device
))
model.eval()
model.to(device)
print("Model loaded!")

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
base_dir = "/Users/hrishikeshbingewar/Downloads/av_project/GRID"
speaker  = "s1"        # use training speaker first to verify
filename = "bbaf2n"    # without extension

# ─────────────────────────────────────────
# Debug each step
# ─────────────────────────────────────────
# Step 1: Check raw wav
raw_wav_path  = f"{base_dir}/{speaker}/audio_mixed/{filename}.wav"
mixed_wav, sr = librosa.load(raw_wav_path, sr=16000)
print(f"Step 1 - raw wav:        min={mixed_wav.min():.4f} max={mixed_wav.max():.4f} mean={mixed_wav.mean():.4f}")
sf.write("/Users/hrishikeshbingewar/Downloads/av_project/test_raw.wav", mixed_wav, 16000)

# Step 2: Check normalized mixed spec
mixed = np.load(f"{base_dir}/{speaker}/audio_preprocessed/{filename}.npy")
print(f"Step 2 - normalized:     min={mixed.min():.4f} max={mixed.max():.4f} mean={mixed.mean():.4f}")

# Step 3: Check norm stats
stats = np.load(f"{base_dir}/{speaker}/norm_stats.npy", allow_pickle=True).item()
mean  = stats["mean"]
std   = stats["std"]
print(f"Step 3 - stats:          mean={mean.mean():.4f} std={std.mean():.4f}")

# Step 4: Undo normalization
mixed_unnorm = (mixed * std) + mean
print(f"Step 4 - unnormalized:   min={mixed_unnorm.min():.4f} max={mixed_unnorm.max():.4f} mean={mixed_unnorm.mean():.4f}")

# Step 5: Undo power law
mixed_mag = np.power(np.abs(mixed_unnorm), 1/0.3)
print(f"Step 5 - magnitude:      min={mixed_mag.min():.4f} max={mixed_mag.max():.4f} mean={mixed_mag.mean():.4f}")

# Step 6: Check pred_iam from model
av        = np.load(f"{base_dir}/{speaker}/concatenated_features/{filename}.npy")
av_tensor = torch.tensor(av, dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    pred_iam = model(av_tensor).squeeze(0).cpu().numpy()
print(f"Step 6 - pred_iam:       min={pred_iam.min():.4f} max={pred_iam.max():.4f} mean={pred_iam.mean():.4f}")

# Step 7: Apply IAM to magnitude
pred_clean = pred_iam * mixed_mag
print(f"Step 7 - pred_clean:     min={pred_clean.min():.4f} max={pred_clean.max():.4f} mean={pred_clean.mean():.4f}")

# Step 8: Get phase from raw wav and reconstruct
stft_mixed   = librosa.stft(mixed_wav, n_fft=512, hop_length=160, win_length=400)
phase        = np.angle(stft_mixed)
pred_clean_T = pred_clean.T                        # (257, time_frames)

min_len      = min(pred_clean_T.shape[1], phase.shape[1])
pred_clean_T = pred_clean_T[:, :min_len]
phase        = phase[:, :min_len]

reconstructed = pred_clean_T * np.exp(1j * phase)
waveform      = librosa.istft(reconstructed, hop_length=160, win_length=400)
print(f"Step 8 - waveform:       min={waveform.min():.4f} max={waveform.max():.4f} mean={waveform.mean():.4f}")

# ─────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────
output_dir = "/Users/hrishikeshbingewar/Downloads/av_project/output"
os.makedirs(output_dir, exist_ok=True)

sf.write(f"{output_dir}/enhanced_speech.wav", waveform,  16000)
sf.write(f"{output_dir}/mixed_speech.wav",    mixed_wav, 16000)
print(f"\nSaved to {output_dir}")
print("Compare mixed_speech.wav vs enhanced_speech.wav")