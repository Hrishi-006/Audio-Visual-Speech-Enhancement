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
# Paths
# ─────────────────────────────────────────
base_dir    = "/Users/hrishikeshbingewar/Downloads/av_project/GRID"
speaker     = "s10"           # use val/test speaker
filename    = "lrbb4p"       # without extension
model_path  = '/Users/hrishikeshbingewar/Downloads/best_av_concat_model (1).pth'

av_path       = f"{base_dir}/{speaker}/concatenated_features/{filename}.npy"
mixed_npy     = f"{base_dir}/{speaker}/audio_preprocessed/{filename}.npy"
raw_wav_path  = f"{base_dir}/{speaker}/audio_mixed/{filename}.wav"
stats_path    = f"{base_dir}/{speaker}/norm_stats.npy"

# ─────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = AVConcatBLSTM(input_size=463, hidden_size=250, num_layers=3, freq_bins=257)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)
print("Model loaded!")

# ─────────────────────────────────────────
# Load norm stats
# ─────────────────────────────────────────
stats = np.load(stats_path, allow_pickle=True).item()
mean  = stats["mean"]   # (257,)
std   = stats["std"]    # (257,)
print(f"Stats loaded — mean: {mean.mean():.4f} std: {std.mean():.4f}")

# ─────────────────────────────────────────
# Load inputs
# ─────────────────────────────────────────
av    = np.load(av_path)     # (time_frames, 463)
mixed = np.load(mixed_npy)   # (time_frames, 257) normalized

print(f"av shape:    {av.shape}")
print(f"mixed shape: {mixed.shape}")

# ─────────────────────────────────────────
# Run model inference
# ─────────────────────────────────────────
av_tensor = torch.tensor(av, dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    pred_iam = model(av_tensor).squeeze(0).cpu().numpy()  # (time_frames, 257)

print(f"pred_iam shape: {pred_iam.shape}")
print(f"pred_iam min:   {pred_iam.min():.4f}")
print(f"pred_iam max:   {pred_iam.max():.4f}")
print(f"pred_iam mean:  {pred_iam.mean():.4f}")

# ─────────────────────────────────────────
# Reconstruct spectrogram
# ─────────────────────────────────────────
# Step 1: Undo normalization
mixed_unnorm = (mixed * std) + mean              # (time_frames, 257)

# Step 2: Undo power law compression
mixed_mag    = np.power(np.abs(mixed_unnorm), 1/0.3)  # (time_frames, 257)

# Step 3: Apply predicted IAM to get clean magnitude
pred_clean   = pred_iam * mixed_mag              # (time_frames, 257)

# Step 4: Transpose to (freq_bins, time_frames) for istft
pred_clean   = pred_clean.T                      # (257, time_frames)

print(f"\npred_clean min:  {pred_clean.min():.4f}")
print(f"pred_clean max:  {pred_clean.max():.4f}")
print(f"pred_clean mean: {pred_clean.mean():.4f}")

# ─────────────────────────────────────────
# Get phase from raw mixed wav
# ─────────────────────────────────────────
mixed_wav, sr = librosa.load(raw_wav_path, sr=16000)
stft_mixed    = librosa.stft(mixed_wav, n_fft=512, hop_length=160, win_length=400)
phase         = np.angle(stft_mixed)             # (257, time_frames)

print(f"\nphase shape:      {phase.shape}")
print(f"pred_clean shape: {pred_clean.shape}")

# ─────────────────────────────────────────
# Trim to same length and reconstruct
# ─────────────────────────────────────────
min_len     = min(pred_clean.shape[1], phase.shape[1])
pred_clean  = pred_clean[:, :min_len]
phase       = phase[:, :min_len]

# Combine magnitude with phase
reconstructed = pred_clean * np.exp(1j * phase)

# Inverse STFT
waveform = librosa.istft(reconstructed, hop_length=160, win_length=400)
output_dir = "/Users/hrishikeshbingewar/Downloads/av_project"
os.makedirs(output_dir, exist_ok=True)

# ─────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────
sf.write("/Users/hrishikeshbingewar/Downloads/av_project/enhanced_speech.wav", waveform,   16000)
sf.write("/Users/hrishikeshbingewar/Downloads/av_project/mixed_speech.wav",    mixed_wav,  16000)
print("\nSaved:")
print("  /kaggle/working/enhanced_speech.wav")
print("  /kaggle/working/mixed_speech.wav")
