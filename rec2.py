import numpy as np
import torch
import librosa
import soundfile as sf

# ─────────────────────────────────────────
# Load and check shapes at each step
# ─────────────────────────────────────────
speaker  = "s9"
filename = "bbaf2n.npy"
base_dir = "/kaggle/input/your-dataset/grid"

# Load inputs
av    = np.load(f"{base_dir}/{speaker}/concatenated_features/{filename}")
mixed = np.load(f"{base_dir}/{speaker}/audio_preprocessed/{filename}")

print(f"av shape:    {av.shape}")     # (time_frames, 463)
print(f"mixed shape: {mixed.shape}")  # (time_frames, 257)

# Run model
av_tensor = torch.tensor(av, dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    pred_iam = model(av_tensor).squeeze(0).cpu().numpy()

print(f"pred_iam shape: {pred_iam.shape}")       # (time_frames, 257)
print(f"pred_iam min:   {pred_iam.min():.4f}")
print(f"pred_iam max:   {pred_iam.max():.4f}")
print(f"pred_iam mean:  {pred_iam.mean():.4f}")

# Apply IAM to mixed
pred_clean = pred_iam * mixed
print(f"\npred_clean min:  {pred_clean.min():.4f}")
print(f"pred_clean max:  {pred_clean.max():.4f}")
print(f"pred_clean mean: {pred_clean.mean():.4f}")

# Undo power law compression
pred_clean_spec = np.power(np.abs(pred_clean), 1/0.3)
print(f"\nafter undo compression min:  {pred_clean_spec.min():.4f}")
print(f"after undo compression max:  {pred_clean_spec.max():.4f}")
print(f"after undo compression mean: {pred_clean_spec.mean():.4f}")