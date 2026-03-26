import numpy as np
import librosa
import os
# Audio / STFT parameters (from the paper)
TARGET_SR = 16000          # Resample to 16 kHz
N_FFT = 512                # FFT size
WIN_LENGTH = 400           # 25 ms × 16000 Hz = 400 samples
HOP_LENGTH = 160           # 10 ms × 16000 Hz = 160 samples
POWER_LAW_P = 0.3          # Power-law compression exponent


def recompute_and_save_stats(speaker, base_dir, input_folder="audio_mixed"):
    input_dir  = os.path.join(base_dir, speaker, input_folder)
    stats_path = os.path.join(base_dir, speaker, "norm_stats.npy")

    mean, std, _ = compute_speaker_stats(input_dir)  # reuse your existing function
    np.save(stats_path, {"mean": mean, "std": std})
    print(f"Saved stats for {speaker}: mean={mean.mean():.4f} std={std.mean():.4f}")


def compute_speaker_stats(audio_dir):
    """
    First pass: compute running mean and std across ALL
    spectrograms of a single speaker for 0-mean 1-std normalization.
    Uses Welford's online algorithm to avoid loading everything into RAM.
    """
    wav_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

    if not wav_files:
        return None, None, 0

    # Online mean/variance accumulators
    n_frames_total = 0
    running_mean = None
    running_m2 = None          # sum of squared differences from mean

    for fname in wav_files:
        filepath = os.path.join(audio_dir, fname)

        # Load & resample
        y, _ = librosa.load(filepath, sr=TARGET_SR, mono=True)

        # STFT → magnitude → power-law compression
        stft = librosa.stft(
            y,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window='hann',
        )
        mag = np.abs(stft)                        # |x|
        compressed = np.power(mag, POWER_LAW_P)   # |x|^p

        # compressed shape: (n_freq_bins, n_time_frames)
        # Transpose → (n_time_frames, n_freq_bins) so each row = 1 frame
        spec = compressed.T  # (T, F)

        # Welford's online update — frame by frame across the file
        for frame in spec:
            n_frames_total += 1
            if running_mean is None:
                running_mean = np.zeros_like(frame, dtype=np.float64)
                running_m2 = np.zeros_like(frame, dtype=np.float64)
            delta = frame - running_mean
            running_mean += delta / n_frames_total
            delta2 = frame - running_mean
            running_m2 += delta * delta2

    if n_frames_total < 2:
        return running_mean, np.ones_like(running_mean), n_frames_total

    variance = running_m2 / n_frames_total         # population variance
    std = np.sqrt(variance)
    std[std < 1e-8] = 1e-8                         # guard against /0

    return running_mean, std, n_frames_total



speakers = ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"]
for speaker in speakers:
    recompute_and_save_stats(speaker, base_dir="/Users/hrishikeshbingewar/Downloads/av_project/GRID")