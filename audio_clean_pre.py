import os
import numpy as np
import librosa

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = "/Users/hrishikeshbingewar/Downloads/av_project/GRID"
SUBDIRS = ["s1", "s2", "s3"]
INPUT_FOLDER = "audio"
OUTPUT_FOLDER = "audio_clean_preprocessed"

# Audio / STFT parameters (from the paper)
TARGET_SR = 16000          # Resample to 16 kHz
N_FFT = 512                # FFT size
WIN_LENGTH = 400           # 25 ms × 16000 Hz = 400 samples
HOP_LENGTH = 160           # 10 ms × 16000 Hz = 160 samples
POWER_LAW_P = 0.3          # Power-law compression exponent

def load_mixed_stats(base_dir, speaker):
    stats_path = os.path.join(base_dir, speaker, "norm_stats.npy")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Missing mixed-audio normalization stats for {speaker}: {stats_path}. "
            "Run save_stats.py on audio_mixed first."
        )
    stats = np.load(stats_path, allow_pickle=True).item()
    return stats["mean"], stats["std"]

# ──────────────────────────────────────────────
# Per-Speaker Statistics (Pass 1)
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# Process & Normalize (Pass 2)
# ──────────────────────────────────────────────
def process_file(input_path, output_path, speaker_mean, speaker_std):
    """
    1. Load waveform, resample to 16 kHz
    2. STFT (512 FFT, 400-sample Hann window, 160-sample hop)
    3. Magnitude → power-law compression |x|^0.3
    4. Per-speaker 0-mean 1-std normalization
    5. Save as .npy
    """
    # Load & resample
    y, _ = librosa.load(input_path, sr=TARGET_SR, mono=True)

    # STFT
    stft = librosa.stft(
        y,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window='hann',
    )

    # Magnitude + power-law compression
    mag = np.abs(stft)
    compressed = np.power(mag, POWER_LAW_P)

    # Transpose to (T, F) — time-major
    spec = compressed.T  # (T, n_fft//2 + 1) = (T, 257)

    # Per-speaker normalization
    spec_norm = (spec - speaker_mean) / speaker_std

    # Save
    np.save(output_path, spec_norm.astype(np.float32))

    return spec_norm.shape


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    total_processed = 0
    total_skipped = 0

    for subdir in SUBDIRS:
        input_dir = os.path.join(BASE_DIR, subdir, INPUT_FOLDER)
        output_dir = os.path.join(BASE_DIR, subdir, OUTPUT_FOLDER)

        if not os.path.isdir(input_dir):
            print(f"[SKIP] Input directory not found: {input_dir}")
            total_skipped += 1
            continue

        wav_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.wav')])
        if not wav_files:
            print(f"[WARN] No .wav files found in: {input_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Speaker: {subdir}  ({len(wav_files)} files)")
        print(f"{'='*60}")

        # Load mixed-domain stats so clean/mixed share the same normalization space
        try:
            speaker_mean, speaker_std = load_mixed_stats(BASE_DIR, subdir)
            print(f"  Loaded mixed-domain stats from {subdir}/norm_stats.npy")
        except Exception as e:
            print(f"  [ERROR] Could not load mixed-domain stats for {subdir}: {e}")
            continue

        # ── Pass 2: process, normalize, save ──
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Pass 2 — Processing & normalizing → {output_dir}")

        for i, fname in enumerate(wav_files, 1):
            input_path = os.path.join(input_dir, fname)
            out_fname = os.path.splitext(fname)[0] + '.npy'
            output_path = os.path.join(output_dir, out_fname)

            try:
                out_shape = process_file(
                    input_path, output_path, speaker_mean, speaker_std
                )
                if i <= 3 or i == len(wav_files):
                    print(f"    [{i}/{len(wav_files)}] {fname} → {out_fname}  shape={out_shape}")
                elif i == 4:
                    print(f"    ...")
                total_processed += 1
            except Exception as e:
                print(f"    [ERROR] {fname}: {e}")
                total_skipped += 1

    print(f"\n{'='*60}")
    print(f"DONE — Processed: {total_processed} | Skipped/Errors: {total_skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
