import os
import random
import numpy as np
import soundfile as sf

GRID_ROOT = "GRID"
SPEAKERS = ["s1", "s2", "s3"]   # can extend later
SNR_DB = -5


def rms(x):
    return np.sqrt(np.mean(x ** 2) + 1e-8)


def match_length(x, y):
    """Trim or pad y to match length of x"""
    if len(y) > len(x):
        return y[:len(x)]
    elif len(y) < len(x):
        pad = len(x) - len(y)
        return np.pad(y, (0, pad))
    return y


def mix_audio(target, interferer, snr_db=0.0):
    """Mix target and interferer at given SNR"""
    target_rms = rms(target)
    interferer_rms = rms(interferer)

    interferer = interferer * (target_rms / (interferer_rms + 1e-8))
    mixed = target + interferer

    # Optional: prevent clipping
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak

    return mixed


def main():
    for target_spk in SPEAKERS:
        target_audio_dir = os.path.join(GRID_ROOT, target_spk, "audio")
        out_dir = os.path.join(GRID_ROOT, target_spk, "audio_mixed")
        os.makedirs(out_dir, exist_ok=True)

        target_files = [f for f in os.listdir(target_audio_dir) if f.endswith(".wav")]

        print(f"\nProcessing {target_spk} ({len(target_files)} files)")

        for wav in target_files:
            target_path = os.path.join(target_audio_dir, wav)
            target_audio, sr = sf.read(target_path)

            # Choose random interferer speaker
            interferer_spk = random.choice([s for s in SPEAKERS if s != target_spk])
            interferer_audio_dir = os.path.join(GRID_ROOT, interferer_spk, "audio")

            interferer_wav = random.choice(
                [f for f in os.listdir(interferer_audio_dir) if f.endswith(".wav")]
            )
            interferer_path = os.path.join(interferer_audio_dir, interferer_wav)
            interferer_audio, _ = sf.read(interferer_path)

            # Match lengths
            interferer_audio = match_length(target_audio, interferer_audio)

            # Mix
            mixed_audio = mix_audio(target_audio, interferer_audio, SNR_DB)

            # Save
            out_path = os.path.join(out_dir, wav)
            sf.write(out_path, mixed_audio, sr)

        print(f"Finished {target_spk}")

    print("\nAll mixtures created!")


if __name__ == "__main__":
    main()
