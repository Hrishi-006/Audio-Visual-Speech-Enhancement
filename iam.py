import numpy as np
import os

def compute_iam(clean_spec, mixed_spec, clip_value=10.0):
    """
    Compute Ideal Amplitude Mask (IAM) = clean / mixed
    Clipped to clip_value for numerical stability (as per paper)
    """
    # Avoid division by zero
    mixed_spec = np.where(mixed_spec == 0, 1e-8, mixed_spec)
    iam = clean_spec / mixed_spec
    iam = np.clip(iam, 0, clip_value)
    return iam

def compute_iam_for_speaker(speaker, base_dir="grid"):
    clean_dir = os.path.join(base_dir, speaker, "audio_clean_preprocessed")
    mixed_dir = os.path.join(base_dir, speaker, "audio_preprocessed")
    iam_dir   = os.path.join(base_dir, speaker, "iam")
    os.makedirs(iam_dir, exist_ok=True)

    clean_files = sorted(os.listdir(clean_dir))
    mixed_files = sorted(os.listdir(mixed_dir))

    # Verify files match
    assert clean_files == mixed_files, \
        f"Mismatch between clean and mixed files for speaker {speaker}!"

    for filename in clean_files:
        if not filename.endswith(".npy"):
            continue

        clean_path = os.path.join(clean_dir, filename)
        mixed_path = os.path.join(mixed_dir, filename)

        clean_spec = np.load(clean_path)   # shape: (freq_bins, time_frames)
        mixed_spec = np.load(mixed_path)

        # Shapes must match
        assert clean_spec.shape == mixed_spec.shape, \
            f"Shape mismatch for {filename}: clean {clean_spec.shape} vs mixed {mixed_spec.shape}"

        iam = compute_iam(clean_spec, mixed_spec)

        iam_path = os.path.join(iam_dir, filename)
        np.save(iam_path, iam)
        print(f"Saved IAM: {iam_path} | shape: {iam.shape} | max: {iam.max():.4f}")

# Run for all speakers
speakers = ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
"s11","s12","s13","s14","s15","s16","s17","s18","s19","s20",
"s22","s23","s24","s25","s26","s27","s28","s29","s30",
"s31","s32","s33","s34"]
for speaker in speakers:
    print(f"\nProcessing speaker: {speaker}")
    compute_iam_for_speaker(speaker, base_dir="grid")

print("\nDone! IAM masks saved for all speakers.")
