import numpy as np
import os

def _to_time_major(mixed, target_time_frames):
    """Convert mixed spectrogram to (time_frames, freq_bins)."""
    if mixed.ndim != 2:
        raise ValueError(f"Expected 2D mixed spectrogram, got shape {mixed.shape}")

    if mixed.shape[0] == target_time_frames:
        return mixed
    if mixed.shape[1] == target_time_frames:
        return mixed.T

    raise ValueError(
        f"Cannot align mixed spectrogram shape {mixed.shape} with landmark time frames "
        f"{target_time_frames}. Expected one axis to match exactly."
    )

def concatenate_av_features(speaker, base_dir="grid"):
    landmark_dir = os.path.join(base_dir, speaker, "landmarks_preprocessed")
    mixed_dir    = os.path.join(base_dir, speaker, "audio_preprocessed")
    concat_dir   = os.path.join(base_dir, speaker, "concatenated_features")
    os.makedirs(concat_dir, exist_ok=True)

    landmark_files = sorted(os.listdir(landmark_dir))
    mixed_files    = sorted(os.listdir(mixed_dir))
    
    assert landmark_files == mixed_files, \
        f"Mismatch between landmark and mixed files for speaker {speaker}!"
        

    for filename in landmark_files:
        if not filename.endswith(".npy"):
            continue

        landmark_path = os.path.join(landmark_dir, filename)
        mixed_path    = os.path.join(mixed_dir, filename)

        landmark = np.load(landmark_path)  # shape: (time_frames, landmark_dims)
        mixed    = np.load(mixed_path)     # shape: (time_frames, freq_bins) or (freq_bins, time_frames)
        mixed_T  = _to_time_major(mixed, landmark.shape[0])

        # Verify time frames match
        if landmark.shape[0] != mixed_T.shape[0]:
            min_frames = min(landmark.shape[0], mixed_T.shape[0])
            landmark = landmark[:min_frames]
            mixed_T  = mixed_T[:min_frames]

        # Concatenate along feature axis -> (time_frames, 136 + freq_bins)
        av_concat = np.concatenate([landmark, mixed_T], axis=1)

        concat_path = os.path.join(concat_dir, filename)
        np.save(concat_path, av_concat)
        print(f"Saved: {concat_path} | shape: {av_concat.shape}")

# Run for all speakers found in base_dir (GRID convention: s1, s2, ...)
base_dir = "grid"
speakers = sorted(
    d for d in os.listdir(base_dir)
    if d.startswith("s") and os.path.isdir(os.path.join(base_dir, d))
)
if not speakers:
    raise ValueError(
        f"No speaker directories found in {base_dir}. "
        "Expected GRID-style directories named like s1, s2, ..."
    )
for speaker in speakers:
    print(f"\nProcessing speaker: {speaker}")
    concatenate_av_features(speaker, base_dir=base_dir)

print("\nDone! AV concat features saved for all speakers.")
