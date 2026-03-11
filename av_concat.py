import numpy as np
import os

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

        landmark = np.load(landmark_path)  # shape: (time_frames, 136)
        mixed    = np.load(mixed_path)     # shape: (freq_bins, time_frames)

        # Transpose mixed to (time_frames, freq_bins) to match landmark's time axis
        mixed_T = mixed.T  # (time_frames, freq_bins)

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

# Run for all speakers
speakers = ["s3"]
for speaker in speakers:
    print(f"\nProcessing speaker: {speaker}")
    concatenate_av_features(speaker, base_dir="grid")

print("\nDone! AV concat features saved for all speakers.")
