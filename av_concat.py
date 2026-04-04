import numpy as np
import os

def concatenate_av_features(speaker, base_dir="grid"):
    landmark_dir = os.path.join(base_dir, speaker, "landmarks_preprocessed")
    mixed_dir    = os.path.join(base_dir, speaker, "audio_preprocessed")
    clean_dir    = os.path.join(base_dir, speaker, "audio_clean_preprocessed")
    iam_dir      = os.path.join(base_dir, speaker, "iam")
    concat_dir   = os.path.join(base_dir, speaker, "concatenated_features")
    os.makedirs(concat_dir, exist_ok=True)

    landmark_files = set(os.listdir(landmark_dir))
    mixed_files    = set(os.listdir(mixed_dir))

    # Only process common files
    common_files = sorted(landmark_files & mixed_files)
    skipped      = sorted(mixed_files - landmark_files)

    if skipped:
        print(f"  Skipping {len(skipped)} files with no landmark: {skipped}")

    for filename in common_files:
        if not filename.endswith(".npy"):
            continue

        landmark_path = os.path.join(landmark_dir, filename)
        mixed_path    = os.path.join(mixed_dir,    filename)
        clean_path    = os.path.join(clean_dir,    filename)
        iam_path      = os.path.join(iam_dir,      filename)
        concat_path   = os.path.join(concat_dir,   filename)

        landmark = np.load(landmark_path)  # (time_frames, 206)
        mixed    = np.load(mixed_path)     # (time_frames, 257) 

        mixed_T = mixed  # already (time_frames, 257)

        # Check time frame mismatch
        if landmark.shape[0] != mixed_T.shape[0]:
            diff = abs(landmark.shape[0] - mixed_T.shape[0])
            print(f"  Time frame mismatch for {filename}: "
                  f"landmark={landmark.shape[0]} mixed={mixed_T.shape[0]} diff={diff}")

            # If difference is too large, delete and skip
            if diff > 10:
                print(f"  Difference too large — deleting {filename} from all folders")
                for path in [landmark_path, mixed_path, clean_path, iam_path]:
                    if os.path.exists(path):
                        os.remove(path)
                        print(f"    Deleted: {path}")
                continue

            # If small difference, truncate to min frames
            min_frames = min(landmark.shape[0], mixed_T.shape[0])
            landmark   = landmark[:min_frames]
            mixed_T    = mixed_T[:min_frames]

        # Concatenate along feature axis -> (time_frames, 206 + 257) = (time_frames, 463)
        av_concat = np.concatenate([landmark, mixed_T], axis=1)

        np.save(concat_path, av_concat)
        print(f"Saved: {concat_path} | shape: {av_concat.shape}")

# Run for all speakers
speakers = ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
"s11","s12","s13","s14","s15","s16","s17","s18","s19","s20",
"s22","s23","s24","s25","s26","s27","s28","s29","s30",
"s31","s32","s33","s34"]
for speaker in speakers:
    print(f"\nProcessing speaker: {speaker}")
    concatenate_av_features(speaker, base_dir="grid")

print("\nDone!")
