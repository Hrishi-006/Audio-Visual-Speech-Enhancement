import os
import numpy as np
from scipy.interpolate import interp1d

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = "/Users/hrishikeshbingewar/Downloads/av_project/GRID"
SUBDIRS = ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10",
"s11","s12","s13","s14","s15","s16","s17","s18","s19","s20",
"s22","s23","s24","s25","s26","s27","s28","s29","s30",
"s31","s32","s33","s34"]
INPUT_FOLDER = "landmarks"
OUTPUT_FOLDER = "landmarks_preprocessed"

ORIGINAL_FPS = 25
TARGET_FPS = 100
FPS_RATIO = TARGET_FPS / ORIGINAL_FPS  # 4x

# ──────────────────────────────────────────────
# Processing
# ──────────────────────────────────────────────
def upsample_motion(data, original_fps, target_fps):
    """
    Upsample temporal data from original_fps to target_fps
    using cubic interpolation along the time axis (axis=0).

    Input:  (T_original, D)
    Output: (T_target, D)
    """
    num_original_frames = data.shape[0]
    num_target_frames = int(num_original_frames * (target_fps / original_fps))

    # Original and target time axes
    t_original = np.linspace(0, 1, num_original_frames)
    t_target = np.linspace(0, 1, num_target_frames)

    # Interpolate each feature dimension independently
    interpolator = interp1d(t_original, data, axis=0, kind='cubic')
    upsampled = interpolator(t_target)

    return upsampled


def process_file(input_path, output_path):
    """Process a single .npy landmark file."""
    # Step 1: Load — expected shape (75, 103, 3)
    landmarks = np.load(input_path)
    assert landmarks.shape == (75, 103, 3), (
        f"Unexpected shape {landmarks.shape} in {input_path}, expected (75, 103, 3)"
    )

    # Step 2: Drop Z coordinate (keep x, y only) → (75, 103, 2)
    landmarks_2d = landmarks[:, :, :2]

    # Step 3: Flatten to (75, -1) → (75, 206)
    num_frames = landmarks_2d.shape[0]
    landmarks_flat = landmarks_2d.reshape(num_frames, -1)  # (75, 206)

    # Step 4: Upsample from 25fps to 100fps → (300, 206)
    landmarks_upsampled = upsample_motion(landmarks_flat, ORIGINAL_FPS, TARGET_FPS)

    # Step 5: Save
    np.save(output_path, landmarks_upsampled)

    return landmarks_upsampled.shape


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

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        npy_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])

        if not npy_files:
            print(f"[WARN] No .npy files found in: {input_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {input_dir}")
        print(f"  Found {len(npy_files)} .npy files")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}")

        for i, fname in enumerate(npy_files, 1):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)

            try:
                out_shape = process_file(input_path, output_path)
                if i <= 3 or i == len(npy_files):  # Print first 3 and last
                    print(f"  [{i}/{len(npy_files)}] {fname}: (75,103,3) → {out_shape}")
                elif i == 4:
                    print(f"  ...")
                total_processed += 1
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")
                total_skipped += 1

    print(f"\n{'='*60}")
    print(f"DONE — Processed: {total_processed} | Skipped/Errors: {total_skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
