import os
import numpy as np

BASE_PATH = "/Users/hrishikeshbingewar/Downloads/av_project/GRID"
SPEAKERS = ["s1", "s2", "s3"]

ORIG_FPS = 25
TARGET_FPS = 100


def upsample_motion(motion, orig_fps=25, target_fps=100):
    T = motion.shape[0]

    # Flatten (T, 478, 3) → (T, 1434)
    motion = motion.reshape(T, -1)

    D = motion.shape[1]
    duration = T / orig_fps

    t_orig = np.linspace(0, duration, T)
    T_new = int(duration * target_fps)
    t_new = np.linspace(0, duration, T_new)

    upsampled = np.zeros((T_new, D))

    for d in range(D):
        upsampled[:, d] = np.interp(t_new, t_orig, motion[:, d])

    return upsampled


for speaker in SPEAKERS:
    input_dir = os.path.join(BASE_PATH, speaker, "landmarks")
    output_dir = os.path.join(BASE_PATH, speaker, "landmarks_100fps")

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".npy"):
            filepath = os.path.join(input_dir, file)
            
            motion_25fps = np.load(filepath)
            motion_100fps = upsample_motion(motion_25fps)

            save_path = os.path.join(output_dir, file)
            np.save(save_path, motion_100fps)

            print(f"Processed {speaker}/{file}")

print("Upsampling complete.")