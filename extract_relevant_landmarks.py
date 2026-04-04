import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

GRID_ROOT = "GRID"
VIDEO_SUBDIR = "video/mpg_6000"
LANDMARK_SUBDIR = "landmarks"

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Speech-relevant landmark indices (lips, mouth, jaw, chin)
# Outer lips
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# Inner lips
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# Upper lip top
UPPER_LIP = [164, 167, 165, 92, 186, 57, 43, 106, 182, 83, 18, 313, 406, 335, 273, 287, 410, 322, 391, 393]

# Lower lip bottom
LOWER_LIP = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 175, 138, 135, 169, 170, 140, 171, 148]

# Jaw outline (for mouth movement context)
JAW = [176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]

# Combine all speech-relevant indices
SPEECH_LANDMARKS = sorted(set(OUTER_LIPS + INNER_LIPS + UPPER_LIP + LOWER_LIP + JAW))

NUM_SPEECH_LANDMARKS = len(SPEECH_LANDMARKS)


def extract_landmark_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    all_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            # Extract only speech-relevant landmarks
            landmarks = np.array(
                [[lm.landmark[i].x, lm.landmark[i].y, lm.landmark[i].z] for i in SPEECH_LANDMARKS],
                dtype=np.float32
            )
        else:
            # If face not detected, repeat last frame or zero
            if len(all_landmarks) > 0:
                landmarks = all_landmarks[-1]
            else:
                landmarks = np.zeros((NUM_SPEECH_LANDMARKS, 3), dtype=np.float32)

        all_landmarks.append(landmarks)

    cap.release()

    all_landmarks = np.stack(all_landmarks)  # (T, NUM_SPEECH_LANDMARKS, 3)

    # Convert to motion features
    motion = np.zeros_like(all_landmarks)
    motion[1:] = all_landmarks[1:] - all_landmarks[:-1]

    return motion


def main():
    speakers = sorted(d for d in os.listdir(GRID_ROOT) if d.startswith("s"))

    print(f"Extracting {NUM_SPEECH_LANDMARKS} speech-relevant landmarks per frame")

    for speaker in speakers:
        video_dir = os.path.join(GRID_ROOT, speaker, VIDEO_SUBDIR)
        landmark_dir = os.path.join(GRID_ROOT, speaker, LANDMARK_SUBDIR)

        if not os.path.isdir(video_dir):
            continue

        os.makedirs(landmark_dir, exist_ok=True)

        videos = sorted(f for f in os.listdir(video_dir) if f.endswith(".mpg"))

        print(f"\nProcessing {speaker} ({len(videos)} videos)")

        for video in tqdm(videos):
            video_path = os.path.join(video_dir, video)
            out_path = os.path.join(
                landmark_dir,
                video.replace(".mpg", ".npy")
            )

            if os.path.exists(out_path):
                continue  # skip if already processed

            motion = extract_landmark_motion(video_path)
            np.save(out_path, motion)

    face_mesh.close()
    print("\nAll speakers processed!")


if __name__ == "__main__":
    main()