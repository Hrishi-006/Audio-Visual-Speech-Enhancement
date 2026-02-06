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
            landmarks = np.array(
                [[p.x, p.y, p.z] for p in lm.landmark],
                dtype=np.float32
            )
        else:
            # If face not detected, repeat last frame or zero
            if len(all_landmarks) > 0:
                landmarks = all_landmarks[-1]
            else:
                landmarks = np.zeros((478, 3), dtype=np.float32)

        all_landmarks.append(landmarks)

    cap.release()

    all_landmarks = np.stack(all_landmarks)  # (T, 478, 3)

    # Convert to motion features
    motion = np.zeros_like(all_landmarks)
    motion[1:] = all_landmarks[1:] - all_landmarks[:-1]

    return motion


def main():
    speakers = sorted(d for d in os.listdir(GRID_ROOT) if d.startswith("s"))

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
