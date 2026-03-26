import os

base_dir = "grid"
speaker = "s1"

landmark_dir = os.path.join(base_dir, speaker, "audio_clean_preprocessed")
mixed_dir    = os.path.join(base_dir, speaker, "audio_preprocessed")

landmark_files = sorted(os.listdir(landmark_dir))
mixed_files    = sorted(os.listdir(mixed_dir))

print(f"Total landmark files: {len(landmark_files)}")
print(f"Total mixed files:    {len(mixed_files)}")

# Files in landmark but not in mixed
only_in_clean = set(landmark_files) - set(mixed_files)
# Files in mixed but not in landmark
only_in_mixed    = set(mixed_files) - set(landmark_files)

print(f"\nFiles only in landmarks_preprocessed: {only_in_clean}")
print(f"Files only in audio_preprocessed:     {only_in_mixed}")

# Preview first few files from each
print(f"\nFirst 5 landmark files: {landmark_files[:5]}")
print(f"First 5 mixed files:    {mixed_files[:5]}")
