#!/bin/bash

GRID_ROOT=GRID

for speaker in "$GRID_ROOT"/*; do
    video_dir="$speaker/video/mpg_6000"
    audio_dir="$speaker/audio"

    # Skip if this speaker doesn't have videos
    if [ ! -d "$video_dir" ]; then
        continue
    fi

    mkdir -p "$audio_dir"

    for video in "$video_dir"/*.mpg; do
        filename=$(basename "$video" .mpg)
        output="$audio_dir/$filename.wav"

        ffmpeg -y -i "$video" -ar 16000 -ac 1 "$output"
    done

    echo "Finished $(basename "$speaker")"
done

echo "All audio extracted!"
