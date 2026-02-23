# Audio-Visual-Speech-Enhancement


Goal- Enhance the speech of a target speaker in a single-channel multi-speaker environment using facial motion cues.

#Features
..


#How to run
Download GRID dataset from: http://spandh.dcs.shef.ac.uk/gridcorpus/
Download the .mpg files.

Extract audio by running extract_audio.sh

Extract landmarks by running extract_landmarks_batch which stores the landmarks motion vectors in npy files with each file storing the motion vectors in a shape of (75, 478, 3).

Create mixed audio for training by running create_audio_mixtures.py which adds the audio from another randomly chosen file.

Run unsample.py to unsample the landmarks from 25fps to 100fps

Run audio_preprocessing.py to sample the audio to 100 from 16000 samples per second and apply power compression law and normalization.
