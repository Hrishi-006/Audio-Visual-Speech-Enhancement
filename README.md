# Audio-Visual-Speech-Enhancement


Goal- Enhance the speech of a target speaker in a single-channel multi-speaker environment using facial motion cues.

#Features
..


**How to run**

Download GRID dataset from: http://spandh.dcs.shef.ac.uk/gridcorpus/
Download the .mpg files.

Extract audio by running extract_audio.sh

Extract landmarks by running extract_relevant_landmarks which stores the landmarks motion vectors of lips and jaw in npy files with each file storing the motion vectors in a shape of (75, 103, 3).

Create mixed audio for training by running create_audio_mixtures.py which adds the audio from another randomly chosen file.

Run vid_preprocessing.py to remove the z coordinate data and unsample the landmarks from 25fps to 100fps

Run audio_pre.py to sample the mixed audio to 100 from 16000 samples per second and apply power compression law and normalization.
Run audio_clean_pre to do the same for the clean audio files.

Run iam.py to calculate the ideal iam( Ideal Amplitude Mask)

Run av_concat.py to concatenate the preprocessed audio and video for training.



