# Audio-Visual Speech Enhancement

Enhance a **target speaker’s speech** from a **single-channel multi-speaker mixture** using **audio-visual fusion**.  
This project uses noisy audio spectrogram features + facial landmark motion features to predict an Ideal Amplitude Mask (IAM), then reconstructs enhanced speech.

---

## Project Goal

Given:
- mixed/noisy speech (`target + interferer`)
- target speaker facial motion (lips/jaw/chin)

Learn a model that estimates a mask to suppress interferer components and recover cleaner target speech.

---

## Core Idea

1. Extract speech-relevant facial landmarks from video.
2. Convert landmarks into motion features.
3. Convert mixed and clean audio into normalized spectrogram-like features.
4. Compute training target IAM = clean / mixed.
5. Concatenate visual and audio features per frame.
6. Train a 3-layer BLSTM to predict IAM.
7. Reconstruct enhanced waveform with predicted magnitude + mixture phase.

---

## Repository Scripts

### Data preparation
- `extract_audio.sh`  
  Extracts mono 16 kHz `.wav` files from GRID `.mpg` videos.

- `extract_relevant_landmarks.py`  
  Uses MediaPipe FaceMesh to extract speech-relevant landmarks (lips/jaw/chin), then saves **motion vectors** in `.npy`.

- `create_audio_mixtures.py`  
  Creates noisy mixtures by adding random interferer speech at configurable SNR (default 0 dB).

### Feature preprocessing
- `vid_preprocessing.py`  
  Drops Z coordinate, flattens landmarks, upsamples from 25 fps to 100 fps.

- `audio_pre.py`  
  Processes **mixed** audio:
  - STFT (n_fft=512, win=400, hop=160)
  - magnitude
  - power-law compression (`|X|^0.3`)
  - per-speaker normalization

- `audio_clean_pre.py`  
  Same preprocessing for **clean target** audio.

- `save_stats.py`  
  Saves per-speaker normalization stats (`norm_stats.npy`) used later in waveform reconstruction.

### Training targets and AV fusion
- `iam.py`  
  Computes IAM target (`clean / mixed`, clipped).

- `av_concat.py`  
  Concatenates visual features + mixed audio features into a single frame-level vector for model input.

### Model / inference / evaluation
- `train.py`  
  Defines dataset, BLSTM model, and training loop.

- `recreate.py`  
  Loads trained model and reconstructs enhanced waveform from predicted IAM.

- `evaluation.py`, `evaluate_train.py`, `evaluation2.py`  
  Evaluate quality with SDR and PESQ.

- `pesq_test.py`, `check_audio.py`, `rec2.py`, `test.py`  
  Utility/debug scripts.

---

## Expected Dataset Structure

Organize each speaker folder like:

```text
GRID/
  s1/
    video/mpg_6000/
    audio/
    audio_mixed/
    landmarks/
    landmarks_preprocessed/
    audio_preprocessed/
    audio_clean_preprocessed/
    iam/
    concatenated_features/
    norm_stats.npy
  s2/
  ...
```

---

## Installation

### 1) Python dependencies

Install required packages (example):

```bash
pip install numpy torch librosa soundfile scipy opencv-python mediapipe tqdm pesq mir_eval
```

### 2) System dependency

`extract_audio.sh` requires `ffmpeg` to be available in PATH.

---

## End-to-End Pipeline (Run Order)

> Run scripts from repository root and adjust dataset paths inside scripts if needed.

1. **Download GRID corpus**  
   http://spandh.dcs.shef.ac.uk/gridcorpus/

2. **Extract audio from videos**
   ```bash
   bash extract_audio.sh
   ```

3. **Extract facial landmark motion**
   ```bash
   python extract_relevant_landmarks.py
   ```

4. **Create noisy mixtures**
   ```bash
   python create_audio_mixtures.py
   ```

5. **Preprocess visual features**
   ```bash
   python vid_preprocessing.py
   ```

6. **Preprocess mixed audio**
   ```bash
   python audio_pre.py
   ```

7. **Preprocess clean audio**
   ```bash
   python audio_clean_pre.py
   ```

8. **Save normalization stats**
   ```bash
   python save_stats.py
   ```

9. **Generate IAM targets**
   ```bash
   python iam.py
   ```

10. **Concatenate AV features**
    ```bash
    python av_concat.py
    ```

11. **Train model**
    ```bash
    python train.py
    ```

12. **Reconstruct enhanced waveform**
    ```bash
    python recreate.py
    ```

13. **Run evaluation**
    ```bash
    python evaluation.py
    ```

---

## Model Architecture

The enhancement model is a Bidirectional LSTM-based mask estimator:

- Input: concatenated AV features per frame  
  (`landmarks_preprocessed` + `audio_preprocessed`)
- Backbone: 3 stacked BLSTM layers
- Hidden size: 250
- Output: frame-wise IAM over frequency bins
- Output activation: sigmoid-scaled mask (`0..10` range)

Predicted clean magnitude is estimated as:

`pred_clean_mag = pred_iam * mixed_mag`

Then enhanced waveform is reconstructed via inverse STFT using phase from the mixed signal.

---

## Facial Landmark Usage for Interferer Suppression

The model relies on the fact that mouth/jaw motion is correlated with the target speaker’s phonetic content:

- Landmark subsets include outer lips, inner lips, upper/lower lip, and jaw/chin region.
- Motion features (`frame_t - frame_{t-1}`) capture articulation dynamics.
- These visual dynamics are fused with mixed audio so the BLSTM can bias mask estimation toward target-consistent spectral regions and suppress competing speakers.

---

## Metrics

Evaluation scripts report:
- **SDR** (signal-to-distortion ratio)
- **PESQ** (Perceptual Evaluation of Speech Quality)

Both noisy baseline and enhanced outputs are compared.

---

## Practical Notes

- Many scripts contain hardcoded dataset/model paths. Update paths before running.
- Ensure file alignment across:
  - `audio_preprocessed`
  - `audio_clean_preprocessed`
  - `landmarks_preprocessed`
  - `iam`
  - `concatenated_features`
- `av_concat.py` handles small frame mismatches by truncation and large mismatches by skipping/deleting inconsistent files.

---

## Troubleshooting

- **No output files generated**
  - Verify source directories exist and contain expected `.wav` / `.mpg` / `.npy` files.

- **MediaPipe face not detected for some frames**
  - Script falls back to previous frame landmarks (or zeros for first frame).

- **Shape mismatch errors**
  - Re-run preprocessing for the affected speaker and confirm consistent frame counts.

- **Poor reconstruction quality**
  - Confirm normalization stats are generated and loaded correctly.
  - Verify model checkpoint corresponds to current feature pipeline.

---

## Acknowledgment

Dataset: GRID Corpus  
http://spandh.dcs.shef.ac.uk/gridcorpus/
