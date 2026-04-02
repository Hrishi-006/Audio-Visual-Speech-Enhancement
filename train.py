import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Your data will be at
base_dir = "/kaggle/input/datasets/hrishikesh3983/speakers-10-2/a"

# Output/saved models go here
save_path = "/kaggle/working/best_av_concat_model.pth"
MAX_GRAD_NORM = 5.0
SYNTHETIC_MAX_IAM = 2.0
SYNTHETIC_EPS = 1e-8
SYNTHETIC_TRAIN_MULTIPLIER = 2
SYNTHETIC_MIN_TRAIN_SAMPLES = 8
SYNTHETIC_MIN_VAL_SAMPLES = 4



# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class AVConcatDataset(Dataset):
    def __init__(self, speakers, base_dir):
        self.samples = []
        for speaker in speakers:
            av_dir    = os.path.join(base_dir, speaker, "concatenated_features")
            iam_dir   = os.path.join(base_dir, speaker, "iam")
            mixed_dir = os.path.join(base_dir, speaker, "audio_preprocessed")
            clean_dir = os.path.join(base_dir, speaker, "audio_clean_preprocessed")

            files = sorted([f for f in os.listdir(av_dir) if f.endswith(".npy")])
            for filename in files:
                self.samples.append({
                    "av":    os.path.join(av_dir,    filename),
                    "iam":   os.path.join(iam_dir,   filename),
                    "mixed": os.path.join(mixed_dir, filename),
                    "clean": os.path.join(clean_dir, filename),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        av    = torch.tensor(np.load(sample["av"]),    dtype=torch.float32)  # (time_frames, 393)
        iam   = torch.tensor(np.load(sample["iam"]), dtype=torch.float32)  # (time_frames, freq_bins)
        mixed = torch.tensor(np.load(sample["mixed"]), dtype=torch.float32)  # (time_frames, freq_bins)
        clean = torch.tensor(np.load(sample["clean"]), dtype=torch.float32)  # (time_frames, freq_bins)
        return av, iam, mixed, clean
    

class SyntheticAVConcatDataset(Dataset):
    def __init__(self, num_samples=8, min_frames=10, max_frames=20, av_dim=393, freq_bins=257):
        self.samples = []
        for _ in range(num_samples):
            t = np.random.randint(min_frames, max_frames + 1)
            av = np.random.randn(t, av_dim).astype(np.float32)
            clean = np.random.randn(t, freq_bins).astype(np.float32)
            noise = np.random.randn(t, freq_bins).astype(np.float32) * 0.3
            mixed = clean + noise
            iam = np.clip(np.abs(clean) / (np.abs(mixed) + SYNTHETIC_EPS), 0.0, SYNTHETIC_MAX_IAM).astype(np.float32)
            self.samples.append((av, iam, mixed, clean))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        av, iam, mixed, clean = self.samples[idx]
        return (
            torch.tensor(av, dtype=torch.float32),
            torch.tensor(iam, dtype=torch.float32),
            torch.tensor(mixed, dtype=torch.float32),
            torch.tensor(clean, dtype=torch.float32),
        )







# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────
class AVConcatBLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=250, num_layers=3, freq_bins=257):
        super(AVConcatBLSTM, self).__init__()
        self.blstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_size * 2, freq_bins)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, time_frames, input_size)
        out, _ = self.blstm(x)           # (batch, time_frames, hidden_size*2)
        out    = self.dropout(out)
        out    = self.fc(out)            # (batch, time_frames, freq_bins)
        out    = self.sigmoid(out) * 10  # IAM range [0, 10]
        return out







def collate_fn(batch):
    avs, iams, mixeds, cleans = zip(*batch)

    # Find max time frames in this batch
    max_len = max(av.shape[0] for av in avs)

    avs_padded    = []
    iams_padded   = []
    mixeds_padded = []
    cleans_padded = []
    lengths       = []

    for av, iam, mixed, clean in zip(avs, iams, mixeds, cleans):
        lengths.append(av.shape[0])
        av_pad    = F.pad(av,    (0, 0, 0, max_len - av.shape[0]))
        iam_pad   = F.pad(iam,   (0, 0, 0, max_len - iam.shape[0]))
        mixed_pad = F.pad(mixed, (0, 0, 0, max_len - mixed.shape[0]))
        clean_pad = F.pad(clean, (0, 0, 0, max_len - clean.shape[0]))

        avs_padded.append(av_pad)
        iams_padded.append(iam_pad)
        mixeds_padded.append(mixed_pad)
        cleans_padded.append(clean_pad)

    return (
        torch.stack(avs_padded),
        torch.stack(iams_padded),
        torch.stack(mixeds_padded),
        torch.stack(cleans_padded),
        torch.tensor(lengths, dtype=torch.long),
    )








# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────
def train(speakers_train, speakers_val, base_dir, epochs, batch_size=8, lr=1e-4, test_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if test_mode:
        train_dataset = SyntheticAVConcatDataset(
            num_samples=max(batch_size * SYNTHETIC_TRAIN_MULTIPLIER, SYNTHETIC_MIN_TRAIN_SAMPLES)
        )
        val_dataset = SyntheticAVConcatDataset(num_samples=max(batch_size, SYNTHETIC_MIN_VAL_SAMPLES))
        print("Running in test mode with synthetic in-memory data.")
    else:
        train_dataset = AVConcatDataset(speakers_train, base_dir)
        val_dataset = AVConcatDataset(speakers_val, base_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # input_size = 136 landmarks + 257 freq_bins
    input_size = train_dataset[0][0].shape[1]
    freq_bins  = train_dataset[0][1].shape[1]

    model     = AVConcatBLSTM(input_size=input_size, hidden_size=250, num_layers=3, freq_bins=freq_bins).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    patience      = 10
    patience_counter = 0

    def masked_mse(pred, target, lengths):
        # pred/target: (B, T, F), lengths: (B,)
        bsz, max_t, _ = pred.shape
        time_idx = torch.arange(max_t, device=pred.device).unsqueeze(0)  # (1, T)
        mask = time_idx < lengths.unsqueeze(1)                            # (B, T)
        mask = mask.unsqueeze(-1)                                          # (B, T, 1)
        sqerr = (pred - target) ** 2
        sqerr = sqerr * mask
        total_valid_elements = mask.sum().float() * pred.shape[-1]
        return sqerr.sum() / torch.clamp(total_valid_elements, min=1e-8)

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for av, iam, mixed, clean, lengths in train_loader:
            av, iam, mixed, clean, lengths = av.to(device), iam.to(device), mixed.to(device), clean.to(device), lengths.to(device)

            optimizer.zero_grad()
            pred_iam = model(av)                        # (batch, time_frames, freq_bins)

            # Loss in spectrogram domain
            pred_clean = pred_iam * mixed               # estimated clean spec
            loss = masked_mse(pred_clean, clean, lengths)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for av, iam, mixed, clean, lengths in val_loader:
                av, iam, mixed, clean, lengths = av.to(device), iam.to(device), mixed.to(device), clean.to(device), lengths.to(device)
                pred_iam   = model(av)
                pred_clean = pred_iam * mixed
                loss       = masked_mse(pred_clean, clean, lengths)
                val_loss  += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_av_concat_model.pth")
            print(f"  → Best model saved (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("\nTraining complete. Best val loss:", best_val_loss)







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Run a quick synthetic-data training pass without external files.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        speakers_train=["s1", "s2","s3","s4", "s5", "s6","s7","s8"],
        speakers_val=["s9","s10"],
        base_dir=base_dir,
        epochs=1 if args.test_mode else args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        test_mode=args.test_mode,
    )
