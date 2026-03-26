import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Your data will be at
base_dir = "/kaggle/input/datasets/hrishikesh3983/speakers-10-2/a"

# Output/saved models go here
save_path = "/kaggle/working/best_av_concat_model.pth"



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







import torch
import torch.nn.functional as F

def collate_fn(batch):
    avs, iams, mixeds, cleans = zip(*batch)

    # Find max time frames in this batch
    max_len = max(av.shape[0] for av in avs)

    avs_padded    = []
    iams_padded   = []
    mixeds_padded = []
    cleans_padded = []

    for av, iam, mixed, clean in zip(avs, iams, mixeds, cleans):
        av_pad    = F.pad(av,    (0, 0, 0, max_len - av.shape[0]))
        iam_pad   = F.pad(iam,   (0, 0, 0, max_len - iam.shape[0]))
        mixed_pad = F.pad(mixed, (0, 0, 0, max_len - mixed.shape[0]))
        clean_pad = F.pad(clean, (0, 0, 0, max_len - clean.shape[0]))

        avs_padded.append(av_pad)
        iams_padded.append(iam_pad)
        mixeds_padded.append(mixed_pad)
        cleans_padded.append(clean_pad)

    return torch.stack(avs_padded), torch.stack(iams_padded), torch.stack(mixeds_padded), torch.stack(cleans_padded)








# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────
def train(speakers_train, speakers_val, base_dir, epochs, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = AVConcatDataset(speakers_train, base_dir)
    val_dataset   = AVConcatDataset(speakers_val,   base_dir)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, collate_fn=collate_fn)

    # input_size = 136 landmarks + 257 freq_bins
    input_size = train_dataset[0][0].shape[1]
    freq_bins  = train_dataset[0][1].shape[1]

    model     = AVConcatBLSTM(input_size=input_size, hidden_size=250, num_layers=3, freq_bins=freq_bins).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience      = 10
    patience_counter = 0

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for av, iam, mixed, clean in train_loader:
            av, iam, mixed, clean = av.to(device), iam.to(device), mixed.to(device), clean.to(device)

            optimizer.zero_grad()
            pred_iam = model(av)                        # (batch, time_frames, freq_bins)

            # Loss in spectrogram domain
            pred_clean = pred_iam * mixed               # estimated clean spec
            loss = criterion(pred_clean, clean)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for av, iam, mixed, clean in val_loader:
                av, iam, mixed, clean = av.to(device), iam.to(device), mixed.to(device), clean.to(device)
                pred_iam   = model(av)
                pred_clean = pred_iam * mixed
                loss       = criterion(pred_clean, clean)
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







# ─────────────────────────────────────────
# Run
# ─────────────────────────────────────────
train(
    speakers_train=["s1", "s2","s3","s4", "s5", "s6","s7","s8"],
    speakers_val=["s9","s10"],
    base_dir=base_dir,
    epochs=50,
    batch_size=8,
    lr=1e-3
)