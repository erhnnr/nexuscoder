"""
Kod-LLM v1 - Ana egitim scripti (Colab icin).

Onceki scriptten farklar:
  1. Model artik causal mask kullaniyor (models/model_v1.py)
  2. Validation seti var, her epoch sonunda val loss olculuyor
  3. Checkpoint'ler Google Drive'a kaydediliyor (Colab oturumu kopsa bile kalici)
  4. Mixed precision (T4'te ~1.5-2x hizlanma)
  5. Model boyutu veri miktarina gore mantikli (varsayilan ~20M parametre)

Colab'da calistirmadan once:
  from google.colab import drive
  drive.mount('/content/drive')
"""
import os
import sys
import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_v1 import KodLLM_v1

# ============================================
# AYARLAR - Colab'daki yol yapinize gore duzenleyin
# ============================================
PROJECT_DIR = "/content/kod-llm"
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/kod-llm-checkpoints"  # Drive'a kaydeder
TOKENIZER_PATH = f"{PROJECT_DIR}/tokenizer/tokenizer_v1.json"
TRAIN_PATH = f"{PROJECT_DIR}/data/train.jsonl"
VAL_PATH = f"{PROJECT_DIR}/data/val.jsonl"

MAX_SEQ_LEN = 512
BATCH_SIZE = 8              # v1 kucuk model oldugu icin buyutulebildi
GRAD_ACCUM_STEPS = 4        # efektif batch = 8*4 = 32
NUM_EPOCHS = 20
LR = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EVAL_EVERY_N_BATCHES = 200
CHECKPOINT_EVERY_N_EPOCHS = 2

MODEL_CONFIG = dict(
    dim=384,
    num_layers=6,
    num_heads=6,
    max_seq_len=MAX_SEQ_LEN,
    dropout=0.1,
)


def check_gpu():
    print("=" * 50)
    print("GPU KONTROL")
    print("=" * 50)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU yok! Runtime -> Change runtime type -> GPU -> T4")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("GPU hazir.\n")


class CodeDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

        skipped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data["text"]
                    ids = [self.bos_id] + tokenizer.encode(text).ids + [self.eos_id]
                    if len(ids) > max_length:
                        ids = ids[:max_length]
                    if len(ids) < 2:
                        skipped += 1
                        continue
                    self.samples.append(ids)
                except Exception:
                    skipped += 1
        print(f"{jsonl_path}: {len(self.samples)} ornek yuklendi, {skipped} atlandi")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        input_ids = ids[:-1]
        target_ids = ids[1:]
        return input_ids, target_ids

    def collate_fn(self, batch):
        max_len = max(len(x[0]) for x in batch)
        input_batch, target_batch = [], []
        for input_ids, target_ids in batch:
            pad_len = max_len - len(input_ids)
            input_batch.append(input_ids + [self.pad_id] * pad_len)
            target_batch.append(target_ids + [self.pad_id] * pad_len)
        return (
            torch.tensor(input_batch, dtype=torch.long),
            torch.tensor(target_batch, dtype=torch.long),
        )


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for input_ids, target_ids in val_loader:
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        _, loss = model(input_ids, target_ids)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def main():
    check_gpu()
    device = torch.device("cuda")
    os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer yuklendi. Vocab: {vocab_size}\n")

    train_ds = CodeDataset(TRAIN_PATH, tokenizer, MAX_SEQ_LEN)
    val_ds = CodeDataset(VAL_PATH, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=train_ds.collate_fn, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=val_ds.collate_fn, num_workers=2,
    )

    model = KodLLM_v1(vocab_size=vocab_size, **MODEL_CONFIG).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayisi: {total_params:,} ({total_params/1e6:.1f}M)\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        epoch_start = time.time()
        optimizer.zero_grad()

        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, loss = model(input_ids, target_ids)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            n_batches += 1

            if batch_idx % EVAL_EVERY_N_BATCHES == 0:
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f}")

        avg_train_loss = epoch_loss / n_batches
        val_loss = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} TAMAMLANDI")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}"
              f"  {'<-- YENI EN IYI' if val_loss < best_val_loss else ''}")
        print(f"  Sure: {epoch_time:.1f} sn")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*50}\n")

        # v1 uyarisi: val_loss train_loss'tan belirgin sekilde yuksek olmaya
        # baslarsa (orn. train 0.5, val 4.0) bu overfitting isaretidir -
        # daha fazla veri veya daha kucuk model gerekebilir.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "model_config": MODEL_CONFIG,
                "vocab_size": vocab_size,
            }, f"{DRIVE_CHECKPOINT_DIR}/model_v1_best.pt")
            print(f"  Yeni en iyi model Drive'a kaydedildi (val_loss={val_loss:.4f})\n")

        if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "model_config": MODEL_CONFIG,
                "vocab_size": vocab_size,
            }, f"{DRIVE_CHECKPOINT_DIR}/model_v1_epoch_{epoch+1}.pt")

        scheduler.step()

    total_time = time.time() - start_time
    print(f"\nEGITIM TAMAMLANDI! Toplam sure: {total_time/60:.1f} dk")
    print(f"En iyi val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
