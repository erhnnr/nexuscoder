"""
Kod-LLM v6 - Instruction fine-tuning.

ADR-0017: model_v5_best.pt'yi baz alip, kucuk instruction veri setiyle
(~500 ornek) DUSUK learning rate ve AZ epoch ile ince ayar yapar.
Amac: modelin "talimat -> kod" formatini ogrenmesi, temel kod bilgisini
KAYBETMEDEN (catastrophic forgetting riski - bu yuzden LR cok dusuk
ve epoch sayisi az tutuluyor, ayrica val loss yakindan izleniyor).
"""
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_v2 import KodLLM_v2

PROJECT_DIR = "/content/nexuscoder"
DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/nexus_checkpoints"
BASE_CHECKPOINT_PATH = f"{DRIVE_CHECKPOINT_DIR}/model_v5_best.pt"
TOKENIZER_PATH = f"{PROJECT_DIR}/tokenizer/tokenizer_v5.json"
TRAIN_PATH = f"{PROJECT_DIR}/data/instruction_train.jsonl"
VAL_PATH = f"{PROJECT_DIR}/data/instruction_val.jsonl"

MAX_SEQ_LEN = 768
BATCH_SIZE = 4               # kucuk veri seti, kucuk batch yeterli
NUM_EPOCHS = 8                # AZ veri -> az epoch (ADR-0017 riski)
LR = 2e-5                     # base egitimden (3e-4) COK DUSUK - catastrophic forgetting'e karsi
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU yok!")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")


class InstructDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length):
        self.samples = []
        self.pad_id = tokenizer.token_to_id("<pad>")
        self.bos_id = tokenizer.token_to_id("<bos>")
        self.eos_id = tokenizer.token_to_id("<eos>")

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    ids = [self.bos_id] + tokenizer.encode(d["text"]).ids + [self.eos_id]
                    if len(ids) > max_length:
                        ids = ids[:max_length]
                    if len(ids) < 2:
                        continue
                    self.samples.append(ids)
                except Exception:
                    pass
        print(f"{jsonl_path}: {len(self.samples)} ornek yuklendi")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        return ids[:-1], ids[1:]

    def collate_fn(self, batch):
        max_len = max(len(x[0]) for x in batch)
        inp, tgt = [], []
        for i, t in batch:
            pad_len = max_len - len(i)
            inp.append(i + [self.pad_id] * pad_len)
            tgt.append(t + [self.pad_id] * pad_len)
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total, n = 0.0, 0
    for inp, tgt in val_loader:
        inp, tgt = inp.to(device), tgt.to(device)
        _, loss, _ = model(inp, tgt)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    check_gpu()
    device = torch.device("cuda")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    base_ckpt = torch.load(BASE_CHECKPOINT_PATH, map_location=device)
    model_config = base_ckpt["model_config"]
    assert base_ckpt["vocab_size"] == vocab_size, "Vocab uyumsuz!"

    model = KodLLM_v2(vocab_size=vocab_size, **model_config).to(device)
    model.load_state_dict(base_ckpt["model_state_dict"])
    print(f"Baz model yuklendi: v5 epoch {base_ckpt['epoch']}, val_loss={base_ckpt['val_loss']:.4f}\n")

    train_ds = InstructDataset(TRAIN_PATH, tokenizer, MAX_SEQ_LEN)
    val_ds = InstructDataset(VAL_PATH, tokenizer, MAX_SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_ds.collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=val_ds.collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_loss = evaluate(model, val_loader, device)
    print(f"Baslangic (fine-tuning oncesi) val_loss: {best_val_loss:.4f}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            optimizer.zero_grad()
            _, loss, _ = model(inp, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {epoch_loss/n_batches:.4f} | "
              f"Val Loss: {val_loss:.4f}" + ("  <-- YENI EN IYI" if val_loss < best_val_loss else ""))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "model_config": model_config,
                "vocab_size": vocab_size,
            }, f"{DRIVE_CHECKPOINT_DIR}/model_v6_instruct_best.pt")
            print(f"  Kaydedildi (val_loss={val_loss:.4f})")

        scheduler.step()

    print(f"\nFINE-TUNING TAMAMLANDI. En iyi val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
