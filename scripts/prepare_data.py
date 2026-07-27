"""
Kod-LLM v3 - Veri hazirlama.
data/*.jsonl dosyalarini birlestirir, karistirir, train/val olarak boler.

ADR-0006: Sadece raw_stack_*.jsonl (temiz, Ingilizce, GitHub kaynakli
Python/JS/C) dosyalari kullaniliyor. Eski raw_1/raw_2.jsonl (karisik
Turkce/Ingilizce icerikli) veriler bilinerek disarida birakildi -
karisik dil, tokenizer vocab budcesini bolup Ingilizce kod
kelimelerinin (factorial, calculate gibi) tam ogrenilmesini
zorlastiriyordu.
"""
import json
import glob
import random

DATA_DIR = "data"
VAL_RATIO = 0.05  # verinin %5'i validation icin ayrilir
SEED = 42


def main():
    random.seed(SEED)
    all_samples = []

    for path in glob.glob(f"{DATA_DIR}/raw_stack_*.jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "").strip()
                    if text:
                        all_samples.append({"text": text})
                except json.JSONDecodeError:
                    pass

    print(f"Toplam {len(all_samples)} ham ornek okundu.")
    random.shuffle(all_samples)

    n_val = max(1, int(len(all_samples) * VAL_RATIO))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    with open(f"{DATA_DIR}/train.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(f"{DATA_DIR}/val.jsonl", "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"train.jsonl: {len(train_samples)} ornek")
    print(f"val.jsonl:   {len(val_samples)} ornek")


if __name__ == "__main__":
    main()
