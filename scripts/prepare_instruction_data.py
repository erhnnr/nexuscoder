"""
Kod-LLM v6 - Instruction verisi hazirlama.
Farkli LLM'lerden (Gemini, GPT, Claude vb.) uretilen JSONL dosyalarini
data/instruction_raw/ klasorunden okuyup birlestirir, egitim formatina
cevirir, train/val olarak boler.
"""
import json
import glob
import random
import os

IN_DIR = "data/instruction_raw"
TRAIN_PATH = "data/instruction_train.jsonl"
VAL_PATH = "data/instruction_val.jsonl"
PROMPT_TEMPLATE = "# Talimat: {instruction}\n# Kod:\n"
VAL_RATIO = 0.1


def main():
    os.makedirs(IN_DIR, exist_ok=True)
    all_samples = []
    for path in glob.glob(f"{IN_DIR}/*.jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    text = PROMPT_TEMPLATE.format(instruction=d["instruction"]) + d["code"]
                    all_samples.append({"text": text})
                except Exception:
                    pass

    random.seed(42)
    random.shuffle(all_samples)

    n_val = max(1, int(len(all_samples) * VAL_RATIO))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(VAL_PATH, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"train: {len(train_samples)} ornek -> {TRAIN_PATH}")
    print(f"val:   {len(val_samples)} ornek -> {VAL_PATH}")


if __name__ == "__main__":
    main()
