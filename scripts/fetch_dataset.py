"""
Kod-LLM v4 - Veri cekme scripti (bigcode/the-stack TAM versiyon, HF token ile).
ADR-0011: dil basina ornek sayisi 8000 -> 30000 (v4 veri buyutme).
"""
import json
import os
from datasets import load_dataset

OUT_DIR = "data"
LANGUAGES = ["python", "javascript", "c"]
SAMPLES_PER_LANGUAGE = 100000  # ADR-0015 (v5): ayni kalabilir, the-stack limiti
MIN_CHARS = 50
MAX_CHARS = 12000  # ADR-0015 (v5): max_seq_len 768e cikti, biraz daha uzun dosya izni


def clean_and_save(lang, samples_target):
    print(f"\n--- {lang} indiriliyor (hedef: {samples_target} ornek) ---")

    ds = load_dataset(
        "bigcode/the-stack",
        data_dir=f"data/{lang}",
        split="train",
        streaming=True,
    )

    out_path = f"{OUT_DIR}/raw_stack_{lang}.jsonl"
    written = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for sample in ds:
            if written >= samples_target:
                break
            content = sample.get("content", "").strip()

            if len(content) < MIN_CHARS:
                skipped += 1
                continue
            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS]

            f.write(json.dumps({"text": content}, ensure_ascii=False) + "\n")
            written += 1

            if written % 2000 == 0:
                print(f"  {written}/{samples_target} yazildi...")

    print(f"{lang}: {written} ornek kaydedildi -> {out_path} ({skipped} atlandi)")
    return written


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for lang in LANGUAGES:
        total += clean_and_save(lang, SAMPLES_PER_LANGUAGE)

    print(f"\n{'='*50}")
    print(f"TOPLAM YENI ORNEK: {total}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
