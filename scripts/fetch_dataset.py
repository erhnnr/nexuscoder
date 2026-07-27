"""
Kod-LLM v2 - Veri cekme scripti.

OGRETICI NOT: "bigcode/the-stack-smol" GATED (korumali) cikti - HF hesabi +
lisans onayi gerektiriyor. Onun yerine ayni mantikta calisan, ACIK
(ungated) bir veri seti kullaniyoruz: "codeparrot/github-code-clean".
Bu da GitHub'dan toplanmis, dile gore filtrelenebilen kod dosyalarindan
olusuyor. Biz sadece Python, JavaScript ve C kismini cekiyoruz - kapsam
daraltmasi = ayni hesaplama butcesiyle daha iyi sonuc (once konustugumuz
"genel LLM degil, kod LLM" prensibi).

Cikti format, mevcut prepare_data.py ile birebir uyumlu: data/raw_*.jsonl
"""
import json
import os
from datasets import load_dataset

OUT_DIR = "data"
# codeparrot/github-code-clean dil isimlerini boyle bekliyor (buyuk harfle basliyor)
LANGUAGES = ["Python", "JavaScript", "C"]
SAMPLES_PER_LANGUAGE = 8000  # dil basina cekilecek dosya sayisi (ayarlanabilir)
MIN_CHARS = 50               # cok kisa/anlamsiz dosyalari ele
MAX_CHARS = 8000             # cok uzun dosyalari kirp (max_seq_len ile uyumlu kalsin)


def clean_and_save(lang, samples_target):
    print(f"\n--- {lang} indiriliyor (hedef: {samples_target} ornek) ---")

    # streaming=True: tum veri setini diske indirmeden, ihtiyac kadar akar
    ds = load_dataset(
        "codeparrot/github-code-clean",
        streaming=True,
        split="train",
        languages=[lang],
    )

    out_path = f"{OUT_DIR}/raw_stack_{lang.lower()}.jsonl"
    written = 0
    skipped = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for sample in ds:
            if written >= samples_target:
                break
            content = sample.get("code", "").strip()

            if len(content) < MIN_CHARS:
                skipped += 1
                continue
            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS]

            f.write(json.dumps({"text": content}, ensure_ascii=False) + "\n")
            written += 1

            if written % 1000 == 0:
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
    print("\nNot: Eski verileriniz (raw_1.jsonl, raw_2.jsonl) data/ klasorunde")
    print("hala duruyor - prepare_data.py hepsini otomatik birlestirecek.")


if __name__ == "__main__":
    main()
