"""
Kod-LLM v1 - Tokenizer egitimi.
data/*.jsonl dosyalarindaki {"text": "..."} satirlarindan BPE tokenizer egitir.
"""
import json
import glob
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

DATA_DIR = "data"
OUT_PATH = "tokenizer/tokenizer_v1.json"
VOCAB_SIZE = 10000


def load_texts():
    texts = []
    files = glob.glob(f"{DATA_DIR}/*.jsonl")
    if not files:
        raise FileNotFoundError(f"{DATA_DIR}/ altinda .jsonl dosyasi bulunamadi.")
    for path in files:
        n_before = len(texts)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("text"):
                        texts.append(data["text"])
                except json.JSONDecodeError:
                    pass
        print(f"{path}: {len(texts) - n_before} ornek okundu")
    return texts


def main():
    texts = load_texts()
    print(f"Toplam {len(texts)} ornek metin.")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(OUT_PATH)

    print(f"Tokenizer kaydedildi: {OUT_PATH}")
    print(f"Gercek vocab boyutu: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
