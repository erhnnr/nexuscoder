"""
Kod-LLM v3 - Tokenizer egitimi.

ADR-0006: Sadece data/train.jsonl kullanilir (prepare_data.py CIKTISI,
yani zaten raw_stack_*.jsonl'den gelen temiz/Ingilizce veri). data/
klasorundeki eski raw_1/raw_2.jsonl gibi ham dosyalar burada
KASITLI olarak OKUNMUYOR - onlari okumak karisik dil sorununu
tokenizer'a geri sizdirir.
"""
import json
import glob
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

DATA_DIR = "data"
OUT_PATH = "tokenizer/tokenizer_v4.json"
VOCAB_SIZE = 32000  # ADR-0012 (v4): 300k cesitli dosyada temel alfabe
                     # bile 16000'i asiyordu (0 merge yapilamiyordu) -
                     # buyutuldu ki gercek alt-kelime birlestirmesi olsun


def load_texts():
    texts = []
    files = [f"{DATA_DIR}/train.jsonl"]  # SADECE temizlenmis train verisi
    for path in files:
        if not glob.glob(path):
            raise FileNotFoundError(f"{path} bulunamadi. Once prepare_data.py calistirin.")
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
