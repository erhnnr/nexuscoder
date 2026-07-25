# Kod-LLM

Python, HTML/CSS/JS ve C odaklı, sıfırdan eğitilen küçük bir kod dil modeli.
Sistematik olarak `model_v1 -> model_v2 -> ...` şeklinde büyütülüyor.

## Versiyonlama mantığı

Her versiyon şunlardan en az birini değiştirir: mimari, veri miktarı, model boyutu.
Bir önceki versiyonun checkpoint'i, veri arttığında yeni versiyona **devam** eğitimi
için taşınabilir (weight'ler uyumluysa).

| Versiyon | Amaç | Veri | Parametre | Durum |
|----------|------|------|-----------|-------|
| v1 | Pipeline'ın doğru çalıştığını kanıtlamak (causal mask, eval loss, checkpoint) | ~2-5k örnek | ~15-25M | 🚧 şu an burada |
| v2 | Veri setini büyütmek (açık kaynak Python/JS/C veri seti) | ~50-200k örnek | ~40-60M | henüz değil |
| v3 | Kapasiteyi büyütmek, gerekirse RoPE/KV-cache ekleyip inference'ı hızlandırmak | v2 ile aynı/daha fazla | ~100M+ | henüz değil |

## v1'de düzeltilen buglar (önceki koddan)

1. **Causal mask eksikliği**: `nn.MultiheadAttention` hiçbir mask almıyordu,
   model geleceği görebiliyordu → loss yapay olarak çok hızlı düşüyordu (ezber).
   v1'de `attn_mask` olarak üst üçgen mask veriliyor.
2. **Model/veri oranı**: 560M parametre, 1374 örnekle eşleşmiyordu (aşırı ezber).
   v1'de model küçültüldü (~20M), veri seti verimli kullanılacak şekilde ayarlandı.
3. **Validation seti yoktu**: Artık train/val ayrımı var, gerçek genelleme takip
   edilebiliyor.
4. **Checkpoint Drive'a değil, Colab'ın geçici diskine kaydediliyordu**: Colab
   oturumu kopunca kaybolabilir. v1'de Drive'a kaydetme talimatı var.

## Klasör yapısı

```
kod-llm/
  models/
    model_v1.py       <- mimari tanımı (bu versiyona özel)
  scripts/
    prepare_data.py   <- veri temizleme / train-val split
    train_tokenizer.py
    train.py           <- ana eğitim scripti
  data/                <- (git'e eklenmez, .gitignore'da) ham/işlenmiş veri
  tokenizer/           <- eğitilmiş tokenizer dosyası
  checkpoints/          <- (git'e eklenmez) model ağırlıkları
```

## Sıradaki adım

Bu README'nin altına her versiyon tamamlandığında bir "sonuçlar" bölümü
eklenecek: val loss grafiği, örnek üretimler, ne öğrendik.
