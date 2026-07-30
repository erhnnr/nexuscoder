# 0012 - v4 Sonucu: Donus Noktasi Tespiti (Epoch 42)

**Durum:** Kabul edildi

## Baglam

v4 (285.000 ornek, 41.8M parametre, 32k vocab) 47 epoch'a kadar
egitildi (birden fazla Colab oturumunda, resume mekanizmasiyla
kesintisiz devam ettirilerek).

## Gozlem

| Epoch | Val Loss |
|-------|----------|
| 10    | 1.6896   |
| 37    | 1.6368   |
| 38    | 1.5288   |
| **42**| **1.4896 (EN IYI)** |
| 43    | 1.5534   |
| 44    | 1.5508   |
| 45    | 1.5351   |
| 46    | 1.5154   |
| 47    | 1.5044   |

Epoch 42'den sonra val loss hicbir zaman 1.4896 seviyesine geri
donmedi - epoch 43-47 arasi gorunen "iyilesme" (1.5534 -> 1.5044),
her oturumun sonunda cosine LR schedule'inin sifira inmesinin dogal
etkisi, gercek genelleme iyilesmesi degil. Bu, v3'teki (epoch 30)
ayni desenin v4'te (cok daha buyuk veri/model ile) epoch 42'de
tekrarlanmasi.

## Karar

v4'un NIHAI/gecerli hali olarak **epoch 42 checkpoint'i**
(`model_v4_best.pt`, val_loss=1.4896) kabul edildi.

## Karsilastirma (tum versiyonlar)

| Versiyon | Veri | Parametre | Donus epoch'u | En iyi val_loss |
|----------|------|-----------|----------------|------------------|
| v1 | 1.374 | 14.7M | 13 | 4.26 |
| v2 | 25.374 (karisik dil) | 14.7M | ~6 (net degil) | 2.76 |
| v3 | 22.800 (temiz) | 17.0M | 30 | 2.42 |
| v4 | 285.000 (temiz) | 41.8M | 42 | **1.49** |

## Sonuclar

- (+) v3 -> v4 arasi val_loss %38 iyilesti (2.42 -> 1.49) - veri ve
  model buyutme kararinin (ADR-0011) somut basarisi.
- (+) Donus noktasinin epoch 13 (v1) -> ~30 (v3) -> 42 (v4) seklinde
  ilerlemesi, ADR-0003'teki "daha fazla veri, ezberlemeyi geciktirir"
  prensibini bir kez daha dogruladi.
- Bir sonraki adim: uretim testi (test_generate.py, model_v4_best.pt
  ile) - gercek kod uretim kalitesini degerlendirmek. Sonrasinda v5
  (daha fazla veri/model) degerlendirilecek.
