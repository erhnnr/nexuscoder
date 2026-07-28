# 0010 - v3 Sonucu: Overfitting Donus Noktasi Tespiti (Epoch 30)

**Durum:** Kabul edildi

## Baglam

ADR-0009'daki LR/checkpoint duzeltmeleri sonrasi, v3 22.800 ornek ve
17M parametre ile 110 epoch'a kadar (kesintisiz, RTX PRO 6000 GPU ile)
egitildi. Amac, modelin bu veri/kapasite kombinasyonuyla ulasabilecegi
gercek sinirini bulmakti.

## Gozlem

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 20    | 2.155     | 2.4403   |
| **30**| **1.937** | **2.4208 (EN IYI)** |
| 40    | 1.803     | 2.4383   |
| 60    | 1.623     | 2.5051   |
| 80    | 1.505     | 2.5655   |
| 110   | 1.435     | 2.5974   |

Desen net: train loss kesintisiz dusmeye devam etti (1.94 -> 1.44),
ama val loss epoch 30'dan sonra SUREKLI ve DUZENLI yukseldi. Bu,
ders-kitabi ornegi bir overfitting egrisi - epoch 30'dan sonraki her
ek egitim, modelin GENELLEME kabiliyetini degil, sadece EZBERINI
artirdi.

## Karar

v3'un NIHAI/gecerli hali olarak **epoch 30 checkpoint'i**
(`model_v3_best.pt`, val_loss=2.4208) kabul edildi. 30'dan sonraki
egitim (80 ek epoch) bosa harcanmis hesaplama olarak kayda gecirildi -
ama zararli degildi, cunku `model_v3_best.pt` bu sure boyunca hep
epoch 30'da sabit kaldi (ADR-0009'daki latest/best ayrimi sayesinde).

Sonuc: v3, elindeki 22.800 orneklik veri ve 17M parametrelik kapasite
ile ulasabilecegi sinira ULASTI. Daha fazla epoch, daha fazla veri
OLMADAN, kaliteyi artirmiyor.

## Karsilastirma (versiyonlar arasi)

| Versiyon | Veri | Parametre | En iyi val_loss |
|----------|------|-----------|------------------|
| v1 | 1.374 | 14.7M | 4.26 |
| v2 | 25.374 (karisik dil) | 14.7M | 2.76 |
| v3 | 22.800 (temiz Ingilizce) | 17.0M | **2.42** |

## Sonuclar

- (+) ADR-0003'teki "model boyutu ve veri miktari birlikte
  buyumeli" prensibi somut veriyle DOGRULANDI: v3, sinirina epoch
  30'da ulasti - bu, v4'te hem VERI hem PARAMETRE artisinin ayni
  anda gerekli oldugunu gosteriyor.
- (+) v4 icin somut, veriyle desteklenmis bir hedef var: daha fazla
  veri (SAMPLES_PER_LANGUAGE artirilarak) + daha fazla parametre
  (dim/num_layers buyutulerek).
- Bir sonraki adim: v4 planlamasi (ayri ADR ile belgelenecek).
