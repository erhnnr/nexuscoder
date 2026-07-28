# 0009 - Oturumlar Arasi LR Sifirlanmasi ve Latest/Best Checkpoint Ayrimi

**Durum:** Kabul edildi

## Baglam

Kullanici Colab Pro sonrasi guclu bir GPU (RTX PRO 6000) ile v3
egitimine devam ettiginde, epoch 59-63 arasinda BEKLENMEDIK bir durum
gozlemlendi: val_loss BEŞ epoch boyunca birebir ayni degerde kaldi
(3.1416, hic degismedi) ve loglarda `LR: 0.000000` yaziyordu.

Kok neden analizi iki BIRBIRINE BAGLI bug ortaya cikardi:

1. **LR sifirlanma bug'i**: Her yeni oturumda `optimizer_state_dict`
   checkpoint'ten yukleniyordu (dogru - momentum bilgisi korunmali).
   Ancak optimizer'in TASIDIGI learning rate degeri de checkpoint'te
   sakliydi - ve onceki oturumun `CosineAnnealingLR` scheduler'i, o
   oturumun sonunda LR'yi ~0'a dusurmustu. Yeni oturumda YENI bir
   `CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)` olusturuluyordu,
   ancak bu scheduler optimizer'in O ANKI (zaten ~0 olan) LR'sini
   "taban" alarak sifirdan sifira iniyordu - yani her yeni oturumda
   GERCEK OGRENME NEREDEYSE HIC OLMUYORDU. Bu, ADR-0008'de "kucuk bir
   verimlilik kaybi" olarak ongorulen riskin, aslinda cok daha ciddi
   (ogrenmeyi TAMAMEN durduran) bir bug oldugunu gosterdi.

2. **Resume/best checkpoint karisikligi**: `RESUME_CHECKPOINT_PATH`,
   `model_v3_best.pt` dosyasini okuyordu - bu dosya SADECE val_loss
   iyilestiginde guncelleniyordu. Bug 1 yuzunden val_loss hicbir zaman
   iyilesmedigi icin, `model_v3_best.pt` epoch 58'de DONMUS durumda
   kaldi. Sonuc: her yeni oturum, hep ayni epoch 58 checkpoint'ini
   yukleyip 5 epoch (bosuna) calisiyor, ama "en iyi" hic
   guncellenmedigi icin BIR SONRAKI oturum yine 58'den basliyordu -
   kullanicinin "63'te kaliyoruz ama yine 59'dan basliyor" gozlemi
   tam olarak buydu.

## Karar

1. **LR bilincli sifirlama**: Resume sirasinda optimizer state
   yuklendikten HEMEN SONRA, her `param_group["lr"]` degeri config'teki
   sabit `LR` degerine (3e-4) MANUEL olarak sifirlaniyor. Boylece her
   yeni oturum, o oturumun kendi `NUM_EPOCHS`'u icin dogru bir cosine
   decay egrisiyle basliyor.
2. **Latest/Best ayrimi**: Iki AYRI checkpoint kavrami tanimlandi:
   - `model_v3_latest.pt`: HER epoch sonunda, val_loss iyilesmis olsun
     olmasin, MUTLAKA guncellenir. RESUME SADECE buradan okur.
   - `model_v3_best.pt`: SADECE val_loss gercekten iyilestiginde
     guncellenir. Uretim testi / degerlendirme icin kullanilir, resume
     icin KULLANILMAZ.

## Sonuclar

- (+) Ilerleme artik val_loss'un iyilesip iyilesmedigine BAKMAKSIZIN
  her oturumda kaydediliyor - resume mekanizmasi asla "takilip
  kalmiyor".
- (+) Her oturum artik GERCEK bir LR decay egrisiyle basliyor,
  onceki oturumun sifira inmis LR'sinden etkilenmiyor.
- (-) v3'un epoch 59-63 arasi (belki daha da oncesi - tam ne zaman
  basladigi belirsiz) egitimi FIILEN BOŞA GITMIS olabilir (LR≈0
  oldugu icin gercek ogrenme olmadi). Model, muhtemelen epoch 58
  civarindaki (LR bug'inin baslamis olabilecegi bir onceki gercek
  ilerleme noktasi) kalitesinde kalmis olabilir.
- Genel ders: bir egitim surecini "kesintili/oturumlu" (resumable)
  hale getirirken, sadece model agirliklari ve optimizer durumu degil,
  learning rate scheduler'in durumu/niyeti de acikca dusunulup
  yonetilmeli - aksi halde sessiz, fark edilmesi zor bir "sifir
  ogrenme" durumu ortaya cikabilir.
