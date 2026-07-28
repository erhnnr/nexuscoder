# 0008 - Egitime Devam Etme (Resume) Ozelligi

**Durum:** Kabul edildi

## Baglam

Kullanicinin gunluk zamani/GPU erisimi kisitli (Colab ucretsiz GPU
kotasi ongorulemez sekilde doluyor, bazen saatlerce beklemek
gerekiyor). Onceki yaklasim, her egitim kosusunu SIFIRDAN baslatiyordu
- bu, kisitli zamanli bir kullanici icin surdurulebilir degil: her
oturumda sadece birkac epoch tamamlanabiliyorsa, ilerleme hic
birikmiyordu.

## Karar

`train.py`'a otomatik RESUME (devam etme) mantigi eklendi:

1. Script baslarken `nexus_checkpoints/model_v3_best.pt` var mi kontrol
   eder.
2. Varsa VE o checkpoint'in `vocab_size`/`model_config` degerleri
   mevcut ayarlarla BIREBIR uyusuyorsa (ADR-0007'deki tokenizer
   uyumsuzlugu hatasinin bir daha yasanmamasi icin guvenlik kontrolu),
   model + optimizer durumu yuklenir ve kumulatif epoch sayaci
   (`start_epoch`) o checkpoint'in kaldigi yerden devam eder.
3. Uyusmuyorsa VEYA hic checkpoint yoksa, SIFIRDAN baslar (sessizce
   yanlis/uyumsuz agirlik yuklemek yerine acikca uyari basar).
4. `NUM_EPOCHS` artik "toplam" degil, "BU OTURUMDA kac epoch
   kosulacak" anlamina geliyor (varsayilan: 5, kisa/hizli oturumlar
   icin). Kullanici zaman buldukca scripti tekrar tekrar calistirarak
   kumulatif olarak ilerleyebilir.

## Sonuclar

- (+) Kullanici artik "3-5 epoch calistir, GPU/zaman bulunca devam et"
  seklinde parca parca ilerleyebilir - ilerleme kaybolmaz.
- (+) ADR-0007'deki tokenizer uyumsuzlugu riski otomatik kontrol
  ediliyor - resume sirasinda sessizce yanlis agirlik yuklenmez.
- (-) Her oturumda learning-rate scheduler'i (`CosineAnnealingLR`)
  SIFIRDAN (`T_max=NUM_EPOCHS`, o oturumun epoch sayisi) baslatiliyor -
  bu, tek bir uzun kosuya kiyasla learning rate egrisinin ideal
  olmamasina yol acabilir (her oturum sonunda LR sifira yakin bir
  yere iner, sonraki oturum yine yuksek LR'den baslar). Kucuk bir
  verimlilik kaybi ama pratiklik icin kabul edilebilir bir odun -
  ileride "toplam hedef epoch sayisini kaydet ve scheduler'i ona gore
  kur" seklinde iyilestirilebilir.
