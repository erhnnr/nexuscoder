# 0011 - v4: Veri ve Model Buyutme, Checkpoint Isimlendirme Temizligi

**Durum:** Kabul edildi

## Baglam

v3'un 110-epoch egitimi (ADR-0010, epoch 30'da val_loss=2.4208 ile
en iyi sonuc) FIILEN KAYBOLDU - ayni Google hesabi/Drive'da
calisildigi dogrulanmis olmasina ragmen, ilgili checkpoint'ler
(`model_v3_latest.pt` dahil) Drive'da bulunamadi. Kucuk bir test
egitimi (2 epoch, kasitli kesilen), kaydetme mekanizmasinin (ADR-0009)
DOGRU calistigini kanitladi - `model_v3_latest.pt` beklendigi gibi
olustu. Bu, kayip egitimin bu ortamda/hesapta hic calismadigini
(farkli bir runtime/sekmede kalip kaybolmus olabilecegini)
gosteriyor - kesin sebep belirsiz kaldi.

Ayni test sirasinda IKINCI bir bug ortaya cikti: `RESUME_CHECKPOINT_
PATH` (latest) bulunamadiginda, script `best_val_loss`'u sifirdan
`infinity` olarak baslatiyor ve var olan (baska bir kaynaktan kalma)
`model_v3_best.pt` dosyasinin UZERINE kontrolsuzce yaziyor - bu,
o dosyanin daha da once bozulmus olan icerigini fark edilmeden
degistirdi.

## Karar

Kayip egitimi ayni sartlarla tekrarlamak yerine (ki zaten epoch
30'daki tavan zaten ADR-0010'da veriyle kanitlanmisti), dogrudan v4'e
gecildi - COK VERI + BUYUK MODEL kombinasyonuyla, TEK bir yeni egitim
kosusuyla:

1. **Veri buyutuldu**: `fetch_dataset.py`'de `SAMPLES_PER_LANGUAGE`
   8.000 -> 30.000 (~4x, toplam ~90.000 ham ornek hedefleniyor).
2. **Model buyutuldu**: `dim` 384->512, `num_layers` 6->8,
   `num_heads` 6->8 (~17M -> ~35-40M parametre, ADR-0003 prensibi:
   veri arttikca kapasite de artmali).
3. **Tum dosya adlari v4'e tasindi**: `tokenizer_v4.json`,
   `model_v4_best.pt`, `model_v4_latest.pt`,
   `model_v4_epoch_{N}.pt`. Bu, ESKI v3 dosyalariyla (bozuk/karisik
   durumdaki) hicbir çakisma olmamasini saglar - v4, v3'un
   checkpoint'lerinden TAMAMEN bagimsiz, temiz bir baslangic.
4. Best/latest baslatma bug'i icin: v4 SIFIRDAN basladigi ve eski
   dosyalarla isim çakismasi olmadigi icin bu spesifik kosuda risk
   yok. Ancak GELECEKTE ayni sorunun tekrarlamamasi icin not: resume
   mantigi ileride "latest yoksa ama best varsa, best'in val_loss'unu
   baslangic degeri olarak al" seklinde saglamlastirilmali (bu
   iyilestirme suana kadar UYGULANMADI, gelecekte ele alinmali).

## Sonuclar

- (+) v3'un kayip egitimini tekrarlamak icin harcanacak GPU suresi,
  dogrudan v4'e yatirilarak verimli kullanildi.
- (+) v4 checkpoint'leri, v3'un karisik/bozuk gecmisinden tamamen
  izole - hata ayiklama ve ilerleme takibi kolaylasti.
- (-) v3'un epoch 30 sonucunun (val_loss=2.4208) somut checkpoint'i
  kalici olarak kayboldu - sadece ADR-0010'daki log kaydi/tablo
  olarak tarihsel referans kaldi.
- Acik risk: best/latest baslatma bug'i tam olarak duzeltilmedi,
  sadece bu kosu icin isim degisikligiyle etkisiz hale getirildi.
  v5+ icin backlog maddesi.
