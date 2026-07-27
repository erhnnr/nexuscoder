0006 - Egitim Verisinin Sadece Ingilizce/Kod Kaynaklarina Daraltilmasi
Durum: Kabul edildi
Baglam
v2 uretim testinde (bkz. model_v2_best, val_loss=2.76) modelin
uretimlerinde tuhaf kelime bolunmeleri gozlemlendi: `factor ial`,
`calcul ate`, `f on k si y on` gibi. Bu, kod yazma kalitesini
dogrudan dusuren bir sorun.
Kok neden analizi: veri seti 5 farkli kaynaktan olusuyordu -
kullanicinin eski, karisik Turkce/Ingilizce icerikli dosyalari
(raw_1.jsonl, raw_2.jsonl, ~1374 ornek) ile GitHub'dan cekilen temiz,
Ingilizce Python/JS/C kodu (raw_stack_*.jsonl, 24.000 ornek) ayni
tokenizer'da birlikte egitiliyordu. Kod, dogasi geregi neredeyse
tamamen Ingilizce anahtar kelime ve isimlendirme kullanir
(`def`, `return`, `calculate`, `factorial`). Karisik dil verisi,
BPE tokenizer'in sinirli vocab budcesini (10.000) gereksiz yere
Turkce kelime parcalarina da ayirmaya zorluyordu - bu da Ingilizce
kod kelimelerinin tam/butun token olarak ogrenilememesine, parcalara
bolunmesine yol acti.
Karar
`prepare_data.py`, sadece `raw_stack_*.jsonl` (temiz, Ingilizce,
GitHub kaynakli) dosyalarini okuyacak sekilde guncellendi. Eski
`raw_1/raw_2.jsonl` egitimden CIKARILDI (silinmedi, sadece
`prepare_data.py`'nin okudugu desenin disinda birakildi).
`train_tokenizer.py`, artik data/ klasorundeki TUM `*.jsonl`
dosyalarini degil, SADECE `prepare_data.py`'nin urettigi temiz
`train.jsonl` dosyasini okuyacak sekilde guncellendi - boylece
ham/karisik veri tokenizer'a hicbir sekilde sizamaz.
Vocab boyutu 10.000 -> 16.000'e cikarildi. Artik tum vocab
budcesi tek bir dile (Ingilizce) ve dar bir alana (kod) ayrildigi
icin, daha buyuk vocab'in daha verimli kullanilmasi beklenir.
Sonuclar
(+) Tokenizer'in ogrendigi kelime parcalari artik kod-ozel
Ingilizce kaliplara odaklanacak.
(+) Karar ve pipeline degisikligi ADR olarak belgelendi - ayni
hata (karisik veri kaynagi) gelecekte fark edilmeden tekrar
girmeyecek.
(-) Egitim verisi 25.374 -> 24.000 ornege kucularak azaldi (~%5),
ama kalan verinin kalitesi/tutarliligi arttigi icin bu kabul
edilebilir bir odun.
Bu degisiklikle model v3 olarak SIFIRDAN egitilecek (tokenizer
vocab'i degistigi icin onceki checkpoint'lerle agirlik uyumu
kalmadi - bkz. "mimari evrimi nasil isler" tartismasi: vocab
degisikligi = yeniden egitim gerektirir).
