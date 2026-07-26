# 0005 - Dil Kapsaminin Python/JavaScript/C ile Sinirlandirilmasi

**Durum:** Kabul edildi

## Baglam

Proje hedefi genel amacli bir dil modeli degil, ozel olarak KOD YAZAN
bir model. Genel amacli LLM'ler dogal dil (Wikipedia, haberler,
kitaplar) ve onlarca programlama dili ile egitilir; bizim
kaynaklarimizla (tek T4 GPU, kucuk model) bu genislikte bir kapsam,
sinirli parametre kapasitesini gereksiz yere boler ve hicbir alanda
yeterli derinlik saglayamaz.

## Karar

Veri seti ve tokenizer sadece uc dile odaklandi: Python (genel amacli
+ veri bilimi), JavaScript (web gelistirme), C (sistem programlama /
temel algoritmalar). Bu, kullanicinin ("gelecekte web sitesi/uygulama
yazabilen bir kod LLM") hedefiyle de uyumlu bir kapsam daraltmasi.

## Sonuclar

- (+) Ayni parametre butcesiyle (14.7M), daha genis bir dil
  yelpazesine kiyasla her dilde daha fazla ornek gorme sansi var.
- (+) Tokenizer, kod-ozel kaliplara (girinti, parantez, `def`/
  `function` gibi anahtar kelimeler) daha fazla vocab kapasitesi
  ayirabiliyor.
- (-) HTML/CSS gibi web'in diger parcalari kapsam disinda kaldi -
  "web sitesi tasarlama" hedefine tam ulasmak icin ileride bu
  kapsamin genisletilmesi gerekebilir.
